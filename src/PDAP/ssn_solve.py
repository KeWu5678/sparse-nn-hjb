"""Trainer-side SSN outer-weight solve, shared by every model.

A model is linear in its outer parameters ``theta`` for this solve. ``theta`` is
just the model's trainable ``nn.Module`` parameters, read and written with torch's
``parameters_to_vector`` / ``vector_to_parameters``; the model only has to supply
the feature maps (``jacobians`` -> ``(Phi_v, Phi_g)``) and the penalized /
nonnegative coordinate masks (``penalty_masks``). The data Hessian is the
Gauss-Newton form ``(1/M)(w1 Phi_v'Phi_v + w2 Phi_g'Phi_g)``; the closure is the
data loss on ``Phi @ theta`` plus the nonconvex penalty on the penalized block.
:class:`src.SSN.SSN` owns the semismooth-Newton step.

SSN hyperparameters (alpha, gamma, th, power, lr, method, line-search/trust-region
tolerances) are read from the model, where the config places them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..SSN import SSN
from ..SSN.penalty import _phi
from .moment import atom_normalizer

if TYPE_CHECKING:
    from ..models.base import PDAPModel

logger = logging.getLogger(__name__)

ALGORITHM2_COEFFICIENT_SOLVER = "global_prox_warmstart_scale"
# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


@dataclass(frozen=True)
class Objective:
    """What is minimized: data fidelity (loss_weights) + the nonconvex penalty.

    The penalty exponent q is *not* here -- it is q = 2/(power+1), derived from
    the model's activation power (the prox closed-forms depend on it), so it lives
    on the model.

    ``normalized`` is an internal model-family property, not a configurable
    objective choice.  It is true only for nonhomogeneous Algorithm 1, whose
    objective ``l^M + alpha * sum phi(w_p(omega_n)|c_n|)`` is realized by the
    substitution ``u = w_p c``.
    """

    alpha: float = 1e-5
    gamma: float = 0.0
    th: float = 0.5
    loss_weights: Tuple[float, float] = (1.0, 1.0)
    moment_order: float = 2.0
    normalized: bool = False


@dataclass(frozen=True)
class SolverConfig:
    """How the SSN outer solve is run (globalization + line-search tolerances)."""

    lr: float = 1.0
    method: str = "levenberg_marquardt"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0


def warmstart_prox_scale(
    coefficients: torch.Tensor,
    penalized: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    q: float,
    rho: float,
) -> float:
    """Choose the fractional proximal scale from the current warm start.

    The warm-start condition is an upper bound.  When the existing
    ``alpha/(1+alpha*gamma)`` scale is already smaller, it is retained.  Zeros
    and unpenalized coordinates do not constrain the bound.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"warm-start proximal scaling requires 0 < q < 1, got {q}")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    magnitudes = coefficients[penalized].abs()
    nonzero = magnitudes[magnitudes > 0]
    if nonzero.numel() == 0:
        raise ValueError("warm-start proximal scaling requires a nonzero coefficient")

    minimum = float(nonzero.min())
    warmstart_bound = rho * minimum ** (2.0 - q) / (2.0 * (1.0 - q))
    existing_scale = alpha / (1.0 + alpha * gamma)
    return min(existing_scale, warmstart_bound)


def nonconvex_penalty(
    theta: torch.Tensor, penalized: torch.Tensor, nonneg: torch.Tensor,
    *, alpha: float, th: float, gamma: float, q: float,
) -> torch.Tensor:
    """The regularizer alpha * sum_i phi(arg_i) over the penalized coordinates.

    arg = base^q with base = |theta| on free-sign coords and clamp(theta, 0) on
    nonnegative ones.  This is the trainer's objective term -- shared by the SSN
    closure and the loss recording so they cannot drift apart.
    """
    pen = theta[penalized]
    if pen.numel() == 0:
        return theta.new_zeros(())
    base = torch.where(nonneg[penalized], pen.clamp_min(0.0), pen.abs())
    arg = (
        base
        if q == 1.0
        else torch.where(
            base > 0,
            base.clamp_min(1e-30) ** q,
            torch.zeros_like(base),
        )
    )
    return alpha * torch.sum(_phi(arg, th, gamma))


def ssn_solve(
    model: "PDAPModel", data_train, objective: Objective, solver: SolverConfig,
    *, iterations: int, verbose: bool = False,
) -> float:
    """Solve for the model's outer weights in place; return the final train loss."""
    X, V, dV = data_train
    Phi_v, Phi_g = model.jacobians(X)
    Phi_v = Phi_v.detach()
    Phi_g = Phi_g.detach()
    Vt = V.reshape(-1).detach()
    dVt = dV.reshape(-1).detach()
    Nx = X.shape[0]  # M, the sample count: the empirical fidelity is 1/(2M) * sum_m
    w1, w2 = objective.loss_weights
    alpha, gamma, th, q = objective.alpha, objective.gamma, objective.th, model.q

    H = (w1 / Nx) * (Phi_v.T @ Phi_v) + (w2 / Nx) * (Phi_g.T @ Phi_g)

    # theta is the model's trainable parameters flattened (output weights for the
    # signed net; [c | C | a | b0] for the semiconcave model).  SSN solves the
    # linear-in-theta subproblem on a standalone copy, then writes it back.
    params = [p for p in model.parameters() if p.requires_grad]
    theta = torch.nn.Parameter(parameters_to_vector(params).detach().clone())
    penalized, nonneg = model.penalty_masks()

    # Normalized objective: solve in u = w_p * c over the dictionary K_p = K/w_p,
    # so the penalty seen below is the ordinary phi(|u|) and none of the SSN math
    # changes.  PDAP restricts this axis to the signed model, whose theta is
    # exactly the per-atom outer weight, so the column scaling aligns with theta.
    scale = None
    if objective.normalized:
        W, b, _ = model.get_atoms()
        scale = atom_normalizer(W, b, p=objective.moment_order)
        if scale.numel() != theta.numel():
            raise RuntimeError(
                "normalized Algorithm 1 expects one trainable coordinate per atom "
                f"(got {theta.numel()} for {scale.numel()} atoms)"
            )
        Phi_v = Phi_v / scale
        Phi_g = Phi_g / scale
        H = (w1 / Nx) * (Phi_v.T @ Phi_v) + (w2 / Nx) * (Phi_g.T @ Phi_g)
        theta = torch.nn.Parameter((theta.detach() * scale).clone())

    prox_scale = None
    initial_nonzero = None
    minimum_nonzero = None
    if q < 1.0:
        prox_scale = warmstart_prox_scale(
            theta.detach(), penalized,
            alpha=alpha, gamma=gamma, q=q, rho=ALGORITHM2_PROX_RHO,
        )
        initial_magnitudes = theta.detach()[penalized].abs()
        initial_active = initial_magnitudes[initial_magnitudes > 0]
        initial_nonzero = int(initial_active.numel())
        minimum_nonzero = float(initial_active.min())

    optimizer = SSN(
        [theta], alpha=alpha, gamma=gamma,
        penalized_mask=penalized, nonneg_mask=nonneg,
        th=th, lr=solver.lr, power=model.power, method=solver.method,
        max_ls_iter=solver.max_ls_iter, tolerance_ls=solver.tolerance_ls,
        tolerance_grad=solver.tolerance_grad, sigmamax=solver.sigmamax,
        prox_scale=prox_scale,
    )
    optimizer.data_hessian = H

    def closure():
        optimizer.zero_grad()
        rv = Phi_v @ theta - Vt
        rg = Phi_g @ theta - dVt
        data = (w1 / (2 * Nx)) * (rv @ rv) + (w2 / (2 * Nx)) * (rg @ rg)
        penalty = nonconvex_penalty(theta, penalized, nonneg, alpha=alpha, th=th, gamma=gamma, q=q)
        return data + penalty

    prev = float(closure().detach())
    failed_steps = 0
    for _ in range(iterations):
        prev = float(optimizer.step(closure).detach())
        failed_steps += int(not optimizer.last_step_success)

    solved = theta.detach() if scale is None else theta.detach() / scale
    vector_to_parameters(solved, params)
    if verbose and q < 1.0:
        final_nonzero = int((theta.detach()[penalized].abs() > 0).sum())
        logger.info(
            "Algorithm 2 correction  q=%.6g  rho=%.3g  min|c|=%.6e  "
            "prox_scale=%.6e  inverse_step=%.6e  nonzero=%d->%d  "
            "failed_steps=%d/%d  train_loss=%.6e",
            q,
            ALGORITHM2_PROX_RHO,
            minimum_nonzero,
            prox_scale,
            alpha / prox_scale,
            initial_nonzero,
            final_nonzero,
            failed_steps,
            iterations,
            prev,
        )
    if verbose:
        logger.debug("Output-weight solve complete  train_loss=%.6e", prev)
    return prev
