"""Damped-Newton and MPCG trust-region globalization for SSN.

Strategies receive tensors, numerical settings, and a trial-loss callback.
They return their acceptance decision for the last trial they evaluated; SSN
keeps or undoes that trial and stores the updated trust-region radius. The
callback installs each trial in the parameters before evaluating it.
The input normal_q is the normal-map value G(q) at the current iterate.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor

from .mpcg import MPCG

logger = logging.getLogger(__name__)

# Keeps the damped-Newton reciprocal damping from underflowing to zero.
_MIN_RECIPROCAL_DAMPING = 1e-12
# Relative objective increase a trust-region trial may make and still be accepted.
_TRUST_REGION_TOLERANCE = 1e-10


def damped_newton_step(
    q: Tensor, normal_q: Tensor, jac_normal: Tensor, loss: Tensor,
    evaluate_trial: Callable[[Tensor], Tensor],
    *, max_ls_iter: int, tolerance_ls: float,
) -> tuple[Tensor, Tensor, bool]:
    """Return (q_trial, loss_trial, accepted), increasing the damping on each retry.

    A trial solves (jac_normal + I/reciprocal_damping) dq = -normal_q. The first
    trial is nearly undamped; its step length seeds the reciprocal damping of
    the retries, and each retry divides it by four, down to a floor. A trial is
    accepted when it raises the objective by at most (tolerance_ls - 1)*|loss|.
    """
    I: Tensor = torch.eye(q.numel(), device=q.device, dtype=q.dtype)
    relative_tolerance = tolerance_ls - 1.0

    def trial(reciprocal_damping: float) -> tuple[Tensor, Tensor, Tensor]:
        dq = _solve_damped_system(jac_normal, normal_q, reciprocal_damping, I)
        q_trial = q + dq
        return dq, q_trial, evaluate_trial(q_trial)

    nearly_undamped: float = 1.0 / (1e-12 * torch.norm(jac_normal, p=float("inf")).item())
    dq, q_trial, loss_trial = trial(nearly_undamped)
    normal_q_norm: float = max(torch.norm(normal_q).item(), 1e-20)  # avoid div-by-zero
    reciprocal_damping: float = max(
        torch.norm(dq).item() / normal_q_norm, nearly_undamped, _MIN_RECIPROCAL_DAMPING,
    )
    retries = 0
    while not _trial_acceptable(loss_trial, loss, relative_tolerance) and retries < max_ls_iter:
        _, q_trial, loss_trial = trial(reciprocal_damping)
        reciprocal_damping = max(reciprocal_damping / 4.0, _MIN_RECIPROCAL_DAMPING)
        retries += 1

    accepted = _trial_acceptable(loss_trial, loss, relative_tolerance)
    if not accepted:
        logger.debug("Damping backtracking exhausted after %d retries", retries)
    return q_trial, loss_trial, accepted


def _solve_damped_system(
    jac_normal: Tensor, normal_q: Tensor, reciprocal_damping: float, I: Tensor,
) -> Tensor:
    """Return dq solving (jac_normal + I/reciprocal_damping) dq = -normal_q."""
    return -torch.linalg.solve(jac_normal + (1 / reciprocal_damping) * I, normal_q)


def _trial_acceptable(loss_trial: Tensor, loss: Tensor, relative_tolerance: float) -> bool:
    """Both strategies' rule: a finite trial objective at most loss + relative_tolerance*|loss|."""
    return bool(torch.isfinite(loss_trial) and loss_trial <= loss + relative_tolerance * torch.abs(loss))


def trust_region_step(
    q: Tensor, normal_q: Tensor, jac_normal: Tensor, jac_prox: Tensor, loss: Tensor,
    evaluate_trial: Callable[[Tensor], Tensor],
    *, sigma: float, sigmamax: float,
) -> tuple[tuple[Tensor, Tensor, bool], float]:
    """Return (q_trial, loss_trial, accepted) and the next trust-region radius."""
    I_active: Tensor = torch.diag(jac_prox) != 0
    kmaxit: int = max(1, int(2 * I_active.sum().item()))

    dq, stop_reason = MPCG(jac_normal, jac_prox).solve(-normal_q, rtol=1e-3, maxit=kmaxit, sigma=sigma)

    q_trial = q + dq
    loss_trial = evaluate_trial(q_trial)

    accepted = _trial_acceptable(loss_trial, loss, _TRUST_REGION_TOLERANCE)
    next_sigma = _next_radius(sigma, sigmamax, accepted, stop_reason)
    if not accepted:
        logger.debug(
            "TR reject: dloss=%.3e %.2e->%.2e",
            (loss_trial - loss).item(), sigma, next_sigma,
        )
    return (q_trial, loss_trial, accepted), next_sigma


def _next_radius(sigma: float, sigmamax: float, accepted: bool, stop_reason: str) -> float:
    """Adapt the radius using acceptance and the MPCG truncation reason.

    The CG quadratic models the normal-map Newton system; its decrease does not
    reliably predict the loss decrease. A 'negdef' boundary step retains the
    historical 1.2 growth factor; only 'radius' gets the factor 2.
    """
    if not accepted:
        return 0.2 * sigma
    growth = 2.0 if stop_reason == "radius" else 1.2
    return min(growth * sigma, sigmamax)


__all__ = ["damped_newton_step", "trust_region_step"]
