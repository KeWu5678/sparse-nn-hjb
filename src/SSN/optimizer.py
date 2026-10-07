"""Complete semismooth Newton coefficient correction for a sparse problem."""

import logging
from typing import Iterable

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer

from ..problems import PreparedObjective, SparseProblem
from .strategies import damped_newton_step, trust_region_step

__all__ = ["SSN"]

logger = logging.getLogger(__name__)

# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


class SSN(Optimizer):
    """Semismooth Newton optimizer consuming an explicit-vector problem.

    ``problem`` owns the objective, proximal operations, and normal-map formulas.
    SSN owns autograd, the parameters, iteration control, and globalization state.
    Each ``step()`` runs a complete configured correction on supplied working
    coefficients. The final guard restores their starting values if the result
    worsens the full objective; nonfinite results are always rejected.

    Args:
        params: parameters to optimize (outer weights only).
        problem: sparse objective evaluated at a supplied flat vector.
        prepared: fixed-support calculations for this coefficient correction.
        max_iter: number of Newton iterations in one complete correction.
        correction_guard: reject an increased final loss; False preserves the
            explicitly configured historical unguarded comparison.
        max_ls_iter: maximum damping retries after the initial trial.
        tolerance_ls: accept damped loss <= tolerance_ls * current loss.
        method: "levenberg_marquardt" (legacy damped-Newton name) or "steihaug_cg".
        sigmamax: maximum trust-region radius.
        tolerance_grad: skip this iteration's Newton work when ||G(q)||_inf is
            at or below the tolerance; the correction still runs max_iter iterations.
            Zero disables this check.
        prox_scale: optional override of the scale chosen from the warm start.
        verbose: report correction iterations, acceptance, and diagnostics.

    q is the proximal preimage, normal_q = G(q), and jac_normal is its selected
    Jacobian. The Newton equation is jac_normal @ dq = -normal_q. inverse_step
    scales the normal map; sigma is the persistent trust-region radius.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        problem: SparseProblem,
        prepared: PreparedObjective,
        *,
        max_iter: int = 20,
        correction_guard: bool = True,
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        method: str = "levenberg_marquardt",
        sigmamax: float = 10.0,
        tolerance_grad: float = 0.0,
        prox_scale: float | None = None,
        verbose: bool = False,
    ) -> None:
        methods = ("levenberg_marquardt", "steihaug_cg")
        if method not in methods:
            raise ValueError(f"unknown method {method!r}; expected one of {list(methods)}")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 0:
            raise ValueError("max_iter must be a nonnegative integer")
        defaults = {
            "max_iter": max_iter,
            "correction_guard": correction_guard,
            "max_ls_iter": max_ls_iter,
            "tolerance_ls": tolerance_ls,
            "method": method,
            "sigmamax": sigmamax,
            "tolerance_grad": tolerance_grad,
            "prox_scale": prox_scale,
        }
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("SSN doesn't support per-parameter options")
        self.problem = problem
        self.prepared = prepared
        self._params = self.param_groups[0]["params"]
        group = self.param_groups[0]
        if prox_scale is None:
            prox_scale = problem.warmstart_prox_scale(
                parameters_to_vector(self._params).detach(), ALGORITHM2_PROX_RHO,
            )
        group["prox_scale"] = prox_scale
        group["inverse_step"] = problem.inverse_step(prox_scale)
        self.verbose = verbose
        self.last_step_success = True
        self.iterations_run = 0
        self.failed_steps = 0
        self.correction_accepted = False

    @torch.no_grad()
    def step(self) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Run a complete correction and return the accepted full objective.

        No convergence claim is made after the fixed iteration budget. Ordinary
        rejected trials continue the correction; exceptions restore the starting
        working coefficients and propagate with their original cause.
        """
        start = parameters_to_vector(self._params).detach().clone()
        self.iterations_run = self.failed_steps = 0
        self.correction_accepted = False
        self.last_step_success = True
        try:
            u, initial_loss = self._evaluate_current()
            if not torch.isfinite(start).all() or not torch.isfinite(initial_loss):
                raise ValueError("SSN requires finite starting coefficients and loss")
            loss = initial_loss
            for iteration in range(self.param_groups[0]["max_iter"]):
                if iteration:
                    u, loss = self._evaluate_current()
                loss = self._iteration(u, loss)
                self.iterations_run += 1
                self.failed_steps += int(not self.last_step_success)
            finite = bool(torch.isfinite(loss) & torch.isfinite(parameters_to_vector(self._params)).all())
            self.correction_accepted = finite and (
                not self.param_groups[0]["correction_guard"] or bool(loss <= initial_loss)
            )
        except Exception:
            vector_to_parameters(start, self._params)
            self.last_step_success = False
            raise
        if not self.correction_accepted:
            vector_to_parameters(start, self._params)
            loss = initial_loss
        if self.verbose:
            self._report(start, initial_loss, loss)
        return loss

    def _evaluate_current(self) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            u = parameters_to_vector(self._params).detach().requires_grad_(True)
            loss = self.problem.loss(u, self.prepared)
        return u, loss

    def _iteration(self, u: Tensor, loss: Tensor) -> Tensor:
        """Build and globalize one normal-map Newton step."""
        self.last_step_success = True
        group = self.param_groups[0]
        inverse_step = group["inverse_step"]
        jac_loss, = torch.autograd.grad(loss, u, retain_graph=True)
        jac_obj_augmented = self.problem.jac_obj_augmented(u, jac_loss)
        q = self.problem.prox_preimage(u, jac_obj_augmented, inverse_step)
        normal_q = self.problem.normal_map(q, u, jac_obj_augmented, inverse_step)

        tolerance_grad = group["tolerance_grad"]
        if tolerance_grad > 0.0 and torch.norm(normal_q, p=float("inf")).item() <= tolerance_grad:
            return loss

        jac_prox = self.problem.jac_prox(q, inverse_step, prox_result=u)
        jac_normal = self.problem.jac_normal(u, jac_prox, inverse_step, self.prepared)
        return self._globalized_step(q, normal_q, jac_normal, jac_prox, loss, inverse_step)

    def _globalized_step(
        self, q: Tensor, normal_q: Tensor, jac_normal: Tensor, jac_prox: Tensor,
        loss: Tensor, inverse_step: float,
    ) -> Tensor:
        """Run a strategy, store its state, and commit an accepted trial."""
        group = self.param_groups[0]

        def evaluate_trial(q_trial: Tensor) -> Tensor:
            return self.problem.loss(self.problem.prox(q_trial, inverse_step), self.prepared)

        if group["method"] == "levenberg_marquardt":
            q_trial, loss_trial, accepted = damped_newton_step(
                q, normal_q, jac_normal, loss, evaluate_trial,
                max_ls_iter=group["max_ls_iter"], tolerance_ls=group["tolerance_ls"],
            )
        else:
            state = self.state[self._params[0]]
            sigma = state.setdefault("sigma", 0.01 * group["sigmamax"])
            trial, state["sigma"] = trust_region_step(
                q, normal_q, jac_normal, jac_prox, loss, evaluate_trial,
                sigma=sigma, sigmamax=group["sigmamax"],
            )
            q_trial, loss_trial, accepted = trial

        self.last_step_success = accepted
        if accepted:
            vector_to_parameters(self.problem.prox(q_trial, inverse_step), self._params)
            return loss_trial
        return loss

    def _report(self, start: Tensor, initial_loss: Tensor, loss: Tensor) -> None:
        group = self.param_groups[0]
        if self.problem.penalty_exponent < 1.0:
            initial_active = self.problem.active_magnitudes(start)
            final_active = self.problem.active_magnitudes(parameters_to_vector(self._params))
            logger.info(
                "Algorithm 2 correction  q=%.6g  rho=%.3g  min|c|=%.6e  "
                "prox_scale=%.6e  inverse_step=%.6e  nonzero=%d->%d  "
                "failed_steps=%d/%d  train_loss=%.6e",
                self.problem.penalty_exponent, ALGORITHM2_PROX_RHO,
                float(initial_active.min()) if initial_active.numel() else 0.0,
                group["prox_scale"], group["inverse_step"], int(initial_active.numel()),
                int(final_active.numel()), self.failed_steps, self.iterations_run, float(loss),
            )
        logger.info(
            "SSN correction  iterations=%d  objective=%.6e->%.6e  accepted=%s",
            self.iterations_run, float(initial_loss.detach()), float(loss.detach()),
            str(self.correction_accepted).lower(),
        )
