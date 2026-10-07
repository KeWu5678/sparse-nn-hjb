"""Semismooth Newton iteration and globalization for a sparse problem."""

from typing import Iterable

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer

from ..problems import PreparedObjective, SparseProblem
from .strategies import damped_newton_step, trust_region_step

__all__ = ["SSN"]


class SSN(Optimizer):
    """Semismooth Newton optimizer consuming an explicit-vector problem.

    ``problem`` owns the objective, proximal operations, and normal-map formulas.
    SSN owns autograd, the parameters, iteration control, and globalization state.
    Call ``step()`` without a closure; trial points are evaluated independently,
    and parameters are written only when a trial is accepted.

    Args:
        params: parameters to optimize (outer weights only).
        problem: sparse objective evaluated at a supplied flat vector.
        prepared: fixed-support calculations for this coefficient correction.
        max_ls_iter: maximum damping retries after the initial trial.
        tolerance_ls: accept damped loss <= tolerance_ls * current loss.
        method: "levenberg_marquardt" (legacy damped-Newton name) or "steihaug_cg".
        sigmamax: maximum trust-region radius.
        tolerance_grad: early-exit tolerance on ||G(q)||_inf; zero disables it.
        prox_scale: fractional proximal scale chosen from the warm start.

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
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        method: str = "levenberg_marquardt",
        sigmamax: float = 10.0,
        tolerance_grad: float = 0.0,
        prox_scale: float | None = None,
    ) -> None:
        methods = ("levenberg_marquardt", "steihaug_cg")
        if method not in methods:
            raise ValueError(f"unknown method {method!r}; expected one of {list(methods)}")
        defaults = {
            "max_ls_iter": max_ls_iter,
            "tolerance_ls": tolerance_ls,
            "method": method,
            "sigmamax": sigmamax,
            "tolerance_grad": tolerance_grad,
            "inverse_step": problem.inverse_step(prox_scale),
        }
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("SSN doesn't support per-parameter options")
        self.problem = problem
        self.prepared = prepared
        self._params = self.param_groups[0]["params"]
        self.last_step_success = True

    @torch.no_grad()
    def step(self) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Build the normal-map system, globalize its step, and return the loss.

        This method uses the supplied problem and prepared calculations and
        always returns a Tensor; it therefore has no closure argument.

        An ordinary rejected trial leaves parameters unchanged and returns the
        original loss with last_step_success=False. Strategy exceptions also
        leave parameters unchanged and propagate to the caller.
        """
        self.last_step_success = True
        group = self.param_groups[0]
        inverse_step = group["inverse_step"]
        with torch.enable_grad():
            u = parameters_to_vector(self._params).detach().requires_grad_(True)
            loss = self.problem.loss(u, self.prepared)
        jac_loss, = torch.autograd.grad(loss, u, retain_graph=True)
        jac_obj_augmented = self.problem.jac_obj_augmented(u, jac_loss)
        q = self.problem.prox_preimage(u, jac_obj_augmented, inverse_step)
        normal_q = self.problem.normal_map(q, u, jac_obj_augmented, inverse_step)

        tolerance_grad = group["tolerance_grad"]
        if tolerance_grad > 0.0 and torch.norm(normal_q, p=float("inf")).item() <= tolerance_grad:
            return loss

        jac_prox = self.problem.jac_prox(q, inverse_step, prox_result=u)
        jac_normal = self.problem.jac_normal(u, jac_prox, inverse_step, self.prepared)
        return self._run_strategy(q, normal_q, jac_normal, jac_prox, loss, inverse_step)

    def _run_strategy(
        self, q: Tensor, normal_q: Tensor, jac_normal: Tensor, jac_prox: Tensor,
        loss: Tensor, inverse_step: float,
    ) -> Tensor:
        """Keep exception handling separate from numerical globalization."""
        try:
            return self._globalized_step(q, normal_q, jac_normal, jac_prox, loss, inverse_step)
        except Exception:
            self.last_step_success = False
            raise

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
