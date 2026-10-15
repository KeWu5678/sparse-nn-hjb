"""Semismooth Newton iterations on the normal map of a proximal splitting."""

import math
from typing import Callable, Iterable, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer

from ..models import Penalty
from .strategies import damped_newton_step, trust_region_step

__all__ = ["SSN"]

# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


class NewtonSystem(NamedTuple):
    """The Newton equation jac_normal @ dq = -normal_q at the preimage q, and P'(q)."""

    q: Tensor
    normal_q: Tensor
    jac_normal: Tensor
    jac_prox: Tensor


class SSN(Optimizer):
    """Semismooth Newton optimizer on the normal map of a proximal splitting.
    Args:
        params: parameters to optimize (outer weights only).
        penalty: the penalty term and its proximal maps, evaluated at a supplied flat vector.
        hess_obj: Hessian of the fitting term only, excluding the penalty.
        max_ls_iter: maximum damping retries after the initial trial.
        tolerance_ls: accept whe objective is at most loss + (tolerance_ls - 1)*|loss|
        method: "levenberg_marquardt" (line-search) or "steihaug_cg".
        sigmamax: maximum trust-region radius.
        tolerance_grad: skip the step when ||G(q)||_inf <= tol_grad. Zero disables this check.
        prox_scale: optional override of the proximal parameter chosen from the warm start.

    q is the proximal preimage, normal_q = G(q), and jac_normal is its selected
    Jacobian. The Newton equation is jac_normal @ dq = -normal_q. tau is the
    proximal step: G(q) = (q - u)/tau + smooth gradient, with u = prox(q) for the
    proximal constant alpha*tau. sigma is the persistent trust-region radius.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        penalty: Penalty,
        hess_obj: Tensor,
        *,
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        method: str = "levenberg_marquardt",
        sigmamax: float = 10.0,
        tolerance_grad: float = 0.0,
        prox_scale: float | None = None,
    ) -> None:
        """Use a proximal splitting whose configuration was validated by training setup."""
        defaults = {
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
        self.penalty = penalty
        self.hess_obj = hess_obj
        self._params = self.param_groups[0]["params"]
        group = self.param_groups[0]
        if prox_scale is None:
            prox_scale = penalty.prox_scale(
                parameters_to_vector(self._params).detach(), ALGORITHM2_PROX_RHO,
            )
        group["prox_scale"] = prox_scale
        group["tau"] = self._tau_from(prox_scale)
        self._last_step_success = True

    def _tau_from(self, prox_scale: float | None) -> float:
        """The penalty's proximal step tau, after checking a supplied scale."""
        if prox_scale is not None and (not math.isfinite(prox_scale) or prox_scale <= 0.0):
            raise ValueError(f"prox_scale must be positive and finite, got {prox_scale}")
        return self.penalty.proximal_step(prox_scale)

    @property
    def last_step_success(self) -> bool:
        """Whether the last step kept a trial or stopped at a stationary point.

        True before the first step; False after a step that raised.
        """
        return self._last_step_success

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Run one globalized Newton iteration and return the objective it keeps.

        The closure is required, as for LBFGS. Each trial is evaluated by writing
        it into the parameters; a rejected trial is undone. An exception restores
        the parameters and propagates with its original cause.
        """
        closure = torch.enable_grad()(closure)
        start = parameters_to_vector(self._params).detach().clone()
        try:
            loss, self._last_step_success = self._newton_iteration(closure, start)
        except Exception:
            vector_to_parameters(start, self._params)
            self._last_step_success = False
            raise
        return loss

    def _newton_iteration(self, closure: Callable[[], Tensor], u: Tensor) -> tuple[Tensor, bool]:
        """Build and globalize one normal-map Newton step from the coefficients u.

        Return the objective it keeps and whether the step succeeded; a stationary
        point succeeds without a trial.
        """
        loss = self.objective(closure)
        q, normal_q = self._normal_map_at(u)
        success = True
        if not self._stationary(normal_q):
            system = self._newton_system(u, q, normal_q)
            loss, success = self._globalize(closure, u, system, loss)
        return loss, success

    def objective(self, closure: Callable[[], Tensor]) -> Tensor:
        """The objective f + alpha*Phi at the parameters; the closure evaluates the fitting term f."""
        fitting = closure()
        with torch.no_grad():
            return fitting + self.penalty.value(parameters_to_vector(self._params))

    def _normal_map_at(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """The preimage q of the coefficients u = P(q) and the normal map G(q)."""
        jac_obj_augmented = self.jac_obj_augmented(u, self._gradient())
        q = self.prox_preimage(u, jac_obj_augmented)
        return q, self.normal_map(q, u, jac_obj_augmented)

    def _gradient(self) -> Tensor:
        """The fitting term's gradient, left on the parameters by the closure's backward()."""
        gradients = [parameter.grad for parameter in self._params]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("the SSN closure must call backward()")
        return parameters_to_vector(gradients)

    def jac_obj_augmented(self, u: Tensor, jac_fitting: Tensor) -> Tensor:
        """Gradient at u of the smooth part: the fitting term plus the penalty's smooth correction."""
        return jac_fitting + self.penalty.jac_correction(u)

    def prox_preimage(self, u: Tensor, jac_obj_augmented: Tensor) -> Tensor:
        """Reconstruct q from a prox-consistent iterate, following MATLAB SSN.m.

        Nonzero entries use the nonsmooth term's stationarity equation;
        zero entries use the smooth gradient.
        """
        active = torch.abs(u) > 0
        grad = torch.where(active, -self.penalty.jac_nonsmooth_term(u), jac_obj_augmented)
        return u - self._tau * grad

    @property
    def _tau(self) -> float:
        """The proximal step tau, fixed at construction; the proximal constant is alpha*tau."""
        return self.param_groups[0]["tau"]

    def normal_map(self, q: Tensor, u: Tensor, jac_obj_augmented: Tensor) -> Tensor:
        """Evaluate G(q), using u = P(q) and the smooth gradient at u."""
        return (q - u) / self._tau + jac_obj_augmented

    def _stationary(self, normal_q: Tensor) -> bool:
        """Whether ||G(q)||_inf is at or below tolerance_grad; zero disables the test."""
        tolerance_grad = self.param_groups[0]["tolerance_grad"]
        return tolerance_grad > 0.0 and (
            torch.norm(normal_q, p=float("inf")).item() <= tolerance_grad
        )

    def _newton_system(self, u: Tensor, q: Tensor, normal_q: Tensor) -> NewtonSystem:
        """Complete the Newton system at q with the selected Jacobians P'(q) and G'(q)."""
        jac_prox = self.penalty.jac_prox(q, self._prox_constant(), prox_image=u)
        return NewtonSystem(q, normal_q, self.jac_normal(u, jac_prox), jac_prox)

    def _prox_constant(self) -> float:
        return self.penalty.alpha * self._tau

    def jac_normal(self, u: Tensor, jac_prox: Tensor) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return (
            (I - jac_prox) / self._tau
            + self.hess_obj_augmented(u) @ jac_prox
        )

    def hess_obj_augmented(self, u: Tensor) -> Tensor:
        """Hessian of the fitting term plus the diagonal Hessian of the correction."""
        return self.penalty.hess_correction(u) + self.hess_obj

    def _globalize(
        self, closure: Callable[[], Tensor], u: Tensor, system: NewtonSystem, loss: Tensor,
    ) -> tuple[Tensor, bool]:
        """Run a strategy, store its state, and keep an accepted trial or undo a rejected one.

        A strategy returns the last trial it evaluated, which the parameters still hold.
        Return the objective kept and whether the trial was accepted.
        """
        group = self.param_groups[0]

        def evaluate_trial(q_trial: Tensor) -> Tensor:
            vector_to_parameters(self.penalty.prox(q_trial, self._prox_constant()), self._params)
            return self.objective(closure)

        if group["method"] == "levenberg_marquardt":
            q_trial, loss_trial, accepted = damped_newton_step(
                system.q, system.normal_q, system.jac_normal, loss, evaluate_trial,
                max_ls_iter=group["max_ls_iter"], tolerance_ls=group["tolerance_ls"],
            )
        else:
            state = self.state[self._params[0]]
            sigma = state.setdefault("sigma", 0.01 * group["sigmamax"])
            trial, state["sigma"] = trust_region_step(
                system.q, system.normal_q, system.jac_normal, system.jac_prox, loss, evaluate_trial,
                sigma=sigma, sigmamax=group["sigmamax"],
            )
            q_trial, loss_trial, accepted = trial

        if accepted:
            loss = loss_trial
        else:
            vector_to_parameters(u, self._params)
        return loss, accepted
