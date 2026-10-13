"""Semismooth Newton iterations on the normal map of a proximal splitting."""

import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer

from ..models import Penalty
from .strategies import damped_newton_step, trust_region_step

__all__ = ["SSN"]

# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


class SSN(Optimizer):
    """Semismooth Newton optimizer on the normal map of a proximal splitting.

    ``penalty`` owns the penalty and its proximal maps. SSN owns the normal map
    built on them, the parameters, and globalization state.
    Each ``step(closure)`` runs one globalized Newton iteration. The closure follows
    the ``torch.optim`` contract: it clears the gradients, evaluates the fitting
    term f at the current parameters, calls ``backward()`` and returns f. SSN adds
    the penalty itself, as AdamW adds its weight decay, and minimizes
    f + alpha*Phi. The caller owns the iteration budget and the acceptance of the
    finished correction.

    Args:
        params: parameters to optimize (outer weights only).
        penalty: the penalty term and its proximal maps, evaluated at a supplied flat vector.
        hess_obj: Hessian of the fitting term only, excluding the penalty.
        max_ls_iter: maximum damping retries after the initial trial.
        tolerance_ls: accept a finite damped trial whose objective is at most
            loss + (tolerance_ls - 1)*|loss|, the current objective loss.
        method: "levenberg_marquardt" (legacy damped-Newton name) or "steihaug_cg".
        sigmamax: maximum trust-region radius.
        tolerance_grad: skip the step's Newton work when ||G(q)||_inf is at or
            below the tolerance. Zero disables this check.
        prox_scale: optional override of the scale chosen from the warm start.

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
        group["tau"] = self._tau(prox_scale)
        self.last_step_success = True

    def _tau(self, prox_scale: float | None) -> float:
        """The penalty's proximal step tau, after checking a supplied scale."""
        if prox_scale is not None and (not math.isfinite(prox_scale) or prox_scale <= 0.0):
            raise ValueError(f"prox_scale must be positive and finite, got {prox_scale}")
        return self.penalty.proximal_step(prox_scale)

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
            return self._iteration(closure, start)
        except Exception:
            vector_to_parameters(start, self._params)
            self.last_step_success = False
            raise

    def _iteration(self, closure: Callable[[], Tensor], u: Tensor) -> Tensor:
        """Build and globalize one normal-map Newton step from the coefficients u."""
        self.last_step_success = True
        group = self.param_groups[0]
        loss = self.objective(closure)
        jac_obj_augmented = self.jac_obj_augmented(u, self._gradient())
        q = self.prox_preimage(u, jac_obj_augmented)
        normal_q = self.normal_map(q, u, jac_obj_augmented)

        tolerance_grad = group["tolerance_grad"]
        if tolerance_grad > 0.0 and torch.norm(normal_q, p=float("inf")).item() <= tolerance_grad:
            return loss

        jac_prox = self.penalty.jac_prox(q, self._prox_constant(), prox_image=u)
        jac_normal = self.jac_normal(u, jac_prox)
        return self._ssn_single_step(closure, u, q, normal_q, jac_normal, jac_prox, loss)

    def objective(self, closure: Callable[[], Tensor]) -> Tensor:
        """The objective f + alpha*Phi at the parameters; the closure evaluates the fitting term f."""
        fitting = closure()
        with torch.no_grad():
            return fitting + self.penalty.value(parameters_to_vector(self._params))

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
        return u - self.param_groups[0]["tau"] * grad

    def normal_map(self, q: Tensor, u: Tensor, jac_obj_augmented: Tensor) -> Tensor:
        """Evaluate G(q), using u = P(q) and the smooth gradient at u."""
        return (q - u) / self.param_groups[0]["tau"] + jac_obj_augmented

    def _prox_constant(self) -> float:
        return self.penalty.alpha * self.param_groups[0]["tau"]

    def jac_normal(self, u: Tensor, jac_prox: Tensor) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return (
            (I - jac_prox) / self.param_groups[0]["tau"]
            + self.hess_obj_augmented(u) @ jac_prox
        )

    def hess_obj_augmented(self, u: Tensor) -> Tensor:
        """Hessian of the fitting term plus the diagonal Hessian of the correction."""
        return self.penalty.hess_correction(u) + self.hess_obj

    def _ssn_single_step(
        self, closure: Callable[[], Tensor], u: Tensor, q: Tensor, normal_q: Tensor,
        jac_normal: Tensor, jac_prox: Tensor, loss: Tensor,
    ) -> Tensor:
        """Run a strategy, store its state, and keep an accepted trial or undo a rejected one.

        A strategy returns the last trial it evaluated, which the parameters still hold.
        """
        group = self.param_groups[0]

        def evaluate_trial(q_trial: Tensor) -> Tensor:
            vector_to_parameters(self.penalty.prox(q_trial, self._prox_constant()), self._params)
            return self.objective(closure)

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
            loss = loss_trial
        else:
            vector_to_parameters(u, self._params)
        return loss
