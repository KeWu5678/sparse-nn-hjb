"""Complete semismooth Newton coefficient correction for a sparse problem."""

import logging
import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer

from ..problems import Proximal
from .strategies import damped_newton_step, trust_region_step

__all__ = ["SSN", "fractional_prox_scale"]

logger = logging.getLogger(__name__)

# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


class SSN(Optimizer):
    """Semismooth Newton optimizer on the normal map of a proximal splitting.

    ``proximal`` owns the penalty and its proximal maps. SSN owns the normal map
    built on them, autograd, the parameters, iteration control, and globalization state.
    Each ``step()`` runs a complete configured correction on supplied working
    coefficients. The final guard restores their starting values if the result
    worsens the full objective; nonfinite results are always rejected.

    Args:
        params: parameters to optimize (outer weights only).
        proximal: penalty and proximal maps evaluated at a supplied flat vector.
        loss: complete loss callable for this fixed-support coefficient correction.
        hess_obj: Hessian of the fitting term only, excluding the penalty.
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
        proximal: Proximal,
        loss: Callable[[Tensor], Tensor],
        hess_obj: Tensor,
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
        """Use a proximal splitting whose configuration was validated by training setup."""
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
        self.proximal = proximal
        self.loss = loss
        self.hess_obj = hess_obj
        self._params = self.param_groups[0]["params"]
        group = self.param_groups[0]
        if prox_scale is None:
            prox_scale = fractional_prox_scale(
                proximal, parameters_to_vector(self._params).detach(), ALGORITHM2_PROX_RHO,
            )
        group["prox_scale"] = prox_scale
        group["inverse_step"] = self._inverse_step(prox_scale)
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
            self._report(initial_loss, loss)
        return loss

    def _evaluate_current(self) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            u = parameters_to_vector(self._params).detach().requires_grad_(True)
            loss = self.loss(u)
        return u, loss

    def _iteration(self, u: Tensor, loss: Tensor) -> Tensor:
        """Build and globalize one normal-map Newton step."""
        self.last_step_success = True
        group = self.param_groups[0]
        jac_loss, = torch.autograd.grad(loss, u, retain_graph=True)
        jac_obj_augmented = self.jac_obj_augmented(u, jac_loss)
        q = self.prox_preimage(u, jac_obj_augmented)
        normal_q = self.normal_map(q, u, jac_obj_augmented)

        tolerance_grad = group["tolerance_grad"]
        if tolerance_grad > 0.0 and torch.norm(normal_q, p=float("inf")).item() <= tolerance_grad:
            return loss

        jac_prox = self.proximal.jac_prox(q, self._prox_constant(), prox_image=u)
        jac_normal = self.jac_normal(u, jac_prox)
        return self._globalized_step(q, normal_q, jac_normal, jac_prox, loss)

    def jac_obj_augmented(self, u: Tensor, jac_loss: Tensor) -> Tensor:
        """Smooth part of the objective's derivative, from the supplied full derivative at u.

        Retain the original full-gradient subtraction and correction arithmetic,
        including its derivative selection at masked nonsmooth coordinates.
        """
        return (jac_loss - self.proximal.jac_penalty(u)) + self.proximal.jac_correction(u)

    def hess_obj_augmented(self, u: Tensor) -> Tensor:
        """Hessian of the fitting term plus the diagonal Hessian of the correction."""
        return self.proximal.hess_correction(u) + self.hess_obj

    def prox_preimage(self, u: Tensor, jac_obj_augmented: Tensor) -> Tensor:
        """Reconstruct q from a prox-consistent iterate, following MATLAB SSN.m.

        Nonzero penalized entries use the fractional penalty's stationarity
        equation; zero entries use the smooth gradient. The proximal map is the
        identity on unpenalized coordinates, whose preimage remains u.
        """
        penalized, _ = self.proximal.masks(u)
        active = (torch.abs(u) > 0) & penalized
        grad = torch.where(active, -self.proximal.jac_fractional_penalty(u), jac_obj_augmented)
        q = u - (1.0 / self.param_groups[0]["inverse_step"]) * grad
        return torch.where(penalized, q, u)

    def normal_map(self, q: Tensor, u: Tensor, jac_obj_augmented: Tensor) -> Tensor:
        """Evaluate G(q), using u = P(q) and the smooth gradient at u."""
        return self.param_groups[0]["inverse_step"] * (q - u) + jac_obj_augmented

    def jac_normal(self, u: Tensor, jac_prox: Tensor) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return (
            self.param_groups[0]["inverse_step"] * (I - jac_prox)
            + self.hess_obj_augmented(u) @ jac_prox
        )

    def _prox_constant(self) -> float:
        return self.proximal.alpha / self.param_groups[0]["inverse_step"]

    def _inverse_step(self, prox_scale: float | None) -> float:
        """Translate the proximal scale to the normal-map inverse step.

        The log family retains the reference scaling 1 + alpha*gamma. The
        fractional family requires the scale chosen from its warm start.
        """
        proximal = self.proximal
        if proximal.penalty_exponent < 1.0 and prox_scale is None:
            raise ValueError("prox_scale is required when penalty_exponent < 1")
        if prox_scale is not None and (not math.isfinite(prox_scale) or prox_scale <= 0.0):
            raise ValueError(f"prox_scale must be positive and finite, got {prox_scale}")
        if proximal.penalty_exponent < 1.0:
            assert prox_scale is not None
            return proximal.alpha / prox_scale
        return 1.0 + proximal.alpha * proximal.gamma

    def _globalized_step(
        self, q: Tensor, normal_q: Tensor, jac_normal: Tensor, jac_prox: Tensor,
        loss: Tensor,
    ) -> Tensor:
        """Run a strategy, store its state, and commit an accepted trial."""
        group = self.param_groups[0]

        def evaluate_trial(q_trial: Tensor) -> Tensor:
            return self.loss(self.proximal.prox(q_trial, self._prox_constant()))

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
            vector_to_parameters(self.proximal.prox(q_trial, self._prox_constant()), self._params)
            return loss_trial
        return loss

    def _report(self, initial_loss: Tensor, loss: Tensor) -> None:
        group = self.param_groups[0]
        logger.debug(
            "SSN correction  iterations=%d  objective=%.6e->%.6e  accepted=%s  "
            "failed_steps=%d  prox_scale=%s  inverse_step=%.6e",
            self.iterations_run, float(initial_loss.detach()), float(loss.detach()),
            str(self.correction_accepted).lower(), self.failed_steps,
            group["prox_scale"], group["inverse_step"],
        )


def fractional_prox_scale(proximal: Proximal, coefficients: Tensor, rho: float) -> float | None:
    """Fractional proximal scale satisfying the warm-start upper bound.

    Return None for the log family, whose scale is fixed by 1 + alpha*gamma.
    """
    if proximal.penalty_exponent == 1.0:
        return None
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
    alpha = proximal.alpha
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    penalized, _ = proximal.masks(coefficients)
    magnitudes = coefficients[penalized].abs()
    nonzero = magnitudes[magnitudes > 0]
    if nonzero.numel() == 0:
        raise ValueError("warm-start proximal scaling requires a nonzero coefficient")
    minimum = float(nonzero.min())
    exponent = proximal.penalty_exponent
    bound = rho * minimum ** (2.0 - exponent) / (2.0 * (1.0 - exponent))
    return min(alpha / (1.0 + alpha * proximal.gamma), bound)
