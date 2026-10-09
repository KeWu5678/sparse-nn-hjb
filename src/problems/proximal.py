"""The coefficient penalty and the proximal maps built on it."""

import torch
from torch import Tensor

from .penalty import (
    _noncvx_correction,
    _noncvx_correction_dd,
    _noncvx_penalty,
    _phi,
    jac_noncvx_penalty,
    phi_prox,
)
from .prox import _power_prox, _power_prox_derivative


class Proximal:
    """Own the penalty term alpha*Phi of the objective and its proximal maps.

    The penalty splits into the fractional penalty alpha*sum|u|^q, which the
    proximal map handles exactly, and a smooth correction. For the log family
    q = 1, so the fractional penalty is alpha*||u||_1 and the correction is
    alpha*sum(phi_gamma(|u|) - |u|). For the power family q < 1 and the
    correction is zero. Proximal maps take the proximal constant lambda_:
    prox(v) = argmin_c (c - v)^2/2 + lambda_*|c|^q.

    Optional masks select penalized and nonnegative coordinates of u and must
    match its length. Unpenalized coordinates carry no penalty; nonnegative
    coordinates are clamped at zero by the proximal maps.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        penalty_exponent: float,
        *,
        penalized_mask: Tensor | None = None,
        nonneg_mask: Tensor | None = None,
    ) -> None:
        self._alpha, self._gamma = alpha, gamma
        self._penalty_exponent = penalty_exponent
        self._penalized_mask = penalized_mask
        self._nonneg_mask = nonneg_mask

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def penalty_exponent(self) -> float:
        return self._penalty_exponent

    def sparsity_functional(self, u: Tensor) -> Tensor:
        """Unweighted log/power functional Phi on the supplied penalty coordinates."""
        penalized, nonneg = self.masks(u)
        return _noncvx_penalty(
            u,
            penalized,
            nonneg,
            alpha=1.0,
            gamma=self._gamma,
            penalty_exponent=self._penalty_exponent,
        )

    def penalty(self, u: Tensor) -> Tensor:
        """The penalty term alpha*Phi(u) of the objective."""
        return self._alpha * self.sparsity_functional(u)

    def jac_penalty(self, u: Tensor) -> Tensor:
        """Derivative of the penalty, selecting zero at u = 0 for the power family."""
        return jac_noncvx_penalty(
            torch.abs(u), torch.sign(u), self._alpha_vec(u), self._gamma,
            penalty_exponent=self._penalty_exponent,
        )

    def jac_fractional_penalty(self, u: Tensor) -> Tensor:
        """Derivative of alpha*sum|u|^q where u != 0; the value at u = 0 is unspecified."""
        alpha_vec = self._alpha_vec(u)
        if self._penalty_exponent == 1.0:
            return alpha_vec * torch.sign(u)
        s = torch.abs(u).clamp(min=1e-30)
        return alpha_vec * self._penalty_exponent * s ** (self._penalty_exponent - 1) * torch.sign(u)

    def correction(self, u: Tensor) -> Tensor:
        """Smooth correction: the penalty minus the fractional penalty."""
        if self._penalty_exponent != 1.0:
            return u.new_zeros(())
        penalized, nonneg = self.masks(u)
        magnitudes = torch.where(nonneg[penalized], u[penalized].clamp_min(0), u[penalized].abs())
        return self._alpha * (_phi(magnitudes, self._gamma) - magnitudes).sum()

    def jac_correction(self, u: Tensor) -> Tensor:
        """Derivative of the smooth correction."""
        correction = _noncvx_correction(
            torch.abs(u), torch.sign(u), self._gamma, penalty_exponent=self._penalty_exponent
        )
        return self._alpha_vec(u) * correction

    def hess_correction(self, u: Tensor) -> Tensor:
        """Diagonal Hessian matrix of the smooth correction."""
        correction = _noncvx_correction_dd(
            torch.abs(u), self._gamma, penalty_exponent=self._penalty_exponent
        )
        return torch.diag(self._alpha_vec(u) * correction)

    def prox(self, q: Tensor, lambda_: float) -> Tensor:
        """Proximal map of the fractional penalty, including nonnegative coordinates."""
        out = _power_prox(q, self._penalized_constant(q, lambda_), penalty_exponent=self._penalty_exponent)
        kill = self.masks(q)[1] & (q < 0)
        return torch.where(kill, torch.zeros_like(out), out)

    def jac_prox(self, q: Tensor, lambda_: float, *, prox_image: Tensor) -> Tensor:
        """Selected proximal Jacobian at q; prox_image must be prox(q, lambda_)."""
        jac = _power_prox_derivative(
            q,
            self._penalized_constant(q, lambda_),
            penalty_exponent=self._penalty_exponent,
            prox_result=prox_image,
        )
        kill = self.masks(q)[1] & (q < 0)
        diagonal = torch.where(kill, torch.zeros_like(prox_image), torch.diagonal(jac))
        return torch.diag(diagonal)

    def penalty_prox(self, value: float, lambda_: float) -> float:
        """Selected nonnegative proximal value for the full scalar penalty.

        The log family's closed-form root requires value >= lambda_ to guarantee
        a minimizer; its caller owns that eligibility check. The vector prox
        instead handles only the fractional penalty, as the normal map requires.
        """
        if self._penalty_exponent == 1.0:
            return phi_prox(value, lambda_, gamma=self._gamma)
        return max(self.fractional_prox(value, lambda_), 0.0)

    def fractional_prox(self, value: float, lambda_: float) -> float:
        """Scalar proximal value of the fractional penalty, including its q=1 safeguard.

        Retain scalar scale arithmetic; the masked vector prox has a different
        floating-point evaluation order and must keep its own numerical contract.
        """
        v = torch.tensor([value], dtype=torch.float64)
        return float(_power_prox(v, lambda_, penalty_exponent=self._penalty_exponent)[0])

    def masks(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Penalized and nonnegative coordinate masks for u, defaulting to all and none."""
        return (
            self._mask(self._penalized_mask, u, True, "penalized_mask"),
            self._mask(self._nonneg_mask, u, False, "nonneg_mask"),
        )

    def _alpha_vec(self, u: Tensor) -> Tensor:
        return self._penalized_constant(u, self._alpha)

    def _penalized_constant(self, u: Tensor, value: float) -> Tensor:
        penalized, _ = self.masks(u)
        return torch.where(penalized, torch.full_like(u, float(value)), torch.zeros_like(u))

    @staticmethod
    def _mask(mask: Tensor | None, coordinates: Tensor, default: bool, name: str) -> Tensor:
        if mask is None:
            return torch.full_like(coordinates, default, dtype=torch.bool)
        if mask.shape != coordinates.shape:
            raise ValueError(f"{name} length must match the problem dimension")
        return mask.to(dtype=torch.bool, device=coordinates.device)
