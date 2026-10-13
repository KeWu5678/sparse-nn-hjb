"""The coefficient penalty and the proximal maps built on it.

log penalty: phi(t) = t/2 + log(1+2*gamma*t)/(4*gamma). At gamma=0, phi(t)=t.
fractional penalty: t^q, with q in {2/3, 1/2} implied by a validated activation power.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Penalty(ABC):
    """The penalty term and its proximal maps.
    """

    def __init__(self, alpha: float, penalty_exponent: float) -> None:
        self.alpha = alpha
        self._penalty_exponent = penalty_exponent

    def value(self, u: Tensor) -> Tensor:
        """The penalty term alpha*Phi(u) of the objective."""
        return self.alpha * self.sparsity_functional(u)

    @abstractmethod
    def sparsity_functional(self, u: Tensor) -> Tensor:
        """Unweighted functional Phi."""

    @abstractmethod
    def jac_nonsmooth_term(self, u: Tensor) -> Tensor:
        """Derivative of alpha*sum|u|^q where u != 0; the value at u = 0 is unspecified."""

    def jac_correction(self, u: Tensor) -> Tensor:
        """Derivative of the smooth correction."""
        return torch.zeros_like(u)

    def hess_correction(self, u: Tensor) -> Tensor:
        """Diagonal Hessian matrix of the smooth correction."""
        return torch.diag(torch.zeros_like(u))

    def prox(self, q: Tensor, lambda_: float) -> Tensor:
        """Proximal map of the nonsmooth term."""
        return self._closed_form_prox(q, torch.full_like(q, float(lambda_)))

    @abstractmethod
    def _closed_form_prox(self, v: Tensor, lambda_) -> Tensor:
        """Global proximal map of lambda_*|.|^q."""

    def jac_prox(self, q: Tensor, lambda_: float, *, prox_image: Tensor) -> Tensor:
        """Selected proximal Jacobian at q; prox_image must be prox(q, lambda_)."""
        constant = torch.full_like(q, float(lambda_))
        return torch.diag(self._prox_derivative(q, constant, prox_image))

    @abstractmethod
    def _prox_derivative(self, q: Tensor, lambda_: Tensor, prox_image: Tensor) -> Tensor:
        """Diagonal of the selected proximal Jacobian."""

    @abstractmethod
    def prox_scale(self, coefficients: Tensor, rho: float) -> float | None:
        """The proximal scale chosen from warm-start coefficients; None if the family fixes it."""

    @abstractmethod
    def proximal_step(self, prox_scale: float | None) -> float:
        """The proximal step tau of the normal map; the proximal constant is alpha*tau."""

    @abstractmethod
    def scalar_penalty(self, c: float) -> float:
        """The penalty alpha*phi(|c|) of one coefficient, in scalar arithmetic."""

    def insertion_weight(self, slope: float, curvature: float) -> float:
        """c* = prox_{(alpha/A)|.|^q}(-P/A), eq. algorithm2-insertion-weight.

        The global minimizer of the one-atom increment c*P + A*c^2/2 + alpha*|c|^q,
        for slope P and curvature A > 0.
        """
        magnitude = self._scalar_prox(abs(slope) / curvature, self.alpha / curvature)
        return magnitude if slope < 0 else -magnitude

    def _scalar_prox(self, value: float, lambda_: float) -> float:
        """Scalar proximal value of the nonsmooth term.

        Retain scalar scale arithmetic; the vector prox has a different
        floating-point evaluation order and must keep its own numerical contract.
        """
        v = torch.tensor([value], dtype=torch.float64)
        return float(self._closed_form_prox(v, lambda_)[0])

    def descent_step(self, slope: float, curvature: float) -> float:
        """The step t >= 0 minimizing slope*t + curvature*t^2/2 + alpha*phi(t).

        Zero unless -slope >= alpha, where the log penalty's closed-form root is
        a minimizer, and the curvature is positive.
        """
        step = 0.0
        if -slope >= self.alpha and curvature > 1e-30:
            step = self.penalty_prox(-slope / curvature, self.alpha / curvature)
        return step

    @abstractmethod
    def penalty_prox(self, value: float, lambda_: float) -> float:
        """Selected nonnegative proximal value for the full scalar penalty.

        The vector prox instead handles only the nonsmooth term, as the normal
        map requires.
        """


class LogPenalty(Penalty):
    """alpha*sum phi_gamma(|u|): the nonsmooth term alpha*||u||_1 plus a smooth correction."""

    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__(alpha, 1.0)
        self._gamma = gamma

    def sparsity_functional(self, u: Tensor) -> Tensor:
        return torch.sum(_phi(torch.abs(u), self._gamma))

    def jac_nonsmooth_term(self, u: Tensor) -> Tensor:
        return self.alpha * torch.sign(u)

    def jac_correction(self, u: Tensor) -> Tensor:
        return self.alpha * (torch.sign(u) * (_dphi(torch.abs(u), self._gamma) - 1.0))

    def hess_correction(self, u: Tensor) -> Tensor:
        return torch.diag(self.alpha * _ddphi(torch.abs(u), self._gamma))

    def _closed_form_prox(self, v: Tensor, lambda_) -> Tensor:
        return _compute_prox_q_one(v, lambda_)

    def _prox_derivative(self, q: Tensor, lambda_: Tensor, prox_image: Tensor) -> Tensor:
        return _compute_prox_derivative_q_one(q, lambda_)

    def prox_scale(self, coefficients: Tensor, rho: float) -> None:
        """None: the log penalty fixes its scale by 1 + alpha*gamma."""
        return None

    def proximal_step(self, prox_scale: float | None) -> float:
        """tau = 1/(1 + alpha*gamma), eq. proximal-residual; prox_scale does not enter."""
        return 1.0 / (1.0 + self.alpha * self._gamma)

    def scalar_penalty(self, c: float) -> float:
        return self.alpha * float(_phi(torch.tensor(abs(c), dtype=torch.float64), self._gamma))

    def penalty_prox(self, value: float, lambda_: float) -> float:
        """The closed-form root, a minimizer when value >= lambda_; its caller checks that."""
        return _phi_prox(value, lambda_, gamma=self._gamma)


class FractionalPenalty(Penalty):
    """alpha*sum|u|^q for q in {1/2, 2/3}; the nonsmooth term is the whole penalty."""

    def sparsity_functional(self, u: Tensor) -> Tensor:
        magnitudes = torch.abs(u)
        powers = torch.where(
            magnitudes > 0,
            magnitudes.clamp_min(1e-30) ** self._penalty_exponent,
            torch.zeros_like(magnitudes),
        )
        return torch.sum(powers)

    def jac_nonsmooth_term(self, u: Tensor) -> Tensor:
        s = torch.abs(u).clamp(min=1e-30)
        return self.alpha * self._penalty_exponent * s ** (self._penalty_exponent - 1) * torch.sign(u)

    def _closed_form_prox(self, v: Tensor, lambda_) -> Tensor:
        """Choose the closed form by the exact q."""
        return {
            0.5: _compute_prox_q_half,
            2.0 / 3.0: _compute_prox_q_twothirds,
        }[self._penalty_exponent](v, lambda_)

    def _prox_derivative(self, q: Tensor, lambda_: Tensor, prox_image: Tensor) -> Tensor:
        return _compute_prox_derivative_fractional(q, lambda_, self._penalty_exponent, prox_image)

    def prox_scale(self, coefficients: Tensor, rho: float) -> float:
        """lambda_t = min(alpha, rho*m^(2-q)/(2(1-q))), eq. algorithm2-prox-scale.

        m is the smallest nonzero magnitude among the warm-start coefficients.
        """
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        magnitudes = coefficients.abs()
        nonzero = magnitudes[magnitudes > 0]
        if nonzero.numel() == 0:
            raise ValueError("warm-start proximal scaling requires a nonzero coefficient")
        minimum = float(nonzero.min())
        exponent = self._penalty_exponent
        bound = rho * minimum ** (2.0 - exponent) / (2.0 * (1.0 - exponent))
        return min(self.alpha, bound)

    def proximal_step(self, prox_scale: float | None) -> float:
        """tau = lambda_t/alpha, eq. algorithm2-normal-map."""
        if prox_scale is None:
            raise ValueError("the fractional penalty requires prox_scale")
        return prox_scale / self.alpha

    def scalar_penalty(self, c: float) -> float:
        """The exact power, without the magnitude floor needed to differentiate the vector penalty."""
        return self.alpha * abs(c) ** self._penalty_exponent

    def penalty_prox(self, value: float, lambda_: float) -> float:
        return max(self._scalar_prox(value, lambda_), 0.0)


# The log penalty phi_gamma, soft thresholding, and the nonnegative step of lambda*phi


def _phi(t, gamma):
    """Fixed half-mixture log penalty on nonnegative magnitudes t."""
    if gamma == 0:
        return t
    gamma_bar = 2.0 * gamma
    return 0.5 * t + 0.5 * torch.log(1 + gamma_bar * t) / gamma_bar


def _dphi(t, gamma):
    """First derivative of phi with respect to its magnitude argument."""
    if gamma == 0:
        return torch.ones_like(t)
    gamma_bar = 2.0 * gamma
    return 0.5 + 0.5 / (1 + gamma_bar * t)


def _ddphi(t, gamma):
    """Second derivative of phi with respect to its magnitude argument."""
    if gamma == 0:
        return torch.zeros_like(t)
    gamma_bar = 2.0 * gamma
    return -0.5 * gamma_bar / ((1 + gamma_bar * t) ** 2)


def _compute_prox_q_one(v, lambda_):
    """Soft thresholding for ``lambda*|.|``."""
    abs_v = torch.abs(v)
    eps = torch.finfo(v.dtype).eps
    abs_v_safe = torch.clamp(abs_v, min=(lambda_ + eps) * eps)
    shrinkage_factor = torch.clamp(1 - lambda_ / abs_v_safe, min=0)
    return shrinkage_factor * v


def _compute_prox_derivative_q_one(v, lambda_):
    """Diagonal generalized Jacobian of soft thresholding: 1 where |v| >= lambda, else 0."""
    abs_v = torch.abs(v)
    eps = torch.finfo(v.dtype).eps
    abs_v_safe = torch.clamp(abs_v, min=(lambda_ + eps) * eps)
    # MATLAB computeDProx.m with N=1:
    #   max(0, 1 - lambda/|v|) + (|v|>=lambda) * lambda/|v|^3 * v^2
    # Preserve this arithmetic for SSN regression compatibility: the reciprocal
    # powers and safeguard can affect floating-point results. An indicator
    # replacement must preserve the >= threshold convention and the identity
    # derivative at lambda_=0 (including v=0), and be tested separately.
    diagonal_term = torch.clamp(1 - lambda_ / abs_v_safe, min=0)
    mask = abs_v >= lambda_
    outer_product_term = mask.to(dtype=v.dtype) * lambda_ / (abs_v_safe ** 3) * (v ** 2)
    return diagonal_term + outer_product_term


def _phi_prox(v: float, lambda_: float, gamma: float) -> float:
    """MATLAB's closed-form
    """
    if gamma == 0:
        return max(v - lambda_, 0.0)

    gamma_bar = 2.0 * gamma
    a = v - lambda_ * 0.5 - 1.0 / gamma_bar
    delta = a * a + 4.0 * (v - lambda_) / gamma_bar
    if delta < 0:
        return 0.0

    t = 0.5 * (a + math.sqrt(delta))
    return max(t, 0.0)


# Closed-form proximal maps of lambda*|.|^q for q in {1/2, 2/3}, and their derivative


def _compute_prox_q_half(v, lambda_):
    """Closed-form global proximal map for ``lambda*|.|^(1/2)``.

    FOC: t + (lambda/2)*t^{-1/2} = |v|.  Substitute s = sqrt(t):
      s^3 - |v|*s + lambda/2 = 0  (depressed cubic)
    Solved via trigonometric method (three real roots when |v| > v_thresh).
    """
    penalty_exponent = 0.5
    scale = torch.as_tensor(lambda_, dtype=v.dtype, device=v.device)
    positive_scale = scale > 0
    safe_scale = torch.where(positive_scale, scale, torch.ones_like(scale))
    normalization = safe_scale ** (1.0 / (2.0 - penalty_exponent))
    normalized_v = v / normalization
    abs_v = torch.abs(normalized_v)
    active = abs_v > 1.5

    # Evaluate the closed form only in its active regime.  The inactive values
    # are replaced before the algebra, then masked back to zero below.
    av = torch.where(active, abs_v, torch.full_like(abs_v, 1.5))
    # Normalized cubic: s^3 - av*s + 1/2 = 0.
    # Use s = 2*sqrt(av/3)*cos(theta), where
    # theta = arccos(-3*sqrt(3) / (4*av^{3/2})) / 3.
    cos_arg = (-3.0 * math.sqrt(3.0) / (4.0 * av ** 1.5)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_arg) / 3.0
    s = 2.0 * torch.sqrt(av / 3.0) * torch.cos(theta)
    t = s ** 2

    t = torch.where(active, normalization * t, torch.zeros_like(t))
    return torch.where(positive_scale, torch.sign(v) * t, v)


def _compute_prox_q_twothirds(v, lambda_):
    """Closed-form global proximal map for ``lambda*|.|^(2/3)``.

    FOC: t + (2lambda/3)*t^{-1/3} = |v|.  Substitute s = t^{1/3}:
      s^4 - |v|*s + 2lambda/3 = 0  (depressed quartic)
    Solved via Ferrari's method:
      1. Resolvent cubic: y^3 - (2lambda/3)*y - |v|^2/8 = 0
      2. Factor quartic into two quadratics using y
      3. Take largest positive root from the quadratic with real roots
    """
    penalty_exponent = 2.0 / 3.0
    scale = torch.as_tensor(lambda_, dtype=v.dtype, device=v.device)
    positive_scale = scale > 0
    safe_scale = torch.where(positive_scale, scale, torch.ones_like(scale))
    normalization = safe_scale ** (1.0 / (2.0 - penalty_exponent))
    normalized_v = v / normalization
    abs_v = torch.abs(normalized_v)
    switch_output = (2.0 / 3.0) ** (3.0 / 4.0)
    switch_input = 2.0 * switch_output
    active = abs_v > switch_input
    av = torch.where(active, abs_v, torch.full_like(abs_v, switch_input))

    # On the global active branch the Ferrari resolvent has one real root.
    # Cardano's cube-root terms u, w satisfy y = u + w and u*w = 2/9.
    # Recover w from that product to avoid subtracting nearly equal numbers.
    cardano_base = av.square() / 16.0
    discriminant_ratio = (8.0 / 729.0) / cardano_base / cardano_base
    sqrt_delta = cardano_base * torch.sqrt((1.0 - discriminant_ratio).clamp_min(0.0))
    u = torch.pow(cardano_base + sqrt_delta, 1.0 / 3.0)
    w = (2.0 / 9.0) / u
    y = u + w

    # Factor normalized quartic: s^2 - r*s + (y - av/(2r)) = 0.
    r = torch.sqrt(2.0 * y)
    disc = (-2.0 * y + 2.0 * av / r).clamp_min(0.0)
    s = (r + torch.sqrt(disc)) / 2.0
    t = s ** 3

    t = torch.where(active, normalization * t, torch.zeros_like(t))
    return torch.where(positive_scale, torch.sign(v) * t, v)


def _compute_prox_derivative_fractional(v, lambda_, penalty_exponent, prox_result):
    """Diagonal derivative from the selected fractional proximal point.

    On the active branch of t + lambda*q*t^(q-1) = |v| the derivative is
    1 / (1 + lambda*q*(q-1)*|prox|^(q-2)); it is zero where the prox is zero
    and the identity at zero scale. At a switching tie the map jumps, so no
    classical derivative exists and the zero selection is returned.
    """
    scale = torch.as_tensor(lambda_, dtype=v.dtype, device=v.device)
    positive_scale = scale > 0
    safe_scale = torch.where(positive_scale, scale, torch.ones_like(scale))
    normalization = safe_scale ** (1.0 / (2.0 - penalty_exponent))
    prox_abs = torch.abs(prox_result)
    active = prox_abs > 0
    normalized_prox = torch.where(
        active,
        prox_abs / normalization,
        torch.ones_like(prox_abs),
    )

    denom = 1.0 + penalty_exponent * (penalty_exponent - 1) * normalized_prox ** (penalty_exponent - 2)
    return torch.where(
        ~positive_scale,
        torch.ones_like(v),
        torch.where(active, 1.0 / denom, torch.zeros_like(v)),
    )
