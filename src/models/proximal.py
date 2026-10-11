"""The coefficient penalty and the proximal maps built on it.

For t >= 0, the log family has the fixed half-mixture used by the MATLAB
2D/stereo references: phi(t) = t/2 + log(1+2*gamma*t)/(4*gamma).
At gamma=0, phi(t)=t. The fractional family is t^q, with gamma=0 and the
exact exponent q in {2/3, 1/2} implied by a validated activation power.
The fractional maps use closed-form stationary roots together with the global
objective-switching threshold; at an exact tie they return zero.

Read top to bottom: Proximal, then the scalar mathematics it uses.
"""

import math

import torch
from torch import Tensor


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
        magnitudes = self._magnitudes(u)
        if magnitudes.numel() == 0:
            return u.new_zeros(())
        if self._penalty_exponent == 1.0:
            return torch.sum(_phi(magnitudes, self._gamma))
        powers = torch.where(
            magnitudes > 0,
            magnitudes.clamp_min(1e-30) ** self._penalty_exponent,
            torch.zeros_like(magnitudes),
        )
        return torch.sum(powers)

    def penalty(self, u: Tensor) -> Tensor:
        """The penalty term alpha*Phi(u) of the objective."""
        return self._alpha * self.sparsity_functional(u)

    def jac_penalty(self, u: Tensor) -> Tensor:
        """Derivative of the penalty, selecting zero at u = 0 for the power family."""
        if self._penalty_exponent == 1.0:
            return self._alpha_vec(u) * _dphi(torch.abs(u), self._gamma) * torch.sign(u)
        return torch.where(torch.abs(u) > 0, self.jac_fractional_penalty(u), torch.zeros_like(u))

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
        magnitudes = self._magnitudes(u)
        return self._alpha * (_phi(magnitudes, self._gamma) - magnitudes).sum()

    def jac_correction(self, u: Tensor) -> Tensor:
        """Derivative of the smooth correction."""
        if self._penalty_exponent != 1.0:
            return torch.zeros_like(u)
        return self._alpha_vec(u) * (torch.sign(u) * (_dphi(torch.abs(u), self._gamma) - 1.0))

    def hess_correction(self, u: Tensor) -> Tensor:
        """Diagonal Hessian matrix of the smooth correction."""
        if self._penalty_exponent != 1.0:
            return torch.diag(torch.zeros_like(u))
        return torch.diag(self._alpha_vec(u) * _ddphi(torch.abs(u), self._gamma))

    def prox(self, q: Tensor, lambda_: float) -> Tensor:
        """Proximal map of the fractional penalty, including nonnegative coordinates."""
        out = self._closed_form_prox(q, self._penalized_constant(q, lambda_))
        kill = self.masks(q)[1] & (q < 0)
        return torch.where(kill, torch.zeros_like(out), out)

    def jac_prox(self, q: Tensor, lambda_: float, *, prox_image: Tensor) -> Tensor:
        """Selected proximal Jacobian at q; prox_image must be prox(q, lambda_)."""
        constant = self._penalized_constant(q, lambda_)
        if self._penalty_exponent == 1.0:
            diagonal = _compute_prox_derivative_q_one(q, constant)
        else:
            diagonal = _compute_prox_derivative_fractional(q, constant, self._penalty_exponent, prox_image)
        kill = self.masks(q)[1] & (q < 0)
        return torch.diag(torch.where(kill, torch.zeros_like(prox_image), diagonal))

    def penalty_prox(self, value: float, lambda_: float) -> float:
        """Selected nonnegative proximal value for the full scalar penalty.

        The log family's closed-form root requires value >= lambda_ to guarantee
        a minimizer; its caller owns that eligibility check. The vector prox
        instead handles only the fractional penalty, as the normal map requires.
        """
        if self._penalty_exponent == 1.0:
            return _phi_prox(value, lambda_, gamma=self._gamma)
        return max(self.fractional_prox(value, lambda_), 0.0)

    def fractional_prox(self, value: float, lambda_: float) -> float:
        """Scalar proximal value of the fractional penalty, including its q=1 safeguard.

        Retain scalar scale arithmetic; the masked vector prox has a different
        floating-point evaluation order and must keep its own numerical contract.
        """
        v = torch.tensor([value], dtype=torch.float64)
        return float(self._closed_form_prox(v, lambda_)[0])

    def masks(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Penalized and nonnegative coordinate masks for u, defaulting to all and none."""
        return (
            self._mask(self._penalized_mask, u, True, "penalized_mask"),
            self._mask(self._nonneg_mask, u, False, "nonneg_mask"),
        )

    def _magnitudes(self, u: Tensor) -> Tensor:
        """Penalized |u|, using max(u, 0) on nonnegative coordinates."""
        penalized, nonneg = self.masks(u)
        coefficients = u[penalized]
        return torch.where(nonneg[penalized], coefficients.clamp_min(0.0), coefficients.abs())

    def _closed_form_prox(self, v: Tensor, lambda_) -> Tensor:
        """Global proximal map of lambda_*|.|^q, choosing the closed form by the exact q."""
        return {
            1.0: _compute_prox_q_one,
            0.5: _compute_prox_q_half,
            2.0 / 3.0: _compute_prox_q_twothirds,
        }[self._penalty_exponent](v, lambda_)

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


# Scalar log-penalty formulas


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


def _phi_prox(v: float, lambda_: float, gamma: float) -> float:
    """MATLAB's closed-form nonnegative step for ``lambda * phi(t)``.

    The objective is J(t) = (t-v)^2/2 + lambda*phi(t), t >= 0.
    At gamma=0 the step is nonnegative soft thresholding. Otherwise, with
    gamma_bar = 2*gamma, stationarity gives
        t^2 - a*t + (lambda-v)/gamma_bar = 0,
        a = v - lambda/2 - 1/gamma_bar.
    Delta is its discriminant; the larger root is clipped to t >= 0, as in
    MATLAB setup_problem_NN_2d.m (theta=1/2).

    For lambda, gamma >= 0, this is a global minimizer when lambda*gamma < 1
    or v >= lambda. PDAP's warm-start acceptance guard ensures the latter.
    Outside these regimes the root need not be globally optimal; no extra
    objective comparison or rejection is performed, matching the reference.
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


# Fractional proximal maps and their derivatives


def _compute_prox_q_one(v, lambda_):
    """Soft thresholding for ``lambda*|.|``."""
    abs_v = torch.abs(v)
    eps = torch.finfo(v.dtype).eps
    abs_v_safe = torch.clamp(abs_v, min=(lambda_ + eps) * eps)
    shrinkage_factor = torch.clamp(1 - lambda_ / abs_v_safe, min=0)
    return shrinkage_factor * v


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
