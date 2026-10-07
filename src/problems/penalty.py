"""Log and power penalties, their derivatives, and SSN correction terms.

Read top to bottom: penalty value and gradient, SSN's smooth remainder,
PDAP's proximal step, then the underlying scalar log-penalty formulas.
The algorithm-facing operations come before the scalar mathematics they use.

For t >= 0, the log family has the fixed half-mixture used by the MATLAB
2D/stereo references: phi(t) = t/2 + log(1+2*gamma*t)/(4*gamma).
At gamma=0, phi(t)=t. The fractional family is t^q, with gamma=0.

``phi_prox`` takes the full nonnegative log-penalty step for PDAP's warm start.
SSN instead splits phi(|u|) into |u| and a smooth remainder, using the signed
``power_prox`` in ``prox.py`` and the correction terms below.
"""

import math

import torch

# Penalty value and gradient


def _noncvx_penalty(
    theta: torch.Tensor, penalized: torch.Tensor, nonneg: torch.Tensor,
    *, alpha: float, gamma: float, penalty_exponent: float,
) -> torch.Tensor:
    """The log or power regularizer over the penalized coordinates.

    For q=1, sum alpha*phi(base); for q!=1 (gamma=0), sum alpha*base^q.
    Here base is |theta| on free-sign coordinates and clamp(theta, 0) on
    nonnegative ones. SparseProblem validates the penalty family and uses this term for
    coefficient evaluation and full-model loss recording.
    """
    coefficients = theta[penalized]
    if coefficients.numel() == 0:
        return theta.new_zeros(())
    base = torch.where(
        nonneg[penalized], coefficients.clamp_min(0.0), coefficients.abs()
    )
    if penalty_exponent == 1.0:
        return alpha * torch.sum(_phi(base, gamma))
    powers = torch.where(
        base > 0,
        base.clamp_min(1e-30) ** penalty_exponent,
        torch.zeros_like(base),
    )
    return alpha * torch.sum(powers)


def jac_noncvx_penalty(abs_u, sign_u, alpha, gamma, penalty_exponent=1.0):
    """Gradient of the selected log or power penalty w.r.t. u.

    For q=1: alpha * dphi(|u|) * sign(u).
    For q!=1 (gamma=0): alpha * q * |u|^{q-1} * sign(u),
    with a singularity guard at u=0. SparseProblem validates this parameter restriction.
    """
    if penalty_exponent == 1.0:
        return alpha * _dphi(abs_u, gamma) * sign_u
    active = abs_u > 0
    s = abs_u.clamp(min=1e-30)
    return torch.where(
        active,
        alpha * penalty_exponent * s ** (penalty_exponent - 1) * sign_u,
        torch.zeros_like(abs_u),
    )


# SSN's smooth remainder


def _noncvx_correction(abs_u, sign_u, gamma, penalty_exponent=1.0):
    """Gradient of the smooth remainder in the SSN splitting.

    For q=1 the remainder is phi(|u|)-|u|. For q!=1 (gamma=0),
    power_prox handles the whole penalty, so there is no remainder.
    """
    if penalty_exponent == 1.0:
        return sign_u * (_dphi(abs_u, gamma) - 1.0)
    return torch.zeros_like(abs_u)


def _noncvx_correction_dd(abs_u, gamma, penalty_exponent=1.0):
    """Diagonal second derivative of the smooth remainder in the SSN splitting.

    For q=1: ddphi(|u|). For q!=1 (gamma=0), the remainder is zero.
    """
    if penalty_exponent == 1.0:
        return _ddphi(abs_u, gamma)
    return torch.zeros_like(abs_u)


# PDAP's nonnegative log-penalty step


def phi_prox(v: float, lambda_: float, gamma: float) -> float:
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
