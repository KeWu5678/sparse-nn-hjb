"""Power-penalty proximal maps for SSN and PDAP, and their derivatives.

The power proximal maps support ``q`` in ``{1, 2/3, 1/2}``.  Their fractional maps
use closed-form stationary roots together with the global objective-switching
threshold; at an exact tie they return zero.
"""

import math

import torch

# Activation powers whose induced exponents q=2/(power+1) have verified global
# scalar proximal maps below.  SparseProblem uses this tuple for configuration
# validation so adding a closed form has one authority.
SUPPORTED_ACTIVATION_POWERS = (1.0, 2.0, 3.0)


def power_prox(v, lambda_, penalty_exponent=1.0):
    """Return the global proximal map of ``lambda*|.|^q``.

    For q=1: soft thresholding, prox(v) = sign(v) * max(|v| - lambda, 0).
    For q=1/2: closed-form via depressed cubic (trigonometric method).
    For q=2/3: closed-form via depressed quartic (Ferrari's method).
    Args:
        v: input tensor
        lambda_: proximal scale (typically alpha / inverse_step in SSN)
        penalty_exponent: power exponent, q = 2/(p+1) where p is the activation power;
            1 exactly, or within 1e-12 of 1/2 or 2/3
    Returns:
        vprox: global minimizer, with zero selected at an exact tie
    """
    penalty_exponent = _canonical_exponent(penalty_exponent)
    if penalty_exponent == 1.0:
        return _compute_prox_q_one(v, lambda_)

    if penalty_exponent == 0.5:
        return _compute_prox_q_half(v, lambda_)

    return _compute_prox_q_twothirds(v, lambda_)


def _canonical_exponent(penalty_exponent):
    """Use the same supported exponent and tolerance for the map and derivative."""
    if penalty_exponent == 1.0:
        return 1.0
    if abs(penalty_exponent - 0.5) < 1e-12:
        return 0.5
    if abs(penalty_exponent - 2.0 / 3.0) < 1e-12:
        return 2.0 / 3.0
    raise ValueError(f"q must be one of {{1, 2/3, 1/2}}, got {penalty_exponent}")


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


def power_prox_derivative(v, lambda_, penalty_exponent=1.0, prox_result=None):
    """Generalized derivative of the scalar proximal map away from its switch.

    Ported from MATLAB computeDProx.m. With N=1 (scalar outer weights),
    each neuron's Jacobian block reduces to a scalar diagonal entry.

    For q=1 (soft thresholding prox(v) = sign(v)*max(|v|-lambda, 0)):
        d prox/dv = 1 for active (|v| > lambda), 0 for inactive.
        (Computed via MATLAB's general N-dim formula specialized to N=1.)

    For q<1 (active global branch from t + lambda*q*t^{q-1} = |v|):
        d prox/dv = 1 / (1 + lambda*q*(q-1)*|prox|^{q-2}) for active, 0 for inactive.
        Requires the precomputed prox_result to identify the selected branch.

    Used in SSN's _DG to form the generalized Jacobian:
        DG = inverse_step*(I - DPc) + alpha*diag(correction_dd)*DPc + H_data*DPc

    Args:
        v: proximal preimage tensor (the SSN variable q_var)
        lambda_: proximal scale (typically alpha / inverse_step)
        penalty_exponent: supported exponent, with the same acceptance tolerance as power_prox
        prox_result: precomputed power_prox(v, lambda_, q) — required for q != 1

    Returns:
        DP: Jacobian matrix (diagonal), shape (n, n)

    At a fractional switching tie the map jumps, so no classical derivative
    exists; the zero selection for the inactive branch is returned.
    """
    assert torch.is_floating_point(v), "Input must be real-valued"

    penalty_exponent = _canonical_exponent(penalty_exponent)
    if penalty_exponent == 1.0:
        return _compute_prox_derivative_q_one(v, lambda_)

    if prox_result is None:
        raise ValueError("prox_result from power_prox is required for fractional q")
    return _compute_prox_derivative_fractional(v, lambda_, penalty_exponent, prox_result)


def _compute_prox_derivative_q_one(v, lambda_):
    """Diagonal generalized Jacobian of soft thresholding."""
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
    return torch.diag(diagonal_term + outer_product_term)


def _compute_prox_derivative_fractional(v, lambda_, penalty_exponent, prox_result):
    """Diagonal derivative from the selected fractional proximal point."""
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
    diag = torch.where(
        ~positive_scale,
        torch.ones_like(v),
        torch.where(active, 1.0 / denom, torch.zeros_like(v)),
    )

    return torch.diag(diag.reshape(-1))
