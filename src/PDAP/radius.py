"""The theorem-derived radius used by Algorithm 1's candidate search.

The quantitative insertion theorem (paper/paper_0805.tex, Section 5) states that
under the growth bound

    w_p(omega) * ||K_p(omega)||  =  ||K^M(omega)||  <=  C (1 + |omega|)^s,   s < p,

every point with ``|P_p(omega)| > alpha*L_phi`` -- hence every candidate the
insertion condition can accept, and every maximizer of ``|P_p|`` -- lies in the
closed ball of radius

    R(mu) = max{ 1, ( 2^(s+1) * C * ||r_mu|| / (alpha*L_phi) )^(1/(p - s)) }.

``s`` is the activation's ``s1`` and ``C`` follows from its ``(C_rho, s0, s1)``
together with the sample extent, so nothing here is calibrated numerically:

    |K(omega)(x)|      <= C_rho * A_M^s0       * (1+|omega|)^s0
    |grad K(omega)(x)| <= C_rho * A_M^(s1 - 1) * (1+|omega|)^s1     (|a| <= |omega|)
    ==>  C = C_rho * (A_M^s0 + A_M^(s1 - 1)),   A_M = max{1, max_m sqrt(1+|x^m|^2)}.

The empirical norm ``||.||_M`` averages over the samples rather than integrating
over ``D``, which is why the ``|D|^(1/2)`` of the population lemma drops out.

Algorithm 1 sets ``R_search = min(R(mu), exp(5))``.  Random starts are sampled
inside that radius, the joint L-BFGS solve over ``omega`` is unconstrained, and
an optimized point is retained only when its final norm is at most
``R_search``.  The point is never projected back to the boundary.  If the
theorem does not supply a finite radius, ``exp(5)`` is used as the fallback.

For the moment orders in use, the theorem radius is typically larger than the
numerical cap and therefore inert; it mainly tightens the final acceptance
region at large ``p``, where ``1/(p - s1)`` is small.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..config.activations import Growth

__all__ = ["FIXED_LOG_CLAMP", "sample_extent", "growth_constant", "certificate_radius"]

# Numerical upper bound used for sampling and final candidate filtering.
FIXED_LOG_CLAMP = 5.0


def sample_extent(X: torch.Tensor) -> float:
    """``A_M = max{1, max_m sqrt(1 + |x^m|^2)}``, the data half of the constant."""
    X = torch.as_tensor(X, dtype=torch.float64)
    if X.numel() == 0:
        return 1.0
    return float(torch.sqrt(1.0 + (X * X).sum(dim=1)).max().clamp_min(1.0))


def growth_constant(growth: Growth, extent: float) -> float:
    """``C = C_rho * (A_M^s0 + A_M^(s1-1))`` from the activation and the samples."""
    return growth.C_rho * (extent ** growth.s0 + extent ** (growth.s1 - 1.0))


def certificate_radius(
    growth: Optional[Growth],
    *,
    extent: float,
    residual_norm: float,
    alpha: float,
    moment_order: float,
    l_phi: float = 1.0,
) -> Optional[float]:
    """Return ``min(R(mu), exp(5))``, or ``None`` when the theorem does not apply.

    ``None`` means that the caller uses the numerical fallback ``exp(5)``.  It is
    returned when the activation declares no growth data, when ``p <= s1`` (so
    the theorem's hypothesis fails), or when the inputs are degenerate.
    """
    if growth is None or alpha <= 0.0 or l_phi <= 0.0:
        return None
    gap = moment_order - growth.s1
    if gap <= 0.0:
        return None
    if residual_norm <= 0.0:
        # A zero residual has no certificate to violate; nothing to search for.
        return math.exp(FIXED_LOG_CLAMP)

    C = growth_constant(growth, extent)
    base = (2.0 ** (growth.s1 + 1.0)) * C * residual_norm / (alpha * l_phi)
    try:
        radius = max(1.0, base ** (1.0 / gap))
    except OverflowError:
        return math.exp(FIXED_LOG_CLAMP)
    if not math.isfinite(radius):
        return math.exp(FIXED_LOG_CLAMP)
    return min(radius, math.exp(FIXED_LOG_CLAMP))
