"""Parameter-moment regularizer primitives (the ``beta * Psi_p`` axis).

The moment penalty adds, on top of the nonconvex penalty ``alpha * Phi_1``, a
weighted total-variation term

    beta * Psi_p(mu) = beta * sum_j (1 + |omega_j|^p) * |c_j|,
    omega_j = (a_j, b_j) in R^{d+1},

from the narrow-convergence proof, now Section 3 of ``paper/paper_0805.tex``
(the standalone draft is archived under ``docs/paper_archive/``).  It is a
*per-atom weighted L1* on the outer coefficients ``c`` with weight
``w_p(omega) = 1 + |omega|^p``; the
weight prices distant neurons out and supplies the tightness the location-blind
``Phi_1`` lacks.  These two helpers are the single home of ``w_p`` and of the
scalar penalty value, shared by the SSN solve, insertion, warm start, and the
recorded objective so they cannot drift apart.

The moment axis is meaningful only for non-homogeneous (``use_sphere=False``)
activations, where the inner-weight scale is a free parameter; for
positively-homogeneous activations the inner weights are gauge-fixed to the unit
sphere and ``|omega_j| = 1`` is constant.  That restriction is enforced in
:class:`src.PDAP.PDAP`, not here.
"""

from __future__ import annotations

import torch

__all__ = [
    "amplitude_mass_radius",
    "moment_weight",
    "moment_penalty",
    "atom_normalizer",
]


def atom_normalizer(
    W: torch.Tensor, b: torch.Tensor, *, normalized: bool, p: float
) -> torch.Tensor:
    """Per-atom divisor turning the dictionary ``K`` into the one the objective uses.

    The revised paper objective is ``J = l^M + alpha * sum phi(w_p(omega_n)|c_n|)``.
    Substituting the normalized coefficient ``u_n = w_p(omega_n) c_n`` turns it into
    the *ordinary* objective ``l^M + alpha * sum phi(|u_n|)`` over the normalized
    dictionary ``K_p = K / w_p``, because

        sum_n c_n K(omega_n) = sum_n u_n K(omega_n)/w_p(omega_n) = sum_n u_n K_p(omega_n).

    So every site that would need a new penalty instead divides its atom columns by
    this vector and works in ``u``, converting back with ``c = u / w_p``.

    Returns ``w_p`` when ``normalized``, and ones otherwise, so call sites can
    divide unconditionally.
    """
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    if not normalized:
        return torch.ones_like(b)
    return moment_weight(W, b, p).reshape(-1)


def moment_weight(W: torch.Tensor, b: torch.Tensor, p: float) -> torch.Tensor:
    """Per-atom moment weight ``w_p(omega) = 1 + |omega|^p`` with ``omega=(a,b)``.

    ``|omega|`` is the Euclidean norm of the *concatenated* inner weight and bias
    (the bias is included, matching the draft's ``omega = (a, b)``).  Written with
    ``|omega|^p = (|a|^2 + b^2)^{p/2}`` so it stays differentiable in the atom
    parameters -- the insertion scale search backpropagates through it.

    Args:
        W: inner weights, shape ``(n, d)`` or ``(d,)`` for a single atom.
        b: biases, shape ``(n,)`` or scalar.
        p: moment order (``moment_order``).

    Returns:
        ``w_p``, shape ``(n,)`` (or scalar for a single atom), matching ``b``.
    """
    if not p > 0.0:
        raise ValueError("moment order p must be positive")
    W = torch.as_tensor(W, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    sq = (W * W).sum(dim=-1) + b * b
    return 1.0 + sq.clamp_min(0.0).pow(p / 2.0)


def moment_penalty(c: torch.Tensor, w_p: torch.Tensor, beta: float) -> torch.Tensor:
    """The scalar moment term ``beta * sum_j w_p(omega_j) * |c_j|``.

    Args:
        c: outer coefficients, shape ``(n,)``.
        w_p: per-atom weights from :func:`moment_weight`, shape ``(n,)``.
        beta: moment weight (``moment_beta``).

    Returns:
        A 0-d tensor.
    """
    if not beta >= 0.0:
        raise ValueError("moment beta must be nonnegative")
    c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
    if c.numel() == 0:
        return c.new_zeros(())
    return beta * torch.sum(w_p.reshape(-1) * c.abs())


def amplitude_mass_radius(
    W: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    mass_fraction: float = 0.95,
) -> torch.Tensor:
    """Smallest parameter radius containing ``mass_fraction`` of ``sum |c_j|``.

    This is the R_0.95-type radius diagnostic used by the moment-penalty
    experiments. Atoms are sorted by
    ``|omega_j| = sqrt(|a_j|^2 + b_j^2)`` and accumulated with weights
    ``|c_j|``. Empty supports and supports with zero total variation have
    radius zero.
    """
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("mass_fraction must lie in (0, 1]")

    W = torch.as_tensor(W, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
    if W.shape[0] != b.numel() or b.numel() != c.numel():
        raise ValueError("W, b, and c must contain the same number of atoms")
    if c.numel() == 0:
        return c.new_zeros(())

    amplitude = c.abs()
    total = amplitude.sum()
    if float(total) == 0.0:
        return total

    radius = torch.sqrt((W * W).sum(dim=-1) + b * b)
    order = torch.argsort(radius)
    radius = radius[order]
    cumulative = torch.cumsum(amplitude[order], dim=0)
    index = torch.searchsorted(cumulative, mass_fraction * total).clamp_max(c.numel() - 1)
    return radius[index]
