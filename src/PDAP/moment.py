"""Parameter-moment regularizer primitives (the ``beta * Psi_p`` axis).

The moment penalty adds, on top of the nonconvex penalty ``alpha * Phi_1``, a
weighted total-variation term

    beta * Psi_p(mu) = beta * sum_j (1 + |omega_j|^p) * |c_j|,
    omega_j = (a_j, b_j) in R^{d+1},

from ``papar/draft/PROOF_NARROW-CONVERGENCE``.  It is a *per-atom weighted L1*
on the outer coefficients ``c`` with weight ``w_p(omega) = 1 + |omega|^p``; the
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

__all__ = ["moment_weight", "moment_penalty"]


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
    c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
    if c.numel() == 0:
        return c.new_zeros(())
    return beta * torch.sum(w_p.reshape(-1) * c.abs())
