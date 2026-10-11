"""Coordinate-descent warm start for newly inserted signed atoms.

One 1-D proximal step (MATLAB PDAPmultisemidiscrete.m:104-147): pick the new
candidate whose descent profile against the current residual is largest, take the
nonconvex prox step along that combined direction, and return initial outer
weights for all new atoms.
"""

from __future__ import annotations

import logging

import torch
from torch.nn import functional as F

from ..models import Penalty, SparseModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def warm_start(W_new: torch.Tensor, b_new: torch.Tensor, residual,
               problem: SparseModel, proximal: Penalty, *, verbose: bool = False) -> torch.Tensor:
    """Select the largest-profile direction and evaluate its proximal step."""
    n_new = W_new.shape[0]
    if n_new == 0:
        return torch.zeros(0, dtype=torch.float64)
    S_val, S_grad = _warm_start_kernel(problem, residual[0], W_new, b_new)
    scale = problem.moment_weight_of_atoms(W_new, b_new)
    S_val, S_grad = S_val / scale, S_grad / scale
    profiles = problem.jac_obj(S_val, S_grad, residual)
    abs_profiles = profiles.abs()
    best = int(abs_profiles.argmax().item())
    eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
    safe = abs_profiles.clamp_min(1e-30)
    base = eps_sqrt * profiles / safe
    base[best] = profiles[best] / safe[best]
    direction = -base
    direction_features = (S_val @ direction).reshape(-1, 1), (S_grad @ direction).reshape(-1, 1)
    slope = float(problem.jac_obj(*direction_features, residual)[0])
    curvature = float(problem.hess_obj(*direction_features)[0, 0])
    tau = 0.0
    if -slope >= proximal.alpha and curvature > 1e-30:
        tau = proximal.penalty_prox(-slope / curvature, proximal.alpha / curvature)
    out = (tau * direction) / scale
    if verbose:
        logger.debug(
            "Warm start        initialized %d new output weights  max |weight|=%.2e",
            n_new, float(out.abs().max().item()) if out.numel() else 0.0,
        )
    return out.reshape(-1)


def _warm_start_kernel(problem: SparseModel, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """Kernel of the new atoms with the historical sphere-comparator derivative.

    On the sphere, the activation derivative in the input-gradient kernel is the
    positive-preactivation indicator, including for leaky_relu and relu2; relu
    is unaffected. Nonhomogeneous activations use the exact kernel.
    """
    value_kernel, gradient_kernel = problem.kernel(X, W, b)
    if not problem.use_sphere:
        return value_kernel, gradient_kernel
    z = F.linear(X, W, b)
    indicator = (z > 0).to(z.dtype)
    power = problem.power
    derivative = indicator if power == 1.0 else power * problem.activation(z) ** (power - 1) * indicator
    return value_kernel, (derivative.unsqueeze(1) * W.T.unsqueeze(0)).reshape(X.shape[0] * X.shape[1], W.shape[0])
