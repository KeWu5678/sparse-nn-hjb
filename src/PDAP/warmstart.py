"""Coordinate-descent warm start for newly inserted signed atoms.

One 1-D proximal step (MATLAB PDAPmultisemidiscrete.m:104-147): pick the new
candidate whose descent profile against the current residual is largest, take the
nonconvex prox step along that combined direction, and set initial outer
weights for all new atoms in place.
"""

from __future__ import annotations

import logging

import torch

from ..models import Penalty, SparseNetwork

logger = logging.getLogger(__name__)


@torch.no_grad()
def warm_start(network: SparseNetwork, penalty: Penalty, X: torch.Tensor, target, n_new: int, *,
               verbose: bool = False) -> None:
    """Set the zero outer weights of the last n_new atoms by a proximal step along the largest-profile direction."""
    if n_new == 0:
        return
    new = slice(network.n_neurons - n_new, None)
    Vp, dVp = network(X)
    residual = X, Vp - target[0], dVp - target[1]
    scale = network.moment_weight_of_atoms()[new]
    jac = network.jac_obj(residual)
    profiles = jac[new] / scale
    abs_profiles = profiles.abs()
    best = int(abs_profiles.argmax().item())
    eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
    safe = abs_profiles.clamp_min(1e-30)
    base = eps_sqrt * profiles / safe
    base[best] = profiles[best] / safe[best]
    direction = torch.zeros_like(jac)
    direction[new] = -base / scale
    slope = float(jac @ direction)
    curvature = float(direction @ network.hess_obj(residual[0]) @ direction)
    tau = penalty.descent_step(slope, curvature)
    network.c[new] = tau * direction[new]
    if verbose:
        logger.debug(
            "Warm start        initialized %d new output weights  max |weight|=%.2e",
            n_new, float(network.c[new].abs().max().item()),
        )
