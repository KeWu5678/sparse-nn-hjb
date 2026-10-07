"""Coordinate-descent warm start for newly inserted signed atoms.

One 1-D proximal step (MATLAB PDAPmultisemidiscrete.m:104-147): pick the new
candidate whose descent profile against the current residual is largest, take the
nonconvex prox step along that combined direction, and return initial outer
weights for all new atoms.
"""

from __future__ import annotations

import logging

import torch

from ..problems import SparseProblem

logger = logging.getLogger(__name__)


@torch.no_grad()
def warm_start(W_new: torch.Tensor, b_new: torch.Tensor, residual,
               problem: SparseProblem, *, verbose: bool = False) -> torch.Tensor:
    """Select the largest-profile direction and evaluate its proximal step."""
    n_new = W_new.shape[0]
    if n_new == 0:
        return torch.zeros(0, dtype=torch.float64)
    profiles = problem.warmstart_profiles(W_new, b_new, residual)
    abs_profiles = profiles.abs()
    best = int(abs_profiles.argmax().item())
    eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
    safe = abs_profiles.clamp_min(1e-30)
    base = eps_sqrt * profiles / safe
    base[best] = profiles[best] / safe[best]
    out = problem.warmstart_coefficients(W_new, b_new, -base, residual)
    if verbose:
        logger.debug(
            "Warm start        initialized %d new output weights  max |weight|=%.2e",
            n_new, float(out.abs().max().item()) if out.numel() else 0.0,
        )
    return out.reshape(-1)
