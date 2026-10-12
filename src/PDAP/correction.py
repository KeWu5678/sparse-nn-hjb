"""The coefficient correction of a PDAP step: SSN iterations and the final discard."""

from __future__ import annotations

import logging
from typing import Callable

import torch

from ..SSN import SSN

logger = logging.getLogger(__name__)


def correct_coefficients(
    optimizer: SSN, coefficients: torch.nn.Parameter, closure: Callable[[], torch.Tensor], *,
    iterations: int, correction_guard: bool, verbose: bool = False,
) -> dict:
    """Run the SSN iteration budget, discarding a correction that increases the objective.

    This is the correction step of Algorithms 1 and 2: if the correction increases
    J, the post-insertion coefficients are kept. A nonfinite result is always
    discarded; correction_guard=False preserves the historical unguarded
    comparison. An exception restores the post-insertion coefficients and
    propagates. No convergence claim is made after the fixed budget. The
    optimizer evaluates the starting objective with the closure.
    """
    start = coefficients.detach().clone()
    initial_loss = optimizer.objective(closure)
    if not torch.isfinite(start).all() or not torch.isfinite(initial_loss):
        raise ValueError("SSN requires finite starting coefficients and loss")
    try:
        loss, failed_steps = _run_fixed_ssn_steps(optimizer, closure, initial_loss, iterations)
    except Exception:
        _restore(coefficients, start)
        raise
    finite = bool(torch.isfinite(loss) & torch.isfinite(coefficients).all())
    accepted = finite and (not correction_guard or bool(loss <= initial_loss))
    if not accepted:
        _restore(coefficients, start)
        loss = initial_loss
    if verbose:
        group = optimizer.param_groups[0]
        logger.debug(
            "SSN correction  iterations=%d  objective=%.6e->%.6e  accepted=%s  "
            "failed_steps=%d  prox_scale=%s  tau=%.6e",
            iterations, float(initial_loss.detach()), float(loss.detach()),
            str(accepted).lower(), failed_steps, group["prox_scale"], group["tau"],
        )
    return {
        "ssn_iterations": iterations,
        "ssn_failed_steps": failed_steps,
        "ssn_last_step_success": optimizer.last_step_success,
        "correction_accepted": accepted,
    }


def _run_fixed_ssn_steps(
    optimizer: SSN, closure: Callable[[], torch.Tensor], initial_loss: torch.Tensor, iterations: int,
) -> tuple[torch.Tensor, int]:
    """Run the fixed number of SSN steps; return the last objective and the failed steps."""
    loss, failed_steps = initial_loss, 0
    for _ in range(iterations):
        loss = optimizer.step(closure)
        failed_steps += int(not optimizer.last_step_success)
    return loss, failed_steps


def _restore(coefficients: torch.nn.Parameter, start: torch.Tensor) -> None:
    with torch.no_grad():
        coefficients.copy_(start)
