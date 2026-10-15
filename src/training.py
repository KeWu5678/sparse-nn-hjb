"""Run PDAP steps, then measure and record their completed network states.

Quality evaluation retains the full-split objective and physical reporting
conventions. Solver time is CPU wall time around PDAP operations; evaluation
time covers metric calculation. The memory value is the lifetime high-water
RSS of this process, including earlier work, not a peak attributable to PDAP.
"""

from __future__ import annotations

import resource
import sys
from collections.abc import Callable
from functools import partial
from time import perf_counter

import structlog
from torch.utils.data import DataLoader

from .data import ValueSampleNormalizer
from .eval import evaluate_checkpoint
from .models import Penalty, SparseNetwork
from .PDAP import PDAP, History

logger = structlog.get_logger(__name__)

_COUNTERS = {
    "searches": "total_insertion_searches",
    "ssn_iterations": "total_ssn_iterations",
    "ssn_failed_steps": "total_ssn_failed_steps",
    "inserted": "total_inserted_neurons",
    "pruned": "total_pruned_neurons",
}


def run_training(
    pdap: PDAP,
    valid_loader: DataLoader | None = None,
    *,
    num_iterations: int,
    normalizer: ValueSampleNormalizer | None = None,
    display_every: int = 2,
) -> History:
    """Run an outer-iteration budget with separate measurement and recording.

    Insertion settings belong to PDAP's configuration. This loop obeys its
    no-candidate stop result and records after correction/pruning. The selected
    checkpoint still minimizes the recorded training objective; no initial row
    is added to an ordinary run. Zero-budget/no-insertion runs record once.
    Progress is logged at checkpoint indices 0, N, 2N,
    where N is ``display_every``. Nonpositive N suppresses periodic progress;
    the terminal summary is still logged. All checkpoints are evaluated and saved.
    """
    network, train_loader = pdap.network, pdap.train_loader
    history = History(stop_reason="iteration_budget")
    history.run = dict.fromkeys(_COUNTERS.values(), 0)
    history.run.update(
        total_ssn_refinements=0, total_accepted_ssn_refinements=0,
        solver_seconds=0.0, evaluation_seconds=0.0,
    )
    record = partial(
        _record, network, pdap.penalty, train_loader, valid_loader, normalizer, history,
        display_every=display_every,
    )
    diagnostics = {}
    if pdap.loop_order == "correction_first":
        # Preserve the historical comparator exactly: initialize, then
        # correct -> prune -> measure -> insert, including the final insertion.
        search = _measure(pdap.initialize, history)
        diagnostics = search
        if search["stop_reason"] is not None:
            history.stop_reason = search["stop_reason"]
        else:
            for _ in range(num_iterations):
                correction = _measure(
                    pdap.train_ssn_outerweights_and_prune, history,
                )
                diagnostics = {**search, **correction}
                diagnostics["solver_seconds"] = search["solver_seconds"] + correction["solver_seconds"]
                record(diagnostics)
                search = _measure(pdap.insert, history)
    else:
        for _ in range(num_iterations):
            diagnostics = _measure(pdap.step, history)
            if diagnostics["stop_reason"] is not None:
                history.stop_reason = diagnostics["stop_reason"]
                break
            record(diagnostics)

    if not history.rows:
        record(diagnostics)
    history.run["final_neurons"] = network.n_neurons
    history.run["process_peak_rss_bytes"] = _process_peak_rss_bytes()
    best = history.rows[history.best_iteration]
    logger.info(
        "training_finished", stop_reason=history.stop_reason,
        checkpoints=len(history.rows), best_iteration=history.best_iteration,
        objective_train=best["objective_train"], neurons=best["neurons"],
        final_neurons=network.n_neurons,
    )
    return history


def _measure(
    operation: Callable[..., dict], history: History,
) -> dict:
    """Time executed solver work and count it even when no checkpoint follows."""
    started = perf_counter()
    diagnostics = operation()
    diagnostics["solver_seconds"] = perf_counter() - started
    for name, total in _COUNTERS.items():
        history.run[total] += diagnostics.get(name, 0)
    if "ssn_refinement_accepted" in diagnostics:
        history.run["total_ssn_refinements"] += 1
        history.run["total_accepted_ssn_refinements"] += int(diagnostics["ssn_refinement_accepted"])
    history.run["solver_seconds"] += diagnostics["solver_seconds"]
    return diagnostics


def _record(
    network: SparseNetwork, penalty: Penalty, train_loader: DataLoader,
    valid_loader: DataLoader | None, normalizer: ValueSampleNormalizer | None,
    history: History, diagnostics: dict, display_every: int,
) -> None:
    """Evaluate the current state before capturing its checkpoint."""
    started = perf_counter()
    row = evaluate_checkpoint(network, penalty, train_loader, valid_loader, normalizer)
    elapsed = perf_counter() - started
    history.run["evaluation_seconds"] += elapsed
    history.record(network, row, {**diagnostics, "evaluation_seconds": elapsed})
    checkpoint = len(history.rows) - 1
    if display_every > 0 and checkpoint % display_every == 0:
        logger.info(
            "training_progress", checkpoint=checkpoint, neurons=row["neurons"],
            objective_train=row["objective_train"], objective_val=row["objective_val"],
            rel_l2_val=row["rel_l2_val"], rel_h1_val=row["rel_h1_val"],
        )


def _process_peak_rss_bytes() -> int:
    """Convert the process lifetime RSS high-water mark to bytes on macOS/Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
