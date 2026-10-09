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
from time import perf_counter

import structlog
from torch.utils.data import DataLoader

from .data import ValueSampleNormalizer
from .eval import metrics
from .PDAP import PDAP, History
from .problems import Proximal, SparseProblem

logger = structlog.get_logger(__name__)

_COUNTERS = {
    "searches": "total_insertion_searches",
    "corrections": "total_coefficient_corrections",
    "ssn_iterations": "total_ssn_iterations",
    "ssn_failed_steps": "total_ssn_failed_steps",
    "inserted": "total_inserted_neurons",
    "pruned": "total_pruned_neurons",
}


def run_training(
    pdap: PDAP,
    problem: SparseProblem,
    train_loader: DataLoader,
    valid_loader: DataLoader | None = None,
    *,
    num_iterations: int,
    reporting_normalizer: ValueSampleNormalizer | None = None,
    verbose: bool = True,
    display_every: int = 2,
) -> History:
    """Run an outer-iteration budget with separate measurement and recording.

    Insertion settings belong to PDAP's configuration. This loop obeys its
    no-candidate stop result and records after correction/pruning. The selected
    checkpoint still minimizes the recorded training objective; no initial row
    is added to an ordinary run. Zero-budget/no-insertion runs record once.
    With ``verbose=True``, progress is logged at checkpoint indices 0, N, 2N,
    where N is ``display_every``. Nonpositive N suppresses periodic progress;
    the terminal summary is still logged. All checkpoints are evaluated and saved.
    """
    history = History(reporting_normalizer=reporting_normalizer)
    history.run_metrics = dict.fromkeys(_COUNTERS.values(), 0)
    history.run_metrics.update(
        total_accepted_corrections=0, solver_seconds=0.0, evaluation_seconds=0.0,
        stop_reason="iteration_budget",
    )
    diagnostics = {}
    if pdap.loop_order == "correction_first":
        # Preserve the historical comparator exactly: initialize, then
        # correct -> prune -> measure -> insert, including the final insertion.
        search = _measure(pdap.initialize, problem, train_loader, history, verbose)
        diagnostics = search
        if search["stop_reason"] is not None:
            history.run_metrics["stop_reason"] = search["stop_reason"]
        else:
            for _ in range(num_iterations):
                correction = _measure(
                    pdap.train_ssn_outerweights_and_prune, problem, train_loader, history, verbose,
                )
                diagnostics = {**search, **correction}
                diagnostics["solver_seconds"] = search["solver_seconds"] + correction["solver_seconds"]
                _record(
                    problem, pdap.proximal, train_loader, valid_loader, history, diagnostics,
                    verbose, display_every,
                )
                search = _measure(pdap.insert, problem, train_loader, history, verbose)
    else:
        for _ in range(num_iterations):
            diagnostics = _measure(pdap.step, problem, train_loader, history, verbose)
            if diagnostics["stop_reason"] is not None:
                history.run_metrics["stop_reason"] = diagnostics["stop_reason"]
                break
            _record(
                problem, pdap.proximal, train_loader, valid_loader, history, diagnostics,
                verbose, display_every,
            )

    if not history.train_loss:
        _record(
            problem, pdap.proximal, train_loader, valid_loader, history, diagnostics,
            verbose, display_every,
        )
    history.final_neurons = problem.n_neurons
    history.run_metrics["process_peak_rss_bytes"] = _process_peak_rss_bytes()
    if verbose:
        logger.info(
            "training_finished", stop_reason=history.run_metrics["stop_reason"],
            checkpoints=len(history.train_loss), best_iteration=history.best_iteration,
            best_train_loss=history.best_train_loss, best_neurons=history.best_neurons,
            final_neurons=history.final_neurons,
        )
    return history


def _measure(
    operation: Callable[..., dict], problem: SparseProblem, loader: DataLoader,
    history: History, verbose: bool,
) -> dict:
    """Time executed solver work and count it even when no checkpoint follows."""
    started = perf_counter()
    diagnostics = operation(problem, loader, verbose=verbose)
    diagnostics["solver_seconds"] = perf_counter() - started
    for name, total in _COUNTERS.items():
        history.run_metrics[total] += diagnostics.get(name, 0)
    history.run_metrics["total_accepted_corrections"] += int(diagnostics.get("correction_accepted") is True)
    history.run_metrics["solver_seconds"] += diagnostics["solver_seconds"]
    return diagnostics


def _record(
    problem: SparseProblem, proximal: Proximal, train_loader: DataLoader,
    valid_loader: DataLoader | None, history: History, diagnostics: dict,
    verbose: bool, display_every: int,
) -> None:
    """Evaluate the current state before capturing its checkpoint."""
    started = perf_counter()
    values = metrics(
        problem, proximal, train_loader, valid_loader,
        reporting_normalizer=history.reporting_normalizer,
    )
    elapsed = perf_counter() - started
    history.run_metrics["evaluation_seconds"] += elapsed
    history.record(problem, values, {**diagnostics, "evaluation_seconds": elapsed})
    checkpoint = len(history.train_loss) - 1
    if verbose and display_every > 0 and checkpoint % display_every == 0:
        logger.info(
            "training_progress", checkpoint=checkpoint, neurons=problem.n_neurons,
            train_loss=values["train_loss"], val_loss=values["val_loss"],
            val_l2=values["err_l2_val"], val_h1=values["err_h1_val"],
        )


def _process_peak_rss_bytes() -> int:
    """Convert the process lifetime RSS high-water mark to bytes on macOS/Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
