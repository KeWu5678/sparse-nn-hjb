"""Optimization work and recorded checkpoints have distinct lifetimes."""

from itertools import count
from types import SimpleNamespace

import pytest
import torch

import src.training as training
from src.config import resolve_model, validate_training
from src.config.schema import TrainingConfig
from src.data import make_dataloader
from src.eval import metrics
from src.PDAP import PDAP, History
from src.SSN import SSN


def _case(monkeypatch, *, order="insertion_first", accepted=(True,)):
    from src.models import SparseModel, make_penalty

    config = resolve_model({"alpha": 0.0}, 1)
    validate_training(config)
    problem = SparseModel(config)
    x = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    loader = make_dataloader((x, x.clone(), torch.ones_like(x)))
    proximal = make_penalty(config["alpha"], config["gamma"], problem.penalty_exponent)
    pdap = PDAP(TrainingConfig(loop_order=order, fit_outer_iterations=0), proximal)
    searches = iter(accepted)
    search_number = count(1)

    def candidates(*args, **kwargs):
        pdap._last_search_radius = 10.0 * next(search_number)
        pdap._last_theorem_applied = 1.0
        size = int(next(searches))
        return x.new_ones((size, 1)), x.new_zeros(size), x.new_ones(size)

    monkeypatch.setattr(pdap, "_insert", candidates)
    return problem, loader, pdap


def test_step_completes_optimization_without_quality_evaluation(monkeypatch):
    problem, loader, pdap = _case(monkeypatch, order="correction_first")

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("PDAP.step must not invoke quality evaluation")

    monkeypatch.setattr(training, "metrics", unexpected_evaluation)
    diagnostics = pdap.step(problem, loader)

    assert problem.n_neurons == 1
    assert diagnostics["inserted"] == diagnostics["corrections"] == 1
    assert diagnostics["pruned"] == 0
    assert diagnostics["stop_reason"] is None
    assert diagnostics["ssn_iterations"] == diagnostics["ssn_failed_steps"] == 0
    # A zero-budget SSN correction accepts its initial coefficients; this differs
    # from a search that never invokes correction at all.
    assert diagnostics["correction_accepted"] is True
    assert metrics(problem, pdap.proximal, loader)["train_loss"] == 0.0


def test_ssn_trial_failure_and_final_acceptance_reach_history_and_run_totals(monkeypatch):
    problem, loader, pdap = _case(monkeypatch)
    pdap.fit_outer_iterations = 4
    pdap.solver["method"] = "steihaug_cg"
    trials = 0

    def objective(u):
        nonlocal trials
        value = .5 * u.square().sum()
        if not u.requires_grad:
            trials += 1
            # Reject one trial, then let the same real SSN correction recover.
            if trials == 2:
                return value + 1.
        return value

    def scripted_correction(params, model, loss, hess_obj, **options):
        return SSN(params, model, objective, torch.eye(1, dtype=torch.float64), **options)

    monkeypatch.setattr("src.PDAP.pdap.SSN", scripted_correction)
    history = training.run_training(pdap, problem, loader, num_iterations=1, verbose=False)

    diagnostics = history.diagnostics[0]
    assert trials == diagnostics["ssn_iterations"] == history.run_metrics["total_ssn_iterations"] == 4
    assert diagnostics["ssn_failed_steps"] == history.run_metrics["total_ssn_failed_steps"] == 1
    assert diagnostics["ssn_last_step_success"] is diagnostics["correction_accepted"] is True
    assert history.run_metrics["total_accepted_corrections"] == 1
    assert history.model_states[0]["c"].item() == pytest.approx(.78)


@pytest.mark.parametrize("order,searches,final_width", [
    ("insertion_first", 2, 2),
    ("correction_first", 3, 3),
])
def test_each_record_scores_its_checkpoint_and_totals_include_trailing_work(
    monkeypatch, order, searches, final_width,
):
    problem, loader, pdap = _case(monkeypatch, order=order, accepted=(True,) * searches)
    history = training.run_training(pdap, problem, loader, num_iterations=2, verbose=False)

    assert [state["c"].numel() for state in history.model_states] == [1, 2]
    assert history.final_neurons == problem.n_neurons == final_width
    assert history.search_radius == [10.0, 20.0]
    assert history.best_iteration == 0
    assert history.run_metrics["total_insertion_searches"] == searches
    assert history.run_metrics["total_inserted_neurons"] == searches
    assert history.run_metrics["total_coefficient_corrections"] == 2
    assert history.run_metrics["total_accepted_corrections"] == 2
    assert history.run_metrics["stop_reason"] == "iteration_budget"
    assert history.summary_metrics()["iterations"] == 2
    for i in range(2):
        history.restore_model(problem, i)
        assert history.train_loss[i] == pytest.approx(metrics(problem, pdap.proximal, loader)["train_loss"])


def test_terminal_search_is_counted_without_a_duplicate_checkpoint(monkeypatch):
    problem, loader, pdap = _case(monkeypatch, accepted=(True, False))
    # First solver step takes 2s, evaluation 3s, terminal search 7s.
    times = iter((0.0, 2.0, 2.0, 5.0, 5.0, 12.0))
    monkeypatch.setattr(training, "perf_counter", lambda: next(times))
    monkeypatch.setattr(training, "_process_peak_rss_bytes", lambda: 4096)
    history = training.run_training(pdap, problem, loader, num_iterations=5, verbose=False)

    assert len(history.train_loss) == len(history.diagnostics) == 1
    assert history.run_metrics["total_insertion_searches"] == 2
    assert history.run_metrics["total_coefficient_corrections"] == 1
    assert history.run_metrics["total_ssn_iterations"] == 0
    assert history.run_metrics["solver_seconds"] == 9.0
    assert history.run_metrics["evaluation_seconds"] == 3.0
    assert history.run_metrics["process_peak_rss_bytes"] == 4096
    assert history.summary_metrics()["stop_reason"] == "no_candidate"
    assert history.search_radius == [10.0]  # The accepted search, not the final failure.


@pytest.mark.parametrize("order", ["insertion_first", "correction_first"])
def test_empty_terminal_model_is_observed_without_inventing_correction_work(monkeypatch, order):
    problem, loader, pdap = _case(monkeypatch, order=order, accepted=(False,))
    history = training.run_training(pdap, problem, loader, num_iterations=3, verbose=False)

    assert history.final_neurons == 0
    assert len(history.train_loss) == 1
    assert history.err_h1_train == [1.0]
    assert history.run_metrics["total_insertion_searches"] == 1
    assert history.run_metrics["total_coefficient_corrections"] == 0
    assert history.run_metrics["total_accepted_corrections"] == 0
    assert history.diagnostics[0].get("correction_accepted") is None
    assert history.run_metrics["stop_reason"] == "no_candidate"


def test_failed_legacy_initialization_keeps_existing_model_and_reports_its_size(monkeypatch):
    problem, loader, pdap = _case(monkeypatch, order="correction_first", accepted=(False,))
    problem.set_atoms(
        torch.tensor([[2.]], dtype=torch.float64),
        torch.tensor([0.], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    history = training.run_training(pdap, problem, loader, num_iterations=3, verbose=False)

    assert history.final_neurons == history.best_neurons == problem.n_neurons == 1
    assert history.model_states[0]["inner_weight"].item() == 2.0
    assert history.train_loss[0] == metrics(problem, pdap.proximal, loader)["train_loss"]
    assert history.run_metrics["total_insertion_searches"] == 1
    assert history.run_metrics["total_coefficient_corrections"] == 0
    assert history.summary_metrics()["stop_reason"] == "no_candidate"


@pytest.mark.parametrize("order,searches", [("insertion_first", 0), ("correction_first", 1)])
def test_zero_budget_records_existing_or_legacy_initialized_state(monkeypatch, order, searches):
    problem, loader, pdap = _case(monkeypatch, order=order)
    problem.set_atoms(
        torch.tensor([[2.]], dtype=torch.float64),
        torch.tensor([0.], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    history = training.run_training(pdap, problem, loader, num_iterations=0, verbose=False)

    assert len(history.train_loss) == 1
    assert history.train_loss[0] == metrics(problem, pdap.proximal, loader)["train_loss"]
    assert history.run_metrics["total_insertion_searches"] == searches
    assert history.run_metrics["total_coefficient_corrections"] == 0
    assert history.run_metrics["stop_reason"] == "iteration_budget"
    assert history.search_radius == ([10.0] if searches else [None])
    assert history.model_states[0]["inner_weight"].item() == (1.0 if searches else 2.0)


def test_history_stores_supplied_values_without_prediction_and_copies_diagnostics(monkeypatch):
    problem, loader, pdap = _case(monkeypatch)
    values = metrics(problem, pdap.proximal, loader)
    diagnostics = {"search_radius": 2.0, "correction_accepted": False}

    def unexpected_prediction(*args, **kwargs):
        pytest.fail("History.record must consume already evaluated metrics")

    monkeypatch.setattr(problem, "forward", unexpected_prediction)
    history = History()
    history.record(problem, values, diagnostics)
    diagnostics["search_radius"] = 99.0
    values["train_loss"] = 99.0

    assert history.train_loss[0] != 99.0
    assert history.search_radius == [2.0]
    assert history.diagnostics[0]["search_radius"] == 2.0


@pytest.mark.parametrize("platform,expected", [("darwin", 12), ("linux", 12 * 1024)])
def test_process_lifetime_rss_units(monkeypatch, platform, expected):
    monkeypatch.setattr(training.sys, "platform", platform)
    monkeypatch.setattr(training.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=12))
    assert training._process_peak_rss_bytes() == expected
