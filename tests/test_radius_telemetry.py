"""Search diagnostics describe the executed search and remain optional in old fits."""

import math
import pickle

import pytest
import torch

from src.config.schema import ExperimentConfig, ModelConfig, TrainingConfig
from src.models import build_model
from src.PDAP import PDAP
from src.PDAP.history import History
from src.PDAP.radius import FIXED_LOG_CLAMP
from src.PDAP.ssn_solve import Objective


@pytest.mark.parametrize("loop_order", ["insertion_first", "correction_first"])
@pytest.mark.parametrize("alpha", [1.0, 1e-8])
def test_nonzero_residual_records_computed_and_capped_theorem_radius(loop_order, alpha):
    cfg = ExperimentConfig(
        model=ModelConfig(activation="softplus", moment_order=4.0, alpha=alpha),
        training=TrainingConfig(radial_cap="theorem", loop_order=loop_order,
                                lbfgs_steps=2, fit_outer_iterations=1),
    )
    model = build_model(cfg, input_dim=2)
    samples = (torch.tensor([[0., 0.], [3., 4.]], dtype=torch.float64),
               torch.ones(2, 1, dtype=torch.float64),
               torch.ones(2, 2, dtype=torch.float64))
    # Softplus: C_rho=s0=s1=1, A_M=sqrt(26), ||r||_M=sqrt(3), p-s1=3.
    expected = min((4 * (math.sqrt(26) + 1) * math.sqrt(3) / alpha) ** (1 / 3),
                   math.exp(FIXED_LOG_CLAMP))
    assert expected > 1
    history = PDAP(cfg).fit(model, samples, samples, num_iterations=1,
                            num_insertion=2, verbose=False)
    assert history.search_radius == pytest.approx([expected])
    assert history.radius_theorem_applied == [1.0]
    assert history.summary_metrics()["search_radius"] == pytest.approx(expected)


@pytest.mark.parametrize("activation,power,insertion,cap,order,expected_radius,applied", [
    ("softplus", 1.0, "profile", "theorem", 4.0, 1.0, 1.0),
    ("gelu_squared", 1.0, "profile", "theorem", 2.0, math.exp(FIXED_LOG_CLAMP), 0.0),
    ("softplus", 1.0, "profile", "fixed", 4.0, math.exp(FIXED_LOG_CLAMP), -1.0),
    ("relu", 2.0, "finite_step", "theorem", 4.0, 1.0, -1.0),
])
def test_zero_measure_fit_records_the_actual_search_radius(
    activation, power, insertion, cap, order, expected_radius, applied,
):
    cfg = ExperimentConfig(
        model=ModelConfig(activation=activation, power=power, insertion=insertion,
                          moment_order=order),
        training=TrainingConfig(radial_cap=cap, lbfgs_steps=2),
    )
    model = build_model(cfg, input_dim=2)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.zeros(3, 1, dtype=torch.float64),
               torch.zeros(3, 2, dtype=torch.float64))
    history = PDAP(cfg).fit(model, samples, samples, num_iterations=1,
                            num_insertion=2, verbose=False)
    assert history.final_neurons == 0
    assert history.search_radius == [expected_radius]
    assert history.radius_theorem_applied == [applied]
    metrics = history.summary_metrics()
    assert metrics["search_radius"] == expected_radius
    assert metrics["radius_theorem_applied"] == applied


def test_old_history_summary_does_not_invent_missing_search_diagnostics():
    model = build_model(ExperimentConfig(), input_dim=2)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.ones(3, 1, dtype=torch.float64),
               torch.ones(3, 2, dtype=torch.float64))
    history = History()
    history.record(model, Objective(), samples, samples, search_radius=1.0, theorem_applied=1.0)
    expected = history.summary_metrics()
    for name in ("search_radius", "radius_theorem_applied"):
        expected.pop(name)
        delattr(history, name)
    legacy = pickle.loads(pickle.dumps(history))
    assert legacy.summary_metrics() == expected


@pytest.mark.parametrize("activation,power,insertion", [
    ("softplus", 1.0, "profile"), ("relu", 2.0, "finite_step"),
])
def test_zero_iteration_fit_does_not_reuse_a_previous_search(activation, power, insertion):
    cfg = ExperimentConfig(
        model=ModelConfig(activation=activation, power=power, insertion=insertion),
        training=TrainingConfig(radial_cap="theorem", loop_order="insertion_first",
                                lbfgs_steps=2),
    )
    model = build_model(cfg, input_dim=2)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.zeros(3, 1, dtype=torch.float64),
               torch.zeros(3, 2, dtype=torch.float64))
    trainer = PDAP(cfg)
    previous = trainer.fit(model, samples, samples, num_iterations=1,
                           num_insertion=2, verbose=False)
    assert previous.search_radius == [1.0]

    history = trainer.fit(model, samples, samples, num_iterations=0,
                          num_insertion=2, verbose=False)
    assert history.search_radius == [None]
    assert history.radius_theorem_applied == [None]
    assert "search_radius" not in history.summary_metrics()
    assert "radius_theorem_applied" not in history.summary_metrics()
