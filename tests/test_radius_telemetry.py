"""Search diagnostics describe the executed search and remain optional in old fits."""

import math
import pickle

import pytest
import torch

from src.config import make_penalty, resolve_model, validate_model_config
from src.config.schema import ExperimentConfig, ModelConfig, TrainingConfig
from src.data import make_dataloader
from src.eval import metrics
from src.models import SparseNetwork
from src.PDAP import PDAP
from src.PDAP.history import History
from src.PDAP.insertion import FIXED_LOG_CLAMP
from src.training import run_training


@pytest.mark.parametrize("loop_order", ["insertion_first", "correction_first"])
@pytest.mark.parametrize("alpha", [1.0, 1e-8])
def test_nonzero_residual_records_computed_and_capped_theorem_radius(loop_order, alpha):
    cfg = ExperimentConfig(
        model=ModelConfig(activation="softplus", moment_order=4.0, alpha=alpha),
        training=TrainingConfig(num_of_initial_pts=2, radial_cap="theorem", loop_order=loop_order,
                                lbfgs_steps=2, fit_outer_iterations=1),
    )
    model_config = resolve_model(cfg.model, input_dim=2)
    validate_model_config(model_config)
    network = SparseNetwork(model_config)
    samples = (torch.tensor([[0., 0.], [3., 4.]], dtype=torch.float64),
               torch.ones(2, 1, dtype=torch.float64),
               torch.ones(2, 2, dtype=torch.float64))
    # Softplus: C_rho=s0=s1=1, A_M=sqrt(26), ||r||_M=sqrt(3), p-s1=3.
    expected = min((4 * (math.sqrt(26) + 1) * math.sqrt(3) / alpha) ** (1 / 3),
                   math.exp(FIXED_LOG_CLAMP))
    assert expected > 1
    history = run_training(
        PDAP(
            cfg.training, make_penalty(model_config), network, make_dataloader(samples),
            insertion=model_config["insertion"],
        ),
        num_iterations=1,
    )
    assert history.search_radius == pytest.approx([expected])
    assert history.radius_theorem_applied == [1.0]
    assert history.summary_metrics()["search_radius"] == pytest.approx(expected)


@pytest.mark.parametrize("activation,power,insertion,cap,order,expected_radius,applied", [
    ("softplus", 1.0, "profile", "theorem", 4.0, 1.0, 1.0),
    ("snake_b0_25", 1.0, "profile", "theorem", 2.0, math.exp(FIXED_LOG_CLAMP), 0.0),
    ("softplus", 1.0, "profile", "fixed", 4.0, math.exp(FIXED_LOG_CLAMP), -1.0),
    ("relu", 2.0, "finite_step", "theorem", 4.0, 1.0, -1.0),
])
def test_zero_measure_fit_records_the_actual_search_radius(
    activation, power, insertion, cap, order, expected_radius, applied,
):
    cfg = ExperimentConfig(
        model=ModelConfig(activation=activation, power=power, insertion=insertion,
                          moment_order=order),
        training=TrainingConfig(num_of_initial_pts=2, radial_cap=cap, lbfgs_steps=2),
    )
    model_config = resolve_model(cfg.model, input_dim=2)
    validate_model_config(model_config)
    network = SparseNetwork(model_config)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.zeros(3, 1, dtype=torch.float64),
               torch.zeros(3, 2, dtype=torch.float64))
    history = run_training(
        PDAP(
            cfg.training, make_penalty(model_config), network, make_dataloader(samples),
            insertion=model_config["insertion"],
        ),
        num_iterations=1,
    )
    assert history.final_neurons == 0
    assert history.search_radius == [expected_radius]
    assert history.radius_theorem_applied == [applied]
    metrics = history.summary_metrics()
    assert metrics["search_radius"] == expected_radius
    assert metrics["radius_theorem_applied"] == applied


def test_old_history_summary_does_not_invent_missing_search_diagnostics():
    config = resolve_model({}, 2)
    network = SparseNetwork(config)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.ones(3, 1, dtype=torch.float64),
               torch.ones(3, 2, dtype=torch.float64))
    history = History()
    history.record(
        network, metrics(network, make_penalty(config), make_dataloader(samples)),
        {"search_radius": 1.0, "radius_theorem_applied": 1.0},
    )
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
        model=ModelConfig(activation=activation, power=power, insertion=insertion, moment_order=2.0),
        training=TrainingConfig(num_of_initial_pts=2, radial_cap="theorem",
                                loop_order="insertion_first", lbfgs_steps=2),
    )
    model_config = resolve_model(cfg.model, input_dim=2)
    validate_model_config(model_config)
    network = SparseNetwork(model_config)
    samples = (torch.zeros(3, 2, dtype=torch.float64),
               torch.zeros(3, 1, dtype=torch.float64),
               torch.zeros(3, 2, dtype=torch.float64))
    trainer = PDAP(
        cfg.training, make_penalty(model_config), network, make_dataloader(samples),
        insertion=model_config["insertion"],
    )
    previous = run_training(trainer, num_iterations=1)
    assert previous.search_radius == [1.0]

    history = run_training(trainer, num_iterations=0)
    assert history.search_radius == [None]
    assert history.radius_theorem_applied == [None]
    assert "search_radius" not in history.summary_metrics()
    assert "radius_theorem_applied" not in history.summary_metrics()
