"""Search diagnostics describe the executed search and remain optional in old fits."""

import math

import pytest
import torch

from src.config import make_penalty, resolve_model, validate_model_config
from src.config.schema import ExperimentConfig, ModelConfig, TrainingConfig
from src.data import make_dataloader
from src.models import SparseNetwork
from src.PDAP import PDAP
from src.PDAP.insertion import FIXED_LOG_CLAMP
from src.training import run_training


@pytest.mark.parametrize("loop_order", ["insertion_first", "correction_first"])
@pytest.mark.parametrize("alpha", [1.0, 1e-8])
def test_nonzero_residual_records_whether_the_cap_bounded_the_theorem_radius(loop_order, alpha):
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
    theorem_radius = (4 * (math.sqrt(26) + 1) * math.sqrt(3) / alpha) ** (1 / 3)
    expected = float(theorem_radius > math.exp(FIXED_LOG_CLAMP))
    history = run_training(
        PDAP(
            cfg.training, make_penalty(model_config), network, make_dataloader(samples),
            insertion=model_config["insertion"],
        ),
        num_iterations=1,
    )
    assert [d.get("cap_of_theorem_applied") for d in history.diagnostics] == [expected]
    assert history.summary()["cap_of_theorem_applied"] == expected


@pytest.mark.parametrize("activation,power,insertion,cap,order,expected", [
    ("softplus", 1.0, "profile", "theorem", 4.0, 0.0),
    ("snake_b0_25", 1.0, "profile", "theorem", 2.0, None),
    ("softplus", 1.0, "profile", "fixed", 4.0, None),
    ("relu", 2.0, "finite_step", "theorem", 4.0, None),
])
def test_zero_measure_fit_records_whether_the_cap_was_applied(
    activation, power, insertion, cap, order, expected,
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
    assert history.run["final_neurons"] == 0
    assert [d.get("cap_of_theorem_applied") for d in history.diagnostics] == [expected]
    assert history.summary().get("cap_of_theorem_applied") == expected


def test_zero_iteration_fit_does_not_reuse_a_previous_search():
    cfg = ExperimentConfig(
        model=ModelConfig(activation="softplus", moment_order=2.0),
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
    assert [d.get("cap_of_theorem_applied") for d in previous.diagnostics] == [0.0]

    history = run_training(trainer, num_iterations=0)
    assert [d.get("cap_of_theorem_applied") for d in history.diagnostics] == [None]
    assert "cap_of_theorem_applied" not in history.summary()
