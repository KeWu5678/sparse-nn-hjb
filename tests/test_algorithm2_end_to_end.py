"""End-to-end regressions for the supported Algorithm 2 powers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config.schema import EnvConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import split_value_samples
from src.models import build_model
from src.PDAP import PDAP


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
def test_algorithm2_fits_an_exact_relu_power_atom(power: float) -> None:
    torch.manual_seed(17)
    np.random.seed(17)
    x = torch.rand(50, 2, dtype=torch.float64) * 2.0 - 1.0
    preactivation = x[:, :1] + 0.2
    value = torch.relu(preactivation).pow(power)
    gradient = torch.zeros_like(x)
    gradient[:, :1] = power * torch.relu(preactivation).pow(power - 1.0)
    data = {"x": x.numpy(), "v": value.numpy(), "dv": gradient.numpy()}
    cfg = ExperimentConfig(
        model=ModelConfig(
            kind="signed",
            insertion="finite_step",
            activation="relu",
            power=power,
            alpha=1e-5,
            gamma=0.0,
        ),
        training=TrainingConfig(
            insert_mode="sequential",
            loop_order="insertion_first",
            correction_guard=True,
            fit_outer_iterations=5,
            lbfgs_steps=50,
        ),
        env=EnvConfig(verbose=False),
    )
    model = build_model(cfg, input_dim=2)
    train, valid = split_value_samples(data, cfg.data.train_fraction)

    history = PDAP(cfg).fit(
        model,
        train,
        valid,
        num_iterations=2,
        num_insertion=20,
        max_insert=1,
        verbose=False,
    )

    assert history.final_neurons > 0
    assert np.all(np.diff(np.asarray(history.train_loss)) <= 1e-12)
