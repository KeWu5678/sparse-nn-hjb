"""Behavioral tests for Algorithm 2's one-node insertion step."""

from __future__ import annotations

import pytest
import torch

from src.config import make_penalty, resolve_model, validate_model_config
from src.models import SparseNetwork
from src.PDAP import insertion


def _insert(monkeypatch, config, profile, curvature=1.0):
    network = SparseNetwork(config)
    W = torch.zeros((1, 1), dtype=torch.float64)
    b = torch.ones(1, dtype=torch.float64)
    monkeypatch.setattr(insertion, "_generate_candidates", lambda *args, **kwargs: (W, b, 0))
    residual = W, W.new_tensor([[profile / curvature]]), torch.zeros_like(W)
    return insertion.finite_step(
        network, make_penalty(config), residual, initial_points=(W, b),
    )[2]


def test_q_half_insertion_minimizes_the_actual_objective_increment(monkeypatch) -> None:
    """For P=4.25, A=alpha=1, the exact nonzero minimizer is c=-4."""
    config = resolve_model({"alpha": 1.0, "power": 3.0, "insertion": "finite_step"}, 1)
    validate_model_config(config)
    result = _insert(monkeypatch, config, 4.25)

    assert result.shape == (1,)
    coefficient = result[0]
    increment = 4.25 * coefficient + 0.5 * coefficient**2 + abs(coefficient) ** 0.5
    assert coefficient == pytest.approx(-4.0)
    assert increment == pytest.approx(-7.0)


def test_q_two_thirds_insertion_minimizes_the_actual_objective_increment(monkeypatch) -> None:
    config = resolve_model({"alpha": 1.5, "power": 2.0, "insertion": "finite_step"}, 1)
    validate_model_config(config)
    result = _insert(monkeypatch, config, 8.5)

    assert result.shape == (1,)
    coefficient = result[0]
    increment = (
        8.5 * coefficient
        + 0.5 * coefficient**2
        + 1.5 * abs(coefficient) ** (2.0 / 3.0)
    )
    assert coefficient == pytest.approx(-8.0)
    assert increment == pytest.approx(-30.0)


def test_q_one_insertion_keeps_the_soft_threshold_rule(monkeypatch) -> None:
    config = resolve_model({"alpha": 1.0, "loss_weights": (2., 1.), "insertion": "finite_step"}, 1)
    validate_model_config(config)
    result = _insert(monkeypatch, config, -3.0, curvature=2.0)

    assert result.shape == (1,)
    coefficient = result[0]
    increment = -3.0 * coefficient + coefficient**2 + abs(coefficient)
    assert coefficient == pytest.approx(1.0)
    assert increment == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("profile", "alpha", "q"),
    [(1.5, 1.0, 0.5), (2.0, 1.5, 2.0 / 3.0)],
)
def test_fractional_insertion_rejects_a_global_proximal_tie(
    monkeypatch, profile: float, alpha: float, q: float
) -> None:
    config = resolve_model({"alpha": alpha, "power": 2.0 / q - 1.0, "insertion": "finite_step"}, 1)
    validate_model_config(config)
    assert _insert(monkeypatch, config, profile).shape == (0,)


@pytest.mark.parametrize("alpha,power,profile,expected", [
    (1e-41, 3., 1e-20, -9.486650001264143e-41),
    (1e-20, 1., 2e-20, -1.9999999959435182e-40),
])
def test_insertion_preserves_the_power_prox_and_exact_tiny_penalty(monkeypatch, alpha, power, profile, expected):
    config = resolve_model({
        "alpha": alpha, "power": power, "loss_weights": (1e20, 1.), "insertion": "finite_step",
    }, 1)
    validate_model_config(config)
    result = _insert(monkeypatch, config, profile, curvature=1e20)
    assert result.shape == (1,)
    c = result[0]
    assert c == pytest.approx(expected, rel=1e-12, abs=0.)
    increment = profile * c + .5 * 1e20 * c**2 + alpha * abs(c)**(2 / (power + 1))
    assert increment < 0.
