"""Tests for Algorithm 1's normalized-measure primitives and network identity."""

from __future__ import annotations

import math

import pytest
import torch

from src.config import resolve_model, validate_model_config
from src.config.schema import ExperimentConfig, ModelConfig
from src.eval import amplitude_mass_radius
from src.models import SparseNetwork


def _cfg(**model_kwargs) -> ExperimentConfig:
    return ExperimentConfig(model=ModelConfig(**model_kwargs))


def moment_weight(W: torch.Tensor, b: torch.Tensor, p: float) -> torch.Tensor:
    """Moment weights of the supplied atoms under a network of moment order p."""
    network = SparseNetwork(resolve_model({"activation": "softplus", "power": 1.0, "moment_order": p}, 2))
    return network.moment_weight_of_atoms(W, b)


def test_moment_weight_p2_is_one_plus_squared_norm() -> None:
    W = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([0.0, 2.0], dtype=torch.float64)

    got = moment_weight(W, b, p=2.0)

    assert torch.allclose(got, torch.tensor([26.0, 5.0], dtype=torch.float64))


def test_moment_weight_general_p() -> None:
    W = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    b = torch.tensor([12.0], dtype=torch.float64)

    got = moment_weight(W, b, p=3.0)

    assert math.isclose(float(got.item()), 1.0 + 13.0**3, rel_tol=1e-12)


def test_moment_weight_single_atom_scalar_bias() -> None:
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)

    got = moment_weight(a, torch.tensor(0.0, dtype=torch.float64), p=2.0)

    assert math.isclose(float(got), 2.0, rel_tol=1e-12)


def test_moment_weight_is_differentiable_in_scale() -> None:
    scale = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0], dtype=torch.float64)

    weight = moment_weight(scale * a, scale * b, p=2.0)
    weight.backward()

    assert math.isclose(float(scale.grad), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("order", [None, 0.0, -1.0])
def test_moment_weight_requires_a_positive_order(order) -> None:
    config = resolve_model({"activation": "softplus", "power": 1.0, "moment_order": order}, 2)
    with pytest.raises(ValueError, match="moment_order"):
        validate_model_config(config)


def test_amplitude_mass_radius_is_weighted_parameter_quantile() -> None:
    radius = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    c = torch.tensor([50.0, -45.0, 5.0], dtype=torch.float64)

    got = amplitude_mass_radius(radius, c, mass_fraction=0.95)

    assert math.isclose(float(got), 2.0, rel_tol=1e-12)


def test_amplitude_mass_radius_is_zero_without_amplitude_mass() -> None:
    radius = torch.tensor([5.0], dtype=torch.float64)
    c = torch.tensor([0.0], dtype=torch.float64)

    assert float(amplitude_mass_radius(radius, c)) == 0.0
    assert float(amplitude_mass_radius(radius[:0], c[:0])) == 0.0


def test_nonhomogeneous_signed_profile_is_normalized_automatically() -> None:
    network = SparseNetwork(resolve_model({"activation": "softplus", "power": 1.0}, 2))

    assert network.normalized


def test_sphere_profile_model_is_not_moment_normalized() -> None:
    sphere = SparseNetwork(resolve_model({"activation": "relu", "power": 1.0}, 2))

    assert not sphere.normalized


def test_algorithm1_rejects_nonunit_activation_power() -> None:
    with pytest.raises(ValueError, match="Algorithm 1.*power == 1"):
        validate_model_config(resolve_model({"activation": "softplus", "power": 2.0}, 2))


def test_model_schema_has_no_additive_objective_axis() -> None:
    fields = ModelConfig.__dataclass_fields__

    assert "objective" not in fields
    assert "moment_beta" not in fields
    assert ModelConfig().moment_order is None
