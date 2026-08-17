"""Behavioral tests for the scalar power-penalty proximal map."""

from __future__ import annotations

import pytest
import torch

from src.SSN.prox import power_prox, power_prox_derivative


def test_q_half_prox_rejects_a_non_global_stationary_point() -> None:
    """The global prox is zero between the fold and the objective switch."""
    value = torch.tensor([1.3], dtype=torch.float64)

    result = power_prox(value, 1.0, q=0.5)

    torch.testing.assert_close(result, torch.zeros_like(value))


def test_q_two_thirds_prox_rejects_a_non_global_stationary_point() -> None:
    """The q=2/3 map also switches by objective value, not root existence."""
    value = torch.tensor([1.8], dtype=torch.float64)

    result = power_prox(value, 1.5, q=2.0 / 3.0)

    torch.testing.assert_close(result, torch.zeros_like(value))


def test_fractional_prox_rejects_an_unsupported_exponent() -> None:
    with pytest.raises(ValueError, match="q must be one of"):
        power_prox(torch.tensor([2.0], dtype=torch.float64), 1.0, q=0.4)


def test_fractional_prox_derivative_is_zero_when_global_prox_is_zero() -> None:
    value = torch.tensor([1.3], dtype=torch.float64)
    prox = power_prox(value, 1.0, q=0.5)

    derivative = power_prox_derivative(value, 1.0, q=0.5, prox_result=prox)

    assert derivative.item() == 0.0


@pytest.mark.parametrize("q", [0.5, 2.0 / 3.0])
def test_fractional_prox_with_zero_scale_is_the_identity(q: float) -> None:
    value = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)
    scale = torch.zeros_like(value)

    result = power_prox(value, scale, q=q)
    derivative = power_prox_derivative(value, scale, q=q, prox_result=result)

    torch.testing.assert_close(result, value)
    torch.testing.assert_close(derivative, torch.eye(3, dtype=torch.float64))


@pytest.mark.parametrize(
    ("q", "scale", "value", "expected", "expected_derivative"),
    [
        (0.5, 1.0, 4.25, 4.0, 32.0 / 31.0),
        (2.0 / 3.0, 1.5, 8.5, 8.0, 48.0 / 47.0),
    ],
)
def test_fractional_prox_closed_forms_on_the_active_branch(
    q: float,
    scale: float,
    value: float,
    expected: float,
    expected_derivative: float,
) -> None:
    inputs = torch.tensor([-value, value], dtype=torch.float64)
    prox = power_prox(inputs, scale, q=q)
    derivative = power_prox_derivative(inputs, scale, q=q, prox_result=prox)

    torch.testing.assert_close(
        prox, torch.tensor([-expected, expected], dtype=torch.float64)
    )
    torch.testing.assert_close(
        torch.diagonal(derivative),
        torch.full((2,), expected_derivative, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("q", "scale", "switch_input"),
    [(0.5, 1.0, 1.5), (2.0 / 3.0, 1.5, 2.0)],
)
def test_fractional_prox_selects_zero_at_the_global_tie(
    q: float, scale: float, switch_input: float
) -> None:
    value = torch.tensor([-switch_input, switch_input], dtype=torch.float64)

    result = power_prox(value, scale, q=q)

    torch.testing.assert_close(result, torch.zeros_like(value))


@pytest.mark.parametrize(
    ("q", "input_value", "normalization"),
    [(0.5, 4.25, 1e-80), (2.0 / 3.0, 8.5, 1e-60)],
)
def test_fractional_prox_respects_its_exact_scaling_law(
    q: float, input_value: float, normalization: float
) -> None:
    base_input = torch.tensor([input_value], dtype=torch.float64)
    base_scale = 1.0
    expected = power_prox(base_input, base_scale, q=q) * normalization
    scaled_mu = normalization ** (2.0 - q)

    result = power_prox(base_input * normalization, scaled_mu, q=q)

    torch.testing.assert_close(result, expected, rtol=1e-12, atol=0.0)


def test_q_two_thirds_closed_form_stays_finite_for_a_large_input() -> None:
    value = torch.tensor([1e100], dtype=torch.float64)

    result = power_prox(value, 1.0, q=2.0 / 3.0)

    assert torch.isfinite(result).all()
    residual = result + (2.0 / 3.0) * result.pow(-1.0 / 3.0) - value
    assert abs(float(residual / value)) < 1e-12
