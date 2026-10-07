"""Behavioral tests for the scalar power-penalty proximal map."""

from __future__ import annotations

import pytest
import torch

from src.problems.prox import power_prox, power_prox_derivative


@pytest.mark.parametrize("tensor_scale", [False, True])
@pytest.mark.parametrize(
    ("scale", "expected_diagonal"),
    [
        (1.0, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        (0.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_soft_threshold_derivative_at_threshold_and_zero_scale(
    scale: float, expected_diagonal: list[float], tensor_scale: bool
) -> None:
    """Keep the active-side choice at the kink and the zero-scale identity."""
    v = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    lambda_ = torch.full_like(v, scale) if tensor_scale else scale

    derivative = power_prox_derivative(v=v, lambda_=lambda_, penalty_exponent=1.0)

    expected = torch.diag(torch.tensor(expected_diagonal, dtype=v.dtype))
    torch.testing.assert_close(derivative, expected, rtol=0.0, atol=0.0)


def test_q_half_prox_rejects_a_non_global_stationary_point() -> None:
    """The global prox is zero between the fold and the objective switch."""
    value = torch.tensor([1.3], dtype=torch.float64)

    result = power_prox(value, 1.0, penalty_exponent=0.5)

    torch.testing.assert_close(result, torch.zeros_like(value))


def test_q_two_thirds_prox_rejects_a_non_global_stationary_point() -> None:
    """The q=2/3 map also switches by objective value, not root existence."""
    value = torch.tensor([1.8], dtype=torch.float64)

    result = power_prox(value, 1.5, penalty_exponent=2.0 / 3.0)

    torch.testing.assert_close(result, torch.zeros_like(value))


@pytest.mark.parametrize("q", [0.4, 1.0 + 5e-13, 0.5 + 2e-12, 2.0 / 3.0 + 2e-12])
def test_prox_and_derivative_reject_the_same_unsupported_exponents(q: float) -> None:
    v = torch.tensor([2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="q must be one of"):
        power_prox(v, 1.0, penalty_exponent=q)
    with pytest.raises(ValueError, match="q must be one of"):
        power_prox_derivative(v, 1.0, penalty_exponent=q, prox_result=v)


@pytest.mark.parametrize("q", [0.5, 2.0 / 3.0])
@pytest.mark.parametrize("offset", [-5e-13, 0.0, 5e-13])
def test_prox_and_derivative_use_the_same_supported_exponent(
    q: float, offset: float
) -> None:
    v = torch.tensor([-4.25, 0.0, 8.5], dtype=torch.float64)
    expected_prox = power_prox(v, 1.0, penalty_exponent=q)
    expected_derivative = power_prox_derivative(v, 1.0, penalty_exponent=q, prox_result=expected_prox)

    prox = power_prox(v, 1.0, penalty_exponent=q + offset)
    derivative = power_prox_derivative(v, 1.0, penalty_exponent=q + offset, prox_result=prox)

    torch.testing.assert_close(prox, expected_prox, rtol=0.0, atol=0.0)
    torch.testing.assert_close(derivative, expected_derivative, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("q", [0.5, 2.0 / 3.0])
def test_fractional_derivative_requires_the_precomputed_prox(q: float) -> None:
    with pytest.raises(ValueError, match="prox_result from power_prox is required"):
        power_prox_derivative(torch.tensor([2.0], dtype=torch.float64), 1.0, penalty_exponent=q)


def test_fractional_prox_derivative_is_zero_when_global_prox_is_zero() -> None:
    value = torch.tensor([1.3], dtype=torch.float64)
    prox = power_prox(value, 1.0, penalty_exponent=0.5)

    derivative = power_prox_derivative(value, 1.0, penalty_exponent=0.5, prox_result=prox)

    assert derivative.item() == 0.0


@pytest.mark.parametrize("q", [0.5, 2.0 / 3.0])
def test_fractional_prox_with_zero_scale_is_the_identity(q: float) -> None:
    value = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)
    scale = torch.zeros_like(value)

    result = power_prox(value, scale, penalty_exponent=q)
    derivative = power_prox_derivative(value, scale, penalty_exponent=q, prox_result=result)

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
    prox = power_prox(inputs, scale, penalty_exponent=q)
    derivative = power_prox_derivative(inputs, scale, penalty_exponent=q, prox_result=prox)

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

    result = power_prox(value, scale, penalty_exponent=q)

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
    expected = power_prox(base_input, base_scale, penalty_exponent=q) * normalization
    scaled_mu = normalization ** (2.0 - q)

    result = power_prox(base_input * normalization, scaled_mu, penalty_exponent=q)

    torch.testing.assert_close(result, expected, rtol=1e-12, atol=0.0)


def test_q_two_thirds_closed_form_stays_finite_for_a_large_input() -> None:
    value = torch.tensor([1e100], dtype=torch.float64)

    result = power_prox(value, 1.0, penalty_exponent=2.0 / 3.0)

    assert torch.isfinite(result).all()
    residual = result + (2.0 / 3.0) * result.pow(-1.0 / 3.0) - value
    assert abs(float(residual / value)) < 1e-12
