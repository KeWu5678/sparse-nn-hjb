"""Behavioral tests for Algorithm 2's outer-weight correction."""

from __future__ import annotations

import pytest
import torch

from src.models import SignedModel
from src.PDAP.ssn_solve import (
    Objective,
    SolverConfig,
    nonconvex_penalty,
    ssn_solve,
    warmstart_prox_scale,
)
from src.SSN import SSN


def test_warmstart_prox_scale_uses_all_nonzero_penalized_coefficients() -> None:
    coefficients = torch.tensor([0.0, -0.5, 2.0, 0.01], dtype=torch.float64)
    penalized = torch.tensor([True, True, True, False])

    scale = warmstart_prox_scale(
        coefficients, penalized, alpha=1.0, gamma=0.0, q=0.5, rho=0.9
    )

    assert scale == pytest.approx(0.3181980515339464)


def test_warmstart_prox_scale_preserves_a_smaller_existing_scale() -> None:
    coefficients = torch.tensor([0.5], dtype=torch.float64)
    penalized = torch.tensor([True])

    scale = warmstart_prox_scale(
        coefficients, penalized, alpha=1e-5, gamma=0.0, q=0.5, rho=0.9
    )

    assert scale == pytest.approx(1e-5)


def test_fractional_ssn_requires_an_explicit_proximal_scale() -> None:
    coefficient = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))

    with pytest.raises(ValueError, match="prox_scale is required"):
        SSN([coefficient], alpha=1.0, gamma=0.0, power=3.0)


def test_fractional_correction_reports_an_all_zero_warm_start() -> None:
    model = SignedModel(activation=torch.relu, power=2.0, verbose=False)
    model.set_atoms(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    X = torch.tensor([[1.0]], dtype=torch.float64)
    data = (
        X,
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.zeros_like(X),
    )

    with pytest.raises(
        ValueError, match="warm-start proximal scaling requires a nonzero coefficient"
    ):
        ssn_solve(
            model,
            data,
            Objective(alpha=1.0, gamma=0.0),
            SolverConfig(),
            iterations=1,
        )


def test_fractional_ssn_preserves_a_prox_consistent_warm_start() -> None:
    q = 0.5
    alpha = 1.0
    curvature = 10.0
    warm_start = -0.5
    profile = curvature * abs(warm_start) + alpha * q * abs(warm_start) ** (q - 1.0)
    coefficient = torch.nn.Parameter(torch.tensor([warm_start], dtype=torch.float64))
    penalized = torch.tensor([True])
    prox_scale = warmstart_prox_scale(
        coefficient.detach(), penalized, alpha=alpha, gamma=0.0, q=q, rho=0.9
    )
    optimizer = SSN(
        [coefficient],
        alpha=alpha,
        gamma=0.0,
        power=3.0,
        prox_scale=prox_scale,
        tolerance_grad=0.0,
    )
    optimizer.data_hessian = torch.tensor([[curvature]], dtype=torch.float64)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = coefficient[0]
        return (
            10.0
            + profile * value
            + 0.5 * curvature * value.square()
            + alpha * value.abs().pow(q)
        )

    before = float(closure().detach())
    optimizer.step(closure)
    after = float(closure().detach())

    assert optimizer.last_step_success
    assert after <= before
    assert coefficient.item() == pytest.approx(warm_start)


def test_fractional_penalty_is_exactly_zero_at_zero() -> None:
    coefficients = torch.zeros(2, dtype=torch.float64)
    penalized = torch.tensor([True, True])
    nonnegative = torch.tensor([False, False])

    penalty = nonconvex_penalty(
        coefficients,
        penalized,
        nonnegative,
        alpha=1.0,
        th=0.5,
        gamma=0.0,
        q=0.5,
    )

    assert penalty.item() == 0.0


@pytest.mark.parametrize(
    ("power", "alpha", "prox_scale", "initial", "linear", "expected"),
    [
        (3.0, 1.0, 1.0, -4.0, 1131.0 / 62.0, -9.0),
        (2.0, 1.5, 1.5, -8.0, 7694.0 / 141.0, -27.0),
    ],
)
def test_fractional_ssn_uses_the_closed_form_prox_derivative(
    power: float,
    alpha: float,
    prox_scale: float,
    initial: float,
    linear: float,
    expected: float,
) -> None:
    coefficient = torch.nn.Parameter(torch.tensor([initial], dtype=torch.float64))
    optimizer = SSN(
        [coefficient],
        alpha=alpha,
        gamma=0.0,
        power=power,
        prox_scale=prox_scale,
        tolerance_grad=0.0,
    )
    optimizer.data_hessian = torch.tensor([[2.0]], dtype=torch.float64)
    q = 2.0 / (power + 1.0)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = coefficient[0]
        return linear * value + value.square() + alpha * value.abs().pow(q)

    optimizer.step(closure)

    assert optimizer.last_step_success
    assert coefficient.item() == pytest.approx(expected, abs=1e-9)
