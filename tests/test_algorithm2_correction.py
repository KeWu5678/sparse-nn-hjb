"""Behavioral tests for Algorithm 2's outer-weight correction."""

from __future__ import annotations

import pytest
import torch
from ssn_test_problem import SyntheticProblem

from src.models import SignedModel
from src.PDAP.ssn_solve import (
    SolverConfig,
    ssn_solve,
)
from src.problems import SparseProblem
from src.problems.penalty import _noncvx_penalty
from src.SSN import SSN


def test_warmstart_prox_scale_uses_all_nonzero_penalized_coefficients() -> None:
    coefficients = torch.tensor([0.0, -0.5, 2.0, 0.01], dtype=torch.float64)
    penalized = torch.tensor([True, True, True, False])

    scale = SparseProblem(None, alpha=1.0, power=3.0, penalized_mask=penalized).warmstart_prox_scale(coefficients, rho=0.9)

    assert scale == pytest.approx(0.3181980515339464)


def test_warmstart_prox_scale_preserves_a_smaller_existing_scale() -> None:
    coefficients = torch.tensor([0.5], dtype=torch.float64)
    penalized = torch.tensor([True])

    scale = SparseProblem(None, alpha=1e-5, power=3.0, penalized_mask=penalized).warmstart_prox_scale(coefficients, rho=0.9)

    assert scale == pytest.approx(1e-5)


def test_fractional_ssn_requires_an_explicit_proximal_scale() -> None:
    coefficient = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    problem = SyntheticProblem(
        lambda u: 0.5 * u.square().sum(), torch.eye(1, dtype=torch.float64),
        alpha=1.0, gamma=0.0, power=3.0,
    )

    with pytest.raises(ValueError, match="prox_scale is required"):
        SSN([coefficient], problem)


@pytest.mark.parametrize("power", [2.0, 3.0])
def test_fractional_ssn_rejects_a_log_penalty(power: float) -> None:
    with pytest.raises(ValueError, match="power penalty requires gamma == 0"):
        SyntheticProblem(
            lambda u: 0.5 * u.square().sum(), torch.eye(1, dtype=torch.float64),
            alpha=1.0, gamma=1.0, power=power,
        )


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
            SparseProblem(data, alpha=1.0, gamma=0.0, power=2.0),
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
    prox_scale = SparseProblem(None, alpha=alpha, power=3.0, penalized_mask=penalized).warmstart_prox_scale(coefficient.detach(), rho=0.9)
    def data_loss(u: torch.Tensor) -> torch.Tensor:
        value = u[0]
        return (
            10.0
            + profile * value
            + 0.5 * curvature * value.square()
        )

    problem = SyntheticProblem(
        data_loss, torch.tensor([[curvature]], dtype=torch.float64),
        alpha=alpha, gamma=0.0, power=3.0,
    )
    optimizer = SSN([coefficient], problem, prox_scale=prox_scale, tolerance_grad=0.0)

    before = float(problem.loss(coefficient).detach())
    optimizer.step()
    after = float(problem.loss(coefficient).detach())

    assert optimizer.last_step_success
    assert after <= before
    assert coefficient.item() == pytest.approx(warm_start)


def test_fractional_penalty_is_exactly_zero_at_zero() -> None:
    coefficients = torch.zeros(2, dtype=torch.float64)
    penalized = torch.tensor([True, True])
    nonnegative = torch.tensor([False, False])

    penalty = _noncvx_penalty(
        coefficients,
        penalized,
        nonnegative,
        alpha=1.0,
        gamma=0.0,
        penalty_exponent=0.5,
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
    def data_loss(u: torch.Tensor) -> torch.Tensor:
        value = u[0]
        return linear * value + value.square()

    problem = SyntheticProblem(
        data_loss, torch.tensor([[2.0]], dtype=torch.float64),
        alpha=alpha, gamma=0.0, power=power,
    )
    optimizer = SSN([coefficient], problem, prox_scale=prox_scale, tolerance_grad=0.0)
    optimizer.step()

    assert optimizer.last_step_success
    assert coefficient.item() == pytest.approx(expected, abs=1e-9)
