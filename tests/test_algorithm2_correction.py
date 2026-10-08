"""Behavioral tests for Algorithm 2's outer-weight correction."""

from __future__ import annotations

import pytest
import torch
from ssn_test_problem import synthetic_problem

from src.config import resolve_model, validate_training
from src.data import make_dataloader
from src.problems import SparseProblem
from src.problems.penalty import _noncvx_penalty
from src.SSN import SSN


def test_prox_scale_for_fractional_penalty_uses_all_nonzero_penalized_coefficients() -> None:
    coefficients = torch.tensor([0.0, -0.5, 2.0, 0.01], dtype=torch.float64)
    penalized = torch.tensor([True, True, True, False])

    problem = SparseProblem(resolve_model({"alpha": 1.0, "power": 3.0}, 4), penalized_mask=penalized)
    scale = problem.prox_scale_for_fractional_penalty(coefficients, rho=0.9)

    assert scale == pytest.approx(0.3181980515339464)


def test_prox_scale_for_fractional_penalty_preserves_a_smaller_existing_scale() -> None:
    coefficients = torch.tensor([0.5], dtype=torch.float64)
    penalized = torch.tensor([True])

    problem = SparseProblem(resolve_model({"alpha": 1e-5, "power": 3.0}, 1), penalized_mask=penalized)
    scale = problem.prox_scale_for_fractional_penalty(coefficients, rho=0.9)

    assert scale == pytest.approx(1e-5)


def test_fractional_ssn_chooses_the_scale_from_its_starting_coefficients() -> None:
    coefficient = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    problem, obj, hess_obj = synthetic_problem(
        lambda u: 0.5 * u.square().sum(), torch.eye(1, dtype=torch.float64),
        alpha=1.0, gamma=0.0, power=3.0,
    )

    optimizer = SSN([coefficient], problem, obj, hess_obj)
    assert optimizer.param_groups[0]["inverse_step"] == pytest.approx(4 * 2 ** .5)


@pytest.mark.parametrize("power", [2.0, 3.0])
def test_fractional_problem_rejects_a_log_penalty(power: float) -> None:
    with pytest.raises(ValueError, match="power penalty requires gamma == 0"):
        validate_training(resolve_model({"alpha": 1.0, "gamma": 1.0, "power": power}, 1))


def test_fractional_correction_reports_an_all_zero_warm_start() -> None:
    config = resolve_model({"alpha": 1.0, "gamma": 0.0, "power": 2.0}, 1)
    validate_training(config)
    problem = SparseProblem(config)
    problem.set_atoms(
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

    obj, hess_obj, _ = problem.prepare_coefficients(make_dataloader(data))
    theta = torch.nn.Parameter(problem.c.detach().clone())
    with pytest.raises(
        ValueError, match="warm-start proximal scaling requires a nonzero coefficient"
    ):
        SSN([theta], problem, obj, hess_obj, max_iter=1)


def test_fractional_ssn_preserves_a_prox_consistent_warm_start() -> None:
    q = 0.5
    alpha = 1.0
    curvature = 10.0
    warm_start = -0.5
    profile = curvature * abs(warm_start) + alpha * q * abs(warm_start) ** (q - 1.0)
    coefficient = torch.nn.Parameter(torch.tensor([warm_start], dtype=torch.float64))
    def data_loss(u: torch.Tensor) -> torch.Tensor:
        value = u[0]
        return (
            10.0
            + profile * value
            + 0.5 * curvature * value.square()
        )

    problem, obj, hess_obj = synthetic_problem(
        data_loss, torch.tensor([[curvature]], dtype=torch.float64),
        alpha=alpha, gamma=0.0, power=3.0,
    )
    prox_scale = problem.prox_scale_for_fractional_penalty(coefficient.detach(), rho=0.9)
    optimizer = SSN([coefficient], problem, obj, hess_obj, prox_scale=prox_scale, tolerance_grad=0.0, max_iter=1)

    before = float(problem.loss(coefficient, obj).detach())
    optimizer.step()
    after = float(problem.loss(coefficient, obj).detach())

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

    problem, obj, hess_obj = synthetic_problem(
        data_loss, torch.tensor([[2.0]], dtype=torch.float64),
        alpha=alpha, gamma=0.0, power=power,
    )
    optimizer = SSN([coefficient], problem, obj, hess_obj, prox_scale=prox_scale, tolerance_grad=0.0, max_iter=1)
    optimizer.step()

    assert optimizer.last_step_success
    assert coefficient.item() == pytest.approx(expected, abs=1e-9)
