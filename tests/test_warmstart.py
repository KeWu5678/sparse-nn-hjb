"""Scalar warm starts for the separate log and power penalty families."""

import pytest
import torch
from ssn_test_problem import penalty_for

from src.config import resolve_model, validate_model_config
from src.models import SparseModel
from src.PDAP.warmstart import warm_start


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize(
    ("power", "gamma", "alpha", "v", "expected"),
    [
        (1.0, 0.5, 0.25, 1.1875, 1.0),
        (1.0, 0.0, 0.5, 3.0, 2.5),
        (2.0, 0.0, 1.5, 8.5, 8.0),
        (3.0, 0.0, 1.0, 4.25, 4.0),
    ],
)
def test_single_atom_warm_start_has_known_minimizer(
    sign: float, power: float, gamma: float, alpha: float, v: float, expected: float
) -> None:
    # A single atom with feature value 1 gives curvature 1 and proximal input |v|.
    X = torch.ones((1, 1), dtype=torch.float64)
    config = resolve_model({
        "power": power, "loss_weights": (1.0, 0.0), "alpha": alpha, "gamma": gamma,
    }, 1)
    validate_model_config(config)
    problem = SparseModel(config)
    problem.set_atoms(torch.ones_like(X), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    target = X.new_tensor([[sign * v]]), torch.zeros_like(X)
    warm_start(problem, penalty_for(config, problem), X, target, 1)

    assert problem.c.item() == pytest.approx(sign * expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize(
    ("feature", "alpha", "v", "expected"),
    [
        # A positive stationary root exists, but -slope < alpha rejects it.
        (1.0, 2.0, 23.0 / 12.0, 0.0),
        # Equality is accepted: at v=lambda=2 the minimizing root is 1/2.
        (1.0, 2.0, 2.0, 0.5),
        # Even when the slope test passes, zero/tiny curvature forbids division.
        (0.0, 0.0, 1.0, 0.0),
        (1e-16, 1e-20, 1.0, 0.0),
    ],
)
def test_log_warm_start_preserves_matlab_acceptance_guards(
    feature: float, alpha: float, v: float, expected: float
) -> None:
    X = torch.tensor([[feature]], dtype=torch.float64)
    config = resolve_model({"loss_weights": (1.0, 0.0), "alpha": alpha, "gamma": 1.0}, 1)
    validate_model_config(config)

    problem = SparseModel(config)
    problem.set_atoms(torch.ones_like(X), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    target = X.new_tensor([[v]]), torch.zeros_like(X)
    warm_start(problem, penalty_for(config, problem), X, target, 1)

    assert problem.c.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_sphere_warm_start_uses_the_exact_activation_derivative() -> None:
    # Gradient-only fit of leaky_relu at x = -1, 2: rho' = (.01, 1), so the
    # slope is -(.01 + 1)/2 and the curvature (.01^2 + 1)/2.
    X = torch.tensor([[-1.], [2.]], dtype=torch.float64)
    config = resolve_model({
        "activation": "leaky_relu", "loss_weights": (0.0, 1.0), "alpha": 0.005, "gamma": 0.0,
    }, 1)
    validate_model_config(config)
    problem = SparseModel(config)
    problem.set_atoms(X.new_ones((1, 1)), X.new_zeros(1), X.new_zeros(1))
    warm_start(problem, penalty_for(config, problem), X, (X.new_zeros((2, 1)), X.new_ones((2, 1))), 1)

    assert problem.c.item() == pytest.approx((0.505 - 0.005) / 0.50005, rel=1e-13)


def test_warm_start_sets_only_the_new_atoms() -> None:
    X = torch.ones((1, 1), dtype=torch.float64)
    config = resolve_model({"loss_weights": (1.0, 0.0), "alpha": 0.5, "gamma": 0.0}, 1)
    validate_model_config(config)
    problem = SparseModel(config)
    problem.set_atoms(X.new_tensor([[-1.], [1.]]), X.new_zeros(2), X.new_tensor([0.7, 0.0]))
    # The existing atom is inactive at x = 1, so the network is zero there.
    warm_start(problem, penalty_for(config, problem), X, (X.new_tensor([[3.]]), X.new_zeros((1, 1))), 1)

    torch.testing.assert_close(problem.c.detach(), X.new_tensor([0.7, 2.5]))
