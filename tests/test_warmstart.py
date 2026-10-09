"""Scalar warm starts for the separate log and power penalty families."""

import pytest
import torch
from ssn_test_problem import proximal_for

from src.config import resolve_model, validate_training
from src.PDAP.warmstart import warm_start
from src.problems import SparseProblem


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
    validate_training(config)
    problem = SparseProblem(config)
    result = warm_start(
        W_new=torch.ones_like(X),
        b_new=torch.zeros(1, dtype=torch.float64),
        residual=(X, torch.tensor([-sign * v], dtype=torch.float64), torch.zeros_like(X)),
        problem=problem,
        proximal=proximal_for(config, problem),
    )

    assert result.item() == pytest.approx(sign * expected, rel=1e-13, abs=1e-13)


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
    validate_training(config)

    problem = SparseProblem(config)
    result = warm_start(
        W_new=torch.ones_like(X),
        b_new=torch.zeros(1, dtype=torch.float64),
        residual=(X, torch.tensor([-v], dtype=torch.float64), torch.zeros_like(X)),
        problem=problem,
        proximal=proximal_for(config, problem),
    )

    assert result.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)
