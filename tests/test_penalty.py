"""Fixed log penalty, its SSN splitting, and objective evaluation."""

import pytest
import torch

from src.config import make_penalty, resolve_model
from src.models import FractionalPenalty, LogPenalty
from src.models.penalty import _ddphi, _dphi, _phi


def _penalty(alpha: float, gamma: float, q: float):
    return LogPenalty(alpha, gamma) if q == 1.0 else FractionalPenalty(alpha, q)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0, 10.0])
def test_fixed_log_penalty_and_derivatives(gamma: float) -> None:
    t = torch.tensor([0.0, 0.01, 0.5, 2.0, 10.0], dtype=torch.float64, requires_grad=True)
    value = _phi(t, gamma)
    expected = t if gamma == 0 else t / 2 + torch.log1p(2 * gamma * t) / (4 * gamma)
    torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-15)

    (first,) = torch.autograd.grad(value.sum(), t, create_graph=True)
    second = torch.zeros_like(t) if gamma == 0 else torch.autograd.grad(first.sum(), t)[0]
    torch.testing.assert_close(_dphi(t, gamma), first)
    torch.testing.assert_close(_ddphi(t, gamma), second)
    assert value[0].item() == 0.0
    assert _dphi(t, gamma)[0].item() == 1.0
    assert _ddphi(t, gamma)[0].item() == -gamma


@pytest.mark.parametrize(("q", "gamma"), [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (2 / 3, 0.0)])
def test_penalty_value_and_ssn_remainder(q: float, gamma: float) -> None:
    u = torch.tensor([-2.0, -0.5, 0.25, 4.0], dtype=torch.float64, requires_grad=True)
    alpha = 0.3
    penalty = _penalty(alpha, gamma, q)
    value = alpha * (_phi(u.abs(), gamma) if q == 1 else u.abs().pow(q)).sum()
    torch.testing.assert_close(penalty.value(u), value)

    # Only the log family has a smooth remainder after the power-prox step.
    remainder = value - alpha * u.abs().sum() if q == 1 else (0 * u.square()).sum()
    (first,) = torch.autograd.grad(remainder, u, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), u)
    torch.testing.assert_close(penalty.jac_correction(u), first)
    torch.testing.assert_close(penalty.hess_correction(u), torch.diag(second))


@pytest.mark.parametrize(("q", "gamma"), [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (2 / 3, 0.0)])
def test_penalty_uses_the_magnitudes_of_signed_coefficients(q: float, gamma: float) -> None:
    coefficients = torch.tensor([-2.0, 4.0, 0.0], dtype=torch.float64)
    magnitudes = torch.tensor([2.0, 4.0], dtype=torch.float64)
    if q != 1:
        expected = magnitudes.pow(q).sum()
    elif gamma == 0:
        expected = magnitudes.sum()
    else:
        expected = (magnitudes / 2 + torch.log1p(2 * gamma * magnitudes) / (4 * gamma)).sum()

    actual = _penalty(0.3, gamma, q).value(coefficients)

    torch.testing.assert_close(actual, 0.3 * expected)


@pytest.mark.parametrize(
    ("power", "gamma", "family"),
    [(1.0, 1.0, LogPenalty), (3.0, 0.0, FractionalPenalty), (2.0, 0.0, FractionalPenalty)],
)
def test_make_penalty_selects_the_family_by_its_exponent(power: float, gamma: float, family: type) -> None:
    penalty = make_penalty(resolve_model({"power": power, "alpha": 0.3, "gamma": gamma}, 1))

    assert type(penalty) is family
    vector = penalty.value(torch.tensor([-2.0], dtype=torch.float64))
    assert penalty.scalar_penalty(-2.0) == pytest.approx(float(vector), rel=1e-15)


def test_insertion_weight_is_the_prox_at_the_newton_point() -> None:
    penalty = FractionalPenalty(1.0, 0.5)

    assert penalty.insertion_weight(-4.25, 1.0) == pytest.approx(4.0)
    assert penalty.insertion_weight(4.25, 1.0) == pytest.approx(-4.0)


def test_descent_step_requires_the_slope_to_exceed_alpha() -> None:
    penalty = LogPenalty(0.5, 0.0)

    assert penalty.descent_step(-3.0, 1.0) == pytest.approx(2.5)
    assert penalty.descent_step(-0.4, 1.0) == 0.0
