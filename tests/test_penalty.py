"""Fixed log penalty, its SSN splitting, and objective evaluation."""

import pytest
import torch

from src.models import FractionalPenalty, LogPenalty, make_penalty
from src.models.proximal import _ddphi, _dphi, _phi


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
def test_penalty_gradient_and_ssn_remainder(q: float, gamma: float) -> None:
    u = torch.tensor([-2.0, -0.5, 0.25, 4.0], dtype=torch.float64, requires_grad=True)
    alpha = 0.3
    proximal = make_penalty(alpha, gamma, q)
    value = alpha * (_phi(u.abs(), gamma) if q == 1 else u.abs().pow(q)).sum()
    (gradient,) = torch.autograd.grad(value, u, create_graph=True)
    torch.testing.assert_close(proximal.jac_penalty(u), gradient)

    # Only the log family has a smooth remainder after the power-prox step.
    remainder = value - alpha * u.abs().sum() if q == 1 else (0 * u.square()).sum()
    (first,) = torch.autograd.grad(remainder, u, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), u)
    torch.testing.assert_close(proximal.jac_correction(u), first)
    torch.testing.assert_close(proximal.hess_correction(u), torch.diag(second))


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

    actual = make_penalty(0.3, gamma, q).penalty(coefficients)

    torch.testing.assert_close(actual, 0.3 * expected)


@pytest.mark.parametrize(("q", "family"), [(1.0, LogPenalty), (0.5, FractionalPenalty), (2 / 3, FractionalPenalty)])
def test_make_penalty_selects_the_family_by_its_exponent(q: float, family: type) -> None:
    penalty = make_penalty(0.3, 0.0, q)

    assert type(penalty) is family
    assert penalty.penalty_exponent == q


def test_fractional_penalty_rejects_a_log_parameter() -> None:
    with pytest.raises(ValueError, match="requires gamma == 0"):
        make_penalty(0.3, 1.0, 0.5)
