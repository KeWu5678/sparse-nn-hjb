"""Fixed log penalty, its SSN splitting, and masked objective evaluation."""

import pytest
import torch

from src.problems.penalty import (
    _ddphi,
    _dphi,
    _noncvx_correction,
    _noncvx_correction_dd,
    _noncvx_penalty,
    _phi,
    jac_noncvx_penalty,
)


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
    value = alpha * (_phi(u.abs(), gamma) if q == 1 else u.abs().pow(q)).sum()
    (gradient,) = torch.autograd.grad(value, u, create_graph=True)
    torch.testing.assert_close(jac_noncvx_penalty(u.abs(), u.sign(), alpha, gamma, q), gradient)

    # Only the log family has a smooth remainder after the power-prox step.
    remainder = value / alpha - u.abs().sum() if q == 1 else (0 * u.square()).sum()
    (first,) = torch.autograd.grad(remainder, u, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), u)
    torch.testing.assert_close(_noncvx_correction(u.abs(), u.sign(), gamma, q), first)
    torch.testing.assert_close(_noncvx_correction_dd(u.abs(), gamma, q), second)


@pytest.mark.parametrize(("q", "gamma"), [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (2 / 3, 0.0)])
def test_penalty_respects_masks_and_signed_magnitudes(q: float, gamma: float) -> None:
    coefficients = torch.tensor([-2.0, -3.0, 4.0, 100.0, 0.0], dtype=torch.float64)
    penalized = torch.tensor([True, True, True, False, True])
    nonneg = torch.tensor([False, True, True, False, False])
    magnitudes = torch.tensor([2.0, 4.0], dtype=torch.float64)
    if q != 1:
        expected = magnitudes.pow(q).sum()
    elif gamma == 0:
        expected = magnitudes.sum()
    else:
        expected = (magnitudes / 2 + torch.log1p(2 * gamma * magnitudes) / (4 * gamma)).sum()

    actual = _noncvx_penalty(coefficients, penalized, nonneg, alpha=0.3, gamma=gamma, penalty_exponent=q)

    torch.testing.assert_close(actual, 0.3 * expected)


def test_unpenalized_coordinates_give_a_scalar_zero() -> None:
    coefficients = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    mask = torch.zeros(2, dtype=torch.bool)

    value = _noncvx_penalty(coefficients, mask, mask, alpha=1.0, gamma=1.0, penalty_exponent=1.0)

    assert value.shape == ()
    assert value.dtype == coefficients.dtype
    assert value.device == coefficients.device
    assert value.item() == 0.0
