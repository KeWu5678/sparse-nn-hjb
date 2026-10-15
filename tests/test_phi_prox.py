"""Closed-form proximal step for the logarithmic penalty and its L1 limit."""

from __future__ import annotations

import math

import pytest

from src.models import LogPenalty


def phi_prox(v: float, lambda_: float, gamma: float) -> float:
    """The log-family penalty prox; alpha does not enter it."""
    return LogPenalty(1.0, gamma).penalty_prox(v, lambda_)


@pytest.mark.parametrize(("v", "expected"), [(-1.0, 0.0), (0.5, 0.0), (3.0, 2.5)])
def test_phi_prox_with_linear_phi_has_known_solution(
    v: float, expected: float
) -> None:
    result = phi_prox(v, 0.5, gamma=0.0)

    assert result == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_phi_prox_mixed_penalty_has_known_stationary_point() -> None:
    # For theta=gamma=1/2, phi'(1)=3/4. With lambda=1/4, this v gives t=1.
    v = 1.0 + 0.25 * 0.75

    result = phi_prox(v, 0.25, gamma=0.5)

    assert result == pytest.approx(1.0, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize(
    ("lambda_", "gamma", "v"),
    [
        (0.5, 1.0, 0.25),
        (0.5, 1.0, 0.5),
        (0.5, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (2.0, 1.0, 3.0),
    ],
)
def test_phi_prox_minimizes_in_paper_and_warmstart_regimes(
    lambda_: float, gamma: float, v: float
) -> None:
    # The paper assumes lambda*gamma < 1; warmstart instead ensures v >= lambda.
    # In these regimes the first-order condition identifies the global minimum.
    t = phi_prox(v, lambda_, gamma)

    if lambda_ * gamma < 1.0 and v <= lambda_:
        assert t == 0.0
        return

    assert t > 0.0
    gamma_bar = 2.0 * gamma
    dphi = 0.5 + 0.5 / (1.0 + gamma_bar * t)
    assert t - v + lambda_ * dphi == pytest.approx(0.0, abs=1e-13)

    phi = 0.5 * t + 0.5 * math.log1p(gamma_bar * t) / gamma_bar
    assert 0.5 * (t - v) ** 2 + lambda_ * phi < 0.5 * v ** 2


def test_phi_prox_returns_zero_when_quadratic_has_no_real_root() -> None:
    assert phi_prox(1.0, 2.0, gamma=1.0) == 0.0


def test_phi_prox_preserves_matlab_root_outside_guaranteed_regimes() -> None:
    # lambda*gamma > 1 and v < lambda: the reference still returns the larger
    # stationary root (1/4), although zero has the lower objective here.
    v, lambda_, gamma = 23.0 / 12.0, 2.0, 1.0

    t = phi_prox(v, lambda_, gamma)

    assert t == pytest.approx(1.0 / 4.0)
    phi = t / 2.0 + math.log1p(2.0 * gamma * t) / (4.0 * gamma)
    assert 0.5 * (t - v) ** 2 + lambda_ * phi > 0.5 * v ** 2
