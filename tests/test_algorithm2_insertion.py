"""Behavioral tests for Algorithm 2's one-node insertion step."""

from __future__ import annotations

import pytest

from src.PDAP import solve_insertion_weight


def test_q_half_insertion_minimizes_the_actual_objective_increment() -> None:
    """For P=4.25, A=alpha=1, the exact nonzero minimizer is c=-4."""
    result = solve_insertion_weight(4.25, 1.0, 1.0, 0.5)

    assert result is not None
    coefficient, increment = result
    actual = 4.25 * coefficient + 0.5 * coefficient**2 + abs(coefficient) ** 0.5
    assert coefficient == pytest.approx(-4.0)
    assert increment == pytest.approx(-7.0)
    assert increment == pytest.approx(actual)


def test_q_two_thirds_insertion_minimizes_the_actual_objective_increment() -> None:
    result = solve_insertion_weight(8.5, 1.0, 1.5, 2.0 / 3.0)

    assert result is not None
    coefficient, increment = result
    actual = (
        8.5 * coefficient
        + 0.5 * coefficient**2
        + 1.5 * abs(coefficient) ** (2.0 / 3.0)
    )
    assert coefficient == pytest.approx(-8.0)
    assert increment == pytest.approx(-30.0)
    assert increment == pytest.approx(actual)


def test_q_one_insertion_keeps_the_soft_threshold_rule() -> None:
    result = solve_insertion_weight(-3.0, 2.0, 1.0, 1.0)

    assert result is not None
    coefficient, increment = result
    actual = -3.0 * coefficient + coefficient**2 + abs(coefficient)
    assert coefficient == pytest.approx(1.0)
    assert increment == pytest.approx(-1.0)
    assert increment == pytest.approx(actual)


@pytest.mark.parametrize(
    ("profile", "alpha", "q"),
    [(1.5, 1.0, 0.5), (2.0, 1.5, 2.0 / 3.0)],
)
def test_fractional_insertion_rejects_a_global_proximal_tie(
    profile: float, alpha: float, q: float
) -> None:
    assert solve_insertion_weight(profile, 1.0, alpha, q) is None
