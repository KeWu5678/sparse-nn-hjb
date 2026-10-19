"""Initial-state designs for open-loop value-sample generation."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def grid_initial_states(
    n_first: int,
    n_second: int,
    first_range: tuple[float, float] = (-3.0, 3.0),
    second_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    """Tensor grid of two-dimensional states, ``n_first * n_second`` rows."""
    first_values = np.linspace(first_range[0], first_range[1], n_first)
    second_values = np.linspace(second_range[0], second_range[1], n_second)
    first_grid, second_grid = np.meshgrid(first_values, second_values)
    return np.column_stack((first_grid.ravel(), second_grid.ravel())).astype(np.float64)


def sobol_initial_states(
    count: int, dimension: int, *, bound: float = 1.0, seed: int = 42,
) -> np.ndarray:
    """Scrambled Sobol points in ``[-bound, bound]^dimension``.

    Powers of two preserve Sobol balance. Every state component is sampled;
    the data are not confined to a few smooth spatial profiles.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count & (count - 1):
        raise ValueError("count must be a power of two, at least 2")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise ValueError("dimension must be an integer of at least 2")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be positive and finite")
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(count.bit_length() - 1)
    return bound * (2.0 * unit - 1.0)


__all__ = ["grid_initial_states", "sobol_initial_states"]
