"""Full-dimensional initial states for the finite-volume Allen–Cahn benchmark."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


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
