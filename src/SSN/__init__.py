"""Semismooth Newton iterations and globalization.

Problem mathematics live in ``src.problems``; MPCG and strategies consume
already evaluated normal-map quantities.
"""

from .optimizer import SSN, fractional_prox_scale

__all__ = ["SSN", "fractional_prox_scale"]
