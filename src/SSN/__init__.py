"""Semismooth Newton iterations and globalization.

Problem mathematics live in ``src.problems``; MPCG and strategies consume
already evaluated normal-map quantities.
"""

from .optimizer import SSN

__all__ = ["SSN"]
