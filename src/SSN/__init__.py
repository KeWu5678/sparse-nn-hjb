"""Sparse semismooth-Newton (SSN) optimizer package.

Layout mirrors ``torch.optim``: ``optimizer.py`` holds the optimizer class,
``penalty.py`` / ``prox.py`` hold the numerical kernels, ``mpcg.py`` holds the
trust-region Krylov solve.  The package is a self-contained leaf (it imports
only ``torch`` and its own submodules), so the rest of ``src`` can depend on it
without creating an import cycle.
"""

from .optimizer import SSN
from .prox import SUPPORTED_ACTIVATION_POWERS

__all__ = ["SSN", "SUPPORTED_ACTIVATION_POWERS"]
