"""Compatibility exports for SSN penalty and proximal helpers.

Historically this module collected unrelated numerical utilities. Those helpers
are no longer used by the project; the active implementations live in
``src.SSN``. Keep the still-supported re-exports here for notebooks and scripts.
"""

from __future__ import annotations

from src.SSN.penalty import (
    _ddphi,
    _dphi,
    _nonconvex_correction,
    _nonconvex_correction_dd,
    _penalty_grad,
    _phi,
)
from src.SSN.prox import (
    _compute_prox_q_half,
    _compute_prox_q_twothirds,
    _phi_prox,
    power_prox,
    power_prox_derivative,
)

__all__ = [
    "power_prox",
    "power_prox_derivative",
    "_compute_prox_q_half",
    "_compute_prox_q_twothirds",
    "_ddphi",
    "_dphi",
    "_nonconvex_correction",
    "_nonconvex_correction_dd",
    "_penalty_grad",
    "_phi",
    "_phi_prox",
]
