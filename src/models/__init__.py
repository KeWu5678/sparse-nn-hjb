"""Parametric value-function models for the PDAP outer loop.

``SignedModel`` represents the shallow network
``V(x) = sum_i c_i sigma(w_i.x+b_i)^p``.
"""

from .build import build_model
from .net import ShallowNetwork
from .signed import SignedModel

__all__ = ["ShallowNetwork", "SignedModel", "build_model"]
