"""The sparse model and its proximal penalty, shared by PDAP and SSN."""

from .proximal import Proximal
from .sparse_model import SparseModel

__all__ = ["Proximal", "SparseModel"]
