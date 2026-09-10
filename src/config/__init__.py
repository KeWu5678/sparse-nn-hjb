"""Centralized PDAP configuration (Hydra structured configs + activation registry)."""

from __future__ import annotations

from .activations import ACTIVATIONS, get_activation, get_use_sphere
from .schema import (
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "ACTIVATIONS",
    "get_activation",
    "get_use_sphere",
]
