"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply obj directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

from src.config import resolve_model, validate_training
from src.problems import SparseProblem


def synthetic_problem(obj, hess_obj, **specifications):
    masks = {name: specifications.pop(name) for name in ("penalized_mask", "nonneg_mask")
             if name in specifications}
    config = resolve_model(specifications, hess_obj.shape[0])
    validate_training(config)
    problem = SparseProblem(config, **masks)
    return problem, obj, hess_obj
