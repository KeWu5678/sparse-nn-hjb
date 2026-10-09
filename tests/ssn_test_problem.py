"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply fitting directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

import pytest

from src.config import resolve_model, validate_training
from src.config.schema import TrainingConfig
from src.PDAP import PDAP
from src.problems import SparseProblem


def synthetic_problem(fitting, hess_obj, **specifications):
    masks = {name: specifications.pop(name) for name in ("penalized_mask", "nonneg_mask")
             if name in specifications}
    config = resolve_model(specifications, hess_obj.shape[0])
    validate_training(config)
    problem = SparseProblem(config, **masks)

    def coefficient_loss(u):
        return fitting(u) + problem.alpha * problem.sparsity_functional(u)

    return problem, coefficient_loss, hess_obj


def capture_coefficient_solve(monkeypatch, problem, loader):
    """Inspect the real PDAP-to-SSN inputs without running or applying a solve."""
    captured = []

    def capture(params, problem, loss, hess_obj, **options):
        captured.extend((loss, hess_obj, params[0].detach().clone()))
        raise StopIteration("captured coefficient correction")

    with monkeypatch.context() as patch:
        patch.setattr("src.PDAP.pdap.SSN", capture)
        with pytest.raises(StopIteration, match="captured coefficient correction"):
            PDAP(TrainingConfig()).train_ssn_outerweights(problem, loader)
    return tuple(captured)
