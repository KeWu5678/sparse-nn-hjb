"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply fitting directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

import pytest
import torch

from src.config import resolve_model, validate_training
from src.config.schema import TrainingConfig
from src.models import SparseModel, make_penalty
from src.PDAP import PDAP
from src.SSN import SSN


def proximal_for(config, problem):
    """The penalty of a resolved configuration, with q supplied by its problem."""
    return make_penalty(config["alpha"], config["gamma"], problem.penalty_exponent)


def synthetic_correction(fitting, hess_obj, **specifications):
    config = resolve_model(specifications, hess_obj.shape[0])
    validate_training(config)
    proximal = proximal_for(config, SparseModel(config))

    def coefficient_loss(u):
        return fitting(u) + proximal.penalty(u)

    return proximal, coefficient_loss, hess_obj


def correction_optimizer(proximal, coefficient_loss, hess_obj, u, **options):
    """An SSN whose normal-map operations are evaluated directly at u."""
    return SSN([torch.nn.Parameter(u.detach().clone())], proximal, coefficient_loss, hess_obj, **options)


def capture_coefficient_solve(monkeypatch, problem, proximal, loader):
    """Inspect the real PDAP-to-SSN inputs without running or applying a solve."""
    captured = []

    def capture(params, proximal, loss, hess_obj, **options):
        captured.extend((loss, hess_obj, params[0].detach().clone()))
        raise StopIteration("captured coefficient correction")

    with monkeypatch.context() as patch:
        patch.setattr("src.PDAP.pdap.SSN", capture)
        with pytest.raises(StopIteration, match="captured coefficient correction"):
            PDAP(TrainingConfig(), proximal).train_ssn_outerweights(problem, loader)
    return tuple(captured)
