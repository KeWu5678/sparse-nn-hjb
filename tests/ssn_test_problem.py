"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply fitting directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

import torch

from src.config import resolve_model, validate_training
from src.config.schema import TrainingConfig
from src.models import SparseModel, make_penalty
from src.PDAP import PDAP
from src.SSN import SSN


def penalty_for(config, problem):
    """The penalty of a resolved configuration, with q supplied by its problem."""
    return make_penalty(config["alpha"], config["gamma"], problem.penalty_exponent)


def synthetic_correction(fitting, hess_obj, **specifications):
    config = resolve_model(specifications, hess_obj.shape[0])
    validate_training(config)
    penalty = penalty_for(config, SparseModel(config))

    def coefficient_loss(u):
        return fitting(u) + penalty.value(u)

    return penalty, coefficient_loss, hess_obj


def correction_optimizer(penalty, hess_obj, u, **options):
    """An SSN whose normal-map operations are evaluated directly at u."""
    return SSN([torch.nn.Parameter(u.detach().clone())], penalty, hess_obj, **options)


def closure_for(optimizer, coefficient, objective):
    """The torch.optim closure that evaluates objective at the coefficient."""
    def closure():
        optimizer.zero_grad()
        loss = objective(coefficient)
        loss.backward()
        return loss

    return closure


def coefficient_problem(problem, penalty, loader):
    """The objective, fitting Hessian and starting coefficients PDAP gives SSN, without a solve."""
    loss, hess_obj, moment_weight = PDAP(TrainingConfig(), penalty).coefficient_objective(problem, loader)
    return loss, hess_obj, problem.c.detach() * moment_weight
