"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply fitting directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

import torch

from src.config import make_penalty, resolve_model, validate_model_config
from src.config.schema import TrainingConfig
from src.PDAP import PDAP
from src.SSN import SSN


def synthetic_correction(fitting, hess_obj, **specifications):
    config = resolve_model(specifications, hess_obj.shape[0])
    validate_model_config(config)
    penalty = make_penalty(config)
    return penalty, fitting, hess_obj


def correction_optimizer(penalty, hess_obj, u, **options):
    """An SSN whose normal-map operations are evaluated directly at u."""
    return SSN([torch.nn.Parameter(u.detach().clone())], penalty, hess_obj, **options)


def closure_for(optimizer, coefficient, fitting):
    """The torch.optim closure that evaluates the fitting term at the coefficient."""
    def closure():
        optimizer.zero_grad()
        value = fitting(coefficient)
        value.backward()
        return value

    return closure


def coefficient_problem(problem, penalty, loader):
    """The objective SSN minimizes, its fitting Hessian and the starting coefficients, without a solve.

    The objective is PDAP's coefficient fitting term plus the penalty SSN adds.
    """
    pdap = PDAP(TrainingConfig(), penalty)
    fitting, hess_obj, moment_weight = pdap.fit_obj_with_outerweights(problem, loader)
    return (lambda u: fitting(u) + penalty.value(u)), hess_obj, problem.c.detach() * moment_weight
