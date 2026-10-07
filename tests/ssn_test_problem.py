"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests supply obj directly
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

from src.problems import PreparedObjective, SparseProblem


def synthetic_problem(obj, hess_obj, **specifications):
    problem = SparseProblem(input_dim=hess_obj.shape[0], **specifications)
    return problem, PreparedObjective(obj, hess_obj)
