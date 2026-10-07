"""Synthetic objectives for solver isolation, including indefinite quadratics.

Production problems are built from samples. These tests override obj/hess_obj
so that negative curvature, additive constants, and historical losses remain
exactly as originally tested.
"""

from types import SimpleNamespace

import torch

from src.problems import SparseProblem


class SyntheticProblem(SparseProblem):
    def __init__(self, obj, hess_obj, **specifications):
        n = hess_obj.shape[0]
        X = hess_obj.new_zeros((n, 1))
        super().__init__((X, X.clone(), X.clone()), **specifications)
        self._test_obj = obj
        self._test_hess_obj = hess_obj
        self.prepare_coefficients(
            SimpleNamespace(
                activation=torch.relu,
                power=specifications.get("power", 1.0),
                jacobians=lambda X: (
                    torch.eye(n, dtype=X.dtype, device=X.device),
                    X.new_zeros((n, n)),
                ),
            )
        )

    def obj(self, u):
        return self._test_obj(u)

    def hess_obj(self):
        return self._test_hess_obj
