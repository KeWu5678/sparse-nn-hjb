"""Pontryagin collocation reference samples for finite-dimensional Allen--Cahn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp

from src.OpenLoop.allen_cahn.problem import AllenCahnProblem
from src.OpenLoop.value_samples import ValueSamples


@dataclass(frozen=True)
class AllenCahnSolverConfig:
    collocation_tol: float = 1e-6
    collocation_max_nodes: int = 10000
    initial_mesh_points: int = 81
    initial_guess_decays: tuple[float, ...] = (1.0, 0.0)
    quadrature_order: int = 5
    store_trajectories: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.collocation_tol) or self.collocation_tol <= 0:
            raise ValueError("collocation_tol must be finite and positive")
        for name in ("initial_mesh_points", "collocation_max_nodes", "quadrature_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{name} must be an integer at least 2")
        if self.collocation_max_nodes < self.initial_mesh_points:
            raise ValueError("collocation_max_nodes must be at least initial_mesh_points")
        decays = tuple(float(decay) for decay in self.initial_guess_decays)
        if not decays or not all(np.isfinite(decay) and decay >= 0 for decay in decays):
            raise ValueError("initial_guess_decays must contain finite nonnegative values")
        object.__setattr__(self, "initial_guess_decays", decays)


@dataclass(frozen=True)
class AllenCahnSampleResult:
    initial_state: np.ndarray
    converged: bool
    message: str
    value: float | None
    gradient: np.ndarray | None
    iterations: int
    collocation_residual: float | None
    boundary_residual: float | None
    nodes: int
    reduced_gradient_norm: float | None
    attempts: int
    time_grid: np.ndarray | None = None
    state_trajectory: np.ndarray | None = None
    adjoint_trajectory: np.ndarray | None = None
    control: np.ndarray | None = None


@dataclass(frozen=True)
class AllenCahnOpenLoopSolution:
    value_samples: ValueSamples
    sample_results: tuple[AllenCahnSampleResult, ...]
    failed_initial_states: np.ndarray
    config: AllenCahnSolverConfig
    problem: AllenCahnProblem


class AllenCahnOpenLoopSolver:
    """Solve the state/adjoint boundary problem from multiple initial guesses.

    Each retained solution satisfies Pontryagin stationarity.  The least-cost
    converged candidate is a numerical reference, not a global-optimality
    certificate for this nonlinear control problem.
    """

    def __init__(self, problem: AllenCahnProblem | None = None, config: AllenCahnSolverConfig | None = None) -> None:
        self.problem = problem or AllenCahnProblem()
        self.config = config or AllenCahnSolverConfig()
        self.time_grid = np.linspace(0.0, self.problem.T_final, self.config.initial_mesh_points)
        self._linear_state = self.problem.nu * self.problem.laplacian + np.eye(self.problem.dimension)
        self._control_adjoint = -(self.problem.control_matrix @ self.problem.control_matrix.T) / (2.0 * self.problem.beta)
        self._quadrature_nodes, self._quadrature_weights = leggauss(self.config.quadrature_order)

    def _rhs(self, _t: np.ndarray, z: np.ndarray) -> np.ndarray:
        d = self.problem.dimension
        y, p = z[:d], z[d:]
        return np.vstack((
            self._linear_state @ y - y**3 + self._control_adjoint @ p,
            -2.0 * self.problem.cell_width * y - self._linear_state.T @ p + 3.0 * y**2 * p,
        ))

    def _rhs_jacobian(self, _t: np.ndarray, z: np.ndarray) -> np.ndarray:
        d = self.problem.dimension
        y, p = z[:d], z[d:]
        jacobian = np.zeros((2 * d, 2 * d, z.shape[1]))
        jacobian[:d, :d] = self._linear_state[:, :, None]
        jacobian[:d, d:] = self._control_adjoint[:, :, None]
        jacobian[d:, d:] = -self._linear_state.T[:, :, None]
        i = np.arange(d)
        jacobian[i, i] -= 3.0 * y**2
        jacobian[d + i, i] = -2.0 * self.problem.cell_width + 6.0 * y * p
        jacobian[d + i, d + i] += 3.0 * y**2
        return jacobian

    def _value(self, solution) -> float:
        widths = np.diff(solution.x)
        midpoints = (solution.x[:-1] + solution.x[1:]) / 2.0
        points = midpoints[:, None] + widths[:, None] * self._quadrature_nodes[None, :] / 2.0
        z = solution.sol(points.reshape(-1))
        d = self.problem.dimension
        control = self.problem.feedback_from_gradient(z[d:])
        cost = self.problem.running_cost(z[:d], control).reshape(points.shape)
        return float(np.sum(widths * (cost @ self._quadrature_weights) / 2.0))

    def solve_sample(self, initial_state: np.ndarray) -> AllenCahnSampleResult:
        initial = np.asarray(initial_state, dtype=np.float64)
        d = self.problem.dimension
        if initial.shape != (d,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"initial_state must be a finite vector of length {d}")

        def boundary(left, right):
            return np.concatenate((left[:d] - initial, right[d:]))

        left_jac = np.zeros((2 * d, 2 * d))
        right_jac = np.zeros_like(left_jac)
        left_jac[:d, :d] = np.eye(d)
        right_jac[d:, d:] = np.eye(d)

        candidates = []
        failures = []
        for decay in self.config.initial_guess_decays:
            guess = np.zeros((2 * d, self.time_grid.size))
            guess[:d] = initial[:, None] * np.exp(-decay * self.time_grid)[None, :]
            solved = solve_bvp(
                self._rhs, boundary, self.time_grid, guess,
                fun_jac=self._rhs_jacobian, bc_jac=lambda _left, _right: (left_jac, right_jac),
                tol=self.config.collocation_tol, bc_tol=self.config.collocation_tol,
                max_nodes=self.config.collocation_max_nodes,
            )
            residual = float(np.max(solved.rms_residuals))
            bc_residual = float(np.max(np.abs(boundary(solved.y[:, 0], solved.y[:, -1]))))
            valid = (
                solved.success and np.all(np.isfinite(solved.y))
                and np.isfinite(residual) and residual <= self.config.collocation_tol
                and np.isfinite(bc_residual) and bc_residual <= self.config.collocation_tol
            )
            value = self._value(solved) if valid else None
            if valid and np.isfinite(value) and value >= 0:
                candidates.append((value, solved, residual, bc_residual))
            else:
                failures.append((solved, residual, bc_residual))

        if not candidates:
            solved, residual, bc_residual = failures[-1]
            return AllenCahnSampleResult(
                initial_state=initial.copy(), converged=False,
                message="; ".join(str(failure[0].message) for failure in failures),
                value=None, gradient=None, iterations=int(solved.niter),
                collocation_residual=residual, boundary_residual=bc_residual,
                nodes=int(solved.x.size), reduced_gradient_norm=None,
                attempts=len(self.config.initial_guess_decays),
            )

        value, solved, residual, bc_residual = min(candidates, key=lambda item: item[0])
        control = self.problem.feedback_from_gradient(solved.y[d:])
        reduced_gradient = self.problem.reduced_gradient(solved.y[d:], control)
        store = self.config.store_trajectories
        return AllenCahnSampleResult(
            initial_state=initial.copy(), converged=True,
            message=f"lowest cost among {len(candidates)} converged Pontryagin candidates",
            value=value, gradient=solved.y[d:, 0].copy(), iterations=int(solved.niter),
            collocation_residual=residual, boundary_residual=bc_residual,
            nodes=int(solved.x.size),
            reduced_gradient_norm=float(np.sqrt(np.trapezoid(np.sum(reduced_gradient**2, axis=0), solved.x))),
            attempts=len(self.config.initial_guess_decays),
            time_grid=solved.x.copy() if store else None,
            state_trajectory=solved.y[:d].copy() if store else None,
            adjoint_trajectory=solved.y[d:].copy() if store else None,
            control=control.copy() if store else None,
        )

    def solve(self, initial_states: np.ndarray, progress: Callable[[int, int], None] | None = None) -> AllenCahnOpenLoopSolution:
        states = np.asarray(initial_states, dtype=np.float64)
        d = self.problem.dimension
        if states.ndim != 2 or states.shape[1] != d or not np.all(np.isfinite(states)):
            raise ValueError(f"initial_states must have finite shape (n, {d})")
        results = []
        for i, initial in enumerate(states):
            if progress is not None:
                progress(i, len(states))
            results.append(self.solve_sample(initial))
        if progress is not None:
            progress(len(states), len(states))
        retained = [result for result in results if result.converged]
        failed = [result.initial_state for result in results if not result.converged]
        return AllenCahnOpenLoopSolution(
            value_samples=ValueSamples(
                x=np.asarray([result.initial_state for result in retained]).reshape(-1, d),
                v=np.asarray([result.value for result in retained]),
                dv=np.asarray([result.gradient for result in retained]).reshape(-1, d),
            ),
            sample_results=tuple(results), failed_initial_states=np.asarray(failed).reshape(-1, d),
            config=self.config, problem=self.problem,
        )


__all__ = ["AllenCahnSolverConfig", "AllenCahnSampleResult", "AllenCahnOpenLoopSolution", "AllenCahnOpenLoopSolver"]
