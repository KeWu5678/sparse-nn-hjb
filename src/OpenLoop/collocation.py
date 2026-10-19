"""Pontryagin collocation reference samples for finite-horizon open-loop problems."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp

from src.data import ValueSamples


class PontryaginProblem(Protocol):
    """Free-terminal problem with no terminal cost and an eliminated control.

    ``pontryagin_rhs`` maps stacked state/costate arrays ``(2 * dimension, n)``
    to their time derivatives after substituting the stationary control
    ``feedback_from_gradient(p)``. Control arrays have shape ``(m, n)``, or
    ``(n,)`` for a single control. A problem may also define
    ``pontryagin_jacobian(t, z)`` with shape ``(2d, 2d, n)``; without it, the
    collocation Jacobian is estimated by finite differences.
    """

    dimension: int
    T_final: float

    def pontryagin_rhs(self, t: np.ndarray, z: np.ndarray) -> np.ndarray: ...

    def feedback_from_gradient(self, gradient: np.ndarray) -> np.ndarray: ...

    def running_cost(self, y: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def reduced_gradient(self, p: np.ndarray, u: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CollocationConfig:
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
class SolveDiagnostics:
    """Numerical diagnostics; unavailable or nonfinite residuals are ``None``."""

    message: str
    iterations: int
    collocation_residual: float | None
    boundary_residual: float | None
    nodes: int
    reduced_gradient_norm: float | None
    attempts: int


@dataclass(frozen=True)
class Trajectory:
    time_grid: np.ndarray
    state_trajectory: np.ndarray
    adjoint_trajectory: np.ndarray
    control: np.ndarray


@dataclass(frozen=True)
class SampleResult:
    """A converged value/gradient pair, with optional retained trajectory data."""

    initial_state: np.ndarray
    value: float
    gradient: np.ndarray
    diagnostics: SolveDiagnostics
    trajectory: Trajectory | None = None


@dataclass(frozen=True)
class SampleFailure:
    initial_state: np.ndarray
    diagnostics: SolveDiagnostics


class ConvergenceError(RuntimeError):
    """No acceptable Pontryagin candidate was found for one initial state."""

    def __init__(self, failure: SampleFailure) -> None:
        super().__init__(failure.diagnostics.message)
        self.failure = failure


class BatchError(RuntimeError):
    """The batch is incomplete; every requested state's outcome is retained."""

    def __init__(self, sample_results: tuple[SampleResult | SampleFailure, ...]) -> None:
        self.sample_results = sample_results
        self.failed_initial_states = np.stack([
            result.initial_state for result in sample_results
            if isinstance(result, SampleFailure)
        ])
        super().__init__(f"{len(self.failed_initial_states)} failed solves; no training dataset assembled")


@dataclass(frozen=True)
class CollocationSolution:
    value_samples: ValueSamples
    sample_results: tuple[SampleResult, ...]
    config: CollocationConfig
    problem: PontryaginProblem


class CollocationSolver:
    """Solve the state/adjoint boundary problem from multiple initial guesses.

    Each retained solution satisfies Pontryagin stationarity. The least-cost
    converged candidate is a numerical reference, not a global-optimality
    certificate for a nonlinear control problem.
    """

    def __init__(self, problem: PontryaginProblem, config: CollocationConfig | None = None) -> None:
        self._problem = problem
        self._config = config or CollocationConfig()
        self._time_grid = np.linspace(0.0, self._problem.T_final, self._config.initial_mesh_points)
        self._quadrature_nodes, self._quadrature_weights = leggauss(self._config.quadrature_order)

    def solve(
        self, initial_states: np.ndarray,
        progress: Callable[[int, int], None] | None = None, *, workers: int = 1,
    ) -> CollocationSolution:
        """Solve all requested states in order, or raise ``BatchError``.

        Progress receives completed and requested counts, starting at zero.
        Numerical failures do not stop remaining states from being attempted;
        they are collected in the exception instead of a partial dataset.
        """
        states = np.asarray(initial_states, dtype=np.float64)
        d = self._problem.dimension
        if states.ndim != 2 or states.shape[1] != d or not np.all(np.isfinite(states)):
            raise ValueError(f"initial_states must have finite shape (n, {d})")
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be an integer at least 1")
        if workers == 1:
            results = _collect_results(map(self._sample_outcome, states), len(states), progress)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = _collect_results(
                    executor.map(self._sample_outcome, states, chunksize=4), len(states), progress,
                )
        if any(isinstance(result, SampleFailure) for result in results):
            raise BatchError(results)
        return CollocationSolution(
            value_samples=_value_samples(results, d), sample_results=results,
            config=self._config, problem=self._problem,
        )

    def solve_sample(self, initial_state: np.ndarray) -> SampleResult:
        """Return a finite value/gradient pair or raise ``ConvergenceError``."""
        initial = np.asarray(initial_state, dtype=np.float64)
        d = self._problem.dimension
        if initial.shape != (d,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"initial_state must be a finite vector of length {d}")
        candidates, failures = [], []
        for decay in self._config.initial_guess_decays:
            try:
                candidates.append(self._solve_candidate(initial, decay))
            except ConvergenceError as error:
                failures.append(error.failure)
        if not candidates:
            diagnostics = replace(
                failures[-1].diagnostics,
                message="; ".join(failure.diagnostics.message for failure in failures),
                attempts=len(self._config.initial_guess_decays),
            )
            raise ConvergenceError(SampleFailure(initial.copy(), diagnostics))
        candidate = min(candidates, key=lambda item: item.value)
        return self._sample_result(initial, candidate, len(candidates))

    def _sample_outcome(self, initial: np.ndarray) -> SampleResult | SampleFailure:
        """Catch numerical failures at the batch boundary, including worker processes."""
        try:
            outcome = self.solve_sample(initial)
        except ConvergenceError as error:
            outcome = error.failure
        return outcome

    def _solve_candidate(self, initial: np.ndarray, decay: float) -> _CollocationCandidate:
        solved, boundary_residual = self._collocate(initial, decay)
        residual = float(np.max(solved.rms_residuals))
        valid = (
            solved.success and np.all(np.isfinite(solved.y))
            and np.isfinite(residual) and residual <= self._config.collocation_tol
            and np.isfinite(boundary_residual) and boundary_residual <= self._config.collocation_tol
        )
        if not valid:
            raise self._convergence_error(initial, solved, residual, boundary_residual, str(solved.message))
        value = self._value(solved)
        if not np.isfinite(value) or value < 0:
            raise self._convergence_error(
                initial, solved, residual, boundary_residual,
                "collocation converged but quadrature cost was not finite and nonnegative",
            )
        return _CollocationCandidate(value, solved, residual, boundary_residual)

    def _collocate(self, initial: np.ndarray, decay: float):
        d = self._problem.dimension

        def boundary(left, right):
            return np.concatenate((left[:d] - initial, right[d:]))

        left_jac = np.zeros((2 * d, 2 * d))
        right_jac = np.zeros_like(left_jac)
        left_jac[:d, :d] = np.eye(d)
        right_jac[d:, d:] = np.eye(d)
        guess = np.zeros((2 * d, self._time_grid.size))
        guess[:d] = initial[:, None] * np.exp(-decay * self._time_grid)[None, :]
        solved = solve_bvp(
            self._problem.pontryagin_rhs, boundary, self._time_grid, guess,
            fun_jac=getattr(self._problem, "pontryagin_jacobian", None),
            bc_jac=lambda _left, _right: (left_jac, right_jac),
            tol=self._config.collocation_tol, bc_tol=self._config.collocation_tol,
            max_nodes=self._config.collocation_max_nodes,
        )
        boundary_residual = float(np.max(np.abs(boundary(solved.y[:, 0], solved.y[:, -1]))))
        return solved, boundary_residual

    def _value(self, solution) -> float:
        widths = np.diff(solution.x)
        midpoints = (solution.x[:-1] + solution.x[1:]) / 2.0
        points = midpoints[:, None] + widths[:, None] * self._quadrature_nodes[None, :] / 2.0
        z = solution.sol(points.reshape(-1))
        d = self._problem.dimension
        control = self._problem.feedback_from_gradient(z[d:])
        cost = self._problem.running_cost(z[:d], control).reshape(points.shape)
        return float(np.sum(widths * (cost @ self._quadrature_weights) / 2.0))

    def _convergence_error(self, initial, solved, residual, boundary_residual, message):
        diagnostics = SolveDiagnostics(
            message=message, iterations=int(solved.niter),
            collocation_residual=residual if np.isfinite(residual) else None,
            boundary_residual=boundary_residual if np.isfinite(boundary_residual) else None,
            nodes=int(solved.x.size), reduced_gradient_norm=None, attempts=1,
        )
        return ConvergenceError(SampleFailure(initial.copy(), diagnostics))

    def _sample_result(self, initial, candidate, converged_count) -> SampleResult:
        solved = candidate.solution
        d = self._problem.dimension
        control = self._problem.feedback_from_gradient(solved.y[d:])
        reduced_gradient = np.atleast_2d(self._problem.reduced_gradient(solved.y[d:], control))
        diagnostics = SolveDiagnostics(
            message=f"lowest cost among {converged_count} converged Pontryagin candidates",
            iterations=int(solved.niter), collocation_residual=candidate.collocation_residual,
            boundary_residual=candidate.boundary_residual, nodes=int(solved.x.size),
            reduced_gradient_norm=float(np.sqrt(np.trapezoid(np.sum(reduced_gradient**2, axis=0), solved.x))),
            attempts=len(self._config.initial_guess_decays),
        )
        trajectory = None
        if self._config.store_trajectories:
            trajectory = Trajectory(
                time_grid=solved.x.copy(), state_trajectory=solved.y[:d].copy(),
                adjoint_trajectory=solved.y[d:].copy(), control=control.copy(),
            )
        return SampleResult(
            initial_state=initial.copy(), value=candidate.value,
            gradient=solved.y[d:, 0].copy(), diagnostics=diagnostics, trajectory=trajectory,
        )


@dataclass(frozen=True)
class _CollocationCandidate:
    value: float
    solution: object
    collocation_residual: float
    boundary_residual: float


def _collect_results(outcomes, count: int, progress) -> tuple:
    results = []
    if progress is not None:
        progress(0, count)
    for outcome in outcomes:
        results.append(outcome)
        if progress is not None:
            progress(len(results), count)
    return tuple(results)


def _value_samples(results: tuple[SampleResult, ...], dimension: int) -> ValueSamples:
    return ValueSamples(
        x=np.asarray([result.initial_state for result in results]).reshape(-1, dimension),
        v=np.asarray([result.value for result in results]),
        dv=np.asarray([result.gradient for result in results]).reshape(-1, dimension),
    )


__all__ = [
    "PontryaginProblem", "CollocationConfig", "SolveDiagnostics", "Trajectory",
    "SampleResult", "SampleFailure", "ConvergenceError", "BatchError",
    "CollocationSolution", "CollocationSolver",
]
