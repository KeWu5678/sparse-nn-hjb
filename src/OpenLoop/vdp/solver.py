"""Pontryagin collocation solver for VDP value samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
from scipy.integrate import solve_bvp

from src.OpenLoop.value_samples import ValueSamples
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem


def _trapz_norm(time_grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=np.float64) ** 2, time_grid)))


@dataclass(frozen=True)
class VdpOpenLoopSolverConfig:
    """Numerical configuration for VDP value-sample generation."""

    time_step: float | None = None
    num_time_points: int | None = None
    store_trajectories: bool = False
    collocation_tol: float = 1e-8
    collocation_max_nodes: int = 20000

    def __post_init__(self) -> None:
        if self.collocation_tol <= 0.0 or self.collocation_max_nodes < 2:
            raise ValueError("invalid PMP collocation tolerance or node limit")
        if self.time_step is not None and self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if self.num_time_points is not None and self.num_time_points < 2:
            raise ValueError("num_time_points must be at least 2")


@dataclass(frozen=True)
class VdpSampleResult:
    initial_state: np.ndarray
    converged: bool
    message: str
    value: float | None
    gradient: np.ndarray | None
    reduced_gradient_norm: float | None
    iterations: int
    control: np.ndarray | None = None
    state_trajectory: np.ndarray | None = None
    adjoint_trajectory: np.ndarray | None = None
    reduced_gradient: np.ndarray | None = None
    collocation_residual: float | None = None


@dataclass(frozen=True)
class VdpOpenLoopSolution:
    value_samples: ValueSamples
    sample_results: tuple[VdpSampleResult, ...]
    failed_initial_states: np.ndarray
    config: VdpOpenLoopSolverConfig
    problem: VdpOptimalControlProblem

    def save_dataset(
        self,
        output_dir: str | Path,
        *,
        grid_shape: tuple[int, int] | None = None,
        date_tag: str | None = None,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        date = datetime.now().strftime("%Y%m%d") if date_tag is None else date_tag
        run_dir = output_path / f"VDP_{date}_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        shape_tag = (
            f"grid_{grid_shape[0]}x{grid_shape[1]}"
            if grid_shape is not None
            else f"N{self.value_samples.size}"
        )
        stem = f"VDP_pmp_{shape_tag}_{date}"
        data_path = self.value_samples.save_npz(run_dir / f"{stem}.npz")
        meta_path = run_dir / f"VDP_pmp_meta_{shape_tag}_{date}.json"
        failed_path = run_dir / f"VDP_pmp_failed_{shape_tag}_{date}.json"

        meta = {
            "problem": asdict(self.problem),
            "solver": asdict(self.config),
            "retained_samples": self.value_samples.size,
            "requested_samples": len(self.sample_results),
            "failed_samples": int(self.failed_initial_states.shape[0]),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        failed_payload = [
            {
                "initial_state": result.initial_state.tolist(),
                "message": result.message,
                "iterations": result.iterations,
                "reduced_gradient_norm": result.reduced_gradient_norm,
            }
            for result in self.sample_results
            if not result.converged
        ]
        failed_path.write_text(json.dumps(failed_payload, indent=2), encoding="utf-8")
        return {"run_dir": run_dir, "data": data_path, "meta": meta_path, "failed": failed_path}


@dataclass(frozen=True)
class _Evaluation:
    value: float
    gradient: np.ndarray
    reduced_gradient_norm: float
    state: np.ndarray
    adjoint: np.ndarray
    control: np.ndarray


class VdpOpenLoopSolver:
    """Generate VDP `(x, V(x), dV(x))` samples for PDAP."""

    def __init__(
        self,
        problem: VdpOptimalControlProblem | None = None,
        config: VdpOpenLoopSolverConfig | None = None,
    ) -> None:
        self.problem = problem or VdpOptimalControlProblem()
        self.config = config or VdpOpenLoopSolverConfig()
        self.time_grid = self._build_time_grid()

    def solve(
        self,
        initial_states: np.ndarray,
        progress: Callable[[int, int], None] | None = None,
    ) -> VdpOpenLoopSolution:
        states = np.asarray(initial_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("initial_states must have shape (n, 2)")

        results: list[VdpSampleResult] = []
        rows_x: list[np.ndarray] = []
        rows_v: list[float] = []
        rows_dv: list[np.ndarray] = []
        failed: list[np.ndarray] = []
        for idx, initial_state in enumerate(states):
            if progress is not None:
                progress(idx, states.shape[0])
            result = self.solve_sample(initial_state)
            results.append(result)
            if result.converged and result.value is not None and result.gradient is not None:
                rows_x.append(initial_state.copy())
                rows_v.append(float(result.value))
                rows_dv.append(result.gradient.copy())
            else:
                failed.append(initial_state.copy())
        if progress is not None:
            progress(states.shape[0], states.shape[0])

        samples = ValueSamples(
            x=np.asarray(rows_x, dtype=np.float64),
            v=np.asarray(rows_v, dtype=np.float64),
            dv=np.asarray(rows_dv, dtype=np.float64),
        )
        failed_states = (
            np.asarray(failed, dtype=np.float64)
            if failed
            else np.empty((0, 2), dtype=np.float64)
        )
        return VdpOpenLoopSolution(
            value_samples=samples,
            sample_results=tuple(results),
            failed_initial_states=failed_states,
            config=self.config,
            problem=self.problem,
        )

    def solve_sample(self, initial_state: np.ndarray) -> VdpSampleResult:
        initial = np.asarray(initial_state, dtype=np.float64)
        if initial.shape != (2,):
            raise ValueError("initial_state must have shape (2,)")
        return self._solve_sample_pmp(initial)

    def _solve_sample_pmp(self, initial_state: np.ndarray) -> VdpSampleResult:
        """Solve the free-terminal PMP equations from two initial guesses.

        Retain the least-cost converged stationary solution. This is a numerical
        reference, not a certificate of global optimality.
        """
        def rhs(t, z):
            control = -z[3] / (2.0 * self.problem.beta)
            return np.vstack((
                self.problem.dynamics(t, z[:2], control),
                self.problem.adjoint_rhs(t, z[:2], z[2:]),
            ))

        def boundary(left, right):
            return np.r_[left[:2] - initial_state, right[2:]]

        nodes, weights = np.polynomial.legendre.leggauss(3)
        best = None
        for decay in (1.0, 0.0):
            guess = np.vstack((
                initial_state[:, None] * np.exp(-decay * self.time_grid),
                np.zeros((2, len(self.time_grid))),
            ))
            solution = solve_bvp(
                rhs, boundary, self.time_grid, guess,
                tol=self.config.collocation_tol,
                max_nodes=self.config.collocation_max_nodes,
            )
            if not solution.success:
                continue
            z = solution.sol(self.time_grid)
            control = -z[3] / (2.0 * self.problem.beta)
            evaluation = self._build_evaluation(z[:2], z[2:], control)
            dt = np.diff(solution.x)
            tq = solution.x[:-1, None] + 0.5 * dt[:, None] * (1.0 + nodes)
            zq = solution.sol(tq.ravel())
            uq = -zq[3] / (2.0 * self.problem.beta)
            rates = self.problem.running_cost(zq[:2], uq).reshape(-1, len(nodes))
            value = float(np.sum(0.5 * dt * (rates @ weights)))
            evaluation = replace(evaluation, value=value)
            result = replace(
                self._success_result(initial_state, evaluation, solution.niter),
                collocation_residual=float(np.max(solution.rms_residuals)),
            )
            if np.isfinite(value) and (best is None or value < best.value):
                best = result
        if best is None:
            return self._failure_result(initial_state, "PMP collocation did not converge", 0)
        return best

    def _build_evaluation(
        self,
        state: np.ndarray,
        adjoint: np.ndarray,
        control: np.ndarray,
    ) -> _Evaluation:
        running = self.problem.running_cost(state, control)
        value = float(np.trapezoid(running, self.time_grid))
        reduced_gradient = self.problem.reduced_gradient(adjoint, control)
        return _Evaluation(
            value=value,
            gradient=reduced_gradient,
            reduced_gradient_norm=_trapz_norm(self.time_grid, reduced_gradient),
            state=state,
            adjoint=adjoint,
            control=control,
        )

    def _success_result(
        self,
        initial_state: np.ndarray,
        evaluation: _Evaluation,
        iterations: int,
    ) -> VdpSampleResult:
        return VdpSampleResult(
            initial_state=initial_state.copy(),
            converged=True,
            message="converged",
            value=float(evaluation.value),
            gradient=evaluation.adjoint[:, 0].copy(),
            reduced_gradient_norm=float(evaluation.reduced_gradient_norm),
            iterations=int(iterations),
            control=evaluation.control.copy() if self.config.store_trajectories else None,
            state_trajectory=evaluation.state.copy() if self.config.store_trajectories else None,
            adjoint_trajectory=evaluation.adjoint.copy() if self.config.store_trajectories else None,
            reduced_gradient=evaluation.gradient.copy() if self.config.store_trajectories else None,
        )

    def _failure_result(
        self,
        initial_state: np.ndarray,
        message: str,
        iterations: int,
    ) -> VdpSampleResult:
        return VdpSampleResult(
            initial_state=initial_state.copy(),
            converged=False,
            message=message,
            value=None,
            gradient=None,
            reduced_gradient_norm=None,
            iterations=int(iterations),
        )

    def _build_time_grid(self) -> np.ndarray:
        if self.config.num_time_points is not None:
            return np.linspace(0.0, self.problem.T_final, self.config.num_time_points)
        if self.config.time_step is not None:
            count = int(round(self.problem.T_final / self.config.time_step))
            return np.linspace(0.0, self.problem.T_final, count + 1)
        return np.linspace(0.0, self.problem.T_final, 301)


__all__ = [
    "VdpOpenLoopSolver",
    "VdpOpenLoopSolverConfig",
    "VdpOpenLoopSolution",
    "VdpSampleResult",
]
