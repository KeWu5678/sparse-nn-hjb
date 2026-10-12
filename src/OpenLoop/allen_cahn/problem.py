"""Finite-volume Allen--Cahn control benchmark with three local actuators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class AllenCahnProblem:
    """Benchmark from Kunisch, Vásquez-Varas, and Walter (2023), §9.3.

    On ``(-1, 1)`` with zero Neumann boundary data, the state obeys
    ``y' = nu L y + y - y**3 + B u``.  The finite-horizon cost is
    ``integral(h |y|² + beta |u|²)`` with no terminal cost.  ``beta`` is
    an explicit benchmark choice; the spatial discretization uses cell
    averages rather than the Chebyshev collocation and Clenshaw--Curtis
    cost quadrature used in the source paper.

    State and adjoint arrays have shape ``(dimension,)`` or
    ``(dimension, n_times)``; control arrays use three rows.
    """

    dimension: int = 6
    nu: float = 0.1
    beta: float = 0.01
    T_final: float = 3.0

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, (int, np.integer)) or self.dimension < 2:
            raise ValueError("dimension must be an integer at least 2")
        for name in ("nu", "beta", "T_final"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def cell_width(self) -> float:
        return 2.0 / self.dimension

    @cached_property
    def cell_centers(self) -> np.ndarray:
        result = -1.0 + (np.arange(self.dimension) + 0.5) * self.cell_width
        result.setflags(write=False)
        return result

    @cached_property
    def laplacian(self) -> np.ndarray:
        diagonal = np.full(self.dimension, -2.0)
        diagonal[[0, -1]] = -1.0
        result = (
            np.diag(diagonal)
            + np.diag(np.ones(self.dimension - 1), 1)
            + np.diag(np.ones(self.dimension - 1), -1)
        ) / self.cell_width**2
        result.setflags(write=False)
        return result

    @cached_property
    def control_matrix(self) -> np.ndarray:
        edges = np.linspace(-1.0, 1.0, self.dimension + 1)
        intervals = np.array([[-0.7, -0.4], [-0.2, 0.2], [0.4, 0.7]])
        overlap = np.minimum(edges[1:, None], intervals[None, :, 1]) - np.maximum(
            edges[:-1, None], intervals[None, :, 0]
        )
        result = np.maximum(overlap, 0.0) / self.cell_width
        result.setflags(write=False)
        return result

    def dynamics(self, _t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return self.nu * (self.laplacian @ y) + y - y**3 + self.control_matrix @ np.asarray(u, dtype=np.float64)

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        return -2.0 * self.cell_width * y - self.nu * (self.laplacian.T @ p) - (1.0 - 3.0 * y**2) * p

    def running_cost(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        return self.cell_width * np.sum(y**2, axis=0) + self.beta * np.sum(u**2, axis=0)

    def feedback_from_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return -(self.control_matrix.T @ np.asarray(gradient, dtype=np.float64)) / (2.0 * self.beta)

    def reduced_gradient(self, p: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.control_matrix.T @ np.asarray(p, dtype=np.float64) + 2.0 * self.beta * np.asarray(u, dtype=np.float64)

    def terminal_gradient(self, y_terminal: np.ndarray) -> np.ndarray:
        return np.zeros_like(y_terminal, dtype=np.float64)

    def rk4_rollout(
        self, u_of_x, x0, *, T: float | None = None, dt: float = 0.002,
        u_clip: float = 50.0,
    ):
        """Roll out a held-per-step feedback and integrate its running cost.

        Returns ``(time, states, controls, cost)`` with states and controls
        stored as rows.  The final step reaches the requested horizon.
        Nonfinite or divergent trajectories raise rather than return a
        misleading partial-horizon cost.
        """
        horizon = self.T_final if T is None else float(T)
        if not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("T and dt must be finite and positive")
        if not np.isfinite(u_clip) or u_clip <= 0:
            raise ValueError("u_clip must be finite and positive")
        initial = np.asarray(x0, dtype=np.float64)
        if initial.shape != (self.dimension,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"x0 must be a finite vector of length {self.dimension}")
        n_steps = int(np.ceil(horizon / dt))
        times = np.linspace(0.0, horizon, n_steps + 1)
        states = np.empty((n_steps + 1, self.dimension))
        controls = np.empty((n_steps + 1, self.control_matrix.shape[1]))
        states[0] = initial
        cost = 0.0
        for i, step in enumerate(np.diff(times)):
            state = states[i]
            control = np.asarray(u_of_x(state), dtype=np.float64).reshape(-1)
            if control.shape != (3,) or not np.all(np.isfinite(control)):
                raise ValueError("feedback must return three finite controls")
            control = np.clip(control, -u_clip, u_clip)
            controls[i] = control
            k1 = self.dynamics(times[i], state, control)
            state2 = state + step * k1 / 2.0
            k2 = self.dynamics(times[i] + step / 2.0, state2, control)
            state3 = state + step * k2 / 2.0
            k3 = self.dynamics(times[i] + step / 2.0, state3, control)
            state4 = state + step * k3
            k4 = self.dynamics(times[i + 1], state4, control)
            states[i + 1] = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            cost += step * float(
                self.running_cost(state, control) + 2.0 * self.running_cost(state2, control)
                + 2.0 * self.running_cost(state3, control) + self.running_cost(state4, control)
            ) / 6.0
            if not np.all(np.isfinite(states[i + 1])) or np.max(np.abs(states[i + 1])) > 1e3 or not np.isfinite(cost):
                raise RuntimeError(f"closed-loop rollout diverged at t={times[i + 1]:g}")
        controls[-1] = controls[-2]
        return times, states, controls, cost


__all__ = ["AllenCahnProblem"]
