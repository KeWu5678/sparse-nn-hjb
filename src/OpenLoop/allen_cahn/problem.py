"""Finite-volume Allen--Cahn control benchmark with three local actuators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class AllenCahnProblem:
    """Benchmark from Kunisch, Vásquez-Varas, and Walter (2023), §9.3.

    On ``(-1, 1)`` with zero Neumann boundary data, the state obeys
    ``y' = nu L y + y - y**3 + B u``. The finite-horizon cost is
    ``integral(h |y|² + beta |u|²)`` with no terminal cost. ``beta`` is
    an explicit benchmark choice; the spatial discretization uses cell
    averages rather than the source paper's Chebyshev collocation and
    Clenshaw--Curtis cost quadrature.

    State and adjoint arrays have shape ``(dimension,)`` or
    ``(dimension, n_times)``; control arrays use three rows. Pontryagin
    operators accept stacked state/adjoint arrays ``(2 * dimension, n_times)``.
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

    def pontryagin_rhs(self, t: np.ndarray, z: np.ndarray) -> np.ndarray:
        """State/costate equations after eliminating the stationary control."""
        y, p = z[:self.dimension], z[self.dimension:]
        return np.vstack((
            self._uncontrolled_dynamics(y) + self._control_adjoint @ p,
            self.adjoint_rhs(t, y, p),
        ))

    def pontryagin_jacobian(self, _t: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Analytic derivative of :meth:`pontryagin_rhs` at each time node."""
        d = self.dimension
        y, p = z[:d], z[d:]
        jacobian = np.zeros((2 * d, 2 * d, z.shape[1]))
        jacobian[:d, :d] = self._linear_state[:, :, None]
        jacobian[:d, d:] = self._control_adjoint[:, :, None]
        jacobian[d:, d:] = -self._linear_state.T[:, :, None]
        i = np.arange(d)
        jacobian[i, i] -= 3.0 * y**2
        jacobian[d + i, i] = -2.0 * self.cell_width + 6.0 * y * p
        jacobian[d + i, d + i] += 3.0 * y**2
        return jacobian

    def dynamics(self, _t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._uncontrolled_dynamics(y) + self.control_matrix @ np.asarray(u, dtype=np.float64)

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        return -2.0 * self.cell_width * y - self._linear_state.T @ p + 3.0 * y**2 * p

    def running_cost(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        return self.cell_width * np.sum(y**2, axis=0) + self.beta * np.sum(u**2, axis=0)

    def feedback_from_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return -(self.control_matrix.T @ np.asarray(gradient, dtype=np.float64)) / (2.0 * self.beta)

    def reduced_gradient(self, p: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Stationarity residual; it vanishes by construction for eliminated controls."""
        return self.control_matrix.T @ np.asarray(p, dtype=np.float64) + 2.0 * self.beta * np.asarray(u, dtype=np.float64)

    def _uncontrolled_dynamics(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return self._linear_state @ y - y**3

    @cached_property
    def _linear_state(self) -> np.ndarray:
        return self.nu * self.laplacian + np.eye(self.dimension)

    @cached_property
    def _control_adjoint(self) -> np.ndarray:
        return -(self.control_matrix @ self.control_matrix.T) / (2.0 * self.beta)

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


__all__ = ["AllenCahnProblem"]
