"""Van der Pol finite-horizon optimal-control problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VdpOptimalControlProblem:
    """Smooth VDP benchmark used to generate PDAP value samples."""

    beta: float = 0.1
    mu: float = 1.0
    state_weight: float = 1.0
    T_final: float = 3.0

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        if self.state_weight <= 0.0:
            raise ValueError("state_weight must be positive")
        if self.T_final <= 0.0:
            raise ValueError("T_final must be positive")

    def dynamics(self, _t: float, y: np.ndarray, u: float) -> np.ndarray:
        y1, y2 = np.asarray(y, dtype=np.float64)
        return np.array(
            [
                y2,
                -y1 + self.mu * (1.0 - y1**2) * y2 + u,
            ],
            dtype=np.float64,
        )

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y1, y2 = np.asarray(y, dtype=np.float64)
        p1, p2 = np.asarray(p, dtype=np.float64)
        return np.array(
            [
                -2.0 * self.state_weight * y1
                + p2 * (1.0 + 2.0 * self.mu * y1 * y2),
                -2.0 * self.state_weight * y2
                - p1
                - self.mu * (1.0 - y1**2) * p2,
            ],
            dtype=np.float64,
        )

    def terminal_gradient(self, _y_terminal: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def running_cost(self, y_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        y_values = np.asarray(y_values, dtype=np.float64)
        u_values = np.asarray(u_values, dtype=np.float64)
        state_cost = self.state_weight * np.sum(y_values * y_values, axis=0)
        control_cost = self.beta * u_values * u_values
        return state_cost + control_cost

    def reduced_gradient(self, p_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        p_values = np.asarray(p_values, dtype=np.float64)
        u_values = np.asarray(u_values, dtype=np.float64)
        return p_values[1, :] + 2.0 * self.beta * u_values

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        gradient = np.asarray(gradient, dtype=np.float64)
        return float(-gradient[1] / (2.0 * self.beta))

    def rk4_rollout(self, u_of_x, x0, *, T: float = 12.0, dt: float = 0.01,
                    u_clip: float = 50.0):
        """Closed-loop RK4 rollout of ``ẏ = f(y, u(y))`` under a state feedback law.

        Integrates with fixed-step RK4 and a zero-order hold on ``u_of_x`` (clipped
        to ``±u_clip`` as a numerical guard). Returns ``(t, xs, us, cost)`` — the time
        grid, state trajectory, applied controls, and accumulated running cost —
        truncated early if the closed loop diverges (non-finite or ``|y| > 1e3``).
        """
        n = int(T / dt)
        xs = np.zeros((n + 1, 2)); us = np.zeros(n + 1); xs[0] = np.asarray(x0, np.float64)
        cost = 0.0
        last = n
        for i in range(n):
            x = xs[i]
            u = float(np.clip(np.ravel(u_of_x(x))[0], -u_clip, u_clip)); us[i] = u
            cost += float(self.running_cost(x.reshape(2, 1), np.array([u]))[0]) * dt
            k1 = self.dynamics(0.0, x, u)
            k2 = self.dynamics(0.0, x + 0.5 * dt * k1, u)
            k3 = self.dynamics(0.0, x + 0.5 * dt * k2, u)
            k4 = self.dynamics(0.0, x + dt * k3, u)
            xs[i + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(xs[i + 1])) or np.abs(xs[i + 1]).max() > 1e3:
                last = i + 1; break
        us[last] = us[last - 1] if last else 0.0
        t = np.arange(last + 1) * dt
        return t, xs[:last + 1], us[:last + 1], cost

    def true_feedback(self, samples):
        """True control u*(x) = feedback_from_gradient(∇V_true) from the dataset costate,
        via a smooth C1 (Clough-Tocher) interpolant of the y2-costate — nearest-neighbour
        would make the gradient piecewise-constant and the control spuriously bang-bang.
        Returns a ``u_of_x`` closure."""
        from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
        smooth = CloughTocher2DInterpolator(samples["x"], samples["dv"][:, 1])
        nearest = NearestNDInterpolator(samples["x"], samples["dv"][:, 1])  # fill outside hull

        def u(x):
            x = np.asarray(x).reshape(1, 2)
            dv2 = smooth(x[:, 0], x[:, 1])[0]
            if not np.isfinite(dv2):
                dv2 = float(nearest(x[:, 0], x[:, 1])[0])
            return self.feedback_from_gradient(np.array([0.0, dv2]))
        return u


__all__ = ["VdpOptimalControlProblem"]
