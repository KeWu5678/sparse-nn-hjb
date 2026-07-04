"""Mathematical pendulum swing-up optimal-control problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class PendulumSwingUpProblem:
    """Infinite-horizon pendulum swing-up OCP from Han and Yang."""

    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.1
    gravity: float = 9.8
    state_weights: tuple[float, float] = (1.0, 1.0)
    control_weight: float = 1.0
    control_limit: float | None = None
    control_gain: float = field(init=False)
    damping_gain: float = field(init=False)
    gravity_gain: float = field(init=False)
    local_lqr_matrix: np.ndarray = field(init=False, repr=False)
    local_lqr_eigvecs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mass = float(self.mass)
        length = float(self.length)
        damping = float(self.damping)
        gravity = float(self.gravity)
        weights = np.asarray(self.state_weights, dtype=np.float64)
        control_weight = float(self.control_weight)
        control_limit = None if self.control_limit is None else float(self.control_limit)

        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if weights.shape != (2,):
            raise ValueError("state_weights must contain two values")
        if np.any(weights < 0.0):
            raise ValueError("state_weights must be nonnegative")
        if control_weight <= 0.0:
            raise ValueError("control_weight must be positive")
        if control_limit is not None and control_limit <= 0.0:
            raise ValueError("control_limit must be positive when provided")

        control_gain = 1.0 / (mass * length**2)
        damping_gain = damping / (mass * length**2)
        gravity_gain = gravity / length
        lqr = self._compute_local_lqr_matrix(
            control_gain,
            damping_gain,
            gravity_gain,
            weights,
            control_weight,
        )

        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "state_weights", (float(weights[0]), float(weights[1])))
        object.__setattr__(self, "control_weight", control_weight)
        object.__setattr__(self, "control_limit", control_limit)
        object.__setattr__(self, "control_gain", control_gain)
        object.__setattr__(self, "damping_gain", damping_gain)
        object.__setattr__(self, "gravity_gain", gravity_gain)
        object.__setattr__(self, "local_lqr_matrix", lqr)
        object.__setattr__(self, "local_lqr_eigvecs", np.linalg.eigh(lqr)[1])

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _compute_local_lqr_matrix(
        control_gain: float,
        damping_gain: float,
        gravity_gain: float,
        state_weights: np.ndarray,
        control_weight: float,
    ) -> np.ndarray:
        linearized_dynamics = np.array(
            [[0.0, 1.0], [gravity_gain, -damping_gain]],
            dtype=np.float64,
        )
        control_matrix = np.array([[0.0], [control_gain]], dtype=np.float64)
        q = np.diag(state_weights)
        r = np.array([[control_weight]], dtype=np.float64)
        return solve_continuous_are(linearized_dynamics, control_matrix, q, r)

    def bounded_control(self, control: float | np.ndarray) -> float | np.ndarray:
        if self.control_limit is None:
            return control
        return np.clip(control, -self.control_limit, self.control_limit)

    def dynamics(self, state: np.ndarray, control: np.ndarray | float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        return np.stack(
            [
                omega,
                -self.damping_gain * omega
                + self.gravity_gain * np.sin(theta)
                + self.control_gain * control_value,
            ],
            axis=-1,
        )

    def costate_rhs(self, state: np.ndarray, costate: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        costate = np.asarray(costate, dtype=np.float64)
        theta = state[..., 0]
        omega = state[..., 1]
        p1 = costate[..., 0]
        p2 = costate[..., 1]
        q1, q2 = self.state_weights
        return np.stack(
            [
                -2.0 * q1 * np.sin(theta) - self.gravity_gain * np.cos(theta) * p2,
                -2.0 * q2 * omega - p1 + self.damping_gain * p2,
            ],
            axis=-1,
        )

    def running_cost(self, state: np.ndarray, control: np.ndarray | float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        q1, q2 = self.state_weights
        return (
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega * omega
            + self.control_weight * control_value * control_value
        )

    def stationarity_residual(
        self,
        costate: np.ndarray,
        control: np.ndarray | float,
    ) -> np.ndarray:
        p2 = np.asarray(costate, dtype=np.float64)[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        return self.control_gain * p2 + 2.0 * self.control_weight * control_value

    def feedback_from_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """The Hamiltonian-minimizing control synthesized from a value gradient.

        Along the optimal trajectory the costate equals ∇V(x), so this maps either
        a PMP costate or a (learned) value gradient to the feedback control
        ``u = -control_gain · ∂_θ̇ V / (2·control_weight)``.
        """
        p2 = np.asarray(gradient, dtype=np.float64)[..., 1]
        control = -self.control_gain * p2 / (2.0 * self.control_weight)
        return np.asarray(self.bounded_control(control), dtype=np.float64)

    def local_lqr_value(self, state: np.ndarray, weight: float = 1.0) -> float:
        state = np.asarray(state, dtype=np.float64).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return float(weight * state @ self.local_lqr_matrix @ state)

    def local_lqr_gradient(self, state: np.ndarray, weight: float = 1.0) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return 2.0 * weight * self.local_lqr_matrix @ state

    def boundary_state(self, angle: float, epsilon: float) -> np.ndarray:
        """Point on the local LQR level set used as PMP terminal data."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        direction = (
            self.local_lqr_eigvecs[:, 0] * np.cos(angle)
            + self.local_lqr_eigvecs[:, 1] * np.sin(angle)
        )
        scale = np.sqrt(epsilon / float(direction @ self.local_lqr_matrix @ direction))
        return np.asarray(scale * direction, dtype=np.float64)

    def boundary_costate(self, state: np.ndarray) -> np.ndarray:
        return self.local_lqr_gradient(state)

    def hjb_residual(
        self,
        state: np.ndarray,
        costate: np.ndarray,
        control: np.ndarray | float | None = None,
    ) -> np.ndarray:
        if control is None:
            control = self.feedback_from_gradient(costate)
        dynamics = self.dynamics(state, control)
        running = self.running_cost(state, control)
        return running + np.sum(np.asarray(costate, dtype=np.float64) * dynamics, axis=-1)

    def rk4_rollout(self, u_of_x, x0, *, T: float = 10.0, dt: float = 0.01,
                    u_clip: float = 30.0):
        """Closed-loop RK4 rollout of ``ẋ = f(x, u(x))`` under a state feedback law.

        Integrates with fixed-step RK4 and a zero-order hold on ``u_of_x`` (clipped
        to ``±u_clip`` as a numerical guard). Returns ``(t, xs, us, cost)`` — the time
        grid, state trajectory, applied controls, and accumulated running cost —
        truncated early if the closed loop diverges (non-finite or ``|x| > 1e3``).
        """
        n = int(T / dt)
        xs = np.zeros((n + 1, 2)); us = np.zeros(n + 1); xs[0] = np.asarray(x0, np.float64)
        cost = 0.0
        last = n
        for i in range(n):
            x = xs[i]
            u = float(np.clip(np.ravel(u_of_x(x))[0], -u_clip, u_clip)); us[i] = u
            cost += float(self.running_cost(x, u)) * dt
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + 0.5 * dt * k1, u)
            k3 = self.dynamics(x + 0.5 * dt * k2, u)
            k4 = self.dynamics(x + dt * k3, u)
            xs[i + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(xs[i + 1])) or np.abs(xs[i + 1]).max() > 1e3:
                last = i + 1; break
        us[last] = us[last - 1] if last else 0.0
        t = np.arange(last + 1) * dt
        return t, xs[:last + 1], us[:last + 1], cost

    def true_feedback(self, samples):
        """True PMP feedback κ*(x) = feedback_from_gradient(∇V_true), nearest-neighbour
        interpolated from the dataset's costate samples. Returns a ``u_of_x`` closure."""
        from scipy.spatial import cKDTree
        tree = cKDTree(samples["x"])

        def u(x):
            _, j = tree.query(np.asarray(x).reshape(1, 2))
            return float(self.feedback_from_gradient(samples["dv"][j])[0])
        return u


__all__ = ["PendulumSwingUpProblem"]
