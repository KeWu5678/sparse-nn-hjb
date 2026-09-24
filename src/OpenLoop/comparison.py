"""Finite-horizon references and costs for feedback rollout comparisons."""

import numpy as np
from scipy.integrate import solve_bvp

from .pendulum.problem import PendulumSwingUpProblem


def _running_cost(problem, states, controls):
    # The VDP problem uses columns; the pendulum problem uses rows.
    states = states if isinstance(problem, PendulumSwingUpProblem) else states.T
    return problem.running_cost(states, controls)


def accumulated_cost(problem, time, states, controls):
    """Integrate running cost with the same RK4 stages and held rollout controls.

    The final control is a plotting endpoint, not an additional time interval.
    No predicted value or terminal-value estimate enters this calculation.
    """
    states, controls = np.asarray(states)[:-1], np.asarray(controls)[:-1]
    dt = np.diff(time)

    def dynamics(x):
        if isinstance(problem, PendulumSwingUpProblem):
            return problem.dynamics(x, controls)
        return np.array([problem.dynamics(0., xi, ui) for xi, ui in zip(x, controls)])

    x2 = states + .5 * dt[:, None] * dynamics(states)
    x3 = states + .5 * dt[:, None] * dynamics(x2)
    x4 = states + dt[:, None] * dynamics(x3)
    rates = (_running_cost(problem, states, controls)
             + 2 * _running_cost(problem, x2, controls)
             + 2 * _running_cost(problem, x3, controls)
             + _running_cost(problem, x4, controls)) / 6
    return np.r_[0.0, np.cumsum(dt * rates)]


def solve_openloop_reference(problem, rollouts, *, tol=1e-6, max_nodes=20000):
    """Lowest-cost converged PMP solve initialized from the supplied rollouts.

    Solve x(0)=x0, p(T)=0 for a free terminal state and zero terminal cost.
    Rollouts must share their initial state and time grid. Multiple guesses help
    explore pendulum branches, but do not certify a global optimum. Returns
    (time, states, controls, accumulated_cost), with a cost array rather than
    the scalar in ``problem.rk4_rollout``. Reference costs integrate the dense
    BVP solution using three-point Gauss quadrature on each output interval.
    """
    time, states, _, _ = rollouts[0]
    time = np.asarray(time)
    x0 = np.asarray(states[0])
    pendulum = isinstance(problem, PendulumSwingUpProblem)
    control_scale = (
        2.0 * problem.control_weight / problem.control_gain if pendulum else 2.0 * problem.beta
    )

    def rhs(t, z):
        if pendulum:
            u = problem.feedback_from_gradient(z[2:].T)
            return np.vstack((problem.dynamics(z[:2].T, u).T,
                              problem.costate_rhs(z[:2].T, z[2:].T).T))
        u = -z[3] / control_scale
        return np.vstack((
            np.column_stack([problem.dynamics(ti, x, ui)
                             for ti, x, ui in zip(t, z[:2].T, u)]),
            np.column_stack([problem.adjoint_rhs(ti, x, p)
                             for ti, x, p in zip(t, z[:2].T, z[2:].T)]),
        ))

    best, best_cost = None, np.inf
    failures = []
    nodes, weights = np.polynomial.legendre.leggauss(3)
    dt = np.diff(time)
    quadrature_time = time[:-1, None] + .5 * dt[:, None] * (1 + nodes)
    for t, xs, us, _ in rollouts:
        if not np.array_equal(t, time) or not np.allclose(xs[0], x0, rtol=0, atol=1e-12):
            raise ValueError("reference guesses must share an initial state and full time grid")
        guess = np.vstack((np.asarray(xs).T, np.zeros(len(time)), -control_scale * us))
        solution = solve_bvp(rhs, lambda a, b: np.r_[a[:2] - x0, b[2:]],
                             time, guess, tol=tol, max_nodes=max_nodes)
        if not solution.success:
            failures.append(solution.message)
            continue
        z = solution.sol(time)
        controls = (problem.feedback_from_gradient(z[2:].T) if pendulum
                    else -z[3] / control_scale)
        zq = solution.sol(quadrature_time.ravel())
        uq = (problem.feedback_from_gradient(zq[2:].T) if pendulum
              else -zq[3] / control_scale)
        rates = _running_cost(problem, zq[:2].T, uq).reshape(-1, len(nodes))
        curve = np.r_[0., np.cumsum(.5 * dt * (rates @ weights))]
        cost = curve[-1]
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best = (time.copy(), z[:2].T, controls, curve)
    if best is None:
        raise RuntimeError(f"no converged finite-horizon reference: {failures}")
    return best
