"""Check actual trajectory costs and finite-horizon PMP references."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.OpenLoop.comparison import accumulated_cost, solve_openloop_reference
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem


@pytest.mark.parametrize("horizon", [3.0, 1.5])
def test_vdp_rollout_defaults_to_the_problem_horizon(horizon):
    problem = VdpOptimalControlProblem(T_final=horizon)
    time, *_ = problem.rk4_rollout(lambda x: 0., [0., 0.])
    assert time[-1] == horizon


@pytest.mark.parametrize("problem", [VdpOptimalControlProblem(), PendulumSwingUpProblem()])
def test_accumulated_cost_uses_own_states_controls_and_elapsed_intervals(problem):
    time = np.array([0., .01, .03])
    states = np.array([[.2, -.3], [.4, .5], [100., 100.]])
    controls = np.array([-2., 3., 100.])
    expected = [0.]
    for dt, state, u in zip(np.diff(time), states, controls):
        def rhs(t, z):
            if isinstance(problem, PendulumSwingUpProblem):
                dynamics = problem.dynamics(z[:2], u)
                rate = 2 - 2 * np.cos(z[0]) + z[1] ** 2 + u ** 2
            else:
                dynamics = problem.dynamics(t, z[:2], u)
                rate = z[0] ** 2 + z[1] ** 2 + .1 * u ** 2
            return np.r_[dynamics, rate]
        solution = solve_ivp(rhs, (0., dt), np.r_[state, 0.], rtol=1e-11, atol=1e-13)
        expected.append(expected[-1] + solution.y[2, -1])
    np.testing.assert_allclose(accumulated_cost(problem, time, states, controls), expected,
                               rtol=1e-7)


@pytest.mark.parametrize("problem", [VdpOptimalControlProblem(), PendulumSwingUpProblem()])
def test_reference_solves_nonzero_start_with_free_terminal_state(problem):
    guess = problem.rk4_rollout(lambda x: -20 * x[0] - 7 * x[1], [.1, .05], T=1., dt=.01)
    reference = solve_openloop_reference(problem, [guess])
    t, states, controls, cost = reference
    np.testing.assert_allclose(states[0], [.1, .05], atol=1e-10)
    assert abs(controls[-1]) < 1e-10  # p(T)=0, not a prescribed terminal state
    assert 0 < cost[-1] < accumulated_cost(problem, *guess[:3])[-1]
    assert np.all(np.diff(cost) >= 0)


def test_reference_rejects_partial_rollouts_and_unconverged_solves():
    problem = PendulumSwingUpProblem()
    guess = problem.rk4_rollout(lambda x: -20 * x[0] - 7 * x[1], [.1, .05], T=1., dt=.01)
    t, states, controls, cost = guess
    with pytest.raises(ValueError, match="full time grid"):
        solve_openloop_reference(problem, [guess, (t[:-1], states[:-1], controls[:-1], cost)])
    with pytest.raises(RuntimeError, match="no converged"):
        solve_openloop_reference(problem, [guess], tol=1e-12, max_nodes=len(t))
