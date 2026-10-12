"""Independent discretization, Pontryagin, and value-gradient checks."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from src.OpenLoop.allen_cahn import (
    AllenCahnOpenLoopSolver,
    AllenCahnProblem,
    AllenCahnSolverConfig,
)


@pytest.mark.parametrize("dimension", [6, 10])
def test_neumann_discretization_preserves_mass_and_actuator_lengths(dimension):
    problem = AllenCahnProblem(dimension=dimension)
    h = problem.cell_width
    np.testing.assert_allclose(problem.laplacian @ np.ones(dimension), 0.0, atol=1e-14)
    np.testing.assert_allclose(problem.laplacian, problem.laplacian.T)
    assert np.max(np.linalg.eigvalsh(problem.laplacian)) < 1e-12
    np.testing.assert_allclose(h * problem.control_matrix.sum(axis=0), [.3, .4, .3])
    assert np.all((problem.control_matrix >= 0) & (problem.control_matrix <= 1))
    np.testing.assert_allclose(problem.control_matrix[::-1, ::-1], problem.control_matrix, atol=1e-14)
    y = np.linspace(-.4, .7, dimension)
    u = np.array([.2, -.1, .4])
    # Diffusion contributes zero to the integral state balance.
    assert h * problem.dynamics(0, y, u).sum() == pytest.approx(
        h * np.sum(y - y**3) + np.dot([.3, .4, .3], u), abs=1e-14,
    )


@pytest.mark.parametrize("dimension", [6, 10])
def test_zero_initial_state_has_zero_value_and_free_terminal_adjoint(dimension):
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=dimension),
        AllenCahnSolverConfig(store_trajectories=True),
    )
    result = solver.solve_sample(np.zeros(dimension))
    assert result.converged
    assert result.value == 0
    np.testing.assert_array_equal(result.gradient, np.zeros(dimension))
    np.testing.assert_array_equal(result.adjoint_trajectory[:, -1], np.zeros(dimension))
    assert result.collocation_residual == 0
    assert result.boundary_residual == 0
    assert result.reduced_gradient_norm == 0
    assert result.state_trajectory.shape == (dimension, result.nodes)
    assert result.control.shape == (3, result.nodes)


@pytest.mark.parametrize("dimension", [6, 10])
def test_value_gradient_matches_independent_initial_state_differences(dimension):
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=dimension),
        AllenCahnSolverConfig(collocation_tol=1e-8),
    )
    initial = np.random.default_rng(41).uniform(-.7, .7, dimension)
    reference = solver.solve_sample(initial)
    assert reference.converged
    epsilon = 2e-5
    differences = []
    for direction in np.eye(dimension):
        plus = solver.solve_sample(initial + epsilon * direction)
        minus = solver.solve_sample(initial - epsilon * direction)
        assert plus.converged and minus.converged
        differences.append((plus.value - minus.value) / (2 * epsilon))
    np.testing.assert_allclose(reference.gradient, differences, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("dimension", [6, 10])
def test_collocation_state_and_cost_match_independent_ivp_rollout(dimension):
    problem = AllenCahnProblem(dimension=dimension)
    solver = AllenCahnOpenLoopSolver(problem, AllenCahnSolverConfig(
        collocation_tol=1e-8, store_trajectories=True,
    ))
    initial = np.linspace(-.6, .8, dimension)
    result = solver.solve_sample(initial)
    assert result.converged
    control = CubicSpline(result.time_grid, result.control, axis=1)

    def rhs(t, state_cost):
        state = state_cost[:dimension]
        u = control(t)
        return np.r_[problem.dynamics(t, state, u), problem.running_cost(state, u)]

    independent = solve_ivp(
        rhs, (0., problem.T_final), np.r_[initial, 0.],
        t_eval=result.time_grid, rtol=1e-10, atol=1e-12,
    )
    assert independent.success
    np.testing.assert_allclose(independent.y[:dimension], result.state_trajectory, atol=3e-7)
    assert independent.y[-1, -1] == pytest.approx(result.value, rel=1e-7, abs=1e-9)
    assert result.boundary_residual <= solver.config.collocation_tol
    assert result.collocation_residual <= solver.config.collocation_tol
    assert result.reduced_gradient_norm < 1e-14


def test_failed_collocation_does_not_enter_dataset():
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=6),
        AllenCahnSolverConfig(initial_mesh_points=2, collocation_max_nodes=2, collocation_tol=1e-10),
    )
    states = np.array([np.linspace(-.7, .9, 6)])
    solution = solver.solve(states)
    assert not solution.sample_results[0].converged
    assert solution.value_samples.x.shape == (0, 6)
    assert solution.value_samples.dv.shape == (0, 6)
    np.testing.assert_array_equal(solution.failed_initial_states, states)


def test_retained_samples_and_progress_preserve_dimension():
    solver = AllenCahnOpenLoopSolver(AllenCahnProblem(dimension=10))
    progress = []
    states = np.vstack((np.zeros(10), np.linspace(-.1, .1, 10)))
    solution = solver.solve(states, lambda current, total: progress.append((current, total)))
    np.testing.assert_array_equal(solution.value_samples.x, states)
    assert solution.value_samples.dv.shape == (2, 10)
    assert solution.failed_initial_states.shape == (0, 10)
    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_rollout_matches_uniform_uncontrolled_scalar_solution_and_exact_horizon():
    problem = AllenCahnProblem(dimension=6, T_final=.025)
    times, states, controls, cost = problem.rk4_rollout(
        lambda _state: np.zeros(3), np.full(6, .5), dt=.01,
    )
    expected = 1 / np.sqrt(1 + 3 * np.exp(-2 * times))
    np.testing.assert_allclose(states, np.repeat(expected[:, None], 6, axis=1), rtol=1e-9)
    assert times[-1] == problem.T_final
    np.testing.assert_array_equal(controls, np.zeros_like(controls))
    # The spatially uniform cost is 2*integral(y(t)**2 dt).
    assert cost == pytest.approx(np.log((np.exp(2 * problem.T_final) + 3) / 4), rel=1e-8)


@pytest.mark.parametrize("kwargs", [{"dimension": 1}, {"dimension": 6.0}, {"nu": np.nan}, {"beta": 0.}, {"T_final": np.inf}])
def test_problem_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AllenCahnProblem(**kwargs)


@pytest.mark.parametrize("kwargs", [{"collocation_tol": np.nan}, {"initial_mesh_points": 1}, {"collocation_max_nodes": 4}, {"quadrature_order": 2.5}, {"initial_guess_decays": ()}, {"initial_guess_decays": (np.inf,)}])
def test_solver_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        AllenCahnSolverConfig(**kwargs)


def test_solver_rejects_invalid_initial_states():
    solver = AllenCahnOpenLoopSolver()
    for state in (np.zeros(10), np.full(6, np.nan)):
        with pytest.raises(ValueError):
            solver.solve_sample(state)
