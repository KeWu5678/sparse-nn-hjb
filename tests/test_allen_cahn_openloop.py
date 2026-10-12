"""Independent discretization, Pontryagin, and value-gradient checks."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from src.OpenLoop.allen_cahn import (
    AllenCahnBatchError,
    AllenCahnConvergenceError,
    AllenCahnOpenLoopSolver,
    AllenCahnProblem,
    AllenCahnSampleFailure,
    AllenCahnSampleResult,
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
    assert result.value == 0
    np.testing.assert_array_equal(result.gradient, np.zeros(dimension))
    trajectory = result.trajectory
    diagnostics = result.diagnostics
    np.testing.assert_array_equal(trajectory.adjoint_trajectory[:, -1], np.zeros(dimension))
    assert diagnostics.collocation_residual == 0
    assert diagnostics.boundary_residual == 0
    assert diagnostics.reduced_gradient_norm == 0
    assert trajectory.state_trajectory.shape == (dimension, diagnostics.nodes)
    assert trajectory.control.shape == (3, diagnostics.nodes)


@pytest.mark.parametrize("dimension", [6, 10])
def test_value_gradient_matches_independent_initial_state_differences(dimension):
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=dimension),
        AllenCahnSolverConfig(collocation_tol=1e-8),
    )
    initial = np.random.default_rng(41).uniform(-.7, .7, dimension)
    reference = solver.solve_sample(initial)
    epsilon = 2e-5
    differences = []
    for direction in np.eye(dimension):
        plus = solver.solve_sample(initial + epsilon * direction)
        minus = solver.solve_sample(initial - epsilon * direction)
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
    trajectory = result.trajectory
    control = CubicSpline(trajectory.time_grid, trajectory.control, axis=1)

    def rhs(t, state_cost):
        state = state_cost[:dimension]
        u = control(t)
        return np.r_[problem.dynamics(t, state, u), problem.running_cost(state, u)]

    independent = solve_ivp(
        rhs, (0., problem.T_final), np.r_[initial, 0.],
        t_eval=trajectory.time_grid, rtol=1e-10, atol=1e-12,
    )
    assert independent.success
    np.testing.assert_allclose(independent.y[:dimension], trajectory.state_trajectory, atol=3e-7)
    assert independent.y[-1, -1] == pytest.approx(result.value, rel=1e-7, abs=1e-9)
    assert result.diagnostics.boundary_residual <= 1e-8
    assert result.diagnostics.collocation_residual <= 1e-8
    assert result.diagnostics.reduced_gradient_norm < 1e-14


def test_failed_collocation_raises_with_initial_state_and_diagnostics():
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=6),
        AllenCahnSolverConfig(initial_mesh_points=2, collocation_max_nodes=2, collocation_tol=1e-10),
    )
    initial = np.linspace(-.7, .9, 6)
    with pytest.raises(AllenCahnConvergenceError) as caught:
        solver.solve_sample(initial)
    failure = caught.value.failure
    np.testing.assert_array_equal(failure.initial_state, initial)
    assert failure.diagnostics.attempts == 2
    assert failure.diagnostics.nodes == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_batch_attempts_every_state_and_reports_ordered_outcomes(workers):
    solver = AllenCahnOpenLoopSolver(
        AllenCahnProblem(dimension=6),
        AllenCahnSolverConfig(initial_mesh_points=2, collocation_max_nodes=2, collocation_tol=1e-10),
    )
    states = np.vstack((np.zeros(6), np.linspace(-.7, .9, 6), np.linspace(.8, -.6, 6)))
    progress = []
    with pytest.raises(AllenCahnBatchError) as caught:
        solver.solve(states, lambda current, total: progress.append((current, total)), workers=workers)
    batch = caught.value
    assert len(batch.sample_results) == len(states)
    assert isinstance(batch.sample_results[0], AllenCahnSampleResult)
    assert all(isinstance(result, AllenCahnSampleFailure) for result in batch.sample_results[1:])
    np.testing.assert_array_equal(batch.failed_initial_states, states[1:])
    np.testing.assert_array_equal([result.initial_state for result in batch.sample_results], states)
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_retained_samples_and_progress_preserve_dimension():
    solver = AllenCahnOpenLoopSolver(AllenCahnProblem(dimension=10))
    progress = []
    states = np.vstack((np.zeros(10), np.linspace(-.1, .1, 10)))
    solution = solver.solve(states, lambda current, total: progress.append((current, total)))
    np.testing.assert_array_equal(solution.value_samples.x, states)
    assert solution.value_samples.dv.shape == (2, 10)
    assert progress == [(0, 2), (1, 2), (2, 2)]


@pytest.mark.parametrize("dimension", [6, 10])
def test_parallel_batch_matches_sequential_values_gradients_and_order(dimension):
    solver = AllenCahnOpenLoopSolver(AllenCahnProblem(dimension=dimension))
    states = np.vstack((np.zeros(dimension), np.linspace(-.3, .2, dimension)))
    sequential = solver.solve(states)
    parallel = solver.solve(states, workers=2)
    for name in ("x", "v", "dv"):
        np.testing.assert_array_equal(getattr(parallel.value_samples, name), getattr(sequential.value_samples, name))


@pytest.mark.parametrize("dimension", [6, 10])
def test_pontryagin_operator_matches_stationary_dynamics_and_analytic_derivative(dimension):
    problem = AllenCahnProblem(dimension=dimension)
    z = np.random.default_rng(72).uniform(-.8, .8, (2 * dimension, 3))
    times = np.linspace(0., problem.T_final, z.shape[1])
    rhs = problem.pontryagin_rhs(times, z)
    control = problem.feedback_from_gradient(z[dimension:])
    np.testing.assert_allclose(rhs[:dimension], problem.dynamics(times, z[:dimension], control), atol=1e-14)
    np.testing.assert_array_equal(rhs[dimension:], problem.adjoint_rhs(times, z[:dimension], z[dimension:]))
    jacobian = problem.pontryagin_jacobian(times, z)
    epsilon = 1e-6
    for column in range(2 * dimension):
        perturbation = np.zeros_like(z)
        perturbation[column] = epsilon
        difference = (problem.pontryagin_rhs(times, z + perturbation) - problem.pontryagin_rhs(times, z - perturbation)) / (2 * epsilon)
        np.testing.assert_allclose(jacobian[:, column], difference, rtol=2e-7, atol=1e-8)


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
