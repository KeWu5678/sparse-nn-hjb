import numpy as np
import pytest

from src.OpenLoop.value_samples import ValueSamples
from src.OpenLoop.vdp import (
    VdpOpenLoopSolver,
    VdpOpenLoopSolverConfig,
    VdpOptimalControlProblem,
    grid_initial_states,
)


def test_vdp_problem_matches_reduced_gradient_equations() -> None:
    problem = VdpOptimalControlProblem(beta=0.1, mu=1.0)
    y = np.array([0.5, -0.25])
    p = np.array([0.2, -0.4])

    dynamics = problem.dynamics(0.0, y, u=0.3)
    adjoint = problem.adjoint_rhs(0.0, y, p)
    residual = problem.reduced_gradient(p.reshape(2, 1), np.array([0.3]))

    assert np.allclose(dynamics, [-0.25, -0.3875])
    assert np.allclose(adjoint, [-1.3, 0.6])
    assert np.allclose(residual, [-0.34])


def test_vdp_running_cost_counts_the_full_state_cost() -> None:
    problem = VdpOptimalControlProblem(beta=0.1)
    # x=(1,0), u=1 is an exact stationary trajectory with running cost 1+0.1.
    # The historical half-state-cost bug would instead report 0.6.
    assert problem.running_cost(np.array([[1.0], [0.0]]), np.array([1.0])) == pytest.approx([1.1])


def test_pmp_solves_zero_initial_state() -> None:
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(time_step=0.05, store_trajectories=True)
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    result = solver.solve_sample(np.array([0.0, 0.0]))

    assert result.converged
    assert result.value == 0.0
    assert np.allclose(result.gradient, [0.0, 0.0])
    assert result.reduced_gradient_norm == 0.0
    assert result.state_trajectory.shape == (2, 3)
    assert result.control.shape == (3,)


def test_pmp_returns_pdap_value_samples_and_save_artifacts(tmp_path) -> None:
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(time_step=0.05)
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    solution = solver.solve(np.array([[0.0, 0.0], [0.1, 0.0]]))
    paths = solution.save_dataset(tmp_path, grid_shape=(1, 2), date_tag="20260605")

    assert solution.value_samples.size >= 1
    assert solution.failed_initial_states.shape[1] == 2
    assert paths["data"].name == "VDP_pmp_grid_1x2_20260605.npz"
    assert paths["run_dir"].name.startswith("VDP_20260605_")
    assert paths["data"].parent == paths["run_dir"]
    assert paths["meta"].parent == paths["run_dir"]
    assert paths["failed"].parent == paths["run_dir"]
    loaded = ValueSamples.load_npz(paths["data"])
    assert loaded.x.shape[1] == 2
    assert loaded.v.ndim == 1
    assert loaded.dv.shape == loaded.x.shape
    assert paths["meta"].exists()
    assert paths["failed"].exists()


def test_pmp_value_gradient_matches_independent_initial_state_differences() -> None:
    solver = VdpOpenLoopSolver(
        VdpOptimalControlProblem(),
        VdpOpenLoopSolverConfig(num_time_points=151),
    )
    initial = np.array([3.0, 3.0])
    result = solver.solve_sample(initial)
    assert result.converged
    assert result.collocation_residual <= solver.config.collocation_tol
    assert result.value == pytest.approx(18.516597, abs=1e-6)
    epsilon = 1e-4
    differences = []
    for direction in np.eye(2):
        plus = solver.solve_sample(initial + epsilon * direction)
        minus = solver.solve_sample(initial - epsilon * direction)
        assert plus.converged and minus.converged
        differences.append((plus.value - minus.value) / (2 * epsilon))
    np.testing.assert_allclose(result.gradient, differences, rtol=1e-6, atol=1e-7)


def test_vdp_grid_sampling_feeds_solver() -> None:
    initial_states = grid_initial_states(1, 2, (0.0, 0.0), (0.0, 0.1))
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(num_time_points=11)
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    solution = solver.solve(initial_states)

    assert len(solution.sample_results) == 2
    assert solution.value_samples.x.shape[1] == 2
