import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import load_value_samples
from src.OpenLoop.collocation import CollocationConfig, CollocationSolver
from src.OpenLoop.vdp import VdpOptimalControlProblem

ROOT = Path(__file__).resolve().parents[1]
# Tight settings for checking labels against independent finite differences.
ACCURATE = CollocationConfig(
    tol=1e-8, max_nodes=20000,
    initial_mesh_points=151, quadrature_order=3,
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


def test_vdp_pontryagin_operator_matches_stationary_dynamics() -> None:
    problem = VdpOptimalControlProblem()
    z = np.random.default_rng(72).uniform(-.8, .8, (4, 3))
    times = np.linspace(0., problem.T_final, 3)
    rhs = problem.pontryagin_rhs(times, z)
    control = problem.feedback_from_gradient(z[2:])
    np.testing.assert_array_equal(control, -z[3] / (2 * problem.beta))
    np.testing.assert_array_equal(rhs[:2], problem.dynamics(times, z[:2], control))
    np.testing.assert_array_equal(rhs[2:], problem.adjoint_rhs(times, z[:2], z[2:]))
    assert problem.reduced_gradient(z[2:], control) == pytest.approx(np.zeros(3), abs=1e-15)


def test_zero_initial_state_has_zero_value_and_free_terminal_adjoint() -> None:
    solver = CollocationSolver(
        VdpOptimalControlProblem(T_final=0.1),
        CollocationConfig(initial_mesh_points=3, store_trajectories=True),
    )

    result = solver.solve_sample(np.array([0.0, 0.0]))

    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, [0.0, 0.0])
    assert result.diagnostics.reduced_gradient_norm == 0.0
    trajectory = result.trajectory
    np.testing.assert_array_equal(trajectory.adjoint_trajectory[:, -1], [0.0, 0.0])
    assert trajectory.state_trajectory.shape == (2, result.diagnostics.nodes)
    assert trajectory.control.shape == (result.diagnostics.nodes,)


def test_value_gradient_matches_independent_initial_state_differences() -> None:
    solver = CollocationSolver(VdpOptimalControlProblem(), ACCURATE)
    initial = np.array([3.0, 3.0])
    result = solver.solve_sample(initial)
    assert result.diagnostics.collocation_residual <= ACCURATE.tol
    assert result.value == pytest.approx(18.516597, abs=1e-6)
    epsilon = 1e-4
    differences = []
    for direction in np.eye(2):
        plus = solver.solve_sample(initial + epsilon * direction)
        minus = solver.solve_sample(initial - epsilon * direction)
        differences.append((plus.value - minus.value) / (2 * epsilon))
    np.testing.assert_allclose(result.gradient, differences, rtol=1e-6, atol=1e-7)


def test_generator_publishes_one_complete_dated_dataset(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_vdp_data.py"),
         "--grid-size", "2", "--tag", "20260605", "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.startswith("VDP_20260605_")
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "VDP_pmp_grid_2x2_20260605.npz", "VDP_pmp_meta_grid_2x2_20260605.json",
    ]
    loaded = load_value_samples(run_dir / "VDP_pmp_grid_2x2_20260605.npz")
    assert loaded.x.shape == (4, 2)
    assert loaded.v.ndim == 1
    assert loaded.dv.shape == loaded.x.shape
    metadata = json.loads((run_dir / "VDP_pmp_meta_grid_2x2_20260605.json").read_text())
    assert metadata["requested_samples"] == metadata["retained_samples"] == 4
    assert metadata["solver"]["quadrature_order"] == 3
