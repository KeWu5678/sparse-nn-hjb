import subprocess
import sys

import numpy as np


def test_grid_initial_states_span_both_ranges() -> None:
    from src.OpenLoop.sampling import grid_initial_states

    grid = grid_initial_states(2, 3, (-1.0, 1.0), (-2.0, 2.0))

    assert grid.shape == (6, 2)
    assert np.allclose(grid[0], [-1.0, -2.0])
    assert np.allclose(grid[-1], [1.0, 2.0])


def test_pendulum_solver_public_imports_are_available() -> None:
    from src.OpenLoop.pendulum import (
        PendulumPmpSolver,
        PendulumPmpSolverConfig,
        PendulumSwingUpProblem,
    )

    problem = PendulumSwingUpProblem(control_limit=2.0)
    config = PendulumPmpSolverConfig(value_max=0.5, t_final=1.0)
    solver = PendulumPmpSolver(problem=problem, config=config)

    assert problem.control_limit == 2.0
    assert solver.config.value_max == 0.5


def test_value_samples_save_npz_uses_explicit_output_path(tmp_path) -> None:
    from src.data import ValueSamples

    samples = ValueSamples(
        x=np.zeros((1, 2)),
        v=np.zeros(1),
        dv=np.zeros((1, 2)),
    )

    path = samples.save_npz(tmp_path / "value_samples.npz")

    assert path == tmp_path / "value_samples.npz"
    assert path.exists()
    with np.load(path) as data:
        assert set(data.files) == {"x", "v", "dv"}


def test_pendulum_pmp_dataset_script_has_help() -> None:
    scripts = ["scripts/run_pendulum_pmp_openloop_example.py"]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "--output-dir" in result.stdout


def test_vdp_solver_public_imports_are_available() -> None:
    from src.OpenLoop.collocation import CollocationConfig, CollocationSolver
    from src.OpenLoop.vdp import VdpOptimalControlProblem

    problem = VdpOptimalControlProblem(T_final=0.1)
    solver = CollocationSolver(problem, CollocationConfig(initial_mesh_points=11))

    assert problem.dimension == 2
    assert solver.solve_sample(np.zeros(2)).value == 0.0
