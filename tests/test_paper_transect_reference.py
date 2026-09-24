"""The local paper transect compares solved PMP branches, independent of axis sign."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.pendulum.trajectories import integrate_pmp_trajectory

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scripts/paper/pendulum_figures.py").exists():
    pytest.skip("local paper renderer is unavailable", allow_module_level=True)
sys.path.insert(0, str(ROOT))
figures = importlib.import_module("scripts.paper.pendulum_figures")


@pytest.fixture(scope="module")
def raw_seeds():
    problem = PendulumSwingUpProblem()
    return tuple(integrate_pmp_trajectory(
        problem, angle, epsilon=2e-4, value_max=100., t_final=50.,
        max_step=.005, rtol=1e-10, atol=1e-12, trajectory_id=i,
    ) for i, angle in enumerate((6.193447431088276, 3.0938475015668527)))


@pytest.mark.parametrize("direction", [1., -1.])
def test_transect_uses_lower_solved_cost_without_spurious_value_jumps(raw_seeds, direction):
    reference = figures._pmp_transect_reference(
        raw_seeds, np.array([.49763437, .53214891]),
        direction * np.array([-.95314081, -.302527]), np.linspace(-.45, .45, 101),
    )
    expected = np.array([25.63859956, 26.63919826, 2.34693281])
    if direction < 0:
        expected = expected[::-1]
    np.testing.assert_allclose(reference["value"][[0, 50, 100]], expected, atol=2e-5)
    assert np.max(np.abs(np.diff(reference["value"]))) < .8
    assert np.count_nonzero(np.diff(reference["selected_target_index"])) == 1
    assert np.max(reference["collocation_residuals"]) <= reference["tolerance"]
    assert np.max(reference["boundary_residuals"]) < 1e-10
    assert np.max(reference["terminal_costs"]) < 1e-12


def test_transect_rejects_failed_pmp_solve(raw_seeds, monkeypatch):
    from types import SimpleNamespace

    import scipy.integrate

    monkeypatch.setattr(scipy.integrate, "solve_bvp", lambda *a, **kw:
                        SimpleNamespace(success=False, message="forced failure"))
    with pytest.raises(RuntimeError, match="forced failure"):
        figures._pmp_transect_reference(
            raw_seeds, np.array([.49763437, .53214891]),
            np.array([-.95314081, -.302527]), np.linspace(-.45, .45, 3),
        )
