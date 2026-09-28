"""Behavioral checks for the finite-horizon numerical stabilization criterion."""

import numpy as np
import pytest

from src.OpenLoop.pendulum.evaluation import assess_upright, select_stabilizing_fit
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem


def test_selection_prioritizes_stabilization_then_global_h1():
    candidates = [dict(record_path="best_region", global_h1=.3, near_h1=.01),
                  dict(record_path="unstable", global_h1=.1, near_h1=.05),
                  dict(record_path="best_stable", global_h1=.2, near_h1=.5)]
    tested = []
    def stabilizes(row):
        tested.append(row["record_path"])
        return row["record_path"] != "unstable"
    chosen, passed = select_stabilizing_fit(candidates, stabilizes)
    assert passed and chosen["record_path"] == "best_stable"
    assert tested == ["unstable", "best_stable"]


def test_selection_reports_failure_when_no_candidate_stabilizes():
    candidates = [dict(record_path="b", global_h1=.4), dict(record_path="a", global_h1=.2)]
    tested = []
    def fails(row):
        tested.append(row["record_path"])
        return False
    chosen, passed = select_stabilizing_fit(candidates, fails)
    assert not passed and chosen["record_path"] == "a"
    assert tested == ["a", "b"]
    with pytest.raises(ValueError, match="finite global"):
        select_stabilizing_fit([], fails)


def test_dwell_window_rejects_a_final_instant_pass_and_a_late_excursion():
    time = np.arange(0., 10.01, .1)
    states = np.zeros((len(time), 2))
    states[:-1, 0] = .3  # The old terminal-only check would pass.
    assert not assess_upright(time, states, horizon=10.)["sustained_upright"]
    states[:, 0] = 2 * np.pi + .02
    states[time < 8., 0] = 1.
    result = assess_upright(time, states, horizon=10.)
    assert result["sustained_upright"]
    assert result["settling_time"] == 8.
    states[90, 1] = .6
    assert not assess_upright(time, states, horizon=10.)["sustained_upright"]


@pytest.mark.parametrize("state", [[.1, 0.], [0., .5], [.1517, 0.]])
def test_tolerances_reject_boundary_and_nonzero_offset(state):
    time = np.arange(11.)
    assert not assess_upright(time, np.tile(state, (11, 1)), horizon=10.)["sustained_upright"]


def test_incomplete_nonfinite_and_short_window_do_not_pass():
    time = np.arange(11.)
    states = np.zeros((11, 2))
    assert not assess_upright(time[:-1], states[:-1], horizon=10.)["sustained_upright"]
    assert not assess_upright(time[9:], states[9:], horizon=10.)["sustained_upright"]
    states[8, 0] = np.nan
    assert not assess_upright(time, states, horizon=10.)["sustained_upright"]
    with pytest.raises(ValueError, match="dwell_time"):
        assess_upright(time, states, horizon=1.)


def test_horizon_prefix_is_not_contaminated_by_later_failure():
    time = np.arange(41.)
    states = np.zeros((41, 2))
    states[19, 0] = .2
    states[40, 0] = np.nan
    assert assess_upright(time, states, horizon=10.)["sustained_upright"]
    assert not assess_upright(time, states, horizon=20.)["sustained_upright"]
    assert not assess_upright(time, states, horizon=40.)["sustained_upright"]


def test_rollout_distinguishes_local_stabilization_from_downward_equilibrium():
    problem = PendulumSwingUpProblem()
    def lqr(x):
        return problem.feedback_from_gradient(problem.local_lqr_gradient(x))
    time, states, *_ = problem.rk4_rollout(lqr, [.1, .05], T=10., dt=.01)
    assert assess_upright(time, states, horizon=10.)["sustained_upright"]
    time, states, *_ = problem.rk4_rollout(lambda x: 0., [np.pi, 0.], T=10., dt=.01)
    assert not assess_upright(time, states, horizon=10.)["sustained_upright"]
