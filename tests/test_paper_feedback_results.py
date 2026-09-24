"""Reject stale manuscript feedback results when local paper traces are available."""

import re
from pathlib import Path

import numpy as np
import pytest

from src.OpenLoop.comparison import accumulated_cost
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key,label", [
    ("gaussian", "Gaussian"),
    ("softplus", "softplus"),
    ("tanh", r"$\tanh$"),
    ("relu^2", r"$\mathrm{ReLU}^{2} + \psi_{2}$"),
    ("relu^3", r"$\mathrm{ReLU}^{3} + \psi_{3}$"),
])
def test_pendulum_feedback_table_matches_its_saved_trajectories(key, label):
    path = ROOT / "experiments/02_pendulum/paper_log_penalty/openloop_a_rollouts.npz"
    if not path.exists():
        pytest.skip("paper trajectories are local artifacts")
    with np.load(path) as traces:
        t, x, u = (traces[f"{key}_{field}"] for field in ("time", "states", "controls"))
    problem = PendulumSwingUpProblem()
    cost = accumulated_cost(problem, t, x, u)[-1]
    upright = abs(problem.wrap_angle(x[-1, 0])) < .4 and abs(x[-1, 1]) < .4
    tex = (ROOT / "paper/paper_0805.tex").read_text()
    match = re.search(r"^" + re.escape(label) + r" & \$([0-9.]+)\$ & (yes|no)", tex, re.M)
    assert match, f"missing feedback-table row for {label}"
    assert match.groups() == (f"{cost:.1f}", "yes" if upright else "no")


def test_pendulum_baseline_table_matches_its_saved_trajectory():
    path = ROOT / "experiments/02_pendulum/paper_log_penalty/baseline_rollouts.npz"
    if not path.exists():
        pytest.skip("paper trajectories are local artifacts")
    with np.load(path) as traces:
        t, x, u = (traces[f"A_{field}"] for field in ("time", "states", "controls"))
    problem = PendulumSwingUpProblem()
    cost = accumulated_cost(problem, t, x, u)[-1]
    upright = abs(problem.wrap_angle(x[-1, 0])) < .4 and abs(x[-1, 1]) < .4
    tex = (ROOT / "paper/paper_0805.tex").read_text()
    match = re.search(r"^" + re.escape(r"ReLU $+\ell^1$") + r" & \$([0-9.]+)\$ & (yes|no)", tex, re.M)
    assert match, "missing baseline feedback-table row"
    assert match.groups() == (f"{cost:.1f}", "yes" if upright else "no")


@pytest.mark.parametrize("folder,stem,horizon", [
    ("01_vdp", "openloop", VdpOptimalControlProblem().T_final),
    ("02_pendulum", "openloop_a", 10.),
    ("02_pendulum", "openloop_b", 10.),
])
def test_control_comparison_traces_use_the_documented_horizon(folder, stem, horizon):
    path = ROOT / "experiments" / folder / "paper_log_penalty" / f"{stem}_rollouts.npz"
    if not path.exists():
        pytest.skip("paper trajectories are local artifacts")
    with np.load(path) as traces:
        times = [traces[key] for key in traces.files if key.endswith("_time")]
        assert len(times) == 6
        for t in times:
            assert t[0] == 0 and t[-1] == horizon
            assert np.all(np.diff(t) > 0)
