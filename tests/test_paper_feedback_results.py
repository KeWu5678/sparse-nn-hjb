"""Reject stale manuscript feedback results when local paper traces are available."""

import importlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.OpenLoop.comparison import accumulated_cost
from src.OpenLoop.pendulum.evaluation import assess_upright
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem

ROOT = Path(__file__).resolve().parents[1]


def test_vdp_feedback_figures_simulate_only_the_benchmark_horizon(tmp_path, monkeypatch):
    if not (ROOT / "scripts/paper/vdp_summary_figures.py").exists():
        pytest.skip("paper renderer is a local artifact")
    monkeypatch.syspath_prepend(str(ROOT))
    summary = importlib.import_module("scripts.paper.vdp_summary_figures")
    horizons = []
    rollout = summary.PROBLEM.rk4_rollout

    def capture(*args, **kwargs):
        result = rollout(*args, **kwargs)
        horizons.append(result[0][-1])
        return result

    monkeypatch.setattr(summary.PROBLEM.__class__, "true_feedback", lambda *a: lambda x: 0.)
    monkeypatch.setattr(summary.PROBLEM.__class__, "rk4_rollout", lambda self, *a, **kw: capture(*a, **kw))
    monkeypatch.setattr(summary, "_model_feedback", lambda *a: lambda x: 0.)
    monkeypatch.setattr(summary, "plot_control_comparison", lambda *a, **kw: {})
    models = {key: {"result_path": "unused"} for key in
              ("softplus", "gaussian", "relu3", "tanh", "relu2")}
    (tmp_path / "figures").mkdir()
    summary.plot_feedback(models, None, None, output_dir=tmp_path)
    assert horizons
    assert set(horizons) == {summary.PROBLEM.T_final}


@pytest.mark.parametrize("key,label", [
    ("true PMP", "reference (PMP)"),
    ("relu_l1", r"ReLU $+\ell^1$"),
    ("gaussian", "Gaussian"),
    ("softplus", "softplus"),
    ("tanh", r"$\tanh$"),
    ("relu^2", r"$\mathrm{ReLU}^{2} + \psi_{2}$"),
    ("relu^3", r"$\mathrm{ReLU}^{3} + \psi_{3}$"),
])
def test_pendulum_feedback_table_matches_its_saved_trajectories(key, label):
    path = ROOT / "experiments/02_pendulum/paper_log_penalty/stability_rollouts.npz"
    if not path.exists():
        pytest.skip("paper trajectories are local artifacts")
    problem = PendulumSwingUpProblem()
    results = []
    with np.load(path) as traces:
        for side in ("A", "B"):
            t, x, u = (traces[f"{side}_{key}_{field}"]
                       for field in ("time", "states", "controls"))
            end = np.flatnonzero(np.isclose(t, 10., rtol=0, atol=1e-10))[0] + 1
            cost = accumulated_cost(problem, t[:end], x[:end], u[:end])[-1]
            results.append(f"{cost:.1f}")
            results.extend("yes" if assess_upright(t, x, horizon=h)["sustained_upright"]
                           else "no" for h in (10., 40.))
    selected = json.loads((path.parent / "controller_selection.json").read_text())["selected"]
    neurons = None if key == "true PMP" else str(selected[key]["neurons"])
    tex = (ROOT / "paper/paper_0805.tex").read_text()
    match = re.search(r"^" + re.escape(label) + r" & (?:\$([0-9]+)\$|---)"
                      + (r" & \$([0-9.]+)\$" + r" & (yes|no)" * 2) * 2, tex, re.M)
    assert match, f"missing multi-horizon feedback-table row for {label}"
    assert match.groups() == (neurons, *results)


def test_pendulum_horizon_report_matches_its_traces():
    directory = ROOT / "experiments/02_pendulum/paper_log_penalty"
    if not (directory / "stability.json").exists():
        pytest.skip("horizon study is a local artifact")
    report = json.loads((directory / "stability.json").read_text())
    assert len(report["results"]) == 42  # Seven laws, two starts, three horizons.
    problem = PendulumSwingUpProblem()
    with np.load(directory / "stability_rollouts.npz") as traces:
        for row in report["results"]:
            prefix = f"{row['start']}_{row['controller']}"
            t, x, u = (traces[f"{prefix}_{field}"] for field in ("time", "states", "controls"))
            metrics = assess_upright(t, x, horizon=row["horizon"])
            assert all(row[key] == value for key, value in metrics.items())
            if row["complete"]:
                end = np.flatnonzero(np.isclose(t, row["horizon"], rtol=0, atol=1e-10))[0] + 1
                cost = accumulated_cost(problem, t[:end], x[:end], u[:end])[-1]
                assert row["cost"] == pytest.approx(cost)
            else:
                assert not row["sustained_upright"] and row["cost"] is None


def test_pendulum_selection_prioritizes_stabilization_then_global_error(monkeypatch):
    directory = ROOT / "experiments/02_pendulum/paper_log_penalty"
    if not (directory / "controller_selection.json").exists():
        pytest.skip("controller selection is a local artifact")
    monkeypatch.syspath_prepend(str(ROOT))
    selector = importlib.import_module("scripts.paper.pendulum_selection")
    selection = selector.load_selection()  # Also verifies all candidate hashes.
    candidates = selector.candidates()
    report = json.loads((directory / "stability.json").read_text())
    for family, chosen in selection["selected"].items():
        ordered = sorted((r for r in candidates if r["activation"] == family),
                         key=lambda r: (r["global_h1"], r["record_path"]))
        tested = [r for r in selection["tested"] if r["activation"] == family]
        assert [r["record_path"] for r in tested] == [
            r["record_path"] for r in ordered[:len(tested)]]
        assert all(not r["stabilized"] for r in tested[:-1])
        if chosen["stabilized"]:
            assert tested[-1]["stabilized"]
            assert chosen["record_path"] == tested[-1]["record_path"]
        else:
            assert len(tested) == len(ordered) and not tested[-1]["stabilized"]
            assert chosen["record_path"] == ordered[0]["record_path"]
        for row in tested:
            outcomes = row["outcomes"]
            assert row["stabilized"] == (
                set(outcomes) == {"A", "B"}
                and all(r["sustained_upright"] for r in outcomes.values()))
        assert ROOT / report["records"][family]["path"] == Path(chosen["record_path"])
        assert report["records"][family]["sha256"] == chosen["record_sha256"]
        results = [r for r in report["results"]
                   if r["controller"] == family and r["horizon"] == selection["horizon"]]
        assert len(results) == 2
        assert all(r["sustained_upright"] for r in results) == chosen["stabilized"]


@pytest.mark.parametrize("folder,stem,horizon,count", [
    ("01_vdp", "openloop", VdpOptimalControlProblem().T_final, 6),
    ("02_pendulum", "openloop_a", 40., 6),
    ("02_pendulum", "openloop_b", 40., 6),
])
def test_control_comparison_traces_use_the_documented_horizon(folder, stem, horizon, count):
    path = ROOT / "experiments" / folder / "paper_log_penalty" / f"{stem}_rollouts.npz"
    if not path.exists():
        pytest.skip("paper trajectories are local artifacts")
    with np.load(path) as traces:
        times = {key: traces[key] for key in traces.files if key.endswith("_time")}
        assert len(times) == count
        for t in times.values():
            assert t[0] == 0 and np.all(np.diff(t) > 0)
            assert t[-1] == horizon
