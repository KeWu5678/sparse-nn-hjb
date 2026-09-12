"""Smoke the retained experiment interfaces without running fits or writing reports."""

import importlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", [
    "scripts/paper/vdp_full_scope.py",
    "scripts/paper/pendulum_full_scope.py",
    "experiments/01_vdp/paper_log_penalty/p_study_figure.py",
])
def test_batch_paper_exports_select_a_consistent_backend(relative):
    path = ROOT / relative
    if not path.exists():
        pytest.skip("paper-workspace entry points are local/ignored under ADR 0011")
    result = subprocess.run(
        [sys.executable, "-c",
         "import runpy, sys, matplotlib; runpy.run_path(sys.argv[1]); "
         "assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()",
         str(path)],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "svg"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("problem", ["vdp", "pendulum"])
def test_openloop_import_does_not_create_output_directories(monkeypatch, problem):
    # Initialize plotting dependencies before testing the experiment's import.
    importlib.import_module("src.plots")

    def unexpected_mkdir(*args, **kwargs):
        pytest.fail("importing an experiment must not create output directories")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)
    namespace = runpy.run_path(str(ROOT / f"experiments/00_openloop/{problem}/generate.py"))
    assert callable(namespace["main"])


@pytest.mark.parametrize("group,problem", [("01_vdp", "vdp"), ("02_pendulum", "pendulum")])
@pytest.mark.parametrize("family", ["paper_log_penalty", "paper_frac_exp_penalty"])
def test_paper_entrypoints_delegate_to_shared_implementation(monkeypatch, group, problem, family):
    path = ROOT / "experiments" / group / family / "analysis.py"
    if not path.exists():
        pytest.skip("paper-workspace entry points are local/ignored under ADR 0011")
    calls = []
    monkeypatch.setattr(sys, "argv", [str(path)])
    module = importlib.import_module(f"scripts.paper.{problem}_full_scope")

    def capture(argv=None):
        calls.append(module._parse_args(argv))
        return 0

    monkeypatch.setattr(module, "main", capture)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(path), run_name="__main__")
    assert result.value.code == 0
    assert len(calls) == 1
    if problem == "vdp":
        assert calls[0].homogeneous_alpha == 1e-6
        monkeypatch.setattr(sys, "argv", [str(path), "--homogeneous-alpha", "1e-4"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(path), run_name="__main__")
        assert calls[1].homogeneous_alpha == 1e-4


def test_p_study_loads_only_the_support_statistics_it_uses(tmp_path):
    path = ROOT / "experiments/01_vdp/paper_log_penalty/p_study_figure.py"
    if not path.exists():
        pytest.skip("paper-workspace entry points are local/ignored under ADR 0011")
    record = {
        "config": {"model": {"activation": "tanh", "moment_order": 2.01, "alpha": 1e-4}},
        "metrics": [{"values": {"radius_max": 5.0, "radius_r95": 3.0, "best_neurons": 4}}],
    }
    (tmp_path / "run.json").write_text(json.dumps(record))
    namespace = runpy.run_path(str(path))
    assert namespace["load"](tmp_path) == {
        ("tanh", 2.01, 1e-4): {"max_radius": 5.0, "r95": 3.0, "neurons": 4.0}
    }
