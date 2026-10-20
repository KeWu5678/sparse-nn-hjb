"""Training-only execution and record safety for the public sweep command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("existing", ["run.jsonl", "legacy_record.json"])
def test_sweep_refuses_to_mix_with_existing_records(tmp_path: Path, existing: str) -> None:
    # Job outputs land in <sweep_dir>/<data>/<EXPERIMENT>/<job>/.
    sweep_dir = tmp_path / "records"
    job_dir = sweep_dir / "vdp" / "log_penalty" / "0"
    job_dir.mkdir(parents=True)
    (job_dir / existing).write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            "make",
            "sweep",
            f"SWEEP_DIR={sweep_dir}",
            "PY=false",
            "MLFLOW_HEALTH=true",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "already contains records; refusing to mix runs" in output


def test_sweep_runs_only_training_without_analysis_scripts(tmp_path: Path) -> None:
    result = subprocess.run(
        ["make", "--silent", "sweep", f"SWEEP_DIR={tmp_path / 'records'}", "PY=echo",
         "MLFLOW_HEALTH=true"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = result.stdout.strip().splitlines()
    assert len(commands) == 1
    assert commands[0].startswith("scripts/train.py -m +experiment=log_penalty ")


def test_sweep_stops_before_training_when_mlflow_is_unreachable(tmp_path: Path) -> None:
    result = subprocess.run(
        ["make", "--silent", "sweep", f"SWEEP_DIR={tmp_path / 'records'}", "PY=echo",
         "MLFLOW_HEALTH=false"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "run scripts/init.sh" in result.stdout
    assert "scripts/train.py" not in result.stdout
