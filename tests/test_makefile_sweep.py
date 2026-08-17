"""Safety checks for the public experiment-sweep command."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sweep_refuses_to_mix_with_existing_records(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "analysis.py").write_text("", encoding="utf-8")
    sweep_dir = tmp_path / "records"
    sweep_dir.mkdir()
    (sweep_dir / "existing.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            "make",
            "sweep",
            f"ANALYSIS_DIR={analysis_dir}",
            f"SWEEP_DIR={sweep_dir}",
            "PY=false",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "already contains records; refusing to mix runs" in output
