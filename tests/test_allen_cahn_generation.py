"""End-to-end generation must publish complete data or inspectable failures."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_allen_cahn_data.py"


def generate(tmp_path, tag, *options):
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--dimension", "6", "--samples", "4",
         "--tag", tag, "--output-dir", str(tmp_path), *options],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_serial_and_parallel_generation_retain_identical_order_and_labels(tmp_path):
    for tag, workers in (("serial", "1"), ("parallel", "2")):
        result = generate(tmp_path, tag, "--workers", workers)
        assert result.returncode == 0, result.stderr
        directory = tmp_path / f"AllenCahn6_{tag}"
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["complete"]
        assert metadata["retained_samples"] == 4
        assert metadata["failed_samples"] == 0
        assert len(metadata["gradient_checks"]) == 3
        assert json.loads((directory / "failures.json").read_text()) == []
    with np.load(tmp_path / "AllenCahn6_serial/value_samples.npz") as serial:
        with np.load(tmp_path / "AllenCahn6_parallel/value_samples.npz") as parallel:
            for name in ("x", "v", "dv"):
                np.testing.assert_array_equal(serial[name], parallel[name])


@pytest.mark.parametrize("workers", ["1", "2"])
def test_failed_batch_persists_diagnostics_without_publishing_training_data(tmp_path, workers):
    result = generate(tmp_path, "failure", "--workers", workers,
                      "--mesh-points", "2", "--max-nodes", "2", "--tol", "1e-12")
    assert result.returncode != 0
    directory = tmp_path / "AllenCahn6_failure"
    assert not (directory / "value_samples.npz").exists()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert not metadata["complete"]
    assert metadata["retained_samples"] == 0
    assert metadata["failed_samples"] == 4
    assert "error" in metadata
    failures = json.loads((directory / "failures.json").read_text())
    assert [row["row"] for row in failures] == [0, 1, 2, 3]
    assert all(row["message"] for row in failures)
    with np.load(directory / "requested_states.npz") as requested:
        np.testing.assert_array_equal([row["initial_state"] for row in failures], requested["x"])


def test_existing_dataset_is_not_overwritten(tmp_path):
    directory = tmp_path / "AllenCahn6_existing"
    directory.mkdir()
    data = directory / "value_samples.npz"
    original = b"existing dataset remains intact"
    data.write_bytes(original)
    result = generate(tmp_path, "existing")
    assert result.returncode != 0
    assert data.read_bytes() == original
    assert not (directory / "metadata.json").exists()
