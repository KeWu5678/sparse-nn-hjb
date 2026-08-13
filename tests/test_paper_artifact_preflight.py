from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PREFLIGHT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper" / "preflight.py"
_SPEC = importlib.util.spec_from_file_location("paper_artifact_preflight", _PREFLIGHT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
preflight = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = preflight
_SPEC.loader.exec_module(preflight)


def _write_record(root: Path, *, job: str = "0") -> Path:
    run_dir = root / job
    run_dir.mkdir(parents=True)
    run_id = f"run_{job}"
    record_path = run_dir / f"{run_id}.json"
    record = {
        "run_id": run_id,
        "status": "completed",
        "config": {
            "name": "pendulum_paper_log_penalty",
            "env": {"seed": 42},
            "data": {"path": "fixture.npz", "normalize": True},
            "model": {
                "kind": "signed",
                "insertion": "profile",
                "activation": "gaussian",
                "power": 1.0,
                "loss_weights": [1.0, 1.0],
                "alpha": 1e-4,
                "gamma": 0.0,
                "moment_order": 2.01,
                "moment_beta": 0.0,
                "objective": "normalized_moment",
            },
            "training": {
                "insert_mode": "sequential",
                "num_iterations": 150,
                "num_insertion": 50,
                "insert_init": "guaranteed",
                "loop_order": "insertion_first",
                "correction_guard": True,
                "ins_merge_tol": 1e-2,
                "lbfgs_lr": 1e-2,
                "lbfgs_steps": 200,
                "radial_cap": "theorem",
            },
        },
        "metrics": [{"values": {"best_neurons": 1}}],
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (run_dir / f"result_{run_id}.pkl").touch()
    return record_path


def _cell(record: preflight.Record) -> tuple[str]:
    return (str(record.model["activation"]),)


def test_grid_validation_requires_region_sidecar(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fixture.npz").touch()
    monkeypatch.setattr(preflight, "DATA_DIR", data_dir)
    monkeypatch.setitem(preflight.EXPECTED_DATA_PATHS, "pendulum", {"fixture.npz"})
    root = tmp_path / "records"
    record_path = _write_record(root)

    preflight._validate_grid(
        root,
        problem="pendulum",
        algorithm=1,
        key=_cell,
        expected=[("gaussian",)],
    )
    with pytest.raises(ValueError, match="missing region-rescore sidecar"):
        preflight._validate_grid(
            root,
            problem="pendulum",
            algorithm=1,
            key=_cell,
            expected=[("gaussian",)],
            require_sidecars=True,
        )

    (data_dir / "fixture_region_eval_pool.npz").touch()
    sidecar = record_path.parent / "region_rescored_run_0.json"
    sidecar.write_text(
        json.dumps(
            {
                "switching_l1_h1": 0.1,
                "rest_l1_h1": 0.1,
                "switching_h1": 0.1,
                "rest_h1": 0.1,
                "switching_count": 1,
                "rest_count": 1,
                "tube_radius": 0.3,
                "pool": "fixture.npz",
                "scoring_version": preflight.REGION_SCORING_VERSION,
                "record_sha256": preflight.hashlib.sha256(record_path.read_bytes()).hexdigest(),
                "fit_history_sha256": preflight._file_sha256(
                    record_path.parent / "result_run_0.pkl"
                ),
                "pool_sha256": preflight._file_sha256(data_dir / "fixture_region_eval_pool.npz"),
            }
        ),
        encoding="utf-8",
    )
    preflight._validate_grid(
        root,
        problem="pendulum",
        algorithm=1,
        key=_cell,
        expected=[("gaussian",)],
        require_sidecars=True,
    )


def test_grid_validation_rejects_duplicate_cells(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fixture.npz").touch()
    monkeypatch.setattr(preflight, "DATA_DIR", data_dir)
    monkeypatch.setitem(preflight.EXPECTED_DATA_PATHS, "pendulum", {"fixture.npz"})
    root = tmp_path / "records"
    _write_record(root, job="0")
    _write_record(root, job="1")

    with pytest.raises(ValueError, match="duplicates=.*gaussian"):
        preflight._validate_grid(
            root,
            problem="pendulum",
            algorithm=1,
            key=_cell,
            expected=[("gaussian",)],
        )


def test_provenance_rejects_another_search_version(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "search_version": "two_stage",
                "merge_tolerance": 1e-2,
                "run_date": "2026-08-13",
                "implementation_commit": "f2d5191",
                "algorithm1_record_roots": sorted(preflight.ALGORITHM1_ROOTS),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "PROVENANCE_MANIFEST", manifest)
    with pytest.raises(ValueError, match="search_version"):
        preflight._validate_provenance()
