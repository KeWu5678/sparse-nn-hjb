"""Check high-dimensional report provenance and physical-coordinate rescoring."""

import copy
import hashlib
import json
import pickle
import runpy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config.schema import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    return runpy.run_path(str(ROOT / "experiments/03_high_dimensional/report.py"))


@pytest.fixture
def saved_example(tmp_path):
    def create(dimension=6, dataset=None):
        dataset = dataset or f"allen_cahn{dimension}"
        scale = np.arange(1, dimension + 1, dtype=float)
        x = np.arange(1, 10 * dimension + 1, dtype=float).reshape(10, dimension)
        data = tmp_path / f"samples{dimension}.npz"
        np.savez(data, x=x, v=7 * np.sum(x / scale, axis=1),
                 dv=np.tile(7 / scale, (len(x), 1)))
        config = asdict(ExperimentConfig())
        config["data"].update(path=str(data), train_fraction=.6)
        config["training"]["num_iterations"] = 2
        record = {
            "run_id": "example", "status": "completed", "config": config,
            "hydra": {"runtime": {"choices": {
                "data": dataset, "experiment": "higher_dim_relu_l1_baseline",
            }}},
            "metric_coordinates": "physical",
            "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "normalization": {"x_scale": scale.tolist(), "v_scale": 7.},
        }
        records = tmp_path / "records"
        records.mkdir(exist_ok=True)
        path = records / "example.json"
        path.write_text(json.dumps(record))
        # Choose the first checkpoint explicitly, although the second has zero
        # approximation error. Reporting must preserve training's selection.
        history = SimpleNamespace(
            best_iteration=0,
            inner_weights=[{"weight": np.ones((1, dimension)), "bias": np.zeros(1)}] * 2,
            outer_weights=[np.array([[.5]]), np.array([[1.]])],
        )
        with (records / "result_example.pkl").open("wb") as handle:
            pickle.dump(history, handle)
        return path, record, {dataset: str(data)}

    return create


@pytest.mark.parametrize("dimension", [6, 10])
def test_report_rescores_all_dimensions_and_keeps_selected_checkpoint(report, saved_example, dimension):
    path, record, paths = saved_example(dimension)
    key = f"allen_cahn{dimension}", "higher_dim_relu_l1_baseline"
    expected = {key: {report["scientific_signature"](record["config"])}}
    rows, curves, found = report["collect"](path.parent, expected, paths)
    assert found == expected
    assert rows[0]["kind"] == "configured"
    assert rows[0]["dimension"] == dimension
    assert rows[0]["validation_count"] == 4
    assert rows[0]["best_iteration"] == 0
    np.testing.assert_allclose([rows[0][key] for key in ("rel_l2", "rel_grad", "rel_h1")], .5)
    history = next(iter(curves.values()))[0]
    assert history["rel_h1"][1] < 1e-14


def test_reduced_budget_is_a_pilot_and_duplicate_cells_are_rejected(report, saved_example):
    path, record, paths = saved_example()
    standard = copy.deepcopy(record["config"])
    standard["training"]["num_iterations"] = 10
    key = "allen_cahn6", "higher_dim_relu_l1_baseline"
    expected = {key: {report["scientific_signature"](standard)}}
    rows, _, found = report["collect"](path.parent, expected, paths)
    assert rows[0]["kind"] == "pilot/override"
    assert not found[key]
    path.with_name("duplicate.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate scientific config"):
        report["collect"](path.parent, expected, paths)


@pytest.mark.parametrize("field,value,message", [
    ("data_sha256", None, "dataset hash"),
    ("metric_coordinates", "training", "physical-coordinate"),
    ("data_sha256", "wrong-hash", "dataset differs"),
])
def test_report_rejects_missing_or_inconsistent_provenance(
    report, saved_example, field, value, message,
):
    path, record, paths = saved_example()
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        report["collect"](path.parent, {("allen_cahn6", "higher_dim_relu_l1_baseline"): set()}, paths)


def test_report_rejects_wrong_state_dimension(report, saved_example):
    path, _, paths = saved_example(10, dataset="allen_cahn6")
    with pytest.raises(ValueError, match="expected 6 state variables"):
        report["collect"](path.parent, {("allen_cahn6", "higher_dim_relu_l1_baseline"): set()}, paths)


def test_frontier_uses_a_neuron_budget_across_unsorted_repeated_counts(report):
    counts, errors = report["lower_envelope"]([3, 1, 2, 1, 4], [.2, .8, .9, .6, .3])
    np.testing.assert_array_equal(counts, [1, 2, 3, 4])
    np.testing.assert_allclose(errors, [.6, .6, .2, .2])
