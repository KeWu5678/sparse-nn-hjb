import hashlib

import numpy as np
import pytest
import torch

from src.PDAP import History
from src.results import find_runs, load_mlflow_run, predict_physical, value_surface_grid


def _atoms(c: float) -> dict[str, torch.Tensor]:
    return {"inner_weight": torch.tensor([[1., 2.]], dtype=torch.float64),
            "inner_bias": torch.tensor([.5], dtype=torch.float64),
            "c": torch.tensor([c], dtype=torch.float64)}


@pytest.fixture
def example(tmp_path):
    """A fit and the fields recorded with it; tests edit them before recording."""
    samples = {
        "x": np.arange(1, 21, dtype=float).reshape(10, 2),
        "v": np.arange(1, 11, dtype=float).reshape(10, 1),
        "dv": np.ones((10, 2)),
    }
    data_path = tmp_path / "data.npz"
    np.savez(data_path, **samples)
    with data_path.open("rb") as stream:
        data_sha256 = hashlib.sha256(stream.read()).hexdigest()
    history = History(
        rows=[{"objective_train": 2., "rel_h1_val": 123.}, {"objective_train": 1., "rel_h1_val": 456.}],
        states=[_atoms(1.), _atoms(3.)],
        diagnostics=[{}, {}],
    )
    return {
        "config": {
            "model": {"kind": "signed", "activation": "relu", "power": 1.,
                      "normalized": True, "moment_order": 99., "th": 0.5},
            "data": {"path": str(data_path), "normalize": True, "train_fraction": .6},
            "env": {"seed": 42},
        },
        "normalization": {"x_scale": [2., 4.], "v_scale": 8.},
        "tags": {"data_sha256": data_sha256, "hydra.output_dir": str(tmp_path / "jobs" / "0")},
        "history": history,
    }


@pytest.fixture
def record(tracking_store, example, tmp_path):
    """Record ``example`` as an MLflow run laid out as ``src.tracking`` writes one."""
    def record(status="FINISHED", experiment="run") -> str:
        found = tracking_store.get_experiment_by_name(experiment)
        experiment_id = found.experiment_id if found else tracking_store.create_experiment(
            experiment, artifact_location=(tmp_path / "mlartifacts" / experiment).as_uri())
        run_id = tracking_store.create_run(experiment_id).info.run_id
        tracking_store.log_dict(run_id, example["config"], "config.json")
        if example["normalization"] is not None:
            tracking_store.log_dict(run_id, example["normalization"], "normalization.json")
        history = tmp_path / run_id / "history.pt"
        history.parent.mkdir()
        torch.save(example["history"].to_dict(), history)
        tracking_store.log_artifact(run_id, str(history))
        for key, value in example["tags"].items():
            tracking_store.set_tag(run_id, key, value)
        tracking_store.set_terminated(run_id, status)
        return run_id
    return record


def test_loading_rejects_dataset_changed_since_training(example, record):
    data_path = example["config"]["data"]["path"]
    run_id = record()
    load_mlflow_run(run_id)
    with np.load(data_path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["v"] = arrays["v"] + 1.0
    np.savez(data_path, **arrays)
    with pytest.raises(ValueError, match="dataset differs"):
        load_mlflow_run(run_id)


def test_only_finished_runs_load(record):
    with pytest.raises(ValueError, match="expected a finished MLflow run"):
        load_mlflow_run(record(status="FAILED"))


def test_saved_run_uses_recorded_scaling_not_dataset_maxima(record):
    run = load_mlflow_run(record())
    v, dv = predict_physical(run, [[2., 4.]])
    torch.testing.assert_close(v, torch.tensor([[84.]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[12., 12.]], dtype=torch.float64))
    assert v.shape == (1, 1) and dv.shape == (1, 2)
    assert not v.requires_grad and not dv.requires_grad
    assert run.iteration == 1
    assert run.history.rows[1]["rel_h1_val"] == 456.
    # Objective/measure normalization is not another transform of V.
    assert run.config["model"]["moment_order"] == 99.
    assert run.model.moment_order is None
    # Historical mixture metadata remains readable; it is not a runtime option.
    assert run.config["model"]["th"] == 0.5


def test_checkpoint_override_does_not_change_saved_model(record):
    run = load_mlflow_run(record())
    before = {k: v.clone() for k, v in run.model.state_dict().items()}
    v, dv = predict_physical(run, [[2., 4.]], iteration=0)
    torch.testing.assert_close(v, torch.tensor([[28.]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[4., 4.]], dtype=torch.float64))
    assert run.iteration == 1
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(IndexError, match="checkpoint"):
        predict_physical(run, [[2., 4.]], iteration=-1)


def test_normalization_disabled_is_identity(example, record):
    example["normalization"] = None
    example["config"]["data"]["normalize"] = False
    v, dv = predict_physical(load_mlflow_run(record()), [[1., 1.]])
    torch.testing.assert_close(v, torch.tensor([[10.5]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[3., 6.]], dtype=torch.float64))


def test_missing_normalization_is_not_guessed(example, record):
    example["normalization"] = None
    with pytest.raises(ValueError, match="normalization"):
        load_mlflow_run(record())


def test_prediction_power_does_not_require_solver_support(example, record):
    example["config"]["model"]["power"] = 4.
    example["normalization"] = None
    example["config"]["data"]["normalize"] = False
    value, gradient = predict_physical(load_mlflow_run(record()), [[1., 1.]])

    torch.testing.assert_close(value, torch.tensor([[3. * 3.5 ** 4]], dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([[12. * 3.5 ** 3, 24. * 3.5 ** 3]], dtype=torch.float64))


def test_saved_nonhomogeneous_predictions_do_not_require_a_training_order(example, record):
    example["config"]["model"]["activation"] = "tanh"
    example["config"]["model"].pop("moment_order")
    run = load_mlflow_run(record())

    value, gradient = predict_physical(run, [[2., 4.]])

    activation = torch.tanh(torch.tensor(3.5, dtype=torch.float64))
    torch.testing.assert_close(value, (24 * activation).reshape(1, 1))
    torch.testing.assert_close(gradient, (12 * (1 - activation.square())).repeat(1, 2))
    assert run.model.moment_order is None


def test_empty_support_and_empty_prediction_batch(example, record):
    example["history"].states[0] = {
        "inner_weight": torch.empty((0, 2), dtype=torch.float64),
        "inner_bias": torch.empty(0, dtype=torch.float64),
        "c": torch.empty(0, dtype=torch.float64),
    }
    run = load_mlflow_run(record())
    v, dv = predict_physical(run, np.empty((0, 2)))
    assert v.shape == (0, 1) and dv.shape == (0, 2)
    v, dv = predict_physical(run, [[1., 2.]], iteration=0)
    assert torch.count_nonzero(v) == 0 and torch.count_nonzero(dv) == 0


def test_surface_grid_uses_same_unclipped_prediction(record):
    run = load_mlflow_run(record())
    x0, x1, value = value_surface_grid(run, x_range=(0., 2.), y_range=(0., 4.), grid_n=3)
    assert x0.shape == x1.shape == value.shape == (3, 3)
    assert value[-1, -1] == 84.  # Above the paper's display ceiling, not clamped.
    expected, _ = predict_physical(run, np.column_stack([x0.ravel(), x1.ravel()]))
    np.testing.assert_array_equal(value.ravel(), expected.numpy().ravel())


def test_find_runs_lists_finished_runs_in_job_folder_order(example, record, tmp_path):
    # String order of the folders, as they list on disk: "p_2.5" sorts before "p_2/".
    folders = [tmp_path / "p_2" / "0", tmp_path / "p_2.5" / "0", tmp_path / "p_3" / "0"]
    run_ids = {}
    for folder in reversed(folders):
        example["tags"]["hydra.output_dir"] = str(folder)
        run_ids[folder] = record(experiment="study")
    example["tags"]["hydra.output_dir"] = str(tmp_path / "p_1" / "0")
    record(status="FAILED", experiment="study")

    runs = find_runs("study")

    expected = [folders[1], folders[0], folders[2]]
    assert [run.run_id for run in runs] == [run_ids[folder] for folder in expected]
    assert runs[0].config == example["config"]
    assert runs[0].artifacts == {"config.json", "normalization.json", "history.pt"}
    with pytest.raises(ValueError, match="no MLflow experiment"):
        find_runs("missing")
