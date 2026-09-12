import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig
from src.data import ValueSampleNormalizer, split_value_samples
from src.eval import distance_binned_error, region_split_errors, relative_errors
from src.experiment_logging import ExperimentRun
from src.logging_config import configure_logging


def load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "experiment": None,
        "run_names": [],
        "params": {},
        "metrics": [],
        "tags": {},
        "ended": [],
    }

    def set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def set_experiment(name):
        calls["experiment"] = name

    def start_run(*, run_name):
        calls["run_names"].append(run_name)

    def log_param(key, value):
        calls["params"][key] = value

    def log_metric(key, value, step=None):
        calls["metrics"].append((key, value, step))

    def set_tag(key, value):
        calls["tags"][key] = value

    def end_run(*, status):
        calls["ended"].append(status)

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=set_tracking_uri,
            set_experiment=set_experiment,
            start_run=start_run,
            log_param=log_param,
            log_metric=log_metric,
            set_tag=set_tag,
            end_run=end_run,
        ),
    )
    return calls


def test_experiment_run_writes_completed_run_record(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="relu_seed42",
        config={"activation": "relu", "seed": 42},
    )

    run.log_metrics({"h1": 0.12, "neurons": 78}, step=0)
    path = run.finish(status="completed")

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["name"] == "activation_search"
    assert record["run_id"] == "relu_seed42"
    assert record["status"] == "completed"
    assert record["config"] == {"activation": "relu", "seed": 42}
    assert record["metrics"] == [{"step": 0, "values": {"h1": 0.12, "neurons": 78}}]
    assert path == tmp_path / "relu_seed42.json"


def test_experiment_run_writes_failed_run_record_with_error(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="bad_activation_seed42",
        config={"activation": "bad_activation", "seed": 42},
    )

    path = run.fail(RuntimeError("unknown activation"))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["message"] == "unknown activation"


def test_experiment_run_records_artifacts(tmp_path):
    plot_path = tmp_path / "plots" / "pareto.png"
    run = ExperimentRun(tmp_path, name="activation_search", run_id="relu_seed42")

    run.add_artifact("pareto_plot", plot_path)
    path = run.finish()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["artifacts"] == [{"name": "pareto_plot", "path": str(plot_path)}]


def test_configure_logging_writes_readable_diagnostics(tmp_path, capsys):
    log_path = tmp_path / "run.log"
    logger = configure_logging(verbose=True, log_file=log_path, level=logging.INFO)

    logger.info("run started: name=activation_search seed=42")

    captured = capsys.readouterr()
    assert "INFO run started: name=activation_search seed=42" in captured.err
    assert "INFO run started: name=activation_search seed=42" in log_path.read_text(encoding="utf-8")


def test_experiment_run_preserves_runner_summary_fields(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="relu_seed42",
        config={"activation": "relu", "seed": 42},
    )

    path = run.finish(summary={"activation": "relu", "seed": 42, "best_score": 18.3})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["activation"] == "relu"
    assert record["seed"] == 42
    assert record["best_score"] == 18.3
    assert record["status"] == "completed"
    assert record["name"] == "activation_search"


@pytest.mark.parametrize("normalization", [None, {"x_scale": [2.0, 4.0], "v_scale": 8.0}])
def test_experiment_run_projects_completed_record_to_mlflow(tmp_path, monkeypatch, normalization):
    calls = fake_mlflow(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    artifact = tmp_path / "result_activationsearch_pendulum_20260611_a3f9.pkl"
    run = ExperimentRun(
        tmp_path,
        name="activationsearch",
        run_id="activationsearch_pendulum_20260611_a3f9",
        config={"model": {"gamma": 1.0}, "data": {"path": "pendulum.npz"}},
        hydra={
            "output_dir": str(tmp_path),
            "job": {"name": "train", "id": "7", "num": 0},
            "runtime": {"choices": {"data": "pendulum", "model": "signed"}},
            "overrides": {"task": ["data=pendulum"]},
        },
    )

    run.add_artifact("fit_history", artifact)
    run.log_metrics({"rel_h1_val": 0.12, "best_neurons": 78, "label": "skip"}, step=3)
    path = run.finish(summary={"best_score": 18.3, "normalization": normalization})

    assert path.exists()
    assert calls["tracking_uri"] == "http://localhost:5000"
    assert calls["experiment"] == "activationsearch"
    assert calls["run_names"] == ["activationsearch_pendulum_20260611_a3f9"]
    assert calls["params"]["model.gamma"] == 1.0
    assert calls["params"]["data.path"] == "pendulum.npz"
    assert calls["params"]["hydra.choice.data"] == "pendulum"
    assert ("rel_h1_val", 0.12, 3) in calls["metrics"]
    assert ("best_neurons", 78.0, 3) in calls["metrics"]
    assert ("best_score", 18.3, None) in calls["metrics"]
    assert all(key != "normalization" for key, _, _ in calls["metrics"])
    assert json.loads(path.read_text())["normalization"] == normalization
    assert calls["tags"]["run_id"] == "activationsearch_pendulum_20260611_a3f9"
    assert calls["tags"]["status"] == "completed"
    assert calls["tags"]["run_record.path"] == str(path)
    assert calls["tags"]["artifact.fit_history.path"] == str(artifact)
    assert calls["tags"]["hydra.output_dir"] == str(tmp_path)
    assert calls["ended"] == ["FINISHED"]


def test_experiment_run_projects_failed_record_to_mlflow(tmp_path, monkeypatch):
    calls = fake_mlflow(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    run = ExperimentRun(
        tmp_path,
        name="activationsearch",
        run_id="activationsearch_pendulum_20260611_dead",
        config={"activation": "bad"},
    )

    path = run.fail(RuntimeError("unknown activation"))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert calls["tags"]["status"] == "failed"
    assert calls["tags"]["error.type"] == "RuntimeError"
    assert calls["tags"]["error.message"] == "unknown activation"
    assert calls["ended"] == ["FAILED"]


def test_run_id_uses_experiment_data_date_and_suffix():
    train = load_train_module()
    cfg = OmegaConf.create({
        "name": "Region Split Pendulum",
        "data": {"path": "fallback_dataset.npz"},
    })
    hydra_cfg = OmegaConf.create({
        "runtime": {"choices": {"data": "pendulum"}},
    })

    run_id = train.run_id_from_config(cfg, hydra_cfg=hydra_cfg, today="20260611", suffix="a3f9")

    assert run_id == "regionsplitpendulum_pendulum_20260611_a3f9"


@pytest.mark.parametrize("normalize", [True, False])
def test_training_record_preserves_fitted_normalization(tmp_path, monkeypatch, normalize):
    train = load_train_module()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    samples = {
        "x": np.array([[2.0, -4.0], [1.0, 3.0], [-1.0, 2.0], [0.0, 1.0]]),
        "v": np.array([[8.0], [4.0], [2.0], [1.0]]),
        "dv": np.array([[3.0, 5.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
    }
    dataset = tmp_path / "samples.npz"
    np.savez(dataset, **samples)
    run_dir = tmp_path / "run"
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(run_dir)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.data.path = str(dataset)
    cfg.data.normalize = normalize
    cfg.training.loop_order = "insertion_first"
    cfg.training.num_iterations = 0
    cfg.env.verbose = False

    # A nonzero fixed model exposes the metric's units without performing an
    # optimizer step. The zero network has relative error 1 in either convention.
    build_model = train.build_model

    def initialized_model(cfg, input_dim):
        model = build_model(cfg, input_dim)
        model.set_atoms(
            torch.tensor([[0.5, -0.2]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.7], dtype=torch.float64),
        )
        return model

    monkeypatch.setattr(train, "build_model", initialized_model)
    train.main.__wrapped__(cfg)

    record = json.loads(next(run_dir.glob("*.json")).read_text())
    assert record["config"]["data"]["normalize"] is normalize
    assert record["metric_coordinates"] == "physical"
    if normalize:
        assert record["normalization"] == {"x_scale": [2.0, 4.0], "v_scale": 8.0}
        saved = record["normalization"]
        restored = ValueSampleNormalizer(np.asarray(saved["x_scale"]), saved["v_scale"])
        normalized = restored.normalize(samples)
        value, gradient = restored.denormalize_prediction(normalized["v"], normalized["dv"])
        np.testing.assert_allclose(value, samples["v"])
        np.testing.assert_allclose(gradient, samples["dv"])
    else:
        assert record["normalization"] is None

    with next(run_dir.glob("result_*.pkl")).open("rb") as file:
        history = pickle.load(file)
    assert (history.reporting_normalizer is not None) is normalize
    model = history.restore_model(build_model(cfg, input_dim=2))
    train.set_seed(cfg.env.seed)
    _, (x, v, dv) = split_value_samples(samples, cfg.data.train_fraction)
    if normalize:
        x = x / torch.as_tensor(saved["x_scale"], dtype=torch.float64)
    vp, dvp = model.predict_tensors(x)
    if normalize:
        vp, dvp = restored.denormalize_tensors(vp, dvp)
    expected = relative_errors(vp, dvp, v, dv)
    metrics = record["metrics"][-1]["values"]
    assert [metrics[f"rel_{name}_val"] for name in ("l2", "grad", "h1")] == pytest.approx(expected)


@pytest.mark.parametrize("normalize", [True, False])
def test_region_reporting_scores_original_value_units(tmp_path, normalize):
    train = load_train_module()
    samples = {
        "x": np.array([[2.0, -4.0], [1.0, 3.0], [-1.0, 2.0], [0.0, 1.0]]),
        "v": np.array([[8.0], [4.0], [2.0], [1.0]]),
        "dv": np.array([[3.0, 5.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
    }
    distance = np.array([0.0, 0.1, 0.3, 0.4])
    pool_path, cache_path = tmp_path / "pool.npz", tmp_path / "distance.npz"
    np.savez(pool_path, **samples, distance=distance)
    np.savez(cache_path, distance=distance)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.eval.eval_pool = str(pool_path)
    cfg.eval.distance_cache = str(cache_path)
    cfg.eval.tube_radius = 0.2
    model = train.build_model(cfg, input_dim=2)
    model.set_atoms(
        torch.tensor([[0.5, -0.2]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.7], dtype=torch.float64),
    )
    normalizer = ValueSampleNormalizer.fit(samples) if normalize else None
    data = normalizer.normalize(samples) if normalizer is not None else samples

    got = train.region_split_metrics(cfg, model, data, normalizer)

    vp, dvp = model.predict_tensors(torch.as_tensor(data["x"], dtype=torch.float64))
    if normalizer is not None:
        # Independent chain-rule calculation, not the reporting implementation.
        vp = vp * normalizer.v_scale
        dvp = dvp * torch.as_tensor(normalizer.v_scale / normalizer.x_scale)
    target = torch.as_tensor(samples["v"]), torch.as_tensor(samples["dv"])
    expected = region_split_errors(vp, dvp, *target, torch.from_numpy(distance <= 0.2))
    expected.update(distance_binned_error(vp, dvp, *target, torch.from_numpy(distance)))
    assert got == pytest.approx(expected, nan_ok=True)


def test_failed_training_keeps_logs_without_a_run_record(tmp_path, monkeypatch):
    train = load_train_module()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.data.path = str(tmp_path / "missing.npz")
    cfg.env.verbose = False

    with pytest.raises(FileNotFoundError):
        train.main.__wrapped__(cfg)

    assert (tmp_path / "run.log").exists()
    assert not list(tmp_path.glob("*.json"))
