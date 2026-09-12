import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig
from src.data import ValueSampleNormalizer
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


def test_experiment_run_projects_completed_record_to_mlflow(tmp_path, monkeypatch):
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
    path = run.finish(summary={"best_score": 18.3})

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

    train.main.__wrapped__(cfg)

    record = json.loads(next(run_dir.glob("*.json")).read_text())
    assert record["config"]["data"]["normalize"] is normalize
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
