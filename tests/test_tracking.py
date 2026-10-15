import hashlib
import importlib.util
import json
import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import pytest
import structlog
import torch
from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig
from src.data import ValueSampleNormalizer, split_value_samples
from src.eval import distance_binned_error, region_metrics, region_split_errors, relative_errors
from src.metrics import CATALOG, TRACE_PREFIX
from src.PDAP import History
from src.results import load_mlflow_run
from src.tracking import git_tags, log_fit

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_train_module():
    path = REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_name_uses_experiment_data_date_and_suffix():
    train = load_train_module()
    cfg = OmegaConf.create({
        "name": "Region Split Pendulum",
        "data": {"path": "fallback_dataset.npz"},
    })
    hydra_cfg = OmegaConf.create({
        "runtime": {"choices": {"data": "pendulum"}},
    })

    run_name = train.run_name_from_config(cfg, hydra_cfg=hydra_cfg, today="20260611", suffix="a3f9")

    assert run_name == "regionsplitpendulum_pendulum_20260611_a3f9"


def test_git_tags_name_the_checked_out_commit_and_uncommitted_changes(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    assert git_tags(tmp_path) == {}
    git("init", "-q")
    (tmp_path / "tracked.txt").write_text("a")
    git("add", "tracked.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (tmp_path / "untracked.txt").write_text("ignored by the dirty check")
    assert git_tags(tmp_path) == {
        "mlflow.source.git.commit": git("rev-parse", "HEAD"), "git_dirty": "false"}

    (tmp_path / "tracked.txt").write_text("b")
    assert git_tags(tmp_path)["git_dirty"] == "true"


@pytest.mark.parametrize("normalize", [True, False])
def test_training_records_its_fit_in_mlflow_and_reloads_it(
    tmp_path, monkeypatch, caplog, capsys, tracking_store, normalize,
):
    train = load_train_module()
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

    # A nonzero fixed network exposes the metric's units without performing an
    # optimizer step. The zero network has relative error 1 in either convention.
    network_class = train.SparseNetwork

    def initialized_network(model_config):
        network = network_class(model_config)
        network.set_atoms(
            torch.tensor([[0.5, -0.2]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.7], dtype=torch.float64),
        )
        return network

    monkeypatch.setattr(train, "SparseNetwork", initialized_network)
    capsys.readouterr()  # Discard the store's own setup messages.
    with _isolated_logging(), structlog.contextvars.bound_contextvars(
        run_name="caller", experiment="outer", seed=-1,
    ):
        context = structlog.contextvars.get_contextvars()
        train.main.__wrapped__(cfg)
        assert structlog.contextvars.get_contextvars() == context
    assert capsys.readouterr().err == ""
    logging.getLogger(__name__).warning("pytest capture restored after training")
    assert "pytest capture restored after training" in caplog.text

    [run] = tracking_store.search_runs([tracking_store.get_experiment_by_name("run").experiment_id])
    assert run.info.status == "FINISHED"
    assert run.data.params["data.normalize"] == str(normalize)
    assert run.data.tags["data_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert "metric_coordinates" not in run.data.tags
    assert {key.removeprefix(TRACE_PREFIX) for key in run.data.metrics} <= set(CATALOG)
    assert run.data.tags["stop_reason"] == "iteration_budget"
    assert run.data.tags["mlflow.source.git.commit"] == subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert run.data.tags["git_dirty"] in ("true", "false")
    assert run.data.tags["hydra.output_dir"] == str(run_dir)
    artifacts = {item.path for item in tracking_store.list_artifacts(run.info.run_id)}
    assert artifacts == {"config.json", "history.pt"} | ({"normalization.json"} if normalize else set())
    assert not list(run_dir.glob("*.json")) and not list(run_dir.glob("*.p*"))

    history = History.from_dict(torch.load(
        tracking_store.download_artifacts(run.info.run_id, "history.pt", str(tmp_path)), weights_only=True))
    network = history.restore(network_class(train.resolve_model(cfg.model, input_dim=2)))
    train.set_seed(cfg.env.seed)
    _, (x, v, dv) = split_value_samples(samples, cfg.data.train_fraction)
    saved = load_mlflow_run(run.info.run_id)
    if normalize:
        restored = ValueSampleNormalizer(np.array([2.0, 4.0]), 8.0)
        np.testing.assert_allclose(saved.normalizer.x_scale, restored.x_scale)
        assert saved.normalizer.v_scale == restored.v_scale
        x = x / torch.as_tensor(restored.x_scale, dtype=torch.float64)
    else:
        assert saved.normalizer is None
    vp, dvp = network(x)
    if normalize:
        vp, dvp = restored.denormalize_tensors(vp, dvp)
    expected = relative_errors(vp, dvp, v, dv)
    assert {kind: run.data.metrics[f"rel_{kind}_val"] for kind in expected} == pytest.approx(expected)
    assert saved.history.rows[saved.iteration]["rel_h1_val"] == run.data.metrics["rel_h1_val"]


@pytest.mark.parametrize("normalize", [True, False])
def test_region_reporting_scores_original_value_units(tmp_path, normalize):
    train = load_train_module()
    samples = {
        "x": np.array([[2.0, -4.0], [1.0, 3.0], [-1.0, 2.0], [0.0, 1.0]]),
        "v": np.array([[8.0], [4.0], [2.0], [1.0]]),
        "dv": np.array([[3.0, 5.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
    }
    distance = np.array([0.0, 0.1, 0.3, 0.4])
    cfg = OmegaConf.structured(ExperimentConfig())
    network = train.SparseNetwork(train.resolve_model(cfg.model, input_dim=2))
    network.set_atoms(
        torch.tensor([[0.5, -0.2]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.7], dtype=torch.float64),
    )
    normalizer = ValueSampleNormalizer.fit(samples) if normalize else None
    data = normalizer.normalize(samples) if normalizer is not None else samples

    got = region_metrics(network, normalizer, {**samples, "distance": distance}, tube_radius=0.2,
                         samples=samples, distance=distance)

    vp, dvp = network(torch.as_tensor(data["x"], dtype=torch.float64))
    if normalizer is not None:
        # Independent chain-rule calculation, not the reporting implementation.
        vp = vp * normalizer.v_scale
        dvp = dvp * torch.as_tensor(normalizer.v_scale / normalizer.x_scale)
    target = torch.as_tensor(samples["v"]), torch.as_tensor(samples["dv"])
    expected = region_split_errors(vp, dvp, *target, torch.from_numpy(distance <= 0.2))
    expected.update(distance_binned_error(vp, dvp, *target, torch.from_numpy(distance)))
    assert got == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("empty_region,distance", [("switching", 1.0), ("rest", 0.0)])
def test_empty_region_logs_null_without_changing_saved_metrics(
    tmp_path, monkeypatch, tracking_store, empty_region, distance,
):
    train = load_train_module()
    run_dir = tmp_path / "run"
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(run_dir)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    dataset = tmp_path / "samples.npz"
    x = np.arange(1.0, 5.0).reshape(-1, 1)
    np.savez(dataset, x=x, v=x**2, dv=2*x, distance=np.full(len(x), distance))
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.data.path = str(dataset)
    cfg.training.loop_order = "insertion_first"
    cfg.training.num_iterations = 0
    cfg.env.verbose = False
    cfg.eval.kind = "region_split"
    cfg.eval.eval_pool = cfg.eval.distance_cache = str(dataset)
    with _isolated_logging():
        train.main.__wrapped__(cfg)

    logs = [json.loads(line, parse_constant=lambda token: pytest.fail(f"non-JSON {token}"))
            for line in (run_dir / "run.jsonl").read_text().splitlines()]
    summary = next(event for event in logs if event["event"] == "region_summary")
    assert summary[f"count_{empty_region}"] == 0
    assert summary[f"rel_h1_{empty_region}"] is None
    run = tracking_store.get_run(logs[-1]["run_id"])
    assert np.isnan(run.data.metrics[f"rel_h1_{empty_region}"])
    assert {key.removeprefix(TRACE_PREFIX) for key in run.data.metrics} <= set(CATALOG)


def test_fit_records_selected_values_for_ranking_and_every_checkpoint_as_a_trace(tracking_store):
    rows = [{"objective_train": objective, "rel_h1_val": error, "neurons": neurons}
            for objective, error, neurons in ((3.0, 0.5, 1), (1.0, 0.2, 2), (2.0, 0.1, 3))]
    experiment = tracking_store.get_experiment_by_name("run").experiment_id
    with mlflow.start_run(experiment_id=experiment) as active:
        log_fit(History(rows=rows, diagnostics=[{}] * 3), normalizer=None)

    run = tracking_store.get_run(active.info.run_id)
    assert (run.data.metrics["rel_h1_val"], run.data.metrics["best_iteration"]) == (0.2, 1)
    for key in rows[0]:
        trace = tracking_store.get_metric_history(run.info.run_id, TRACE_PREFIX + key)
        assert sorted((point.step, point.value) for point in trace) == [
            (step, row[key]) for step, row in enumerate(rows)]
    assert {key.removeprefix(TRACE_PREFIX) for key in run.data.metrics} <= set(CATALOG)


def test_metrics_outside_the_catalog_are_refused_before_logging():
    history = History(rows=[{"objective_train": 0.0, "unlisted_error": 1.0}], diagnostics=[{}])
    with pytest.raises(ValueError, match="unlisted_error"):
        log_fit(history, normalizer=None)


def test_failed_training_is_recorded_as_a_failed_run_with_its_config(
    tmp_path, monkeypatch, tracking_store,
):
    train = load_train_module()
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.data.path = str(tmp_path / "missing.npz")
    cfg.env.verbose = False

    with _isolated_logging(), structlog.contextvars.bound_contextvars(
        run_name="caller", experiment="outer", seed=-1,
    ):
        context = structlog.contextvars.get_contextvars()
        with pytest.raises(FileNotFoundError):
            train.main.__wrapped__(cfg)
        assert structlog.contextvars.get_contextvars() == context

    assert (tmp_path / "run.jsonl").exists()
    [run] = tracking_store.search_runs([tracking_store.get_experiment_by_name("run").experiment_id])
    assert run.info.status == "FAILED"
    assert run.data.params["data.path"] == cfg.data.path
    assert {item.path for item in tracking_store.list_artifacts(run.info.run_id)} == {"config.json"}


def test_training_refuses_to_start_without_a_tracking_store(tmp_path, monkeypatch):
    train = load_train_module()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.env.verbose = False

    def premature_start(*args, **kwargs):
        pytest.fail("training started without a tracking store")

    monkeypatch.setattr(train.mlflow, "start_run", premature_start)
    with _isolated_logging(), pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI is unset"):
        train.main.__wrapped__(cfg)


@pytest.mark.parametrize("settings,message", [
    ({"activation": "unknown"}, "unknown activation"),
    ({"power": 4.}, "supports activation powers 1, 2, 3"),
    ({"activation": "softplus"}, "explicit moment_order"),
    ({"activation": "softplus", "moment_order": 1.}, "moment_order must exceed"),
    ({"insertion": "finite_step", "activation": "softplus", "moment_order": 2.},
     "requires a sphere activation"),
    ({"insertion": "finite_step", "gamma": .1}, "requires gamma == 0"),
])
def test_invalid_model_config_fails_before_model_solver_or_loader_setup(
    tmp_path, monkeypatch, tracking_store, settings, message,
):
    train = load_train_module()
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    dataset = tmp_path / "samples.npz"
    np.savez(dataset, x=np.ones((2, 2)), v=np.ones((2, 1)), dv=np.ones((2, 2)))
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.data.path = str(dataset)
    cfg.env.verbose = False
    cfg.model.update(settings)
    raw_model = OmegaConf.to_container(cfg.model, resolve=True)

    def premature_setup(*args, **kwargs):
        pytest.fail("invalid model configuration reached model, solver or sample-loader setup")

    for name in ("SparseNetwork", "PDAP", "make_dataloaders"):
        monkeypatch.setattr(train, name, premature_setup)
    with _isolated_logging(), pytest.raises(ValueError, match=message):
        train.main.__wrapped__(cfg)

    assert OmegaConf.to_container(cfg.model, resolve=True) == raw_model


@contextmanager
def _isolated_logging():
    """Detach pytest's handlers while the runner owns the process-wide setup."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    logger_class = logging.getLoggerClass()
    configured, config = structlog.is_configured(), structlog.get_config()
    context = structlog.contextvars.get_contextvars()
    root.handlers.clear()
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        logging.setLoggerClass(logger_class)
        structlog.reset_defaults()
        if configured:
            structlog.configure(**config)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context)
