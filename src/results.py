"""Find saved fits and evaluate the original-scale value function.

An MLflow run owns the configuration, checkpoint History and data transform of
a fit (``src.tracking`` records them). This module lists and restores runs; it
does not select experiments, change the objective, or write artifacts.
"""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from .config import resolve_model
from .data import DATA_DIR, ValueSampleNormalizer, file_sha256
from .eval import physical_prediction
from .models import SparseNetwork
from .PDAP import History
from .tracking import CONFIG_ARTIFACT, HISTORY_ARTIFACT, NORMALIZATION_ARTIFACT


@dataclass(frozen=True)
class RunSummary:
    """A finished MLflow run's configuration and stored results, without its fit.

    ``config`` is the typed resolved configuration. ``metrics`` hold numbers
    only, and integer counts come back as floats; non-numeric results are tags.
    """

    run_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    artifacts: frozenset[str]


def find_runs(experiment: str) -> list[RunSummary]:
    """Finished runs of one MLflow experiment, in Hydra job-folder order.

    Runs sort by their Hydra output folder, the order the job folders list on
    disk, so tie-breaks and table rows are stable across reruns. Unfinished and
    deleted runs are excluded; an unknown experiment is an error.
    """
    client = mlflow.MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"no MLflow experiment named {experiment!r}")
    runs, token = [], None
    while True:
        page = client.search_runs(
            [found.experiment_id], "attributes.status = 'FINISHED'", page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            break
    # Each summary needs two artifact requests; issue them concurrently.
    with ThreadPoolExecutor(max_workers=16) as pool:
        summaries = list(pool.map(lambda run: _summarize(client, run), runs))
    return sorted(summaries, key=lambda run: f"{run.tags['hydra.output_dir']}/{run.name}")


def _summarize(client: mlflow.MlflowClient, run) -> RunSummary:
    return RunSummary(
        run.info.run_id, run.info.run_name,
        mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/{CONFIG_ARTIFACT}"),
        dict(run.data.metrics), dict(run.data.params), dict(run.data.tags),
        frozenset(item.path for item in client.list_artifacts(run.info.run_id)),
    )


@dataclass(frozen=True)
class SavedRun:
    """A finished MLflow run's fit, restored at its selected checkpoint.

    ``metrics`` are the values recorded at training time (``src.metrics``);
    the History holds every checkpoint's metrics and atoms.
    """

    run_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    history: History
    model: SparseNetwork
    normalizer: ValueSampleNormalizer | None

    @property
    def iteration(self) -> int:
        return self.history.best_iteration


def load_mlflow_run(run_id: str) -> SavedRun:
    """Restore a finished MLflow run's selected checkpoint and data transform.

    The run is read from the store named by ``MLFLOW_TRACKING_URI``. Its dataset
    must still have the bytes it was trained on.
    """
    run = mlflow.MlflowClient().get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"expected a finished MLflow run: {run_id}")
    with tempfile.TemporaryDirectory() as directory:
        artifacts = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=directory))
        config = json.loads((artifacts / CONFIG_ARTIFACT).read_text(encoding="utf-8"))
        history = History.from_dict(torch.load(artifacts / HISTORY_ARTIFACT, weights_only=True))
        transform = artifacts / NORMALIZATION_ARTIFACT
        normalizer = (ValueSampleNormalizer.from_dict(json.loads(transform.read_text(encoding="utf-8")))
                      if transform.is_file() else None)
    if config["data"]["normalize"] != (normalizer is not None):
        raise ValueError(f"normalization artifact contradicts data.normalize: MLflow run {run_id}")
    if file_sha256(DATA_DIR / config["data"]["path"]) != run.data.tags["data_sha256"]:
        raise ValueError(f"dataset differs from the bytes used for training: MLflow run {run_id}")
    return SavedRun(run_id, config, dict(run.data.metrics), history,
                    _network(config, history), normalizer)


def _network(config: dict[str, Any], history: History, iteration: int | None = None) -> SparseNetwork:
    """The fitted network at one checkpoint, built for prediction only."""
    model = config["model"]
    if model["kind"] != "signed":
        raise ValueError(f"unsupported saved model kind: {model['kind']!r}")
    input_dim = history.states[0]["inner_weight"].shape[1]
    network = SparseNetwork(resolve_model(
        {"activation": model["activation"], "power": model["power"]}, input_dim))
    history.restore(network, iteration)
    return network.eval()


def predict_physical(
    run: SavedRun, x: Any, iteration: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unclipped original-scale (V, ∇V) of one checkpoint, the selected one by default."""
    network = (run.model if iteration is None or iteration == run.iteration
               else _network(run.config, run.history, iteration))
    return physical_prediction(network, x, run.normalizer)


def value_surface_grid(
    run: SavedRun, *, x_range: tuple[float, float], y_range: tuple[float, float],
    grid_n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the existing 2D mesh with unclipped original-scale predictions."""
    if run.model.input_dim != 2:
        raise ValueError("value surfaces require a two-dimensional model")
    x0, x1 = np.meshgrid(np.linspace(*x_range, grid_n), np.linspace(*y_range, grid_n))
    value, _ = predict_physical(run, np.column_stack([x0.ravel(), x1.ravel()]))
    return x0, x1, value.cpu().numpy().reshape(x0.shape)
