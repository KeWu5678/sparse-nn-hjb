"""Record training runs in MLflow, the system of record for results.

``scripts/train.py`` opens one MLflow run per job; these helpers translate the
project's configuration and fitted History into MLflow params, tags, metrics
and artifacts. Only names in ``src.metrics.CATALOG`` and their traces are
recorded as metrics.
``src.results.load_mlflow_run`` reads them back.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.entities import Metric
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT
from mlflow.utils.validation import MAX_METRICS_PER_BATCH

from .data import ValueSampleNormalizer
from .metrics import CATALOG, TRACE_PREFIX
from .PDAP import History

CONFIG_ARTIFACT = "config.json"
NORMALIZATION_ARTIFACT = "normalization.json"
HISTORY_ARTIFACT = "history.pt"


def require_tracking_store() -> None:
    """Fail before training unless ``MLFLOW_TRACKING_URI`` names a reachable store.

    MLflow holds the only copy of a run's results, so no run starts without one.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is unset; start the server "
            "(docker compose -f deploy/docker/compose.yaml up -d) "
            "and export MLFLOW_TRACKING_URI=http://127.0.0.1:5000"
        )
    try:
        mlflow.MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
    except Exception as error:  # noqa: BLE001 -- any failure to reach it is fatal here
        raise RuntimeError(f"MLflow tracking store unreachable at {uri}: {error}") from error


def git_tags(repo: Path) -> dict[str, str]:
    """The commit checked out in ``repo`` and whether tracked files differ from it.

    Read when the run starts, so the tags describe the code that trains it.
    MLflow's own detection follows ``sys.argv[0]`` and caches per process, which
    is unreliable in sweep workers; these tags take precedence over it.
    Outside a git checkout no tags are returned.
    """
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        changes = _git(repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {MLFLOW_GIT_COMMIT: commit, "git_dirty": str(bool(changes)).lower()}


def log_config(config: dict[str, Any]) -> None:
    """Record the resolved configuration as searchable params and as typed JSON.

    Params flatten nested keys (``model.activation``) and hold strings; the JSON
    artifact keeps the original types for reconstruction.
    """
    mlflow.log_params({key: _param_value(value) for key, value in _flatten(config)})
    mlflow.log_dict(config, CONFIG_ARTIFACT)


def log_fit(
    history: History, *, normalizer: ValueSampleNormalizer | None,
    extra: dict[str, float] | None = None,
) -> None:
    """Record the History's summary, its traces and ``extra`` metrics, the transform and the History.

    ``extra`` holds post-fit metrics of the selected checkpoint, such as
    regional errors. A name outside the catalog fails before anything is
    logged. A run trained without normalization has no transform artifact.
    """
    metrics = {**history.summary(), **(extra or {})}
    traced = {key for row in history.rows for key in row}
    if unknown := sorted((set(metrics) | traced) - set(CATALOG)):
        raise ValueError(f"metrics missing from src.metrics.CATALOG: {unknown}")
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
    log_traces(mlflow.active_run().info.run_id, history.rows)
    if history.stop_reason is not None:
        mlflow.set_tag("stop_reason", history.stop_reason)
    if normalizer is not None:
        mlflow.log_dict(normalizer.to_dict(), NORMALIZATION_ARTIFACT)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / HISTORY_ARTIFACT
        torch.save(history.to_dict(), path)
        mlflow.log_artifact(str(path))


def log_traces(run_id: str, rows: list[dict[str, float]], *, timestamp: int | None = None) -> None:
    """Record each checkpoint metric as the curve ``trace.<name>``, step = checkpoint index."""
    timestamp = int(time.time() * 1000) if timestamp is None else timestamp
    points = [Metric(TRACE_PREFIX + key, float(value), timestamp, step)
              for step, row in enumerate(rows) for key, value in row.items()]
    client = mlflow.MlflowClient()
    for start in range(0, len(points), MAX_METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=points[start:start + MAX_METRICS_PER_BATCH])


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            items.extend(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return items
    return [(prefix, value)]


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, default=str)
