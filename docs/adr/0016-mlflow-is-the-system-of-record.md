---
status: accepted
---

# MLflow is the system of record for training runs

Supersedes [ADR 0002](0002-mlflow-as-run-record-backend.md). The run contents
listed below are amended by [ADR 0017](0017-metric-catalog.md): the fit artifact
is `history.pt` and the `metric_coordinates` tag is gone.

ADR 0002 kept a local JSON Run Record per job as the source of truth and
projected it into MLflow for the dashboard. Keeping the two stores consistent
was manual: on 2026-10-15 the 588 active MLflow runs matched the 588 local
records only because 592 runs whose sweep folders had been deleted were removed
from MLflow by hand. The projection also omitted the data transform, dataset
hashes and metric coordinates, so no fit could be restored from MLflow alone.
Keeping both required a runtime class, a projection layer and an upload script,
and uploads tagged runs with the commit checked out at upload time instead of at
training time.

## Decision

- `scripts/train.py` records each job as one MLflow run, and nothing else stores
  results. `MLFLOW_TRACKING_URI` is required; a job fails at startup when the
  store is unset or unreachable.
- Configuration and provenance are recorded before training. A job that raises
  stays in MLflow with status `FAILED`; its traceback is in `run.jsonl`.
- Each run holds:
  - **Name:** `{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}`; the MLflow
    run ID is the storage identity.
  - **Params:** flattened resolved config and Hydra runtime choices.
  - **Tags:** `mlflow.source.git.commit` read from the repository at training
    time and `git_dirty` (tracked files differ from it); `data_sha256`,
    `eval_pool_sha256`, `distance_cache_sha256`; `metric_coordinates`; Hydra job
    metadata; non-numeric results such as `stop_reason`.
  - **Metrics:** selected-checkpoint metrics, region metrics and run counters.
  - **Artifacts:** `config.json` (typed config), `normalization.json` (normalized
    runs only) and `history.pkl`.
- `src.results.load_mlflow_run(run_id)` restores a finished run with the same
  provenance checks the JSON reader applied.
- Hydra job folders keep only `.hydra/` and `run.jsonl`.
- Existing JSON records were attached to their MLflow runs (artifacts and
  provenance tags), and the paper scripts list runs with
  `src.results.find_runs`. The JSON reader was removed once every paper output
  regenerated from MLflow matched the JSON-based outputs exactly (2026-10-15).

## Considered Options

- **Keep local JSON as the source of truth** — rejected: it keeps two writers
  and needs reconciliation, which had been manual.
- **Store the JSON record as an MLflow artifact** — rejected: it duplicates the
  params and metrics MLflow already holds natively. Only the typed config and
  the transform need JSON artifacts.

## Consequences

- `deploy/docker/mlflow-data/` (`mlflow.db` and `mlartifacts/`) is the only
  copy of results and must be backed up; see [mlflow.md](mlflow.md).
- Training needs a tracking store. Parallel sweeps go through the server,
  since SQLite accepts one writer at a time.
- Failed runs are visible and searchable.
- Analysis code reads runs through MLflow (`mlflow.search_runs`,
  `load_mlflow_run`) instead of globbing folders.
