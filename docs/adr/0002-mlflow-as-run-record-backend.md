---
status: accepted
---

# MLflow is an optional dashboard projection of local Run Records

See [mlflow.md](mlflow.md) for day-to-day usage.

`ExperimentRun` is the central runtime API for experiment recording. Every completed run
writes a local JSON **Run Record** and local **Run Artifacts** in the Hydra
output directory. When `MLFLOW_TRACKING_URI` is set, the completed Run Record is
also projected to MLflow so the MLflow UI can compare runs.

MLflow is therefore a dashboard/index, not the source of truth. Local JSON and
local artifacts remain authoritative. The MLflow SQLite backend lives on a
local bind mount per [ADR 0014](0014-retire-the-ec2-mlflow-target.md). If the
backend DB is lost, the dashboard can be rebuilt from local Run Records later.

## Decision

- Write and keep the local JSON Run Record on successful completion (`finish()`).
- Failed training runs retain diagnostic logs but do not produce a Run Record
  or a dashboard entry. The generic runner does not call `fail()`.
- If `MLFLOW_TRACKING_URI` is set, publish dashboard metadata to MLflow after the
  JSON record is written.
- Do not upload artifacts to MLflow or S3 in v1. MLflow stores local artifact
  paths as tags only.
- Use MLflow's own run ID as the storage identity. The project `run_id` is a
  human-readable run key and MLflow run name.
- Generate project run IDs as
  `{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}`, using the local date.

## What MLflow Stores

The intended MLflow use is cross-run comparison, not per-iteration training
monitoring:

- **Params:** flattened resolved config values and Hydra runtime choices.
- **Metrics:** scalar summary metrics from the Run Record.
- **Tags:** project `run_id`, status, local Run Record path, local artifact
  paths, Hydra output directory/job metadata, and failure metadata when present.

Per-iteration curves and the full `PDAP.fit()` result stay local in the saved
pickle artifact. MLflow records the pickle path, but does not read or upload the
pickle.

## Considered Options

- **MLflow replaces local JSON** — rejected because the SQLite store
  is a dashboard index and may be rebuilt; local records are the durable research
  source of truth.
- **Always write local JSON and MLflow** — rejected as a hard requirement because
  offline runs should not need an MLflow server.
- **Local JSON plus optional MLflow projection** — chosen because it keeps one
  runtime API, preserves offline reproducibility, and still enables the MLflow
  comparison UI when a tracking server is available.
- **Upload artifacts to S3 via MLflow** — deferred. The current workflow keeps
  artifacts local and stores dashboard history in the SQLite backend.

## Consequences

- A run launched without `MLFLOW_TRACKING_URI` still produces complete local
  records.
- A run launched with `MLFLOW_TRACKING_URI` produces the same local records and
  additionally appears in the MLflow UI.
- Existing local Run Records can be uploaded without retraining via
  `scripts/upload_run_records_to_mlflow.py`.
- Stopping the container preserves MLflow history on the bind mount.
  Destroying/replacing the backend requires either DB backup/restore
  or replaying local JSON Run Records with
  `scripts/upload_run_records_to_mlflow.py`.
- MLflow logging must remain downstream of the Run Record and must not reach into
  PDAP internals.
