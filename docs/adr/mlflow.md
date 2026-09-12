# MLflow Pipeline

Operational guide for experiment tracking (local container per ADR 0002/0014).

`ExperimentRun` always writes a local JSON Run Record in the Hydra output
directory. MLflow is filled by **importing** those completed local records after
a sweep, so training never depends on the dashboard being up. MLflow stores
params, scalar metrics, status, Hydra metadata, and local artifact paths; it does
not upload the `result_<run_id>.pkl` artifact in the current pipeline.

### Start The Dashboard

```bash
docker compose -f deploy/docker/compose.yaml up -d
```

`restart: unless-stopped` keeps it running across reboots without a terminal, so
this is normally a one-time command. The server listens on `127.0.0.1:5000`
only. The SQLite backend and server-uploaded artifacts (`mlartifacts/`) both
live under `deploy/docker/mlflow-data/` (gitignored).

### Normal Workflow

Run experiments normally. Local Run Records are written whether the dashboard
exists or not:

```bash
make sweep EXPERIMENT=log_penalty
```

Then project the sweep into MLflow:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
.venv/bin/python scripts/upload_run_records_to_mlflow.py \
  rawdata/logs/multirun/vdp/log_penalty
```

The importer reads each Run Record JSON and, for older records, enriches it with
adjacent `.hydra/overrides.yaml` metadata before uploading. Re-running the
importer creates additional MLflow runs for the same JSON; deduplication is not
implemented yet.

Point it at a single record, or at a whole experiment tree, by changing the path
argument.

Project run IDs use:

```text
{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}
```

### Optional Live Logging

Live logging is available but is not the default workflow. With the container
up, set the tracking URI before training:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make sweep EXPERIMENT=log_penalty
```

At `run.finish()`, the script writes the local JSON Run Record, keeps it on
disk, and publishes dashboard data to MLflow. If `MLFLOW_TRACKING_URI` is unset,
the run is local-only.

### Without A Container

A foreground server is fine for a one-off look, and uses the same client
contract:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Resetting

Run Records can restore dashboard metadata, but not uploaded artifact bytes.
Stop the server before resetting:

```bash
docker compose -f deploy/docker/compose.yaml down
```

Move the entire `deploy/docker/mlflow-data/` directory to a backup location
outside the repository. Keep the database and `mlartifacts/` together so the
original run-to-artifact associations can be restored. Then start a fresh store
and re-import the Run Records:

```bash
docker compose -f deploy/docker/compose.yaml up -d
```

### History

Earlier revisions of this guide described an EC2 deployment driven by
`make mlflow-deploy` / `make mlflow-backfill` over an SSM tunnel. That target was
retired by [ADR 0014](0014-retire-the-ec2-mlflow-target.md).
