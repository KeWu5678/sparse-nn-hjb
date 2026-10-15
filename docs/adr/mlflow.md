# MLflow Pipeline

Operational guide for experiment tracking (local container per ADR 0014).

MLflow is the system of record for training runs
([ADR 0016](0016-mlflow-is-the-system-of-record.md)). Every training job records
one MLflow run: config and Hydra choices as params; git commit, dataset hashes
and Hydra job metadata as tags; selected-checkpoint metrics, by which runs are
ranked, and the training curves `trace.<name>`; and `config.json`,
`normalization.json` and `history.pt` as artifacts. Nothing else stores results,
so a job refuses to start without a reachable tracking store.

### Start The Server

```bash
docker compose -f deploy/docker/compose.yaml up -d
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`restart: unless-stopped` keeps it running across reboots without a terminal, so
this is normally a one-time command. The server listens on `127.0.0.1:5000`
only. The SQLite backend and the artifacts (`mlartifacts/`) both live under
`deploy/docker/mlflow-data/` (gitignored).

### Normal Workflow

```bash
make sweep EXPERIMENT=log_penalty
```

The Makefile sets `MLFLOW_TRACKING_URI` to the local server. Runs appear in the
UI as they finish; a job that raises is kept with status `FAILED`, and its
traceback is in the job's `run.jsonl`.

Run names use:

```text
{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}
```

The `mlflow.source.git.commit` tag is the commit checked out when the job
started; `git_dirty=true` means tracked files had uncommitted changes.

### Reading Runs

```python
import mlflow
from src.results import load_mlflow_run

runs = mlflow.search_runs(experiment_names=["log_penalty"],
                          filter_string="params.`model.activation` = 'relu'")
saved = load_mlflow_run(runs.loc[0, "run_id"])
```

`search_runs` returns a DataFrame with `params.*`, `metrics.*` and `tags.*`
columns. `load_mlflow_run` downloads the artifacts and restores the selected
checkpoint with its recorded data transform.

### Without The Server

A single, non-parallel run can write straight to the database:

```bash
export MLFLOW_TRACKING_URI=sqlite:///deploy/docker/mlflow-data/mlflow.db
```

Parallel sweeps must go through the server, because SQLite accepts one writer
at a time. Experiments created by the server store artifacts under the
`mlflow-artifacts:` scheme, which only the server can resolve.

### Backups

`deploy/docker/mlflow-data/` is the only copy of every result. Back up the
database and `mlartifacts/` together so runs keep their artifacts:

```bash
sqlite3 deploy/docker/mlflow-data/mlflow.db ".backup 'mlflow-backup.db'"
```

`.backup` is safe while the server runs. Copy `mlartifacts/` alongside it.

### History

Before ADR 0016, local JSON Run Records were the source of truth and
`scripts/upload_run_records_to_mlflow.py` projected them into MLflow. Earlier
still, an EC2 deployment driven by `make mlflow-deploy` / `make mlflow-backfill`
was retired by [ADR 0014](0014-retire-the-ec2-mlflow-target.md).
