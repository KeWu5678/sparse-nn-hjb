# MLflow Deployment

Self-hosted MLflow tracking server for a central comparison dashboard, run as a
container on the developer machine.

```
your machine --> container (mlflow server, 127.0.0.1:5000)
                  └─ backend store: sqlite on a bind mount
client config: MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

MLflow is a *projection* of the local JSON Run Records, not the system of
record. Run Records under `rawdata/logs/multirun/` are the source of truth, and
any sweep can be re-projected at any time, so the dashboard's SQLite store is
disposable.

The application logs params, scalar metrics, tags, and local artifact paths. It
does not upload `result_<run_id>.pkl` or JSON records as MLflow artifacts.

See:

- [`docs/adr/0002-mlflow-as-run-record-backend.md`](../docs/adr/0002-mlflow-as-run-record-backend.md)
- [`docs/adr/0014-retire-the-ec2-mlflow-target.md`](../docs/adr/0014-retire-the-ec2-mlflow-target.md)
- [`docs/adr/mlflow.md`](../docs/adr/mlflow.md) — operational guide

## Prerequisites

Docker Desktop, with *Start Docker Desktop when you sign in* enabled — the
container's restart policy cannot fire until the Docker daemon is up.

## Run

```bash
docker compose -f deploy/docker/compose.yaml up -d
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`restart: unless-stopped` brings the server back after a reboot or a Docker
Desktop restart, so nothing has to stay in a terminal. The port is bound to
`127.0.0.1`, not `0.0.0.0` — the dashboard is not reachable from the LAN.

Stop it with:

```bash
docker compose -f deploy/docker/compose.yaml down
```

## Publish Runs

Run experiments normally; local Run Records are written whether the dashboard is
up or not. Then project them into MLflow:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
.venv/bin/python scripts/upload_run_records_to_mlflow.py \
  rawdata/logs/multirun/vdp/log_penalty
```

Re-running the importer creates additional MLflow runs for the same JSON;
deduplication is not implemented.

## State And Versions

State lives in `deploy/docker/mlflow-data/` (gitignored) and survives
`docker compose down`. Both the SQLite database and any server-uploaded artifacts
(`mlartifacts/`) use this bind mount; the current importer still uploads only
metadata. Re-importing Run Records restores the dashboard metadata, not uploaded
artifact bytes, so preserve those files if artifact uploads are used.

The image tag in `compose.yaml` is pinned; keep it aligned with the resolved
`mlflow` version in `uv.lock`.

## History

An earlier target ran the server on a stoppable EC2 instance reached over an SSM
tunnel. It was
retired by [ADR 0014](../docs/adr/0014-retire-the-ec2-mlflow-target.md); the
`mlflow-deploy` / `mlflow-backfill` Makefile targets and
`scripts/mlflow_backfill_session.sh` went with it.

Docker Compose is the supported deployment; Terraform is no longer required.
The former Terraform configuration and local state are archived on this machine
under `outdated/terraform/` (gitignored) for any remaining AWS cleanup. Removing
the configuration from the repository does not destroy AWS resources.
