# MLflow Deployment

Self-hosted MLflow tracking server for a central comparison dashboard, run as a
container on the developer machine.

```
your machine --> container (mlflow server, 127.0.0.1:5000)
                  └─ backend store: sqlite on a bind mount
client config: MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

MLflow is the system of record for training runs: each job logs its params,
metrics, tags and artifacts (`config.json`, `normalization.json`, `history.pt`)
to this server, and nothing else keeps a copy. Training refuses to start when
the server is unreachable.

See:

- [`docs/adr/0016-mlflow-is-the-system-of-record.md`](../docs/adr/0016-mlflow-is-the-system-of-record.md)
- [`docs/adr/0014-retire-the-ec2-mlflow-target.md`](../docs/adr/0014-retire-the-ec2-mlflow-target.md)
- [`docs/adr/mlflow.md`](../docs/adr/mlflow.md) — operational guide

## Prerequisites

Docker Desktop, with *Start Docker Desktop when you sign in* enabled — the
container's restart policy cannot fire until the Docker daemon is up.

## Run

`scripts/init.sh` runs the start command below, syncs the Python environment,
and waits until the server answers. To start only the server:

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

## State And Versions

State lives in `deploy/docker/mlflow-data/` (gitignored) and survives
`docker compose down`. The SQLite database and the artifacts (`mlartifacts/`)
both use this bind mount. It is the only copy of every result and cannot be
rebuilt, so back it up; see [`docs/adr/mlflow.md`](../docs/adr/mlflow.md).

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
