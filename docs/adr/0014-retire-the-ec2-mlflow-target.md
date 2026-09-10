---
status: accepted
---

# Retire the EC2 MLflow target in favour of a local container

See [mlflow.md](mlflow.md) for day-to-day usage.

The former deployment placed the tracking server on a stoppable EC2 instance
reached over an SSM tunnel. In practice the dashboard is
read by one person on one machine, and the deployment's cost was never the
burden — the instance sits stopped between backfills, leaving a 10 GB EBS volume
at well under a dollar a month. The burden was the operating procedure: every
look at the dashboard required starting the instance, opening an SSM tunnel, and
stopping it again, and that ceremony is what kept the dashboard from being
consulted.

We therefore serve MLflow from a container on the developer machine
(`deploy/docker/compose.yaml`). A `restart: unless-stopped` policy keeps it
available across reboots without holding a terminal open, and the port is bound
to `127.0.0.1`, so the exposure story matches the SSM-only posture it replaces.
The backend store remains SQLite, now on a bind mount.

This is safe precisely because of
[ADR 0002](0002-mlflow-as-run-record-backend.md): MLflow is a projection of the
local JSON Run Records, not the system of record. Discarding the EC2 backend
store loses no experimental data — any sweep can be re-projected from
`rawdata/logs/multirun/` with `scripts/upload_run_records_to_mlflow.py`.

What is given up is the off-machine copy of the projection and the ability to
point a collaborator at a shared dashboard. If a shared server is needed later,
the same compose file runs on any Linux host, so the decision is reversible
without returning to Terraform.

Retired with this ADR: `deploy/terraform/`, `scripts/mlflow_backfill_session.sh`,
and the `mlflow-deploy` / `mlflow-backfill` Makefile targets. Kept unchanged:
`src/experiment_logging.py`, `scripts/upload_run_records_to_mlflow.py`, and the
`MLFLOW_TRACKING_URI` client contract.
