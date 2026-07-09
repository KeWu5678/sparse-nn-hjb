# MLflow Pipeline

Operational guide for experiment tracking (EC2/Terraform deployment per ADR
0001/0002).

`ExperimentRun` always writes a local JSON Run Record in the Hydra output
directory. MLflow is normally filled by **backfilling** those completed local
records after a sweep, so training does not depend on the EC2 server being up.
MLflow stores params, scalar metrics, status, Hydra metadata, and local artifact
paths; it does not upload the `result_<run_id>.pkl` artifact in the current
pipeline.

Provision the EC2/Terraform deployment once:

```bash
make mlflow-deploy
```

### Normal Workflow

Run experiments normally. Local Run Records are written whether MLflow exists or
not:

```bash
make sweep EXPERIMENT=activationsearch DATA=pendulum
```

Backfill the latest full sweep into the EC2 MLflow dashboard:

```bash
make mlflow-backfill EXPERIMENT=activationsearch DATA=pendulum
```

`mlflow-backfill` starts the EC2 instance, opens an SSM tunnel, uploads local Run
Records, and stops the instance in a cleanup trap. By default it uploads the
latest full Hydra sweep under the current `EXPERIMENT`/`DATA` log directory.

Preview without starting EC2 or uploading:

```bash
make mlflow-backfill EXPERIMENT=activationsearch DATA=pendulum MLFLOW_DRY_RUN=true
```

Limit or broaden the source records:

```bash
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch/1069
make mlflow-backfill MLFLOW_RECORDS=rawdata/logs/multirun/activationsearch MLFLOW_LATEST=false
```

The importer reads each Run Record JSON and, for older records, enriches it with
adjacent `.hydra/overrides.yaml` metadata before uploading. Re-running the
importer creates additional MLflow runs for the same JSON; deduplication is not
implemented yet.

Project run IDs use:

```text
{experiment_name}_{data_choice}_{YYYYMMDD}_{4hex}
```

### Optional Live Logging

Live logging is still available, but it is not the default workflow. Start the
server and tunnel manually, then set `MLFLOW_TRACKING_URI` before training:

```bash
aws ec2 start-instances --instance-ids "$(terraform -chdir=deploy/terraform output -raw instance_id)"
eval "$(terraform -chdir=deploy/terraform output -raw ssm_port_forward_command)"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make sweep EXPERIMENT=activationsearch DATA=pendulum
aws ec2 stop-instances --instance-ids "$(terraform -chdir=deploy/terraform output -raw instance_id)"
```

At `run.finish()`, the script writes the local JSON Run Record, keeps it on disk,
and publishes dashboard data to MLflow. If `MLFLOW_TRACKING_URI` is unset, the
run is local-only.

For a purely local server instead:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then point at that local server and use the lower-level uploader:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
.venv/bin/python scripts/upload_run_records_to_mlflow.py rawdata/logs/multirun/activationsearch
```
