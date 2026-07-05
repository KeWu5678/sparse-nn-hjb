#!/usr/bin/env bash
set -euo pipefail

tf_dir="${TF_DIR:-deploy/terraform}"
py="${PY:-.venv/bin/python}"
latest="${MLFLOW_LATEST:-true}"
dry_run="${MLFLOW_DRY_RUN:-false}"
stop_after="${MLFLOW_STOP_AFTER:-true}"

records=("$@")
if [ "${#records[@]}" -eq 0 ]; then
  records=("rawdata/logs/multirun")
fi

is_true() {
  case "$1" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

flags=()
if is_true "$latest"; then
  flags+=(--latest-run)
fi

if is_true "$dry_run"; then
  flags+=(--dry-run)
  exec "$py" scripts/upload_run_records_to_mlflow.py "${flags[@]}" "${records[@]}"
fi

instance_id=""
tunnel_pid=""
tunnel_log="${MLFLOW_TUNNEL_LOG:-${TMPDIR:-/tmp}/sparsehjb-mlflow-ssm-tunnel.log}"

cleanup() {
  status=$?
  if [ -n "$tunnel_pid" ]; then
    kill "$tunnel_pid" >/dev/null 2>&1 || true
    wait "$tunnel_pid" >/dev/null 2>&1 || true
  fi
  if is_true "$stop_after" && [ -n "$instance_id" ]; then
    echo "Stopping MLflow EC2 instance $instance_id"
    aws ec2 stop-instances --instance-ids "$instance_id" >/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

instance_id="$(terraform -chdir="$tf_dir" output -raw instance_id)"
port="$(terraform -chdir="$tf_dir" output -raw mlflow_port)"
tracking_uri="http://127.0.0.1:$port"

state="$(aws ec2 describe-instances \
  --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"

if [ "$state" != "running" ]; then
  echo "Starting MLflow EC2 instance $instance_id"
  aws ec2 start-instances --instance-ids "$instance_id" >/dev/null
fi
aws ec2 wait instance-running --instance-ids "$instance_id"

echo "Waiting for SSM agent on $instance_id"
ping_status=""
for _ in {1..60}; do
  ping_status="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$instance_id" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [ "$ping_status" = "Online" ]; then
    break
  fi
  sleep 5
done
if [ "$ping_status" != "Online" ]; then
  echo "SSM agent did not become Online for $instance_id" >&2
  exit 1
fi

tunnel_cmd="$(terraform -chdir="$tf_dir" output -raw ssm_port_forward_command)"
echo "Opening SSM tunnel to $tracking_uri"
( eval "$tunnel_cmd" ) >"$tunnel_log" 2>&1 &
tunnel_pid=$!

for _ in {1..60}; do
  if curl -fsS "$tracking_uri/version" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    echo "SSM tunnel exited early. Last tunnel log lines:" >&2
    tail -40 "$tunnel_log" >&2 || true
    exit 1
  fi
  sleep 2
done
if ! curl -fsS "$tracking_uri/version" >/dev/null 2>&1; then
  echo "MLflow server was not reachable at $tracking_uri. Last tunnel log lines:" >&2
  tail -40 "$tunnel_log" >&2 || true
  exit 1
fi

"$py" scripts/upload_run_records_to_mlflow.py \
  --tracking-uri "$tracking_uri" \
  "${flags[@]}" \
  "${records[@]}"
