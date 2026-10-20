#!/usr/bin/env bash
# Prepare this machine to run experiments: the Python environment and the
# local MLflow tracking server. Safe to re-run.
#
#   scripts/init.sh
set -euo pipefail

cd "$(dirname "$0")/.."
MLFLOW_URL=http://127.0.0.1:5000
COMPOSE_FILE=deploy/docker/compose.yaml

fail() {
  printf 'init: %s\n' "$1" >&2
  exit 1
}

command -v uv >/dev/null || fail "uv is not installed: https://docs.astral.sh/uv/"
command -v docker >/dev/null || fail "Docker is not installed; install Docker Desktop."
docker info >/dev/null 2>&1 || fail "Docker is not running; start Docker Desktop and retry."

echo "init: syncing the Python environment"
uv sync --extra dev

echo "init: starting the MLflow server"
docker compose -f "$COMPOSE_FILE" up -d

echo "init: waiting for MLflow at $MLFLOW_URL"
for _ in $(seq 60); do
  if curl -fsS "$MLFLOW_URL/health" >/dev/null 2>&1; then
    echo "init: ready; dashboard at $MLFLOW_URL"
    exit 0
  fi
  sleep 1
done
fail "MLflow did not answer within 60 s; inspect: docker compose -f $COMPOSE_FILE logs mlflow"
