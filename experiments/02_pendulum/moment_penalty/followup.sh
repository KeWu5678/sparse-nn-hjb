#!/usr/bin/env bash
set -euo pipefail

analysis_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$analysis_dir/../../.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
record_root="$repo_root/rawdata/logs/multirun/pendulum/moment_penalty/followup"
jobs="${JOBS:-8}"
verbose="${VERBOSE:-false}"

if find "$record_root" -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "$record_root already contains records; refusing to overwrite or duplicate them."
  exit 2
fi
mkdir -p "$record_root"

run_stage() {
  local stage="$1"
  local activation="$2"
  local alpha="$3"
  local beta="$4"
  local order="$5"
  local loss_weights="$6"
  local gamma="$7"

  OMP_NUM_THREADS=1 "$python_bin" "$repo_root/scripts/train.py" -m \
    +experiment=pendulum/moment_penalty \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$jobs" \
    "hydra.sweep.dir=$record_root/$stage" \
    "env.verbose=$verbose" env.seed=42 \
    "model.activation=$activation" "model.alpha=$alpha" \
    "model.moment_beta=$beta" "model.moment_order=$order" \
    "model.loss_weights=$loss_weights" "model.gamma=$gamma"
}

# H1/gamma=1 is reused from the screen or adaptive refinement.
run_stage tanh_h1 tanh 1e-4 1e-5 3 '[1.0,1.0]' 0,0.1,10
run_stage softplus_h1 softplus 1e-4 1e-10 2.01 '[1.0,1.0]' 0,0.1,10
run_stage gaussian_h1 gaussian 1e-4 1e-4 2.01 '[1.0,1.0]' 0,0.1,10
run_stage gelu_h1 gelu_squared 1e-5 1e-10 2.01 '[1.0,1.0]' 0,0.1,10
run_stage matern_h1 matern52 1e-5 1e-7 2.01 '[1.0,1.0]' 0,0.1,10

run_stage tanh_l2 tanh 1e-4 1e-5 3 '[1.0,0.0]' 0,0.1,1,10
run_stage softplus_l2 softplus 1e-4 1e-10 2.01 '[1.0,0.0]' 0,0.1,1,10
run_stage gaussian_l2 gaussian 1e-4 1e-4 2.01 '[1.0,0.0]' 0,0.1,1,10
run_stage gelu_l2 gelu_squared 1e-5 1e-10 2.01 '[1.0,0.0]' 0,0.1,1,10
run_stage matern_l2 matern52 1e-5 1e-7 2.01 '[1.0,0.0]' 0,0.1,1,10

"$python_bin" "$analysis_dir/followup.py"
