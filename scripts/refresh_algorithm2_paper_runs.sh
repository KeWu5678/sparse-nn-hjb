#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

python_cmd=.venv/bin/python
jobs=${JOBS:-8}
verbose=${VERBOSE:-false}
seq_iterations=${SEQ_ITERATIONS:-150}
stage=rawdata/logs/multirun/.algorithm2_global_prox_stage
archive=rawdata/logs/archive/algorithm2_pre_global_prox_20260817
canonical=rawdata/logs/multirun

if [[ -e "$stage" ]]; then
  echo "$stage already exists; refusing to mix or overwrite staged runs."
  exit 2
fi
if [[ -e "$archive" ]]; then
  echo "$archive already exists; refusing to overwrite the recoverable archive."
  exit 2
fi

for problem in vdp pendulum; do
  for mode in batch sequential; do
    if [[ "$mode" == batch ]]; then
      iterations=10
    else
      iterations=$seq_iterations
    fi
    out="$stage/$problem/paper_frac_exp_penalty/$mode"
    OMP_NUM_THREADS=1 "$python_cmd" scripts/train.py -m \
      +experiment="$problem/paper_frac_exp_penalty" \
      hydra/launcher=joblib hydra.launcher.n_jobs="$jobs" \
      hydra.sweep.dir="$out" \
      env.verbose="$verbose" env.seed=42 \
      training.insert_mode="$mode" training.num_iterations="$iterations"
  done

  out="$stage/$problem/paper_frac_exp_penalty/relu_l1"
  OMP_NUM_THREADS=1 "$python_cmd" scripts/train.py -m \
    +experiment="$problem/paper_frac_exp_penalty" \
    hydra/launcher=joblib hydra.launcher.n_jobs="$jobs" \
    hydra.sweep.dir="$out" \
    env.verbose="$verbose" env.seed=42 \
    training.insert_mode=sequential training.num_iterations="$seq_iterations" \
    model.activation=relu model.power=1 model.gamma=0 \
    model.alpha=1e-1,1e-2,1e-3,1e-4,1e-5,1e-6 \
    model.loss_weights='[1.0,1.0]'
done

for variant in base6k band40 band60 add2k; do
  out="$stage/pendulum/paper_frac_exp_penalty/oversampling/$variant"
  OMP_NUM_THREADS=1 "$python_cmd" scripts/train.py -m \
    +experiment=pendulum/paper_frac_exp_penalty \
    hydra/launcher=joblib hydra.launcher.n_jobs="$jobs" \
    hydra.sweep.dir="$out" \
    env.verbose="$verbose" env.seed=42 \
    training.insert_mode=sequential training.num_iterations="$seq_iterations" \
    model.activation=relu model.power=2 model.gamma=0 \
    model.alpha=1e-4,1e-5,1e-6 model.loss_weights='[1.0,1.0]' \
    data.path="Pendulum_2sided_oversample_20260704/$variant.npz" \
    eval.distance_cache="Pendulum_2sided_oversample_20260704/${variant}_region_distances.npz"
done

"$python_cmd" scripts/paper/preflight.py --algorithm2-root "$stage"

mkdir -p "$archive/vdp" "$archive/pendulum"
for problem in vdp pendulum; do
  mv "$canonical/$problem/paper_frac_exp_penalty" \
    "$archive/$problem/paper_frac_exp_penalty"
  mv "$stage/$problem/paper_frac_exp_penalty" \
    "$canonical/$problem/paper_frac_exp_penalty"
done

"$python_cmd" scripts/paper/rescore_regions.py --force \
  "$canonical/pendulum/paper_frac_exp_penalty/sequential"
"$python_cmd" scripts/paper/preflight.py \
  --algorithm2-root "$canonical" --require-sidecars

echo "Algorithm 2 records replaced; previous records archived under $archive."
