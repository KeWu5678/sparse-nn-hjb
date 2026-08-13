#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

export MPLCONFIGDIR=/private/tmp/sparsenn-mpl-cache
mkdir -p "$MPLCONFIGDIR"

multirun=rawdata/logs/multirun
vdp_alg1="$multirun/vdp/paper_log_penalty/sequential"
pendulum_alg1="$multirun/pendulum/paper_log_penalty/sequential"
vdp_alg2="$multirun/vdp/paper_frac_exp_penalty/sequential"
pendulum_alg2="$multirun/pendulum/paper_frac_exp_penalty/sequential"

# Region tables use un-normalized physical coordinates from these sidecars.
.venv/bin/python scripts/investigation/rescore_region_metrics.py \
  "$pendulum_alg1" \
  "$multirun/pendulum/paper_log_penalty/oversampling"

.venv/bin/python experiments/01_vdp/paper_log_penalty/analysis.py
.venv/bin/python experiments/02_pendulum/paper_log_penalty/analysis.py

.venv/bin/python experiments/01_vdp/paper_log_penalty/p_study_figure.py

.venv/bin/python experiments/01_vdp/moment_penalty/full_scope.py \
  --records "$vdp_alg1" \
  --homogeneous-records "$vdp_alg2" \
  --traditional-records "$multirun/vdp/paper_frac_exp_penalty/relu_l1" \
  --out experiments/01_vdp/paper_log_penalty \
  --alpha 1e-4 \
  --gamma 10 \
  --order 2.01 \
  --beta 0 \
  --free-homogeneous-alpha

.venv/bin/python experiments/02_pendulum/moment_penalty/full_scope.py \
  --records-alg1 "$pendulum_alg1" \
  --records-alg2 "$pendulum_alg2" \
  --traditional-records "$multirun/pendulum/paper_frac_exp_penalty/relu_l1" \
  --out experiments/02_pendulum/paper_log_penalty \
  --operating-point 1e-4,10,2.01

echo "Algorithm 1 paper artifacts regenerated."
