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
vdp_l1="$multirun/vdp/paper_frac_exp_penalty/relu_l1"
pendulum_l1="$multirun/pendulum/paper_frac_exp_penalty/relu_l1"
pendulum_oversampling_alg1="$multirun/pendulum/paper_log_penalty/oversampling"
pendulum_oversampling_alg2="$multirun/pendulum/paper_frac_exp_penalty/oversampling"

# Validate every source grid before writing sidecars or manuscript artifacts.
.venv/bin/python scripts/paper/preflight.py

# Region tables use physical coordinates from mandatory per-run sidecars.
.venv/bin/python scripts/paper/rescore_regions.py \
  --force \
  "$pendulum_alg1" \
  "$pendulum_alg2"

# Refuse to render if rescoring left even one current record without a sidecar.
.venv/bin/python scripts/paper/preflight.py --require-sidecars

.venv/bin/python experiments/01_vdp/paper_log_penalty/analysis.py
.venv/bin/python experiments/02_pendulum/paper_log_penalty/analysis.py

.venv/bin/python experiments/01_vdp/paper_log_penalty/p_study_figure.py

.venv/bin/python scripts/paper/vdp_full_scope.py \
  --records "$vdp_alg1" \
  --homogeneous-records "$vdp_alg2" \
  --traditional-records "$vdp_l1" \
  --out experiments/01_vdp/paper_log_penalty \
  --alpha 1e-4 \
  --gamma 10 \
  --order 2.01 \
  --free-homogeneous-alpha

.venv/bin/python scripts/paper/pendulum_full_scope.py \
  --records-alg1 "$pendulum_alg1" \
  --oversampling-alg1-records "$pendulum_oversampling_alg1" \
  --oversampling-alg2-records "$pendulum_oversampling_alg2" \
  --records-alg2 "$pendulum_alg2" \
  --traditional-records "$pendulum_l1" \
  --out experiments/02_pendulum/paper_log_penalty \
  --operating-point 1e-4,10,2.01

echo "Algorithm 1 paper artifacts regenerated."
