PY := .venv/bin/python

VERBOSE ?= false
JOBS ?= 8
EXPERIMENT ?= log_penalty
# One invocation writes a subtree per dataset; the layout is set by
# hydra.sweep.subdir in each conf/experiment/<name>.yaml.
SWEEP_DIR = rawdata/logs/multirun

.PHONY: help openloop sweep

help:
	@printf '%s\n' \
	  'make sweep [EXPERIMENT=log_penalty] [JOBS=8] [VERBOSE=false] — train and save records' \
	  '  EXPERIMENT selects conf/experiment/<name>.yaml; JOBS sets parallel workers.' \
	  '  Records land in rawdata/logs/multirun/<data>/<EXPERIMENT>/<job>/.' \
	  'make openloop — regenerate VDP and pendulum data figures'

openloop:
	$(PY) "experiments/00_openloop/vdp/generate.py"
	$(PY) "experiments/00_openloop/pendulum/generate.py"

sweep:
	@if find "$(SWEEP_DIR)" -path '*/$(EXPERIMENT)/*' -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR)/*/$(EXPERIMENT) already contains records; refusing to mix runs."; exit 2; \
	fi
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR) \
	  env.verbose=$(VERBOSE) \
	  env.seed=42
