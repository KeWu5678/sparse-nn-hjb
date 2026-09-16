PY := .venv/bin/python

VERBOSE ?= false
JOBS ?= 8
EXPERIMENT ?= log_penalty
# Records publish to MLflow live; env.require_mlflow makes a sweep abort at startup
# rather than run for hours with the dashboard silently off.
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
# One invocation writes a subtree per dataset; the layout is set by
# hydra.sweep.subdir in each conf/experiment/<name>.yaml.
SWEEP_DIR = rawdata/logs/multirun

.PHONY: help openloop sweep

help:
	@printf 'Usage: make <target> [VAR=value]\n\n'
	@printf 'Targets:\n'
	@printf '  %-12s %s\n' 'sweep'    'train one experiment sweep and save run records'
	@printf '  %-12s %s\n' 'openloop' 'regenerate the VDP and pendulum datasets and figures'
	@printf '\nVariables for sweep:\n'
	@printf '  %-12s %-44s %s\n' 'EXPERIMENT' 'conf/experiment/<name>.yaml' '(default: $(EXPERIMENT))'
	@printf '  %-12s %-44s %s\n' 'JOBS'       'parallel workers'            '(default: $(JOBS))'
	@printf '  %-12s %-44s %s\n' 'VERBOSE'    'stream progress to the console' '(default: $(VERBOSE))'
	@printf '\nAvailable experiments:\n'
	@ls conf/experiment/*.yaml | sed 's|.*/||; s|\.yaml$$||' | sort | sed 's/^/  /'
	@printf '\nRecords land in %s/<dataset>/<experiment>/<axis>/<job>/\n' '$(SWEEP_DIR)'
	@printf 'The <axis> level is set by hydra.sweep.subdir in each experiment file.\n'

openloop:
	$(PY) "experiments/00_openloop/vdp/generate.py"
	$(PY) "experiments/00_openloop/pendulum/generate.py"

sweep:
	@if find "$(SWEEP_DIR)" -path '*/$(EXPERIMENT)/*' -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR)/*/$(EXPERIMENT) already contains records; refusing to mix runs."; exit 2; \
	fi
	OMP_NUM_THREADS=1 MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	  $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR) \
	  env.verbose=$(VERBOSE) \
	  env.require_mlflow=true \
	  env.seed=42
