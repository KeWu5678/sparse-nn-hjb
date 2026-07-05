PY := .venv/bin/python
TF_DIR ?= deploy/terraform

# This root Makefile is the project's experiment entrypoint.
#
# Convention:
#   - Use `sweep EXPERIMENT=<name>` for curated multirun experiments.
#   - Keep the experiment's config in `conf/experiment/<name>.yaml`.
#   - Keep experiment-owned analysis/results under the `experiments/` tree.
#   - Use `scripts/train.py` only for a single Hydra-composed training run.
#
# If this file grows too large, split target bodies into included fragments such
# as `experiments/<name>/experiment.mk`; keep the root Makefile as the discoverable
# interface for humans and CI.
#
# Per-run logging. Quiet by default; pass `VERBOSE=true` to print each run's
# PDAP progress tables to the console, e.g. `make sweep VERBOSE=true`.
VERBOSE ?= false
# Parallelism for the multirun sweeps (Hydra joblib launcher). JOBS runs are
# launched at once; each is pinned to one BLAS thread (OMP_NUM_THREADS=1 in the
# recipes) so the workers don't oversubscribe the cores. Override per-invocation,
# e.g. `make sweep JOBS=10`; JOBS=1 is effectively serial.
JOBS ?= 8
# Experiments are named <problem>/<model-family>, matching conf/experiment/:
#   vdp/log_penalty, vdp/frac_exp_penalty,
#   pendulum/log_penalty, pendulum/frac_exp_penalty
# Each config pins its own data, so there is no DATA variable.
EXPERIMENT ?= vdp/log_penalty
# Curated sweeps use convention-based paths:
#   results/analysis: experiments/<numbered problem dir>/<model family>
#   run records:      rawdata/logs/multirun/<problem>/<model family>
PROBLEM_DIR_vdp = 01_vdp
PROBLEM_DIR_pendulum = 02_pendulum
ANALYSIS_DIR = experiments/$(PROBLEM_DIR_$(patsubst %/,%,$(dir $(EXPERIMENT))))/$(notdir $(EXPERIMENT))
SWEEP_DIR = rawdata/logs/multirun/$(EXPERIMENT)
# MLflow backfill publishes local Run Records to the EC2 dashboard. By default it
# uploads the current experiment/dataset sweep directory, keeps only the latest
# full Hydra sweep, and stops the EC2 instance when done.
MLFLOW_RECORDS ?= $(SWEEP_DIR)
MLFLOW_LATEST ?= true
MLFLOW_DRY_RUN ?= false
MLFLOW_STOP_AFTER ?= true
.PHONY: help openloop sweep region-split paper-figures mlflow-deploy mlflow-backfill

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  %-20s %s\n", $$1, $$2}'
	@printf "\n  %-20s %s\n" "VERBOSE=true" "also stream PDAP tables to console (always in per-run run.log)"
	@printf "  %-20s %s\n" "JOBS=N" "parallel sweep workers (default 8)"
	@printf "  %-20s %s\n" "EXPERIMENT=name" "sweep to run: {vdp,pendulum}/{log_penalty,frac_exp_penalty}"
	@printf "  %-20s %s\n" "MLFLOW_RECORDS=PATH" "JSON record file/dir for backfill (default current sweep dir)"
	@printf "  %-20s %s\n" "MLFLOW_LATEST=false" "backfill all records under MLFLOW_RECORDS instead of the latest full sweep"
	@printf "  %-20s %s\n" "MLFLOW_DRY_RUN=true" "preview MLflow backfill records without starting EC2 or uploading"

paper-figures:  ## refresh papar/plot/ from curated experiment figures (matched by basename; ambiguous names pinned below)
	@for f in papar/plot/*.png; do \
	  b=$$(basename "$$f"); \
	  case "$$b" in \
	    value_surface_softplus.png|value_surface_gaussian.png) \
	      src="experiments/01_vdp/log_penalty/figures/$$b";; \
	    value_surface_p2.png|value_surface_p3.png|value_surface_p5.png) \
	      src="experiments/01_vdp/frac_exp_penalty/figures/$$b";; \
	    frontier.png) \
	      src="experiments/01_vdp/summary/figures/$$b";; \
	    pendulum_insertion_frontier.png) \
	      src="experiments/02_pendulum/region_split/figures/frontier.png";; \
	    *) src=$$(find experiments -path "*/figures/$$b");; \
	  esac; \
	  n=$$(printf '%s\n' "$$src" | grep -c '[^ ]' || true); \
	  if [ "$$n" -gt 1 ]; then echo "  AMBIGUOUS $$b — pin it in the paper-figures recipe:"; printf '%s\n' "$$src" | sed 's/^/    /'; \
	  elif [ -n "$$src" ]; then cp "$$src" "$$f" && echo "  $$src -> $$f"; \
	  else echo "  (no experiment source for $$b — left as-is)"; fi; \
	done

mlflow-deploy:  ## provision/update EC2 MLflow tracking server with Terraform
	terraform -chdir=$(TF_DIR) init
	terraform -chdir=$(TF_DIR) apply

mlflow-backfill:  ## start EC2, upload local Run Records to MLflow, then stop EC2
	@PY=$(PY) TF_DIR=$(TF_DIR) MLFLOW_LATEST=$(MLFLOW_LATEST) MLFLOW_DRY_RUN=$(MLFLOW_DRY_RUN) MLFLOW_STOP_AFTER=$(MLFLOW_STOP_AFTER) \
	  bash scripts/mlflow_backfill_session.sh $(MLFLOW_RECORDS)

openloop:  ## regenerate the centralized open-loop data figures (vdp + pendulum) into experiments/00_openloop
	$(PY) "experiments/00_openloop/vdp/generate.py"
	$(PY) "experiments/00_openloop/pendulum/generate.py"

sweep:  ## run Hydra multirun EXPERIMENT ({vdp,pendulum}/{log_penalty,frac_exp_penalty}), then regenerate its results.md
	@test -f "$(ANALYSIS_DIR)/analysis.py" || { \
	  echo "Unsupported EXPERIMENT=$(EXPERIMENT)."; \
	  echo "Supported: vdp/log_penalty, vdp/frac_exp_penalty, pendulum/log_penalty, pendulum/frac_exp_penalty."; \
	  exit 2; \
	}
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR) \
	  env.verbose=$(VERBOSE) \
	  env.seed=42
	$(PY) "$(ANALYSIS_DIR)/analysis.py"

region-split:  ## regenerate the pendulum region-split analysis (no sweep of its own; reads the pendulum/* records)
	$(PY) experiments/02_pendulum/region_split/analysis.py
