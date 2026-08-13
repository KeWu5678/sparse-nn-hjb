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
#   vdp/log_penalty, vdp/frac_exp_penalty, vdp/moment_penalty,
#   pendulum/log_penalty, pendulum/frac_exp_penalty,
#   pendulum/moment_penalty
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
# Paper-conforming sweeps (paper/paper_0805.tex Algorithms 1 and 2) run each cell
# in both insertion modes, into separate record subdirectories so one analysis can
# compare them.  Sequential admits one atom per outer iteration, so it needs a
# matched neuron budget rather than a matched iteration count (docs/adr/0008):
# SEQ_ITERATIONS defaults to T_out * N_ins = 10 * 15, the batch cap.
PAPER_MODE ?= both
SEQ_ITERATIONS ?= 150

.PHONY: help openloop sweep paper-sweep paper-p-study moment-sweep moment-refine moment-followup moment-oversampling region-split paper-figures mlflow-deploy mlflow-backfill

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  %-20s %s\n", $$1, $$2}'
	@printf "\n  %-20s %s\n" "VERBOSE=true" "also stream PDAP tables to console (always in per-run run.log)"
	@printf "  %-20s %s\n" "JOBS=N" "parallel sweep workers (default 8)"
	@printf "  %-20s %s\n" "EXPERIMENT=name" "sweep to run: {vdp,pendulum}/{log_penalty,frac_exp_penalty,moment_penalty,paper_log_penalty,paper_frac_exp_penalty}"
	@printf "  %-20s %s\n" "PAPER_MODE=m" "paper-sweep insertion mode: batch|sequential|both (default both)"
	@printf "  %-20s %s\n" "SEQ_ITERATIONS=N" "T_out for sequential insertion (default 150 = batch neuron budget)"
	@printf "  %-20s %s\n" "MLFLOW_RECORDS=PATH" "JSON record file/dir for backfill (default current sweep dir)"
	@printf "  %-20s %s\n" "MLFLOW_LATEST=false" "backfill all records under MLFLOW_RECORDS instead of the latest full sweep"
	@printf "  %-20s %s\n" "MLFLOW_DRY_RUN=true" "preview MLflow backfill records without starting EC2 or uploading"

paper-figures:  ## refresh paper/plot/ from curated experiment figures (matched by basename; ambiguous names pinned below)
	@for f in paper/plot/*.png; do \
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

paper-sweep:  ## run a paper-conforming sweep ({vdp,pendulum}/paper_{log,frac_exp}_penalty) in both insertion modes
	@case "$(EXPERIMENT)" in \
	  */paper_log_penalty|*/paper_frac_exp_penalty) ;; \
	  *) echo "Use EXPERIMENT={vdp,pendulum}/paper_{log,frac_exp}_penalty."; exit 2 ;; \
	esac
	@test -f "$(ANALYSIS_DIR)/analysis.py" || { \
	  echo "Missing $(ANALYSIS_DIR)/analysis.py, which this target runs once the"; \
	  echo "sweep finishes.  Per-study analysis.py files are gitignored, so a clean"; \
	  echo "checkout does not carry them.  Checked here rather than after the"; \
	  echo "sweep, which costs hours."; \
	  exit 2; \
	}
	@case "$(PAPER_MODE)" in \
	  batch|both) \
	    echo "== batch insertion (Algorithm as printed, N_ins=$${NINS:-15}) =="; \
	    OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	      hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	      hydra.sweep.dir=$(SWEEP_DIR)/batch \
	      env.verbose=$(VERBOSE) env.seed=42 \
	      training.insert_mode=batch ;; \
	esac
	@case "$(PAPER_MODE)" in \
	  sequential|both) \
	    echo "== sequential insertion (one atom per iteration, T_out=$(SEQ_ITERATIONS)) =="; \
	    OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	      hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	      hydra.sweep.dir=$(SWEEP_DIR)/sequential \
	      env.verbose=$(VERBOSE) env.seed=42 \
	      training.insert_mode=sequential \
	      training.num_iterations=$(SEQ_ITERATIONS) ;; \
	esac
	$(PY) "$(ANALYSIS_DIR)/analysis.py"

paper-p-study:  ## sweep the moment order p, which is live under the normalized objective even at beta=0
	@case "$(EXPERIMENT)" in \
	  */paper_log_penalty) ;; \
	  *) echo "Use EXPERIMENT={vdp,pendulum}/paper_log_penalty (p is inert on the sphere)."; exit 2 ;; \
	esac
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/p_study_sequential \
	  env.verbose=$(VERBOSE) env.seed=42 \
	  training.insert_mode=sequential \
	  training.num_iterations=$(SEQ_ITERATIONS) \
	  model.activation=softplus,tanh,gaussian,gelu_squared \
	  model.moment_order=2.01,2.5,3,4 \
	  model.alpha=1e-4,1e-5 \
	  model.gamma=1 model.loss_weights='[1.0,1.0]'
	$(PY) "$(ANALYSIS_DIR)/analysis.py"

moment-sweep:  ## run the seed-42 moment screen; beta=0 baseline is deduplicated across p
	@case "$(EXPERIMENT)" in \
	  vdp/moment_penalty|pendulum/moment_penalty) ;; \
	  *) echo "Use EXPERIMENT=vdp/moment_penalty or pendulum/moment_penalty."; exit 2 ;; \
	esac
	@$(PY) -c "from src.config.schema import ModelConfig; assert 'moment_beta' in ModelConfig.__dataclass_fields__, 'merge or check out the moment-penalty implementation before running this sweep'"
	@test -f "$(ANALYSIS_DIR)/analysis.py" || { \
	  echo "Missing $(ANALYSIS_DIR)/analysis.py, which this target runs on the"; \
	  echo "records once the sweep finishes.  Per-study analysis.py files are"; \
	  echo "gitignored (see .gitignore), so a clean checkout does not carry them."; \
	  echo "Checked here rather than after the sweep, which costs hours."; \
	  exit 2; \
	}
	@if find "$(SWEEP_DIR)/baseline" "$(SWEEP_DIR)/screen" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR) already contains first-pass records; refusing to overwrite or duplicate them."; \
	  exit 2; \
	fi
	@mkdir -p "$(SWEEP_DIR)/baseline" "$(SWEEP_DIR)/screen"
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/baseline \
	  env.verbose=$(VERBOSE) env.seed=42 \
	  model.activation=tanh,softplus,gaussian,gelu_squared \
	  model.alpha=1e-2,1e-3,1e-4,1e-5 \
	  model.gamma=1 model.loss_weights='[1.0,1.0]' \
	  model.moment_beta=0 model.moment_order=2.01
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/screen \
	  env.verbose=$(VERBOSE) env.seed=42 \
	  model.activation=tanh,softplus,gaussian,gelu_squared \
	  model.alpha=1e-2,1e-3,1e-4,1e-5 \
	  model.gamma=1 model.loss_weights='[1.0,1.0]' \
	  model.moment_beta=1e-10,1e-5,1e-2,1e-1 \
	  model.moment_order=2.01,2.5,3,4
	$(PY) "$(ANALYSIS_DIR)/analysis.py"

moment-refine:  ## fill selected beta gaps and add Matern-5/2 at seed 42
	@case "$(EXPERIMENT)" in \
	  vdp/moment_penalty|pendulum/moment_penalty) ;; \
	  *) echo "Use EXPERIMENT=vdp/moment_penalty or pendulum/moment_penalty."; exit 2 ;; \
	esac
	@if find "$(SWEEP_DIR)/refine" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR)/refine already contains records; refusing to overwrite or duplicate them."; \
	  exit 2; \
	fi
	@mkdir -p "$(SWEEP_DIR)/refine"
	@set -e; if [ "$(EXPERIMENT)" = "vdp/moment_penalty" ]; then \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/tanh_early \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=tanh \
	    model.alpha=1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-9,1e-8,1e-7,1e-6 model.moment_order=2.01; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/softplus_early \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=softplus \
	    model.alpha=1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-9,1e-8,1e-7,1e-6 model.moment_order=2.01; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/gaussian_late \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=gaussian \
	    model.alpha=1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-4,1e-3 model.moment_order=3; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/gelu_late \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=gelu_squared \
	    model.alpha=1e-3 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-4,1e-3 model.moment_order=2.01; \
	else \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/softplus_early \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=softplus \
	    model.alpha=1e-4 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-9,1e-8,1e-7,1e-6 model.moment_order=2.01; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/gelu_early \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=gelu_squared \
	    model.alpha=1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-9,1e-8,1e-7,1e-6 model.moment_order=2.01; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/tanh_late \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=tanh \
	    model.alpha=1e-4 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-4,1e-3 model.moment_order=3; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/refine/gaussian_late \
	    env.verbose=$(VERBOSE) env.seed=42 model.activation=gaussian \
	    model.alpha=1e-4 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-4,1e-3 model.moment_order=2.01; \
	fi
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/refine/matern_baseline \
	  env.verbose=$(VERBOSE) env.seed=42 model.activation=matern52 \
	  model.alpha=1e-4,1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	  model.moment_beta=0 model.moment_order=2.01
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/refine/matern_positive \
	  env.verbose=$(VERBOSE) env.seed=42 model.activation=matern52 \
	  model.alpha=1e-4,1e-5 model.gamma=1 model.loss_weights='[1.0,1.0]' \
	  model.moment_beta=1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1 \
	  model.moment_order=2.01
	$(PY) "$(ANALYSIS_DIR)/refinement.py"

moment-followup:  ## compare gamma and value-only/H1 losses at selected moment cells
	@case "$(EXPERIMENT)" in \
	  vdp/moment_penalty|pendulum/moment_penalty) ;; \
	  *) echo "Use EXPERIMENT=vdp/moment_penalty or pendulum/moment_penalty."; exit 2 ;; \
	esac
	JOBS=$(JOBS) VERBOSE=$(VERBOSE) bash "$(ANALYSIS_DIR)/followup.sh"

moment-oversampling:  ## rerun the pendulum Gaussian arm on the four switching-band datasets
	@test "$(EXPERIMENT)" = "pendulum/moment_penalty" || { \
	  echo "Use EXPERIMENT=pendulum/moment_penalty."; exit 2; \
	}
	@if find "$(SWEEP_DIR)/oversampling" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR)/oversampling already contains records; refusing to overwrite or duplicate them."; \
	  exit 2; \
	fi
	@set -e; for variant in base6k band40 band60 add2k; do \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$(SWEEP_DIR)/oversampling/$$variant \
	    env.verbose=$(VERBOSE) env.seed=42 \
	    data.path=Pendulum_2sided_oversample_20260704/$$variant.npz \
	    eval.distance_cache=Pendulum_2sided_oversample_20260704/$${variant}_region_distances.npz \
	    model.kind=signed model.insertion=profile model.activation=gaussian \
	    model.alpha=1e-3,1e-4,1e-5 model.gamma=0 \
	    model.loss_weights='[1.0,1.0]' \
	    model.moment_beta=1e-4 model.moment_order=2.01; \
	done

region-split:  ## regenerate the pendulum region-split analysis (no sweep of its own; reads the pendulum/* records)
	$(PY) experiments/02_pendulum/region_split/analysis.py
