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

.PHONY: help openloop sweep paper-sweep paper-p-study paper-radius-study paper-oversampling-study paper-l1-study paper-algorithm2-refresh paper-artifacts moment-sweep moment-refine moment-followup moment-oversampling region-split paper-figures mlflow-deploy mlflow-backfill

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
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
	    *) src=$$(find experiments -path "*/figures/$$b");; \
	  esac; \
	  n=$$(printf '%s\n' "$$src" | grep -c '[^ ]' || true); \
	  if [ "$$n" -gt 1 ]; then echo "  AMBIGUOUS $$b — pin it in the paper-figures recipe:"; printf '%s\n' "$$src" | sed 's/^/    /'; \
	  elif [ -n "$$src" ]; then cp "$$src" "$$f" && echo "  $$src -> $$f"; \
	  else echo "  (no experiment source for $$b — left as-is)"; fi; \
	done

paper-artifacts:  ## regenerate current manuscript analyses and figures from validated records
	./scripts/regenerate_paper_artifacts.sh

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
	@if find "$(SWEEP_DIR)" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "$(SWEEP_DIR) already contains records; refusing to mix runs."; exit 2; \
	fi
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR) \
	  env.verbose=$(VERBOSE) \
	  env.seed=42
	$(PY) "$(ANALYSIS_DIR)/analysis.py"

paper-sweep:  ## run an Algorithm 1 paper sweep ({vdp,pendulum}/paper_log_penalty) in both insertion modes
	@case "$(EXPERIMENT)" in \
	  */paper_log_penalty) ;; \
	  *) echo "Use EXPERIMENT={vdp,pendulum}/paper_log_penalty; use paper-algorithm2-refresh for Algorithm 2."; exit 2 ;; \
	esac
	@case "$(PAPER_MODE)" in batch|sequential|both) ;; \
	  *) echo "Use PAPER_MODE=batch, sequential, or both."; exit 2 ;; \
	esac
	@test -f "$(ANALYSIS_DIR)/analysis.py" || { \
	  echo "Missing tracked analysis entry point $(ANALYSIS_DIR)/analysis.py."; \
	  exit 2; \
	}
	@for mode in batch sequential; do \
	  case "$(PAPER_MODE)" in $$mode|both) \
	    if find "$(SWEEP_DIR)/$$mode" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	      echo "$(SWEEP_DIR)/$$mode already contains records; refusing to mix runs."; exit 2; \
	    fi ;; \
	  esac; \
	done
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

paper-p-study:  ## sweep the moment order p in Algorithm 1's normalized objective
	@case "$(EXPERIMENT)" in \
	  */paper_log_penalty) ;; \
	  *) echo "Use EXPERIMENT={vdp,pendulum}/paper_log_penalty (p is inert on the sphere)."; exit 2 ;; \
	esac
	@if find "$(SWEEP_DIR)/p_study" "$(SWEEP_DIR)/p_study_sequential" \
	  -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "The p-study output already contains records; refusing to mix runs."; exit 2; \
	fi
	OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$(EXPERIMENT) \
	  hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	  hydra.sweep.dir=$(SWEEP_DIR)/p_study \
	  env.verbose=$(VERBOSE) env.seed=42 \
	  training.insert_mode=batch \
	  training.num_iterations=10 \
	  model.activation=softplus,tanh,gaussian,gelu_squared \
	  model.moment_order=2.01,2.5,3,4 \
	  model.alpha=1e-4,1e-5 \
	  model.gamma=1 model.loss_weights='[1.0,1.0]'
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

paper-radius-study:  ## compare fixed and theorem search radii for Algorithm 1 on VDP
	@if find rawdata/logs/multirun/vdp/paper_log_penalty/radius_ablation \
	  rawdata/logs/multirun/vdp/paper_log_penalty/radius_ablation_sequential \
	  -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "The radius-ablation output already contains records; refusing to mix runs."; exit 2; \
	fi
	@for mode in batch sequential; do \
	  if [ "$$mode" = batch ]; then iterations=10; suffix=radius_ablation; \
	  else iterations=$(SEQ_ITERATIONS); suffix=radius_ablation_sequential; fi; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=vdp/paper_log_penalty \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=rawdata/logs/multirun/vdp/paper_log_penalty/$$suffix \
	    env.verbose=$(VERBOSE) env.seed=42 \
	    training.insert_mode=$$mode training.num_iterations=$$iterations \
	    training.radial_cap=fixed,theorem \
	    model.activation=softplus,tanh,gaussian,gelu_squared \
	    model.moment_order=2.01,2.5,3,4 model.alpha=1e-4,1e-5 \
	    model.gamma=1 model.loss_weights='[1.0,1.0]' || exit $$?; \
	done

paper-oversampling-study:  ## rerun both current algorithms on four pendulum switching-band datasets
	@if find rawdata/logs/multirun/pendulum/paper_log_penalty/oversampling \
	  rawdata/logs/multirun/pendulum/paper_frac_exp_penalty/oversampling \
	  -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	  echo "The oversampling output already contains records; refusing to mix runs."; exit 2; \
	fi
	@set -e; for variant in base6k band40 band60 add2k; do \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=pendulum/paper_log_penalty \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=rawdata/logs/multirun/pendulum/paper_log_penalty/oversampling/$$variant \
	    env.verbose=$(VERBOSE) env.seed=42 \
	    training.insert_mode=sequential training.num_iterations=$(SEQ_ITERATIONS) \
	    model.activation=gaussian model.gamma=0 model.moment_order=2.01 \
	    model.alpha=1e-3,1e-4,1e-5 model.loss_weights='[1.0,1.0]' \
	    data.path=Pendulum_2sided_oversample_20260704/$$variant.npz \
	    eval.distance_cache=Pendulum_2sided_oversample_20260704/$${variant}_region_distances.npz; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=pendulum/paper_frac_exp_penalty \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=rawdata/logs/multirun/pendulum/paper_frac_exp_penalty/oversampling/$$variant \
	    env.verbose=$(VERBOSE) env.seed=42 \
	    training.insert_mode=sequential training.num_iterations=$(SEQ_ITERATIONS) \
	    model.activation=relu model.power=2 model.gamma=0 \
	    model.alpha=1e-4,1e-5,1e-6 model.loss_weights='[1.0,1.0]' \
	    data.path=Pendulum_2sided_oversample_20260704/$$variant.npz \
	    eval.distance_cache=Pendulum_2sided_oversample_20260704/$${variant}_region_distances.npz; \
	done

paper-l1-study:  ## run the current ReLU+l1 baselines used in both paper frontiers
	@set -e; for problem in vdp pendulum; do \
	  out=rawdata/logs/multirun/$$problem/paper_frac_exp_penalty/relu_l1; \
	  if find "$$out" -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
	    echo "$$out already contains records; refusing to mix runs."; exit 2; \
	  fi; \
	  OMP_NUM_THREADS=1 $(PY) scripts/train.py -m +experiment=$$problem/paper_frac_exp_penalty \
	    hydra/launcher=joblib hydra.launcher.n_jobs=$(JOBS) \
	    hydra.sweep.dir=$$out env.verbose=$(VERBOSE) env.seed=42 \
	    training.insert_mode=sequential training.num_iterations=$(SEQ_ITERATIONS) \
	    model.activation=relu model.power=1 model.gamma=0 \
	    model.alpha=1e-1,1e-2,1e-3,1e-4,1e-5,1e-6 \
	    model.loss_weights='[1.0,1.0]'; \
	done

paper-algorithm2-refresh:  ## stage, validate, archive, and replace all current Algorithm 2 paper runs
	JOBS=$(JOBS) VERBOSE=$(VERBOSE) SEQ_ITERATIONS=$(SEQ_ITERATIONS) \
	  ./scripts/refresh_algorithm2_paper_runs.sh

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
