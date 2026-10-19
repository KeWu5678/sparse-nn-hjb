# Allen–Cahn at 6 and 10 state dimensions

This study asks how the two sparse-network algorithms behave when the state
dimension grows beyond the current two-dimensional VDP and pendulum examples.
It compares value/gradient approximation and network size at a fixed data
budget, using a coupled nonlinear control system rather than copies of the
same independent two-dimensional problem.

The example adapts §9.3 of Kunisch, Vásquez-Varas, and Walter,
[*Learning Optimal Feedback Operators and their Sparse Polynomial
Approximations*, JMLR 24(301), 2023](https://jmlr.org/papers/v24/22-1010.html).
The published example has 19 Chebyshev states. These benchmarks use six and ten
finite-volume cell values, retaining the published diffusion `nu=0.1`,
horizon `T=3`, three actuator intervals, and Neumann boundaries on `(-1,1)`.
Cell averages preserve all three actuator channels on these coarse meshes.
The control penalty `beta=0.01` and Sobol cube `[-1,1]^d` are explicit new
choices. See the [source research](../../vault/higher_dimensional_benchmarks_research.md)
and [open-loop guide](../00_openloop/allen_cahn/README.md) for the equations,
provenance, solver diagnostics, and limitations.

## Generate the data

From the repository root:

```sh
uv run python scripts/generate_allen_cahn_data.py --dimension 6 --samples 2048 --workers 4 --tag 20261012
uv run python scripts/generate_allen_cahn_data.py --dimension 10 --samples 2048 --workers 4 --tag 20261012
```

The data groups `allen_cahn6` and `allen_cahn10` select respectively
`rawdata/data/AllenCahn6_20261012/value_samples.npz` and
`rawdata/data/AllenCahn10_20261012/value_samples.npz`. Each generation records
the requested initial states, exact parameters, solver results, source and
dataset hashes, and independently re-solved directional finite differences.
The generator fails instead of silently dropping unsolved initial states.
Choose another tag and update the data configs to regenerate existing data.

Each dataset has 2,048 independently initialized trajectories and contributes
one fixed-horizon `(x,V,dV)` sample per trajectory. The seed-42 90/10 split
uses 1,843 samples for training and 205 for validation. This matches the
existing training convention; the validation set is a subset of the same
Sobol design, not an independently scrambled test set. Hyperparameter
selection on it does not provide an independent test error estimate.

## Run the inherited experiment grids

All training still uses `scripts/train.py`. The three new presets inherit the
existing algorithms and sweep axes and change only the dataset choices and
run names. The current VDP/pendulum grids remain separate.

| Preset | Runs per dimension | Total |
| --- | ---: | ---: |
| `higher_dim_log_penalty` | 224 | 448 |
| `higher_dim_frac_exp_penalty` | 16 | 32 |
| `higher_dim_relu_l1_baseline` | 10 | 20 |

Start the [local MLflow service](../../deploy/README.md), then run:

```sh
make sweep EXPERIMENT=higher_dim_log_penalty
make sweep EXPERIMENT=higher_dim_frac_exp_penalty
make sweep EXPERIMENT=higher_dim_relu_l1_baseline
```

Algorithms 1 and 2 retain 150 outer iterations, 50 candidate directions,
sequential insertion, and guarded coefficient correction. Algorithm 1 retains
moment order 2.01 and its theorem-based radius. Algorithm 2 uses sphere
normalization and ReLU powers two and three. Both compare value-only and
gradient-augmented loss. Data normalization and physical-coordinate error
reporting use the shared reversible transform.

The traditional ReLU-L1 baseline retains its original ten correction-first
iterations, batch insertion (up to 15 neurons), and warm start. Its iteration
budget is not equal to the other two methods' sequential budget; report
network size and timing alongside errors. It is not Algorithm 2 with power
one. Full definitions are in `conf/experiment/`; inherited settings are not
reimplemented in an experiment-specific trainer.

## Smoke runs and restricted pilots

These six small fits check both dimensions and all three training paths. They
deliberately reduce the optimization budget and write outside the production
sweep directories. Like every training job, they are recorded in MLflow, so
start the local service first:

```sh
uv run python scripts/train.py -m +experiment=higher_dim_log_penalty \
  +model=profile +data=allen_cahn6,allen_cahn10 model.activation=tanh \
  model.alpha=1e-4 model.gamma=1 'model.loss_weights=[1.0,1.0]' \
  training.num_iterations=3 training.num_of_initial_pts=5 training.lbfgs_steps=10 \
  training.fit_outer_iterations=5 env.verbose=false \
  hydra.sweep.dir=rawdata/logs/allen_cahn_smoke

uv run python scripts/train.py -m +experiment=higher_dim_frac_exp_penalty \
  +model=finite_step +data=allen_cahn6,allen_cahn10 model.activation=relu \
  model.power=2 model.alpha=1e-4 model.gamma=0 'model.loss_weights=[1.0,1.0]' \
  training.num_iterations=3 training.num_of_initial_pts=5 training.lbfgs_steps=10 \
  training.fit_outer_iterations=5 env.verbose=false \
  hydra.sweep.dir=rawdata/logs/allen_cahn_smoke

uv run python scripts/train.py -m +experiment=higher_dim_relu_l1_baseline \
  +model=profile +data=allen_cahn6,allen_cahn10 model.activation=relu \
  model.power=1 model.alpha=1e-4 model.gamma=0 'model.loss_weights=[1.0,1.0]' \
  training.num_iterations=3 training.num_of_initial_pts=5 training.lbfgs_steps=10 \
  training.fit_outer_iterations=5 env.verbose=false \
  hydra.sweep.dir=rawdata/logs/allen_cahn_smoke
```

Hydra command-line axes replace the corresponding preset axes. Pin every
listed model axis when repeating these commands; removing one can expand
the multirun unexpectedly. For a scientific pilot, keep the normal training
budget and restrict only the parameter grid, writing to a fresh output root.

Six such smoke fits completed on October 12, 2026, before runs were recorded
only in MLflow. The full 500-cell study has not been run.

## Analysis

Analysis is local and ignored, as for VDP. It reuses the VDP paper pipeline and
its fixed operating points, and skips figures that need a two-dimensional state.

These experiments evaluate a fixed finite-horizon value slice `V(0,x)`.
Reusing its gradient as feedback would define a receding-horizon controller
and would need a separate rollout study. Open-loop convergence and regression
errors do not establish feedback stabilization, global optimality, or PDE mesh
convergence. No manuscript claims or existing figures are changed by these
new experiment definitions.
