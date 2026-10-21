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
choices. See the [open-loop guide](../00_openloop/allen_cahn/README.md) for the
equations, provenance, solver diagnostics, and limitations.

## Run

Prepare once with `scripts/init.sh` (see the [experiment guide](../README.md)).
The datasets `rawdata/data/AllenCahn6_20261012/` and
`rawdata/data/AllenCahn10_20261012/` must exist; generate them with the
[Allen–Cahn data guide](../00_openloop/allen_cahn/README.md). Then:

```sh
make sweep EXPERIMENT=higher_dim_log_penalty
make sweep EXPERIMENT=higher_dim_frac_exp_penalty
make sweep EXPERIMENT=higher_dim_relu_l1_baseline
```

| Preset | Runs per dimension | Total |
| --- | ---: | ---: |
| `higher_dim_log_penalty` | 224 | 448 |
| `higher_dim_frac_exp_penalty` | 16 | 32 |
| `higher_dim_relu_l1_baseline` | 10 | 20 |

Each preset extends the VDP/pendulum preset without the `higher_dim_` prefix
and keeps its algorithm, settings, and sweep axes; only the datasets and run
names change. The
ReLU–L1 baseline keeps its own correction-first, batch-insertion budget, so
compare network size and timing alongside errors.

## Analysis

Analysis is local and git-ignored, as for VDP. It reuses the VDP paper pipeline
and its fixed operating points, and skips value surfaces and weight portraits
that need a two-dimensional state. Each dimension's tables and frontier plots
are written to `higher_dim_allen_cahn6/` and `higher_dim_allen_cahn10/` in this folder.

The manuscript uses only the 10D case, at the same fixed operating points as
VDP Tables 1–3 and Figures 6–7. Regenerate its figures and verify the table
metrics against the saved checkpoints with:

```sh
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python scripts/paper/allen_cahn_paper.py
```

This writes the selected run IDs, exact table metrics, feedback arrays, and
four figure panels under `higher_dim_allen_cahn10/paper/`. Algorithm 1 uses
`alpha=1e-4, gamma=10, p=2.01`; Algorithm 2 uses `alpha=1e-5` for its table
and `alpha=1e-6` for the figures. The feedback plot starts from the constant
profile `(1,...,1)` over `T=3` with `dt=0.01`, reusing the PMP reference in
`feedback/openloop_references.npz` from the evaluation below. Its control
panel shows the norm of the three control channels.

Evaluate feedback from the saved models with the existing MLflow server:

```sh
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python scripts/paper/allen_cahn_feedback.py
```

This separate comparison uses the lowest validation-H1 fit in each completed
softplus, Gaussian, tanh, ReLU², ReLU³, and ReLU–L1 sweep. It does not select
models using stabilization results or retrain them. It simulates all 205
validation states and 11 prescribed profiles through `T=40`, with the
unmodified feedback `u(x) = -B.T @ grad(V_hat)(x) / (2 beta)`. Controls are held
for each RK4 step without clipping; results use `dt=0.005` and are compared
with `dt=0.01`. The running cost uses the same RK4 stages.
The discontinuous ReLU–L1 feedback is additionally evaluated at `dt=0.0025`
and `dt=0.00125` to measure its control-update sensitivity.

Numerical stabilization requires the spatial L² state norm
`sqrt((2/d) * sum(x**2)) < 0.05` throughout the final two seconds at each
reporting horizon. Reports also show thresholds 0.01 and 0.1, the control at
the origin, and any change in outcome under time-step refinement. Three-second
costs are compared with the stored PMP values; only the 11 extra profiles
need new reference solves. Reports, plots, run IDs, and full state/control/cost
arrays are saved in each dimension's `feedback/` subdirectory.

## Scope

The validation set is the seed-42 10% holdout of the same Sobol design, so
selection on it is not an independent test estimate. These experiments
approximate a fixed finite-horizon value slice `V(0,x)`. Its repeated use in
the feedback evaluation defines an approximate receding-horizon controller.
Finite-time numerical stabilization does not establish asymptotic stability;
in particular, an unmodified learned gradient may apply nonzero control at
the origin. These comparisons do not establish global optimality or PDE mesh
convergence. The feedback reports are separate from the manuscript figures.
