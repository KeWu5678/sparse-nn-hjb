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
and its fixed operating points, and skips figures that need a two-dimensional
state.

## Scope

The validation set is the seed-42 10% holdout of the same Sobol design, so
selection on it is not an independent test estimate. These experiments
evaluate a fixed finite-horizon value slice `V(0,x)`. Reusing its gradient as
feedback would define a receding-horizon controller and would need a separate
rollout study. Open-loop convergence and regression
errors do not establish feedback stabilization, global optimality, or PDE mesh
convergence. No manuscript claims or existing figures are changed by these
new experiment definitions.
