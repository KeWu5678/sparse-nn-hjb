# Open-loop data — Allen–Cahn at 6 and 10 states

These coupled nonlinear systems adapt the controlled Allen–Cahn problem in
Kunisch, Vásquez-Varas, and Walter, *Learning Optimal Feedback Operators and
their Sparse Polynomial Approximations*, JMLR 24(301), 2023,
[§9.3, Eq. (58)](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=20).

The published diffusion `nu=0.1`, horizon `T=3`, Neumann boundaries on `(-1,1)`,
and three actuator intervals are retained. The article uses 19 Chebyshev
degrees of freedom. Here, `d=6` or `d=10` cell-centered finite volumes use
cell-averaged actuators so the narrow middle actuator is preserved on both
meshes. The control-cost coefficient `beta=0.01` is an explicit repository
choice; the source does not specify its numerical Allen–Cahn value.

With cell width `h=2/d`, the dynamics and running cost are

```text
y' = 0.1 L y + y - y^3 + B u
ell(y,u) = h sum_j y_j^2 + 0.01 sum_k u_k^2
```

`L` is the zero-flux discrete Laplacian, and `B[j,k]` is the fraction of cell
`j` covered by actuator `k`. The spatial cost weight prevents dimension from
changing the state/control weighting. States are cell values in spatial order.

## Generate

From the repository root:

```sh
uv run python scripts/generate_allen_cahn_data.py --dimension 6 --samples 2048 --workers 4 --tag 20261012
uv run python scripts/generate_allen_cahn_data.py --dimension 10 --samples 2048 --workers 4 --tag 20261012
```

Each produces `rawdata/data/AllenCahn<d>_20261012/value_samples.npz`, containing
`x` and `dv` of shape `(2048,d)` and `v` of shape `(2048,)`. Initial states are
scrambled Sobol points throughout `[-1,1]^d`, seed 42. This is a smaller domain
and a different sampling design from the article. Use the same count at both
dimensions to compare the algorithms under a fixed data budget.

The generator refuses an existing output directory. For a fresh generation,
choose a new tag and update the corresponding `conf/data/allen_cahn*.yaml`
on the experiment branch. It retains every requested state or fails; a failed
run never silently becomes a smaller training dataset. A quick generation
check uses `--samples 8 --tag smoke`.

`src/OpenLoop/allen_cahn/` solves the PMP boundary-value equations by adaptive
collocation with two initial guesses and an analytic Jacobian. The minimum
cost among converged stationary candidates is retained. Five-point Gauss
quadrature on each adaptive interval computes the cost; `p(0)` is the value
gradient. Before saving the dataset, directional finite differences at three
sampled states check the gradient labels by re-solving with the same
collocation method. This checks sensitivity consistency, not agreement with
an independent optimizer.

The directory also holds requested initial states, per-sample diagnostics,
failure records, and metadata with all problem/solver/sampling settings,
package versions, source hashes, dataset hash, and gradient-validation errors.
Generated arrays remain ignored local artifacts, like the existing datasets.

## Generated data audit, October 12, 2026

Both configured datasets were generated successfully from generator revision
`5d712a0`. Subsequent changes are recorded below;
the metadata retains hashes of the source bytes used during generation.

| States | Samples | Failed solves | Maximum collocation residual | Maximum gradient spot-check error |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 2,048 | 0 | 9.99985e-7 | 9.93e-10 |
| 10 | 2,048 | 0 | 9.99999e-7 | 4.29e-11 |

The gradient errors above are absolute directional errors at three sampled
states per dataset, not a bound over all samples. Independent tests also
perturb every coordinate at representative 6D and 10D states and replay the
computed open-loop controls with a separate initial-value integrator.

Dataset SHA-256:

```text
AllenCahn6_20261012/value_samples.npz
6f1f36bc87ad9cc5597f1d569f7a52ce7d67b7fb729fffc8906d7bad62d4ab22
AllenCahn10_20261012/value_samples.npz
4c233f374dc36bbdb235dda373350b6dd714cda9b3bc5192bf93e3e44071efb0
```

## Extended numerical and code review

The initial generation checks were supplemented with the following audits.
Detailed results and reproducible local scripts are kept in the ignored
`rawdata/reports/allen_cahn_audit/` directory of the open-loop worktree.

| Check | Coverage | Largest observed discrepancy |
| --- | --- | ---: |
| Finer PMP solve: tolerance `1e-9`, 161 initial mesh points | 25 dispersed/extreme rows per dimension | value 1.25e-9; gradient 1.49e-9 |
| Independently integrated state/cost under the computed control | same 50 rows | state 3.77e-11; cost 7.95e-13 |
| Sign and spatial-reflection symmetries | same 50 rows | 7.22e-16 |
| Independent direct control optimization, 300 RK4 steps | 3 actual rows per dimension, zero/random starts | value 7.08e-9; gradient 2.04e-7 |

The direct method separately implements the dynamics and discrete adjoint;
Torch automatic differentiation and finite differences check its derivatives.
All 24 optimizations at 150/300 time steps converged, and every selected
sample's value error decreased on refinement. Twelve additional re-optimized
initial-state perturbations agreed with stored directional costates within
5.71e-9. The detailed report is `direct_control_results.json`.

All 4,096 stored rows pass finite-array, requested-state, full-dimensional-rank,
hash, and independently integrated zero-control upper-bound checks. The
smallest improvement over zero control is 0.0381 (6D) and 0.0392 (10D).
Another 200 boundary-value solves used nonzero-costate initial guesses:
194 converged to the same costs; six exploratory starts did not converge.
These six failed attempts are separate from the complete production datasets.
No cheaper branch was found in this audit. See `scientific_audit_summary.json`.

The algebraic `reduced_gradient_norm` is zero by construction after control
elimination; it is not independent evidence of optimality. Independent
optimization and refinement supply the additional numerical evidence.
Tests also compare the nonlinear solver near zero to an independently solved
finite-horizon Riccati equation, checking the value/costate factor of two.

Review against the current `refactor.md` found duplicated equations, duplicated
batch assembly, nullable failure results, mixed generation/reporting
responsibilities, and explanatory ordering issues. The refactor puts the PMP
operator and Jacobian in the problem, centralizes serial/parallel batch solving,
raises diagnostic exceptions for failures, and separates generation stages.
Single-sample results now guarantee value/gradient fields; a failed batch
returns no partial training dataset. The unused feedback rollout was removed.

After refactoring, all 4,096 original initial states were solved again with
four workers. Every state, value, and gradient entry is exactly equal to the
stored arrays; both batches completed with zero failures and the original
dataset hashes remained unchanged. This full comparison took 270 seconds on
the local machine. `refactor_equivalence.json` records the configurations,
package versions, timings, and source hashes. `refactor_acceptance.json`
records eleven rejection checks for invalid solver output and a mixed batch
that retained failure diagnostics without producing partial training data.

## Interpretation

These are finite-horizon `V(0,x)` samples with free terminal state and zero
terminal cost (`p(T)=0`), as in the existing VDP example. Intermediate trajectory
points are not added to this state-only dataset because their remaining
horizons differ. Applying the learned fixed-horizon gradient repeatedly gives
a receding-horizon policy, not the original time-dependent optimal feedback.

A converged PMP solution is a numerical stationary reference, not proof of a
global optimum. Six and ten cells are separate coarse ODE benchmarks; this
comparison does not establish PDE mesh convergence. The experiment branch adds
training presets and reporting without altering the current two-example study.
