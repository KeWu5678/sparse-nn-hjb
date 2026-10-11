# Algorithm & Implementation Details

Sub-level disclosure for the "Code — core algorithm" section of `../CLAUDE.md`.
Covers the full PDAP/SSN pipeline, the MATLAB mapping, known differences,
critical parameters, implementation gotchas, and a script index.

## PDAP loop (one `PDAP` class, configured by `model=` / `insertion=`)

```
PDAP.step() — one complete paper iteration:
  1. Candidate search
       - Algorithm 1: random starts inside R_search, one unconstrained joint
         L-BFGS solve, final-radius filter, Euclidean deduplication.
       - Algorithm 2: sphere starts, normalized L-BFGS refinement, cosine
         deduplication.
  2. Acceptance and warm start
       - Algorithm 1: normalized-profile threshold and candidate-specific
         guaranteed-decrease coefficient.
       - Algorithm 2: minimize the actual one-atom increment with the selected
         global scalar prox; insert only a nonzero negative-increment candidate.
  3. Outer-weight correction with fixed inner weights
       - Algorithm 1: soft-threshold normal map in normalized coefficients.
       - Algorithm 2: global-prox normal map with a warm-start-derived fixed
         scale for q in {2/3,1/2}; q=1 uses soft thresholding.
       - Retain the correction only when it does not raise the post-insertion
         objective.
  4. Prune coefficients at or below the amplitude tolerance.
```

The plain `PDAP` class updates `SparseModel` using the insertion strategy from
`src/PDAP/insertion.py`. `run_training` in `src/training.py` repeats the steps,
evaluates their results, and records History. Its explicit historical
`correction_first` branch preserves initialization, correction/pruning,
observation, and then insertion, including the last uncorrected insertion.
The historical semiconcave
parametrization was retired by ADR 0012; its archived design and empirical notes
remain in `semiconcave_model.md`. The insertion-candidate merge tolerance
(1e-2) is distinct from the prune amplitude tolerance.

### Scope checked against the manuscript

The authoritative source is [paper_0805.tex](../paper/paper_0805.tex),
Algorithm 1 (`alg:log`, lines 2898–2924) and Algorithm 2 (`alg:q`, lines
3011–3037). PDAP owns their outer procedure: search and filter candidates, test
and rank them, choose initial coefficients and insert, request correction with
inner weights fixed, prune, and stop when no candidate is accepted or the outer
iteration budget is exhausted. Both algorithms require a correction that does
not increase the post-insertion objective. SSN implements that correction
contract; the paper does not dictate the Python class organization.

Algorithm 1 uses the normalized derivative threshold and initializes each
accepted neuron from its certificate violation and per-neuron curvature
(lines 2838–2874). Algorithm 2 minimizes the finite one-neuron objective
increment, accepts a negative increment, and uses that coefficient to initialize
correction (lines 2959–2979). The coordinate-descent warm start is described for
the separately run ReLU/l1 baseline (lines 3005–3009); it is not a universal
extra stage required by both paper algorithms. Historical configurable paths
must remain identified separately from these manuscript procedures.

"Candidate-profile evaluation" in the refactor discussion means evaluating the
fitting-loss derivative for a supplied candidate neuron at zero inserted
coefficient (lines 2613–2616). It is one signed value; Algorithm 1 uses its
normalized version (lines 2636–2645 and 2818–2822). SparseModel supplies this
problem-specific calculation. PDAP searches over candidates and applies the
relevant acceptance and initialization rules. The derivative alone is neither
the full regularized objective increment nor an insertion decision.

Fixed-support preparation is only reuse within the coefficient correction:
features and the data-fitting Hessian are calculated once while inner weights
and samples remain fixed. Its small return record does not own candidate search,
warm-start, acceptance or a second network; it is not a paper algorithm stage.
It also holds the per-neuron normalization weights, using ones on the sphere.
These convert physical outer coefficients to the coordinates used by SSN and
back; they do not modify inner weights or provide rollback storage.

## MATLAB reference

Reference implementation: `/Users/ruizhechao/Documents/NonConvexSparseNN/`

| MATLAB | Python | Notes |
|--------|--------|-------|
| `PDAPmultisemidiscrete.m` | `src/PDAP/pdap.py` | Main PDAP loop (NOT `PDAPsemidiscrete.m`) |
| `SSN.m` | `src/SSN/optimizer.py` | Semismooth Newton optimizer (package) |
| `SSN_TR.m` | `src/SSN/strategies.py` (`steihaug_cg`) | Trust-region globalization |
| `setup_problem_NN_2d_from_xhat.m` | `src/models/sparse_model.py` + `src/PDAP/insertion.py` | Kernel/loss evaluation and candidate search, respectively |
| `run_vdp_2d_from_mat.m` | `notebook/pdpa_vdp.ipynb` | Experiment runner |

### Known differences from MATLAB
- **Kernel**: MATLAB uses smoothed ReLU (`delta=0.001`), Python uses exact `torch.relu`.
- **Parameterization**: MATLAB optimizes in stereographic R^d; Python on sphere S^d (`use_sphere=True`).
- **Postprocess**: MATLAB merges by Euclidean distance in stereographic space; Python by cosine similarity on S^d.
- **Gamma sweep**: MATLAB warm-starts each gamma from previous solution; Python runs independently.
- **Fractional powers**: Algorithm 2 supports `ReLU^2` and `ReLU^3`, with
  `q=2/3` and `q=1/2`; `ReLU`, `q=1` is the L1 endpoint. See
  `power_q_penalty.md`.

## Critical parameters (matching MATLAB)

| Parameter | Value | Source |
|-----------|-------|--------|
| `alpha` | `1e-5` | `run_vdp_2d_from_mat.m:105` |
| `th` | `0.5` | `setup_problem_NN_2d_from_xhat.m:183` |
| `delta` (MATLAB only) | `0.001` | `setup_problem_NN_2d_from_xhat.m:147` |
| Loss normalization | `M = N_points` | empirical objective in the paper |
| Max PDAP iterations | `15` | `run_vdp_2d_from_mat.m:12` |
| Max insert per step | `15` | `PDAPmultisemidiscrete.m:95` |
| SSN iterations | `20` | Configured correction budget; not a convergence guarantee |

## Critical implementation notes

### SSN step control (clarified 2026-09-09)

The project's SSN correction does not use a tunable learning rate. Step control
comes from adaptive Levenberg–Marquardt damping or the Steihaug trust-region
radius. The redundant `training.lr` / `SolverConfig.lr` / `SSN(lr=...)` plumbing
has been removed. This preserves the former `lr=1` behavior used by the tracked
experiments. The separate `training.lbfgs_lr` still controls the insertion
search's L-BFGS optimizer.

### Numerical safeguards

1. **Fractional SSN requires a nonzero warm start**: Algorithm 2 first computes
   the exact one-atom coefficient. Its smallest nonzero magnitude fixes the
   proximal scale for the entire correction.
2. **Use the normal-map smooth gradient**: Autograd differentiates the full
   objective `obj + Penalty.penalty` built by PDAP; `SSN.jac_obj_augmented` subtracts the full penalty gradient
   and applies the smooth nonconvex correction. SSN uses this returned gradient.
3. **SSN line search uses full objective**: Data loss + regularization. Without regularization, sparsifying steps get rejected.
4. **Loss normalization**: The empirical fidelity divides the value and full
   gradient sums by the sample count `M`, not by the number of scalar input
   coordinates.
5. **Insertion deduplication**: Algorithm 1 uses absolute Euclidean distance
   after its final-radius filter. Algorithm 2 uses cosine distance on the sphere
   and may repeat refinement after a merge.
6. **Sphere parameterization (`use_sphere`)**: For positively homogeneous activations (ReLU), inner weights must lie on the unit sphere S^d (`use_sphere=True`). Without this, L-BFGS pushes weights to extreme norms (1e+15), causing kernel-matrix overflow (`what=nan`) and coordinate-descent failure.
7. **Data scale / normalization (samples, not analytic targets)**: external datasets (e.g. pendulum, value O(100) over a large domain) must be normalized so `alpha`/`gamma` regularize as on the O(1) analytic targets. The data loss grows quadratically in the value scale while the log-penalty grows at most linearly, so an unscaled large value makes `alpha=1e-5` effectively zero. Normalize `x -> [-1,1]^d`, `V -> ~[0,1]`, and rescale `dV` by `s_x/s_v` consistently; un-normalize for any physical HJB residual.

## Modules (`src/`)

The implemented ownership and its verification are recorded in
[REFACTOR_HANDOFF.md](../REFACTOR_HANDOFF.md). PDAP owns insertion acceptance and
warm-start selection; SparseModel and the Penalty supply their mathematical evaluations.

- `models/` — the network, its fitting and its proximal penalty:
  - `sparse_model.py` — `SparseModel(nn.Module)` reads resolved configuration
    and registers `inner_weight` and
    `inner_bias` as buffers and `c` as its parameter. `forward` returns network
    values and input gradients; `kernel` supplies the kernel matrices of the
    current or supplied atoms, i.e. their coefficient Jacobians, and
    `moment_weight_of_atoms` their moment weights w_p. `obj`, `jac_obj`,
    and `hess_obj` supply fitting and its coefficient derivatives; the problem
    also supplies the penalty exponent q = 2/(power+1). `PDAP.train_ssn_outerweights`
    prepares the coefficient loss, Hessian and normalization for one correction.
    There is no objective-carrier class or persistent feature cache.
  - `proximal.py` — `make_penalty(alpha, gamma, penalty_exponent)` selects
    `LogPenalty(alpha, gamma)` (q = 1) or `FractionalPenalty(alpha, q)`. Their
    base `Penalty` holds the penalty αΦ, its split into the nonsmooth term
    α·Σ|u|^q and a smooth correction, and the vector and scalar proximal maps
    (taking the proximal constant λ). Shared by PDAP, SSN and evaluation. The
    scalar formulas sit below the classes, grouped by family.
- `PDAP/` — candidate search/acceptance, warm-start, support changes, and correction scheduling:
  - `pdap.py` — plain `PDAP` with `step(problem, train_loader)` for one complete
    paper iteration, configured by `insertion=` ("profile" or "finite_step").
    Returns solver facts and an algorithmic stop reason; evaluation and History
    are outside the class.
  - `insertion.py` — profile-threshold and finite-step candidate selection and
    insertion objective change.
  - `correction.py` — `correct_coefficients` runs the SSN iteration budget and
    discards a correction that increases the objective (or is nonfinite).
  - `radius.py` — theorem certificate radius; PDAP applies its cap and fallback.
  - `warmstart.py` — warm-start procedure and initialization using problem calculations,
    including the historical indicator derivative of its sphere kernel.
  - `history.py` — stores supplied measurements and one detached snapshot per
    checkpoint with artifact views; selects the earliest minimum training
    objective. Execution totals also include work without a new checkpoint.
- `SSN/` — the semismooth Newton optimizer:
  - `optimizer.py` — `SSN(Optimizer)` on a `Penalty`. One `step()` runs one
    globalized Newton iteration on supplied coefficients; SSN owns the normal
    map, its Jacobian and the prox preimage. The penalty family supplies the proximal scale and step
    (`prox_scale`, `proximal_step`).
  - `strategies.py` — Levenberg–Marquardt and Steihaug CG globalization.
  - `mpcg.py` — projected/trust-region CG inner solve.
- `data.py` — reading, normalization, splitting, standard DataLoader factories,
  and complete-sample assembly. Loaders and samples remain external to the model.
- `eval.py` — relative errors and metric/report assembly, using the problem's
  canonical `obj` and the Penalty's sparsity functional for its objective mathematics.
- `training.py` — external `run_training` loop, timing, evaluation, recording
  and progress output. The budget belongs here; PDAP decides whether candidate
  search provides another admissible update. Native PyTorch evaluates each full
  split without a metric-framework dependency or streaming accumulator.
- `results.py` — saved-history reconstruction and physical-coordinate prediction.
- `metric.py` — experiment-analysis utilities (per-gamma neuron/loss tables, plots).
- `config/` — Hydra structured configuration for PDAP training:
  - `schema.py` — typed model/training/data/env config objects, activation and
    geometry resolution, and training compatibility validation. Setup resolves
    and validates configuration before constructing `SparseModel`; the model
    and optimizers do not initiate those configuration operations. Raw Hydra
    settings remain unchanged for serialization.
  - `activations.py` — canonical activation registry used by configs and scripts.
  - `scripts/train.py` registers the Hydra `config_schema` before composition.
- Open-loop data subsystem: `src/OpenLoop/` — shared `ValueSamples`, VDP smooth
  data generation under `src/OpenLoop/vdp/`, and paper-backed infinite-horizon
  pendulum data generation under `src/OpenLoop/pendulum/`.

## Script index (`scripts/`)

| script | role |
|--------|------|
| `train.py` | Hydra entry point for PDAP training on key-based value/gradient datasets |
| `run_activation_experiment.py` | base activation registry + VDP-HJB activation search |
| `run_discontinuous_activation_experiment.py` | discontinuous-gradient activation search (extends `ACTIVATIONS`, `set_seed`) |
| `run_pendulum_pmp_openloop_example.py` | generate PMP backward-sampler pendulum dataset (infinite-horizon) |
| `visualize_proximal_deadzone.py` | 3-panel global-prox switching diagnostic (see `power_q_penalty.md`) |
| `append_pendulum_pilot_plots.py` | plotting helper |

## Data flow

```
VDP / pendulum samples -> external preprocessing -> DataLoaders
SparseModel + Penalty + PDAP + DataLoaders -> run_training -> History -> saved-run evaluation
```

```python
pdap = PDAP(cfg.training, insertion=model_config["insertion"])
history = run_training(
    pdap, problem, train_loader, valid_loader,
    num_iterations=cfg.training.num_iterations,
    reporting_normalizer=normalizer,
)
```

The runner passes any reporting normalizer directly to evaluation. History
receives the resulting scalar values and never initiates data loading or metric
evaluation. Existing quality keys and best-checkpoint selection stay unchanged.
`iterations` remains the historical checkpoint-row count; explicit execution
counters describe searches and coefficient corrections, including a terminal
failed search or the legacy final insertion. No-candidate termination and budget
exhaustion are distinct from a mathematical convergence guarantee.

Runtime fields distinguish wall time spent in solver operations from evaluation.
`process_peak_rss_bytes` is the current process's lifetime high-water resident
memory on macOS/Linux, excluding child workers. It is not a solver-only peak or
an isolated per-run allocation measurement. Detailed profiling is separate.

Training data may be a legacy key-based `.npy` file or a newer `.npz` file; both must expose `x`, `v`, and `dv` arrays.
VDP experiments: `notebook/pdpa_vdp.ipynb` (pickles in `models/experiment_N/`).
Activation/model studies: `scripts/*` -> `autoresearch/*`.
