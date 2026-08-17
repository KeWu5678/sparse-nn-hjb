# Algorithm & Implementation Details

Sub-level disclosure for the "Code — core algorithm" section of `../CLAUDE.md`.
Covers the full PDAP/SSN pipeline, the MATLAB mapping, known differences,
critical parameters, implementation gotchas, and a script index.

## PDAP loop (one `PDAP` class, configured by `model=` / `insertion=`)

```
PDAP.fit() outer iteration:
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

The single `PDAP` loop drives the signed model via its uniform interface and the
insertion strategy from `src/PDAP/insertion.py`. The historical semiconcave
parametrization was retired by ADR 0012; its archived design and empirical notes
remain in `semiconcave_model.md`. The insertion-candidate merge tolerance
(1e-2) is distinct from the prune amplitude tolerance.

## MATLAB reference

Reference implementation: `/Users/ruizhechao/Documents/NonConvexSparseNN/`

| MATLAB | Python | Notes |
|--------|--------|-------|
| `PDAPmultisemidiscrete.m` | `src/PDAP/pdap.py` | Main PDAP loop (NOT `PDAPsemidiscrete.m`) |
| `SSN.m` | `src/SSN/optimizer.py` | Semismooth Newton optimizer (package) |
| `SSN_TR.m` | `src/SSN/strategies.py` (`steihaug_cg`) | Trust-region globalization |
| `setup_problem_NN_2d_from_xhat.m` | `src/models/signed.py` + `src/net.py` | Kernel, loss, find_max |
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
| SSN iterations | `20` | Converges in 3-4 steps |

## Critical implementation notes

1. **Fractional SSN requires a nonzero warm start**: Algorithm 2 first computes
   the exact one-atom coefficient. Its smallest nonzero magnitude fixes the
   proximal scale for the entire correction.
2. **SSN gradient must be data-only**: Autograd gives the full objective gradient; SSN adds `alpha*dphi` separately. Double-counting breaks convergence. (The closure returns the full objective; SSN subtracts the penalty gradient internally.)
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

- `PDAP/` — the unified outer-loop package:
  - `pdap.py` — `PDAP` class + `fit()` (matches `PDAPmultisemidiscrete.m`); configured
    by `insertion=` ("profile"|"finite_step").
    Holds the shared `sample_uniform_sphere_points` / `prune_small_weights` /
    `check_linearity_neurons` helpers.
  - `insertion.py` — `profile_threshold` / `finite_step` strategies, their
    distinct nonhomogeneous/sphere candidate searches, and
    `solve_insertion_weight` for the actual Algorithm 2 increment.
- `models/` — the signed parametric value-function model behind the protocol in `base.py`:
  - `signed.py` — `SignedModel` (pure network; matches `setup_problem_NN_2d_from_xhat.m`).
- `net.py` — `ShallowNetwork`: `input -> hidden (ReLU^p) -> output`; `forward_network_matrix()` and `forward_gradient_kernel()` build the SSN data Hessian.
- `SSN/` — the semismooth-Newton optimizer package (one configurable class):
  - `optimizer.py` — `SSN`. Stores `q=2/(p+1)` and implements the proximal
    normal map using the supplied fractional proximal scale.
    Optional masks remain part of the optimizer's generic coordinate API.
  - `strategies.py` — `levenberg_marquardt` (damped Newton) and `steihaug_cg`
    (trust-region MPCG) globalizations, selected by `method=`.
  - `prox.py`, `penalty.py` — proximal / penalty kernels (re-exported by `utils.py`).
  - `mpcg.py` — projected/trust-region CG inner solve for `steihaug_cg`.
- `utils.py` — compatibility re-exports for the `SSN` penalty/proximal kernels,
  including `power_prox` and `power_prox_derivative`.
- `metric.py` — experiment-analysis utilities (per-gamma neuron/loss tables, plots).
- `config/` — Hydra structured configuration for PDAP training:
  - `schema.py` — typed model/training/data/env config objects.
  - `activations.py` — canonical activation registry used by configs and scripts.
  - `store.py` — registers the Hydra `config_schema`.
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
VDP / pendulum data generator -> dict {x, v, dv} -> PDAP(model=, insertion=).fit() -> results
```
Training data may be a legacy key-based `.npy` file or a newer `.npz` file; both must expose `x`, `v`, and `dv` arrays.
VDP experiments: `notebook/pdpa_vdp.ipynb` (pickles in `models/experiment_N/`).
Activation/model studies: `scripts/*` -> `autoresearch/*`.
