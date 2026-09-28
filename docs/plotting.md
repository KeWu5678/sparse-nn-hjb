# Saved-run evaluation and plotting

Current-paper generators reconstruct models through `src.results` and render
prepared arrays through `src.plots`. They consume the dated datasets selected
by `conf/data/*.yaml` and the current experiment records. Publication output
is restricted to PNGs included by `paper/paper_0805.tex`.

## Ownership

| Module | Responsibility |
| --- | --- |
| `src.results` | Load recorded fits, restore checkpoints, and evaluate physical values/gradients |
| `src.data` | Load value samples and apply/reverse data normalization |
| `src.eval` | Compute errors from prediction/target tensors |
| `src.plots` | Render prepared arrays and export figures |
| `src.plotstyle` | Shared palette, typography, and axes conventions |
| Paper scripts | Select runs and evaluation inputs, construct references, simulate feedback, and write reports |

## Evaluate a current saved fit

```python
from src.results import load_run, predict_physical, validation_samples
from src.eval import relative_errors

run = load_run(record_path)
x, v_true, dv_true = validation_samples(run)
v_pred, dv_pred = predict_physical(run, x)
l2, gradient, h1 = relative_errors(v_pred, dv_pred, v_true, dv_true)
```

`load_run` restores the saved `best_iteration` using the recorded model and
normalization. Current fits are stored beside their records as
`result_<run_id>.pkl`. `predict_physical` accepts physical states `(N,d)` and
returns detached float64 tensors `(N,1)` and `(N,d)`. An optional `iteration`
evaluates another checkpoint without changing the selected model. Predictions
are not clipped, and loading/evaluation preserve the caller's random state.

`validation_samples` reproduces the recorded seed and training fraction.
`evaluate_history(run)` scores all checkpoints on that same holdout and returns
`iteration`, `neurons`, `rel_l2`, `rel_grad`, and `rel_h1` in iteration order.
A sorted support frontier is a reporting view, not a different checkpoint
selection rule.

### Normalization and regional evaluation

Training uses its normalized objective, fidelity, and regularizer. Reported
approximation errors compare physical values and gradients, with no
regularization or moment-weight factors. For `x_n=x/s_x` and `V_n=V/s_v`, the
inverse transform is `V=s_v V_n` and `dV=(s_v/s_x) dV_n`, implemented by
`ValueSampleNormalizer`. Input scaling can change a relative H1 error even when
a common value scale leaves relative L2 unchanged.

Current records include explicit normalization and
`metric_coordinates: physical`. The runner passes `reporting_normalizer` to
`PDAP.fit`; direct callers that normalize samples should do the same. Current
publication preflight rejects missing normalization and nonphysical metrics.
Archived normalization recovery is not part of the fresh publication workflow.

Pendulum regional scores use the shared pool in `conf/eval/region_split.yaml`.
The pool excludes the union of the production dataset and all four sampling
variants. The switching tube has distance at most 0.3 to the saved switching
points and their adjacent periodic translates. The dataset-aligned distance
cache serves distance-binned diagnostics. These artifacts are hashed before
training and their hashes are recorded. Current records already contain
physical regional metrics; no rescoring sidecar is required.

Within each run, the selected checkpoint minimizes the training objective.
Within each pendulum activation family, cross-run selection first requires
numerical stabilization from both A and B at T = 40, then minimizes global
physical-coordinate H1 validation error. If none passes, the lowest-global-error
fit is explicitly reported as a failed comparator. Regional errors describe
these selected fits; they do not rank controllers. A/B are selection states,
so their outcomes do not independently measure control generalization. See the
[selection protocol](pendulum_stability.md#controller-selection).
The separate oversampling study still compares the smallest switching-region
error for each training set; it evaluates approximation rather than feedback.

## Render prepared arrays

```python
from src.results import value_surface_grid
from src.plots import plot_value_surface, save_figure

grid = value_surface_grid(run, x_range=(-8, 8), y_range=(-8, 8), grid_n=220)
fig, ax = plot_value_surface(*grid, clip=(0, 60))
save_figure(fig, output_path, tight=False, bbox_inches="tight")
```

Clipping and interpolation used for a surface are display operations. They do
not change metric targets, predictions used for scoring, or feedback laws.
Display windows need not equal dataset support; the current pendulum data
extend beyond `[-9,9]^2`. The reference-surface construction and its cropped
window are documented in the [pendulum data guide](../experiments/00_openloop/pendulum/README.md).

Import the palette and publication style from `src.plotstyle`; renderers return
figure/axes and `save_figure` handles export and closing. New exports use PNG at
300 dpi, without in-figure titles. Preserve the explicit layout and bounding-box
options. Batch generators select Matplotlib's Agg backend before importing
pyplot helpers so text layout does not depend on a desktop backend. Shared
renderers remain usable interactively.

## Signed controls and realized costs

The two benchmarks compare all five selected activations. Each learned law is
simulated along its own state trajectory from the same initial state as its
numerical open-loop reference. The comparison horizons and steps are:

| Comparison | Horizon | Step |
| --- | ---: | ---: |
| Van der Pol control/cost | 3 | 0.01 |
| Pendulum control/cost, each of two starts | 40 | 0.005 |

Pendulum success uses the [numerical stabilization protocol](pendulum_stability.md):
angle error below 0.1 rad and speed below 0.5 rad/s at every sample in the
final 2 seconds. The same checkpoints are evaluated at 10, 20, and 40 seconds
by `scripts/paper/pendulum_stability.py`. These are finite-horizon diagnostics,
not asymptotic stability certificates.

All VDP state, control, and cost plots use the benchmark horizon `T=3`.
Its signed-control, state-norm, and cost panels show the same five learned
controllers and numerical open-loop reference.

`src.OpenLoop.comparison.accumulated_cost` integrates running costs at the
held-control RK4 stages of each learned rollout. It does not substitute a
network's value prediction. The numerical reference uses three-point Gauss
quadrature on its dense PMP boundary-value solution.

`solve_openloop_reference` fixes the initial state and comparison horizon,
with a free terminal state and zero terminal cost. It retains the least-cost
converged solve initialized from the supplied rollouts. This is a numerical
reference, not a global-optimality certificate. Reference controls are
unconstrained, matching the OCP; learned rollouts use their configured numerical
control guards. Pendulum's finite-horizon comparison is a truncation of the
infinite-horizon problem used for its training data.

`scripts/paper/control_comparison.py` writes signed-control and cumulative-cost
PNGs, plus the VDP state-norm panel. It preserves time/state/control/cost arrays
in NPZ files and writes final costs to CSV files beside the reports.
The pendulum figure uses newly solved 40-second open-loop references; only
complete rollouts initialize those solves. Divergent feedback traces end at
their actual stopping time (the CSV horizon is their achieved horizon).
Table 4 retains the cost at 10 seconds and gives separate numerical stabilization
columns for 10 and 40 seconds; the full horizon study also reports 20 seconds.

## Current publication pipeline

The [experiment workflow](../experiments/README.md) describes dataset generation
and the six current sweep presets. Local/ignored paper commands are:

```sh
uv run python scripts/paper/preflight.py
make openloop
uv run python experiments/01_vdp/paper_log_penalty/p_study_figure.py
uv run python scripts/paper/vdp_full_scope.py --homogeneous-alpha 1e-6
uv run python scripts/paper/pendulum_full_scope.py
uv run python scripts/paper/preflight.py --require-figures --write-manifest
```

The VDP command explicitly uses homogeneous alpha `1e-6` for frontier/feedback
comparisons; its separate surface/table comparison uses `1e-5`. The two benchmark
generators read current `log_penalty`, `frac_exp_penalty`, and relevant baseline
or oversampling record roots. They validate the corresponding benchmark before
writing outputs. The moment-order figure reads both `moment_order_study` roots.
The local Algorithm 1/2 `analysis.py` wrappers delegate to those same generators.

Preflight constructs the expected grid dynamically from `conf/experiment/`
(currently 588 cells). It checks cell completeness/uniqueness, current dataset
paths and recorded hashes, completed status, physical metrics, normalization,
fit-file presence, training settings, and pendulum evaluation-input hashes.
It rejects the legacy VDP dataset hash.

`--require-figures` derives the allowed paths from TeX and rejects missing
included PNGs or unreferenced PNGs under `experiments/` and `paper/plot/`.
`--write-manifest` writes `scripts/paper/current_run_manifest.json` with exact
SHA-256 hashes of records, adjacent fits, datasets, evaluation inputs, and the
checked figures. Numerical correctness still requires separate verification.

Relevant checks include:

```sh
uv run pytest tests/test_results.py tests/test_history.py tests/test_experiment_logging.py
uv run pytest tests/test_control_comparison.py tests/test_paper_feedback_results.py
```

Reusable code and tests are tracked; paper-specific generators and artifacts
remain local/ignored under ADR 0011. Superseded inputs, records, reports,
figures, pipeline files, and diagnosis notes are preserved with hash manifests
under `outdated/experiment-refresh-20260924/`. The current pipeline does not
load that archive. An empty checkout requires data generation and training
before these local publication commands can run.
