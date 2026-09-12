# Saved-run evaluation and plotting

The current-paper generators share model reconstruction through `src.results`
and rendering through `src.plots`. Their existing numerical implementations,
layouts and manuscript figure names are retained. The active generators now
produce only the 36 PNGs included by `paper/paper_0805.tex`; surplus diagnostic
renderers and their images are archived, not regenerated.

## Ownership

| Module | Responsibility |
| --- | --- |
| `src.results` | Read fits, restore checkpoints, predict physical V/dV, orchestrate historical rescoring; no artifact writes |
| `src.data` | Load value samples and apply/reverse the data transform |
| `src.eval` | Pure error calculations on prediction/target tensors; no coordinate guessing or regularizer |
| `src.plots` | Render prepared arrays and save figures |
| `src.plotstyle` | Shared palette and axes/typography conventions |
| Paper scripts | Select runs/evaluation sets, construct references, simulate feedback, choose labels and paths |

## Evaluate one saved fit

```python
from src.results import load_run, predict_physical, validation_samples
from src.eval import relative_errors

run = load_run(record_path)
x, v_true, dv_true = validation_samples(run)
v_pred, dv_pred = predict_physical(run, x)
l2, gradient, h1 = relative_errors(v_pred, dv_pred, v_true, dv_true)
```

`load_run` reads activation, power and fitted data scaling from the Run Record
and restores its saved `best_iteration`. `predict_physical` accepts physical
states `(N,d)` and returns detached float64 tensors `(N,1)` and `(N,d)`. Its
optional `iteration` argument evaluates another checkpoint without changing
the selected model. Predictions are not clipped. Loading/evaluation leave the
caller's random-number generators unchanged.

`validation_samples` reproduces the runner's original split using its recorded
seed and train fraction, not a fresh split. The recorded dataset must remain
available. `evaluate_history(run)` rescores every checkpoint on that holdout,
returning arrays `iteration`, `neurons`, `rel_l2`, `rel_grad`, `rel_h1` in
iteration order. Sorting a frontier and taking its lower envelope remain
reporting choices, not a change of the selected checkpoint.

### Objective normalization is not evaluation normalization

Training retains its normalized objective, fidelity, regularizer, acceptance
decisions, and best-checkpoint selection. Approximation errors compare the
original-scale V and gradient, without regularizer or moment-weight factors.

For data scaling `x_n=x/s_x`, `V_n=V/s_v`, the inverse prediction transform is
`V=s_v V_n`, `dV=(s_v/s_x) dV_n`. It lives in `ValueSampleNormalizer`, including
`denormalize_tensors`. H1 remains the paper's relative error. Relative L2 can
be unchanged by a common value scale while H1 changes with input scaling.

The generic runner passes `reporting_normalizer` to `PDAP.fit`; new records
declare `metric_coordinates: physical`. Direct trainer callers that normalize
their samples should supply that same transform. The default `None` preserves
the untransformed caller contract. Historical errors are never relabeled or
overwritten by reporting.

`run.stored_metric_coordinates` identifies the convention of the saved errors.
An absent `metric_coordinates` tag means `training`, including records that
already contain `normalization`; unknown labels are rejected. Loading a
normalized run with training-coordinate metrics warns about this distinction.
Use `validation_metrics(run)` or `evaluate_history(run)` to obtain physical
errors instead of comparing those results with stored training-coordinate H1
values. Loading does not alter the original record or its saved errors.

### Historical records without normalization metadata

The current paper's old sweeps use an explicit compatibility opt-in:

```python
run = load_run(record_path, recover_legacy_normalization=True)
```

Recovery reconstructs the historical max-absolute transform from the recorded
dataset and must reproduce all three saved validation errors at the saved
checkpoint, in the old training coordinates. It rejects disagreement,
contradictory/null metadata, and records declaring another metric-coordinate
convention. It does not change the old record; `run.normalization_recovered`
marks the recovery. Default loading remains strict.

This consistency check does not independently prove the original dataset's
provenance. Retain original datasets and record/artifact provenance checks;
a matching filename alone is not sufficient evidence.

## Render prepared arrays

```python
from src.results import value_surface_grid
from src.plots import plot_value_surface, save_figure

grid = value_surface_grid(run, x_range=(-8, 8), y_range=(-8, 8), grid_n=220)
fig, ax = plot_value_surface(*grid, clip=(0, 60))
save_figure(fig, output_path, tight=False, bbox_inches="tight")
```

Surface clipping changes a display copy, never the arrays used for metrics or
feedback. Existing colormaps, sparse ticks, camera angles and panes are retained.
Plotters return figure/axes; `save_figure` handles export and closing. Layout
and bounding-box options remain explicit to preserve geometry. New exports use
PNG at 300 dpi; legacy wrappers retain their existing return shapes.

Batch experiment/paper entry points select Matplotlib's Agg backend before
importing pyplot-dependent helpers. Backend selection affects text layout and
tight bounding boxes even for PNG export: the macOS desktop backend produced
different VDP image dimensions with the same font and numerical data. Shared
plotting functions remain usable from interactive callers without forcing their
backend.

`plot_neuron_h1_frontier` retains its series-dictionary interface. The summary
frontier has a separate renderer because its dimensions and marker placement
differ. Cross-sections, feedback paths, weight portraits, regional comparisons
and reference-data figures use extracted renderers in the same module.
Reference/PMP calculations stay outside them.

Legacy `plot_model_value_surface` and `_best_iteration_atoms` imports remain
compatibility wrappers. New callers should use bound SavedRuns and prepared
arrays, not separately supplied activations or re-fitted normalizers.

## Reproduction and scope

See the local [paper pipeline](../scripts/paper/README.md) for current manuscript
commands. Its VDP Algorithm 2 frontier comparison uses alpha=1e-6; the surface/table comparison
uses alpha=1e-5. These are deliberately distinct. Existing run selection stays
fixed during rescoring; retuning is a separate experiment.

The VDP and pendulum full-scope generators emit 13 and 17 PNGs respectively.
The open-loop generators write five reference images directly to `paper/plot/`,
and the joint moment-order study writes one `p_study.png`. Both Algorithm 1 and
Algorithm 2 study `analysis.py` wrappers delegate to the same full-scope generator
for their benchmark. They no longer produce separate all-grid or diagnostic plots.

After regeneration, check the TeX-derived image allowlist as well as provenance:

```sh
uv run python scripts/paper/preflight.py --require-sidecars --require-figures
```

This rejects missing included images and surplus PNGs in `experiments/` and
`paper/plot/`. It does not check the mathematical correctness of their contents.

```sh
uv run pytest tests/test_results.py tests/test_history.py tests/test_experiment_logging.py
uv run pytest
```

Checks cover anisotropic scaling, metadata recovery/failure, holdout and RNG
preservation, unchanged training decisions, and display-only clipping.
The earlier physical-metric correction changed error frontiers. The subsequent
paper-only cleanup preserves all 36 included outputs byte-for-byte against
controlled post-correction, pre-trim generator baselines, and leaves the stored
manuscript PNGs untouched. Snapshot equality does not certify a reference solution:
Figure 12's known reference-selection defect remains a separate correction.

Per ADR 0011, reusable code/tests are tracked; paper-specific generators and
artifacts remain local/ignored. Only manuscript TeX/PDF are tracked in `paper/`.
The subsequent [cleanup](../experiments/README.md) also trimmed `scripts/paper/`
to the manuscript pipeline. Nineteen retired experiment source files, the shared
all-grid analyzer, pre-trim source/documentation copies, and 170 unreferenced PNGs
are recoverable under `outdated/experiment-code-cleanup-2026-09-12/`. The images
retain their bytes and repository-relative paths under `unreferenced_pngs/`;
its `PNG_SHA256SUMS` verifies them. Historical reports label superseded metrics
and note locally archived images without linking to unavailable files.
Investigation scripts outside the manuscript pipeline were not removed.
