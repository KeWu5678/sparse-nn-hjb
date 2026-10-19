---
status: accepted
---

# One metric catalog; History is plain data

Amends the run contents of [ADR 0016](0016-mlflow-is-the-system-of-record.md).

Metric names were defined implicitly in several places that had to change
together: the keys returned by evaluation, the `History` dataclass fields
(matched by string), a hand-written rename table in `History.summary_metrics`,
the region evaluation inside `scripts/train.py` and the readers in
`src.results`. One quantity had up to four names (`err_h1_val`, `rel_h1_val`,
`val_h1`, `rel_h1`). No metric had a written meaning or unit, and the run-wide
tag `metric_coordinates=physical` was wrong for every loss, which is measured
in training coordinates. `history.pkl` pickled the project class, so each
refactor of `History` left a reader branch for older layouts; on 2026-10-15
every stored run used the pre-refactor layer layout and the current code read
them only through those branches. The readers also recomputed physical errors
that the runs already stored (identical to 3e-16 on sampled runs).

## Decision

- `src/metrics.py` defines every recorded metric once: name, meaning,
  coordinates (`physical`, `training` or `none`) and which direction is better.
  Names follow `<quantity>_<split>`; the switching tube and the rest of the
  region-evaluation pool are splits (`rel_h1_switching`, `count_rest`).
  `python -m src.metrics` prints the catalog.
- `src.tracking.log_fit` refuses a metric name outside the catalog before
  logging anything. `stop_reason` is set as a tag explicitly; values are no
  longer routed to metrics or tags by Python type.
- `History` stores one row of checkpoint metrics, one atom state and one
  diagnostics record per checkpoint, plus the run totals. The selected
  checkpoint (`best_iteration`) is derived from the rows. The artifact is
  `history.pt`, written with `torch.save` from plain containers and read with
  `torch.load(weights_only=True)`, so it does not depend on project classes.
- `src.eval` owns all metric computation, including the regional evaluation
  (`region_metrics`) and physical-unit prediction (`physical_prediction`).
- Removed: the derived aliases `phi_1`, `alpha_phi_1`, `data_*_term_*` and
  `iterations`; `best_neurons` (now `neurons` at the selected checkpoint); the
  `metric_coordinates` tag; `validation_metrics`, `evaluate_history`,
  `history_atoms`, `model_from_history`, legacy normalization recovery and the
  old-layout restore branches. `src/metric.py` became `src/tables.py`.
- Each catalog name holds its selected-checkpoint value, by which runs are
  compared and ranked. Each checkpoint metric is also logged as the curve
  `trace.<name>` with step = checkpoint index (`src.metrics.TRACE_PREFIX`).
  The curve needs its own key because MLflow reports a key's last-logged step
  as the run's value, which would replace the selected value with the final
  checkpoint's. Regional metrics and run totals have one value and no trace.

## Migration (2026-10-15)

The 588 active runs were converted by a one-time script (not kept in the
repository) into new runs in the same experiments, with the same names,
params, start and end times, provenance tags and `config.json` /
`normalization.json` bytes, plus a `migrated_from` tag naming the old run.
New runs were created rather than edited because MLflow cannot delete a metric
key or an artifact. For every run the script checked that the old output bias
was zero, that the recomputed selection equals the recorded `best_iteration`,
that every carried scalar equals its stored value, and that every checkpoint's
stored train and validation errors equal a fresh physical rescoring. It dropped
the metrics `psi_p`, `search_radius` and `radius_theorem_applied` (no current
producer) and the tags `run_id`, `run_record.path`, `artifact.fit_history.path`
and `status` (pointers to the retired JSON run records). The old runs were
soft-deleted. The database was backed up first as
`deploy/docker/mlflow-data/mlflow-backup-before-adr0017-20261015.db`.

The traces were then added to the same 588 runs from their `history.pt` with
`src.tracking.log_traces` (780,980 points), after a backup as
`mlflow-backup-before-traces-20261015.db`. Adding keys left the run IDs and
every untraced value unchanged; the database grew from 66 MB to 299 MB.

## Consequences

- Adding a metric means computing it in `src.eval` (or the runner's counters)
  and adding its catalog entry; a missing entry fails at logging time and in the
  tracking tests.
- Analysis reads stored metrics and `history.rows` instead of recomputing them.
- Runs recorded before this change cannot be loaded by the current code.
