"""The quantities a training run records, each defined once.

``CATALOG`` lists every metric name a run may record: evaluation produces
these names, History stores them, ``src.tracking`` refuses any other name, and
analysis reads them back from MLflow. Names follow ``<quantity>_<split>``. The
splits are the training and validation samples and, for regional evaluation,
the switching tube and the rest of the region-evaluation pool.

``coordinates`` names where a value is measured: ``physical`` (original V and
gradient units), ``training`` (the normalized samples and coefficients the
objective uses) or ``none`` (counts, indices, times and memory).

A run records each name once, at its selected checkpoint, and each checkpoint
metric also as a training curve named with ``TRACE_PREFIX``.

Run ``python -m src.metrics`` to print the catalog as a Markdown table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SPLITS = ("train", "val")
REGIONS = ("switching", "rest")
DISTANCE_BINS = 30


@dataclass(frozen=True)
class Metric:
    name: str
    meaning: str
    coordinates: Literal["physical", "training", "none"] = "none"
    better: Literal["lower", "higher", "none"] = "none"


_RELATIVE = {
    "l2": "relative value error ‖V̂ − V‖ / ‖V‖",
    "grad": "relative gradient error ‖∇V̂ − ∇V‖ / ‖∇V‖",
    "h1": "relative H1 error sqrt((‖V̂ − V‖² + ‖∇V̂ − ∇V‖²) / (‖V‖² + ‖∇V‖²))",
}
_REGIONAL_L1 = {
    "value": "mean |V̂ − V| in the region / mean |V| over the pool",
    "grad": "mean ‖∇V̂ − ∇V‖ in the region / mean ‖∇V‖ over the pool",
    "h1": "(mean |V̂ − V| + mean ‖∇V̂ − ∇V‖) in the region / the same means of the targets over the pool",
}
_REGION = {"switching": "in the switching tube of the region-evaluation pool",
           "rest": "outside the switching tube of the region-evaluation pool"}

_CHECKPOINT = (
    *(Metric(f"objective_{split}", f"data_loss_{split} + αΦ, the PDAP objective", "training", "lower")
      for split in SPLITS),
    *(Metric(f"data_loss_{split}", "weighted H1 fitting loss w₁·value_loss + w₂·gradient_loss",
             "training", "lower") for split in SPLITS),
    *(Metric(f"value_loss_{split}", "Σ (V̂ − V)² / 2N", "training", "lower") for split in SPLITS),
    *(Metric(f"gradient_loss_{split}", "Σ ‖∇V̂ − ∇V‖² / 2N", "training", "lower") for split in SPLITS),
    *(Metric(f"rel_{kind}_{split}", f"{meaning}, {split} split", "physical", "lower")
      for split in SPLITS for kind, meaning in _RELATIVE.items()),
    Metric("neurons", "support size N"),
    Metric("sparsity_functional", "Φ(u) of the moment-weighted coefficients u = c·w_p", "training"),
    Metric("alpha_phi", "the penalty term αΦ(u)", "training"),
    Metric("total_variation", "Σ |c_j|", "training"),
    Metric("radius_r95", "smallest atom radius ‖(a_j, b_j)‖ holding 95% of Σ |c_j|", "training"),
    Metric("radius_max", "largest atom radius ‖(a_j, b_j)‖", "training"),
)
"""Measured at every checkpoint by ``src.eval.evaluate_checkpoint``."""

_RUN = (
    Metric("best_iteration", "selected checkpoint: the least objective_train"),
    Metric("final_neurons", "support size when the run stopped, after any final uncorrected insertion"),
    Metric("cap_of_theorem_applied",
           "1 if the exp(5) cap bounded the insertion search radius, 0 if the theorem's radius did"),
    Metric("total_insertion_searches", "candidate searches run"),
    Metric("total_inserted_neurons", "atoms inserted"),
    Metric("total_pruned_neurons", "atoms pruned"),
    Metric("total_ssn_iterations", "semismooth Newton iterations"),
    Metric("total_ssn_failed_steps", "rejected semismooth Newton trial steps"),
    Metric("total_ssn_refinements", "coefficient corrections attempted"),
    Metric("total_accepted_ssn_refinements", "coefficient corrections accepted"),
    Metric("solver_seconds", "wall time of PDAP operations"),
    Metric("evaluation_seconds", "wall time of checkpoint evaluation"),
    Metric("process_peak_rss_bytes", "lifetime peak resident memory of the training process"),
)
"""Recorded once per run by ``src.training.run_training``."""

_REGIONAL = (
    *(Metric(f"rel_{kind}_{region}", f"{meaning} {_REGION[region]}", "physical", "lower")
      for region in REGIONS for kind, meaning in _RELATIVE.items()),
    *(Metric(f"l1_{kind}_{region}", f"{meaning}, {_REGION[region]}", "physical", "lower")
      for region in REGIONS for kind, meaning in _REGIONAL_L1.items()),
    *(Metric(f"count_{region}", f"samples {_REGION[region]}") for region in REGIONS),
    *(Metric(f"distbin{k}_ratio",
             f"mean |V̂ − V| + ‖∇V̂ − ∇V‖ in distance bin {k} (1 = nearest the switching set) "
             "/ that mean over the dataset", "physical") for k in range(1, DISTANCE_BINS + 1)),
)
"""Measured at the selected checkpoint by ``src.eval.region_metrics``."""

CHECKPOINT_METRICS = tuple(metric.name for metric in _CHECKPOINT)
CATALOG = {metric.name: metric for metric in (*_CHECKPOINT, *_RUN, *_REGIONAL)}

TRACE_PREFIX = "trace."
"""``trace.<name>`` is checkpoint metric ``<name>`` at every checkpoint, with
step = checkpoint index. MLflow reports a key's last-logged value as the run's
value, so ``<name>`` itself keeps the selected checkpoint's value for ranking."""


if __name__ == "__main__":
    from .tables import format_table

    print(format_table([vars(metric) for metric in CATALOG.values()],
                       ["name", "meaning", "coordinates", "better"]))
