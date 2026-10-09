"""The training record produced by ``run_training``.

Evaluation computes the metrics; ``History.record`` receives those values,
stores them alongside parameter snapshots and selects the best checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from ..data import ValueSampleNormalizer
from ..models import SparseModel


@dataclass
class History:
    """Per-iteration losses, relative errors, and support snapshots.

    ``reporting_normalizer`` records the transform used by the external evaluator
    for relative errors. Losses and checkpoint selection use training coordinates.
    """

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    err_l2_train: List[float] = field(default_factory=list)
    err_l2_val: List[float] = field(default_factory=list)
    err_grad_train: List[float] = field(default_factory=list)
    err_grad_val: List[float] = field(default_factory=list)
    err_h1_train: List[float] = field(default_factory=list)
    err_h1_val: List[float] = field(default_factory=list)
    data_loss_train: List[float] = field(default_factory=list)
    data_loss_val: List[float] = field(default_factory=list)
    value_loss_train: List[float] = field(default_factory=list)
    value_loss_val: List[float] = field(default_factory=list)
    gradient_loss_train: List[float] = field(default_factory=list)
    gradient_loss_val: List[float] = field(default_factory=list)
    sparsity_functional: List[float] = field(default_factory=list)
    psi_p: List[float | None] = field(default_factory=list)
    alpha_phi: List[float] = field(default_factory=list)
    total_variation: List[float] = field(default_factory=list)
    radius_r95: List[float] = field(default_factory=list)
    radius_max: List[float] = field(default_factory=list)
    #: Cap of the search that produced the support being evaluated:
    #: min(theorem radius, exp(5)), fallback exp(5), or 1 on the unit sphere.
    #: None means no search. In correction-first order, this is the preceding
    #: insertion (initialization for row 0); the final insertion is unrecorded.
    search_radius: List[float | None] = field(default_factory=list)
    #: 1 = theorem radius available, 0 = requested but a required hypothesis
    #: failed (fallback used), -1 = not requested (radial_cap=fixed, or a
    #: sphere activation, where there is no radial search to bound).
    radius_theorem_applied: List[float | None] = field(default_factory=list)
    inner_weights: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    outer_weights: List[torch.Tensor] = field(default_factory=list)
    model_states: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    penalty_exponent: float | None = None
    loss_weights: tuple[float, float] = (1.0, 1.0)
    best_iteration: int = 0
    best_train_loss: float = float("inf")
    final_neurons: int = 0
    reporting_normalizer: ValueSampleNormalizer | None = None
    diagnostics: List[dict[str, Any]] = field(default_factory=list)
    #: Actual solver work and runtime, including searches without a checkpoint.
    #: Seconds are wall time; process_peak_rss_bytes is the process lifetime peak.
    run_metrics: dict[str, Any] = field(default_factory=dict)

    def record(
        self, problem: SparseModel, values: dict[str, float | None],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Store supplied measurements and an independent checkpoint.

        Search diagnostics describe the search that produced
        the support being evaluated, in either loop order. They separate a
        ``radial_cap=theorem`` run whose hypothesis failed from a
        ``radial_cap=fixed`` run: both search
        inside ``exp(5)``, and without these the records are identical.
        """
        for name, value in values.items():
            getattr(self, name).append(value)
        diagnostics = dict(diagnostics or {})
        self.diagnostics.append(diagnostics)
        search_radius = diagnostics.get("search_radius")
        theorem_applied = diagnostics.get("radius_theorem_applied")
        self.search_radius.append(None if search_radius is None else float(search_radius))
        self.radius_theorem_applied.append(None if theorem_applied is None else float(theorem_applied))
        self.penalty_exponent = problem.penalty_exponent
        self.loss_weights = problem.loss_weights
        state = {name: tensor.detach().clone() for name, tensor in problem.state_dict().items()}
        self.model_states.append(state)
        self.inner_weights.append({"weight": state["inner_weight"], "bias": state["inner_bias"]})
        self.outer_weights.append(state["c"].reshape(1, -1))
        i = len(self.train_loss) - 1
        if self.train_loss[i] < self.best_train_loss:
            self.best_train_loss = self.train_loss[i]
            self.best_iteration = i

    @property
    def best_neurons(self) -> int:
        return int(self.inner_weights[self.best_iteration]["weight"].shape[0])

    @property
    def best_err_l2_train(self) -> float:
        return self.err_l2_train[self.best_iteration]

    @property
    def best_err_h1_train(self) -> float:
        return self.err_h1_train[self.best_iteration]

    def restore_model(self, model, iteration: int | None = None):
        """Restore ``model`` to one recorded iteration (the selected best by default).

        A zero-measure snapshot is restored as such: ``model`` is emptied rather
        than left carrying whatever support it held before, so the restored model
        always represents the recorded iteration.
        """
        i = self.best_iteration if iteration is None else int(iteration)
        states = getattr(self, "model_states", [])
        state = states[i] if states else {}
        if "hidden.weight" in state:
            # Pre-refactor snapshots stored two Linear layers. Translate at the
            # artifact reader rather than retaining the removed model classes.
            state = {"inner_weight": state["hidden.weight"], "inner_bias": state["hidden.bias"],
                     "c": state["output.weight"].reshape(-1)}
        elif "W" in state:
            state = {"inner_weight": state["W"], "inner_bias": state["b"], "c": state["c"]}
        if not state:
            state = {"inner_weight": self.inner_weights[i]["weight"],
                     "inner_bias": self.inner_weights[i]["bias"],
                     "c": self.outer_weights[i].reshape(-1)}
        model.set_atoms(state["inner_weight"], state["inner_bias"], state["c"])
        return model

    def summary_metrics(self) -> dict[str, Any]:
        """Selected-checkpoint metrics plus independently counted run work."""
        i = int(self.best_iteration)
        metrics: dict[str, Any] = {
            "rel_l2_train": float(self.err_l2_train[i]),
            "rel_l2_val": float(self.err_l2_val[i]),
            "rel_grad_train": float(self.err_grad_train[i]),
            "rel_grad_val": float(self.err_grad_val[i]),
            "rel_h1_train": float(self.err_h1_train[i]),
            "rel_h1_val": float(self.err_h1_val[i]),
            "best_iteration": i,
            "best_neurons": int(self.best_neurons),
            "final_neurons": int(self.final_neurons),
            # Historical artifact field: checkpoint rows, including a terminal
            # zero-model row when no correction ran. Actual work is counted in
            # run_metrics, independently of observation/checkpoint positions.
            "iterations": int(len(self.train_loss)),
        }
        if getattr(self, "data_loss_train", []):
            w1, w2 = getattr(self, "loss_weights", (1.0, 1.0))
            psi_p = self.psi_p[i]
            metrics.update(
                {
                    "objective_train": float(self.train_loss[i]),
                    "objective_val": float(self.val_loss[i]),
                    "data_loss_train": float(self.data_loss_train[i]),
                    "data_loss_val": float(self.data_loss_val[i]),
                    "value_loss_train": float(self.value_loss_train[i]),
                    "value_loss_val": float(self.value_loss_val[i]),
                    "gradient_loss_train": float(self.gradient_loss_train[i]),
                    "gradient_loss_val": float(self.gradient_loss_val[i]),
                    "data_value_term_train": float(w1 * self.value_loss_train[i]),
                    "data_value_term_val": float(w1 * self.value_loss_val[i]),
                    "data_gradient_term_train": float(w2 * self.gradient_loss_train[i]),
                    "data_gradient_term_val": float(w2 * self.gradient_loss_val[i]),
                    "sparsity_functional": float(self.sparsity_functional[i]),
                    "psi_p": None if psi_p is None else float(psi_p),
                    "alpha_phi": float(self.alpha_phi[i]),
                    "total_variation": float(self.total_variation[i]),
                    "radius_r95": float(self.radius_r95[i]),
                    "radius_max": float(self.radius_max[i]),
                }
            )
            if self.penalty_exponent == 1.0:
                metrics["phi_1"] = float(self.sparsity_functional[i])
                metrics["alpha_phi_1"] = float(self.alpha_phi[i])
        # Old fit pickles have no search telemetry; do not invent provenance.
        for name in ("search_radius", "radius_theorem_applied"):
            values = getattr(self, name, [])
            if len(values) > i and values[i] is not None:
                metrics[name] = float(values[i])
        metrics.update(getattr(self, "run_metrics", {}))
        return metrics
