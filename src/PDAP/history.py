"""The training record produced by ``run_training``.

Evaluation computes each checkpoint's metrics; ``History.record`` stores them
with the checkpoint's atoms and the diagnostics of the search and correction
that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..models import SparseNetwork


@dataclass
class History:
    """One row of metrics per checkpoint, with its atoms and solver diagnostics.

    ``rows[i]`` holds the ``src.metrics`` checkpoint metrics of checkpoint i and
    ``states[i]`` its atoms (``inner_weight``, ``inner_bias``, ``c``).
    ``diagnostics[i]`` describes the search that produced the support: in
    correction-first order, the preceding insertion (initialization for row 0).
    ``run`` holds the run totals of solver work, runtime and final support.
    """

    rows: list[dict[str, float]] = field(default_factory=list)
    states: list[dict[str, torch.Tensor]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    run: dict[str, float] = field(default_factory=dict)
    stop_reason: str | None = None

    def record(
        self, network: SparseNetwork, row: dict[str, float],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Store one checkpoint's metrics, diagnostics and an independent copy of its atoms."""
        self.rows.append(dict(row))
        self.diagnostics.append(dict(diagnostics or {}))
        self.states.append({name: tensor.detach().clone() for name, tensor in network.state_dict().items()})

    @property
    def best_iteration(self) -> int:
        """The first checkpoint with the least finite training objective, else 0."""
        objective = [row["objective_train"] for row in self.rows]
        return min((i for i, value in enumerate(objective) if value < math.inf),
                   key=objective.__getitem__, default=0)

    def restore(self, network: SparseNetwork, iteration: int | None = None) -> SparseNetwork:
        """Set ``network`` to one recorded checkpoint, the selected one by default."""
        i = self.best_iteration if iteration is None else int(iteration)
        if not 0 <= i < len(self.states):
            raise IndexError(f"checkpoint {i} outside recorded history of length {len(self.states)}")
        state = self.states[i]
        network.set_atoms(state["inner_weight"], state["inner_bias"], state["c"])
        return network

    def summary(self) -> dict[str, float]:
        """Selected-checkpoint metrics and run totals, as recorded in MLflow."""
        i = self.best_iteration
        summary = {**self.rows[i], "best_iteration": i, **self.run}
        cap_applied = self.diagnostics[i].get("cap_of_theorem_applied")
        if cap_applied is not None:
            summary["cap_of_theorem_applied"] = float(cap_applied)
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Plain containers and tensors, loadable with ``torch.load(weights_only=True)``."""
        return {"rows": self.rows, "states": self.states, "diagnostics": self.diagnostics,
                "run": self.run, "stop_reason": self.stop_reason}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> History:
        return cls(**values)
