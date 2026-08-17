"""The training record produced by ``PDAP.fit``.

``PDAP`` is a pure trainer: the per-iteration metrics and weight snapshots are
not its state, they are returned to the caller in a :class:`History` (the Keras
``fit``-returns-``History`` convention).  The metric *computation* is pure
evaluation (``src.eval``) plus the regularizer; ``History.record`` orchestrates
it once per iteration so the training loop stays free of evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch.nn.utils import parameters_to_vector

from ..eval import data_loss_terms, relative_errors
from .moment import (
    amplitude_mass_radius,
    atom_normalizer,
    moment_weight,
)
from .ssn_solve import Objective, nonconvex_penalty


def _regularizer_terms(model, objective: Objective) -> dict[str, float]:
    """Raw functionals and weighted regularizer terms on the current support."""
    W, b, c = model.get_atoms()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if params:
        theta = parameters_to_vector(params).detach()
        penalized, nonneg = model.penalty_masks()
        # Normalized objective: the penalty is evaluated at the normalized measure,
        # phi(w_p(omega_n)|c_n|) = phi(|u_n|), so score it on u = w_p * c -- the same
        # quantity ssn_solve minimizes, so the recorded objective cannot drift from it.
        if objective.normalized:
            scale = atom_normalizer(W, b, p=objective.moment_order)
            if scale.numel() != theta.numel():
                raise RuntimeError(
                    "normalized Algorithm 1 expects one trainable coordinate per atom "
                    f"(got {theta.numel()} for {scale.numel()} atoms)"
                )
            theta = theta * scale
        phi = nonconvex_penalty(
            theta,
            penalized,
            nonneg,
            alpha=1.0,
            th=objective.th,
            gamma=objective.gamma,
            q=model.q,
        )
    else:
        phi = c.new_zeros(())
    weights = moment_weight(W, b, objective.moment_order)
    psi = torch.sum(weights.reshape(-1) * c.reshape(-1).abs())
    total_variation = c.reshape(-1).abs().sum()
    if c.numel():
        radius = torch.sqrt((W * W).sum(dim=-1) + b * b)
        radius_max = radius.max()
    else:
        radius_max = c.new_zeros(())

    return {
        "sparsity_functional": float(phi.detach()),
        "psi_p": float(psi.detach()),
        "alpha_phi": float((objective.alpha * phi).detach()),
        "total_variation": float(total_variation.detach()),
        "radius_r95": float(amplitude_mass_radius(W, b, c).detach()),
        "radius_max": float(radius_max.detach()),
    }


def objective_value(model, objective: Objective, data) -> float:
    """Return data fidelity plus the configured model-family penalty."""
    X, V, dV = data
    Vp, dVp = model.predict_tensors(X)
    data_loss = data_loss_terms(Vp, dVp, V, dV, objective.loss_weights)[0]
    regularizer = _regularizer_terms(model, objective)
    return float(data_loss.detach()) + regularizer["alpha_phi"]


@dataclass
class History:
    """Per-iteration losses, relative errors, and support snapshots."""

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
    psi_p: List[float] = field(default_factory=list)
    alpha_phi: List[float] = field(default_factory=list)
    total_variation: List[float] = field(default_factory=list)
    radius_r95: List[float] = field(default_factory=list)
    radius_max: List[float] = field(default_factory=list)
    inner_weights: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    outer_weights: List[torch.Tensor] = field(default_factory=list)
    model_states: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    penalty_exponent: float | None = None
    loss_weights: tuple[float, float] = (1.0, 1.0)
    best_iteration: int = 0
    best_train_loss: float = float("inf")
    final_neurons: int = 0

    def record(self, model, objective: Objective, data_train, data_valid) -> None:
        """Evaluate the current model and append one iteration's record."""
        train_pred = model.predict_tensors(data_train[0])
        valid_pred = model.predict_tensors(data_valid[0])
        train_terms = data_loss_terms(
            *train_pred, *data_train[1:], objective.loss_weights
        )
        valid_terms = data_loss_terms(
            *valid_pred, *data_valid[1:], objective.loss_weights
        )
        regularizer = _regularizer_terms(model, objective)
        tl = float(train_terms[0].detach()) + regularizer["alpha_phi"]
        vl = float(valid_terms[0].detach()) + regularizer["alpha_phi"]
        self.train_loss.append(tl)
        self.val_loss.append(vl)
        self.data_loss_train.append(float(train_terms[0].detach()))
        self.data_loss_val.append(float(valid_terms[0].detach()))
        self.value_loss_train.append(float(train_terms[1].detach()))
        self.value_loss_val.append(float(valid_terms[1].detach()))
        self.gradient_loss_train.append(float(train_terms[2].detach()))
        self.gradient_loss_val.append(float(valid_terms[2].detach()))
        self.sparsity_functional.append(regularizer["sparsity_functional"])
        self.psi_p.append(regularizer["psi_p"])
        self.alpha_phi.append(regularizer["alpha_phi"])
        self.total_variation.append(regularizer["total_variation"])
        self.radius_r95.append(regularizer["radius_r95"])
        self.radius_max.append(regularizer["radius_max"])
        self.penalty_exponent = float(model.q)
        self.loss_weights = tuple(float(weight) for weight in objective.loss_weights)
        l2t, gt, h1t = relative_errors(*train_pred, *data_train[1:])
        l2v, gv, h1v = relative_errors(*valid_pred, *data_valid[1:])
        self.err_l2_train.append(l2t); self.err_l2_val.append(l2v)
        self.err_grad_train.append(gt); self.err_grad_val.append(gv)
        self.err_h1_train.append(h1t); self.err_h1_val.append(h1v)
        W, b, c = model.get_atoms()
        self.inner_weights.append({"weight": W, "bias": b})
        self.outer_weights.append(c.reshape(1, -1))
        self.model_states.append(
            {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        )
        i = len(self.train_loss) - 1
        if tl < self.best_train_loss:
            self.best_train_loss = tl
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
        state = self.model_states[i]
        W = self.inner_weights[i]["weight"]
        b = self.inner_weights[i]["bias"]
        c = self.outer_weights[i]
        model.set_atoms(W, b, c)
        if state:
            model.load_state_dict(state)
        return model

    def summary_metrics(self) -> dict[str, float | int]:
        """Scalar comparison metrics at the selected best iteration."""
        i = int(self.best_iteration)
        metrics: dict[str, float | int] = {
            "rel_l2_train": float(self.err_l2_train[i]),
            "rel_l2_val": float(self.err_l2_val[i]),
            "rel_grad_train": float(self.err_grad_train[i]),
            "rel_grad_val": float(self.err_grad_val[i]),
            "rel_h1_train": float(self.err_h1_train[i]),
            "rel_h1_val": float(self.err_h1_val[i]),
            "best_iteration": i,
            "best_neurons": int(self.best_neurons),
            "final_neurons": int(self.final_neurons),
            # How many outer iterations were actually recorded.  Under the paper's
            # loop order this is less than the configured T_out exactly when the
            # run stopped on its own rule -- no candidate cleared the insertion
            # threshold -- which is the difference between converging in the
            # algorithm's terms and merely exhausting the budget.
            "iterations": int(len(self.train_loss)),
        }
        if getattr(self, "data_loss_train", []):
            w1, w2 = getattr(self, "loss_weights", (1.0, 1.0))
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
                    "psi_p": float(self.psi_p[i]),
                    "alpha_phi": float(self.alpha_phi[i]),
                    "total_variation": float(self.total_variation[i]),
                    "radius_r95": float(self.radius_r95[i]),
                    "radius_max": float(self.radius_max[i]),
                }
            )
            if self.penalty_exponent == 1.0:
                metrics["phi_1"] = float(self.sparsity_functional[i])
                metrics["alpha_phi_1"] = float(self.alpha_phi[i])
        return metrics
