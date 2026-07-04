"""Performance evaluation of a model's predictions against value samples.

Pure functions of (prediction, target) tensors with no model state: a model
produces ``(V, dV)`` via ``predict_tensors`` and the trainer scores them here.
This is the single home for the data-fidelity numbers: the relative errors and
the value/gradient loss split (the data term of the training objective).

The regularizer is *not* here: the nonconvex penalty lives with the trainer
(``src.PDAP.ssn_solve.nonconvex_penalty``), which adds it to ``data_loss`` to
form the full objective.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def data_loss_terms(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    loss_weights: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(data_loss, value_loss, grad_loss)`` for the data-fidelity term.

    MSE is normalized by ``Nx = N * d`` (matching the MATLAB reference, where
    ``Nx = numel(xhat) = d * N``) and halved.  ``data_loss`` weights the two by
    ``loss_weights = (w1, w2)``.  Differentiable (plain torch ops), so it can sit
    inside an SSN closure as well as serve evaluation.
    """
    nx = v_true.shape[0] * dv_true.shape[1]
    value_loss = torch.sum((v_pred - v_true) ** 2) / (2 * nx)
    grad_loss = torch.sum((dv_pred - dv_true) ** 2) / (2 * nx)
    w1, w2 = loss_weights
    data_loss = w1 * value_loss + w2 * grad_loss
    return data_loss, value_loss, grad_loss


@torch.no_grad()
def relative_errors(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return relative ``(L2, gradient, H1)`` errors as plain floats.

    Each is ``||pred - true|| / ||true||`` in the relevant norm; denominators are
    clamped to ``1e-30`` to stay finite on all-zero targets.
    """
    v_diff = torch.sum((v_pred - v_true) ** 2)
    dv_diff = torch.sum((dv_pred - dv_true) ** 2)
    v_ref = torch.sum(v_true ** 2)
    dv_ref = torch.sum(dv_true ** 2)

    err_l2 = torch.sqrt(v_diff / v_ref.clamp_min(1e-30))
    err_grad = torch.sqrt(dv_diff / dv_ref.clamp_min(1e-30))
    err_h1 = torch.sqrt((v_diff + dv_diff) / (v_ref + dv_ref).clamp_min(1e-30))
    return float(err_l2.item()), float(err_grad.item()), float(err_h1.item())


@torch.no_grad()
def region_split_errors(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    switching_mask: torch.Tensor,
) -> Dict[str, float]:
    """Errors split by a switching-tube / rest boolean mask.

    ``switching_mask`` is a length-``N`` boolean tensor over the same samples as
    the predictions: ``True`` = inside the switching tube, ``False`` = rest.
    (Records written before the tube consolidation carry the same metrics under
    the legacy ``near``/``far`` key prefixes.)

    Two families of per-region numbers are returned (see
    ``experiments/02_pendulum/region_split/README.md`` for the rationale):

      * ``{switching,rest}_{l2,grad,h1}`` — region-local *relative* errors (each
        region normalized by its own ‖true‖). Well posed when the switching
        region carries large |V| (the two-sided pool); historically confounded
        when the region held only small-|V| samples.
      * ``{switching,rest}_l1_{value,grad,h1}`` — region **mean** per-sample
        absolute (L1) error, normalized by the **global mean** ‖true‖ over *all*
        samples. L1 is the norm of choice near discontinuities (L2 worst-case
        bounds do not generalize across a shock); using the per-sample *mean*
        (not the sum) keeps the regions count-fair; the shared global denominator
        removes the per-region V→0 confound while keeping the regions on one
        comparable scale.

    Region counts are included so an empty/degenerate region is visible.
    """
    switching = switching_mask.to(torch.bool)
    rest = ~switching
    metrics: Dict[str, float] = {
        "switching_count": float(int(switching.sum().item())),
        "rest_count": float(int(rest.sum().item())),
    }
    for tag, sel in (("switching", switching), ("rest", rest)):
        if not bool(sel.any()):
            for suffix in ("l2", "grad", "h1", "l1_value", "l1_grad", "l1_h1"):
                metrics[f"{tag}_{suffix}"] = float("nan")
            continue
        l2, grad, h1 = relative_errors(
            v_pred[sel], dv_pred[sel], v_true[sel], dv_true[sel]
        )
        metrics[f"{tag}_l2"] = l2
        metrics[f"{tag}_grad"] = grad
        metrics[f"{tag}_h1"] = h1

    # Mean per-sample L1 (absolute), normalized by global mean ‖true‖ — count-fair
    # across regions and robust to V→0.
    v_err = (v_pred - v_true).abs().reshape(-1)
    g_err = (dv_pred - dv_true).norm(dim=1)
    v_den = v_true.abs().mean().clamp_min(1e-30)
    g_den = dv_true.norm(dim=1).mean().clamp_min(1e-30)
    for tag, sel in (("switching", switching), ("rest", rest)):
        if not bool(sel.any()):
            continue
        nv = v_err[sel].mean()
        ng = g_err[sel].mean()
        metrics[f"{tag}_l1_value"] = float((nv / v_den).item())
        metrics[f"{tag}_l1_grad"] = float((ng / g_den).item())
        metrics[f"{tag}_l1_h1"] = float(((nv + ng) / (v_den + g_den)).item())
    return metrics


@torch.no_grad()
def distance_binned_error(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    distance: torch.Tensor,
    n_bins: int = 30,
) -> Dict[str, float]:
    """Per-sample absolute H1 error in equal-width bins of distance-to-switching-set.

    The literature-standard diagnostic for behaviour near a discontinuity (Gibbs
    error decays with distance from the kink). Bins are equal-*width* in distance
    (``n_bins`` uniform intervals spanning the dataset's distance range), so the
    x-axis is a true spatial coordinate. Sample counts per bin are therefore uneven
    — dense near the switching set, sparse in the far tail (the analysis overlays
    the counts). The reported value is the bin's mean per-sample absolute error
    ``|ΔV| + ‖Δ∇V‖`` divided by that mean over *all* samples — a scale-free ratio,
    so ``bin_ratio > 1`` means worse-than-average there, and the curve is comparable
    across models. Bins are indexed from nearest the switching set (``bin1``)
    outward; empty bins report NaN.
    """
    per_sample = (v_pred - v_true).abs().reshape(-1) + (dv_pred - dv_true).norm(dim=1)
    overall = per_sample.mean().clamp_min(1e-30)
    d = distance.reshape(-1)
    edges = torch.linspace(d.min(), d.max(), n_bins + 1, dtype=d.dtype)
    metrics: Dict[str, float] = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (d >= lo) & (d <= hi) if i == n_bins - 1 else (d >= lo) & (d < hi)
        ratio = (per_sample[in_bin].mean() / overall) if bool(in_bin.any()) else torch.tensor(float("nan"))
        metrics[f"distbin{i + 1}_ratio"] = float(ratio.item())
    return metrics
