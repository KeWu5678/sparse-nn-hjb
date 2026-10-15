"""Evaluate a network against value samples under the names in ``src.metrics``.

``evaluate_checkpoint`` measures each recorded checkpoint of a run;
``region_metrics`` scores the selected checkpoint on the region-evaluation
pool. SparseNetwork supplies the fitting objective and Penalty the sparsity
functional. Losses use the training samples' coordinates; relative errors
compare original V and gradient units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import numpy as np
import torch

from .data import full_batch
from .metrics import DISTANCE_BINS

if TYPE_CHECKING:
    from .data import ValueSampleNormalizer
    from .models import Penalty, SparseNetwork

PREDICTION_CHUNK = 100_000


@torch.no_grad()
def evaluate_checkpoint(
    network: SparseNetwork, penalty: Penalty, train_loader, valid_loader=None,
    normalizer: ValueSampleNormalizer | None = None,
) -> dict[str, float]:
    """Checkpoint metrics of both splits and the support.

    Without ``valid_loader`` the validation split is the training split.
    ``normalizer`` converts predictions and targets back to original units
    for the relative errors.
    """
    inner_weight, inner_bias, c = network.inner_weight, network.inner_bias, network.c.detach()
    u = c * network.moment_weight_of_atoms()
    phi = penalty.sparsity_functional(u) if c.numel() else c.new_zeros(())
    alpha_phi = float(penalty.alpha * phi)
    radius = torch.sqrt((inner_weight * inner_weight).sum(dim=-1) + inner_bias * inner_bias)
    values: dict[str, float] = {
        "neurons": network.n_neurons,
        "sparsity_functional": float(phi),
        "alpha_phi": alpha_phi,
        "total_variation": float(c.reshape(-1).abs().sum()),
        "radius_r95": float(amplitude_mass_radius(radius, c)),
        "radius_max": float(radius.max()) if c.numel() else 0.0,
    }
    if valid_loader is None:
        valid_loader = train_loader
    for split, loader in (("train", train_loader), ("val", valid_loader)):
        data = full_batch(loader)
        prediction = network(data[0])
        target = data[1:]
        fitting = network.obj(data[0], target)
        n = data[0].shape[0]
        values[f"objective_{split}"] = float(fitting) + alpha_phi
        values[f"data_loss_{split}"] = float(fitting)
        values[f"value_loss_{split}"] = float(torch.sum((prediction[0] - target[0]) ** 2) / (2 * n))
        values[f"gradient_loss_{split}"] = float(torch.sum((prediction[1] - target[1]) ** 2) / (2 * n))
        if normalizer is not None:
            prediction = normalizer.denormalize_tensors(*prediction)
            target = normalizer.denormalize_tensors(*target)
        for kind, error in relative_errors(*prediction, *target).items():
            values[f"rel_{kind}_{split}"] = error
    return values


def amplitude_mass_radius(
    radius: torch.Tensor, c: torch.Tensor, mass_fraction: float = 0.95,
) -> torch.Tensor:
    """Smallest atom radius ``|omega_j|`` containing ``mass_fraction`` of ``sum |c_j|``.

    This is the R_0.95-type radius diagnostic for normalized Algorithm 1. Atoms
    are sorted by radius and accumulated with weights ``|c_j|``. Empty supports
    and supports with zero total variation have radius zero.
    """
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("mass_fraction must lie in (0, 1]")
    if c.numel() == 0:
        return c.new_zeros(())

    amplitude = c.abs()
    total = amplitude.sum()
    if float(total) == 0.0:
        return total

    order = torch.argsort(radius)
    cumulative = torch.cumsum(amplitude[order], dim=0)
    index = torch.searchsorted(cumulative, mass_fraction * total).clamp_max(c.numel() - 1)
    return radius[order][index]


@torch.no_grad()
def relative_errors(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
) -> dict[str, float]:
    """Relative ``l2``, ``grad`` and ``h1`` errors as plain floats.

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
    return {"l2": float(err_l2.item()), "grad": float(err_grad.item()), "h1": float(err_h1.item())}


@torch.no_grad()
def physical_prediction(
    network: SparseNetwork, x: Any, normalizer: ValueSampleNormalizer | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unclipped original-unit (V, ∇V), shapes (N,1) and (N,d), at physical states x.

    States are scaled into the network's training coordinates and predictions
    scaled back; inputs are evaluated in chunks of ``PREDICTION_CHUNK`` rows.
    """
    x = torch.as_tensor(x, dtype=torch.float64, device=network.inner_weight.device)
    if x.ndim != 2 or x.shape[1] != network.input_dim:
        raise ValueError("physical states must have shape (N, model input dimension)")
    if normalizer is not None:
        x = x / torch.as_tensor(normalizer.x_scale, dtype=x.dtype, device=x.device)
    parts = [network(x[lo:lo + PREDICTION_CHUNK]) for lo in range(0, len(x), PREDICTION_CHUNK)]
    v, dv = (torch.cat(chunks) for chunks in zip(*parts)) if parts else network(x)
    if normalizer is not None:
        v, dv = normalizer.denormalize_tensors(v, dv)
    return v, dv


@torch.no_grad()
def region_metrics(
    network: SparseNetwork, normalizer: ValueSampleNormalizer | None,
    pool: Mapping[str, np.ndarray], *, tube_radius: float,
    samples: Mapping[str, np.ndarray] | None = None, distance: np.ndarray | None = None,
) -> Dict[str, float]:
    """Switching-tube and rest errors on the pool, and the distance-binned profile.

    ``pool`` holds the region-evaluation pool's physical ``x``, ``v``, ``dv``
    and ``distance`` to the switching set; the tube is ``distance <= tube_radius``.
    The pool excludes the training rows, so these errors are out of sample.
    The profile is scored on the physical training dataset ``samples``, whose
    row-aligned ``distance`` comes from the dataset's distance cache.
    """
    v_pred, dv_pred = physical_prediction(network, pool["x"], normalizer)
    values = region_split_errors(
        v_pred, dv_pred,
        torch.as_tensor(pool["v"], dtype=torch.float64).reshape(-1, 1),
        torch.as_tensor(pool["dv"], dtype=torch.float64),
        torch.from_numpy(pool["distance"] <= tube_radius),
    )
    if distance is not None:
        if samples is None or distance.shape[0] != samples["x"].shape[0]:
            raise ValueError("the distance cache is not aligned to the dataset")
        v_pred, dv_pred = physical_prediction(network, samples["x"], normalizer)
        values.update(distance_binned_error(
            v_pred, dv_pred,
            torch.as_tensor(samples["v"], dtype=torch.float64).reshape(-1, 1),
            torch.as_tensor(samples["dv"], dtype=torch.float64),
            torch.as_tensor(distance, dtype=torch.float64),
        ))
    return values


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

    Two families of per-region numbers are returned (see
    ``experiments/02_pendulum/region_split/README.md`` for the rationale):

      * ``rel_{l2,grad,h1}_{switching,rest}`` — region-local *relative* errors
        (each region normalized by its own ‖true‖). Well posed when the
        switching region carries large |V| (the two-sided pool); historically
        confounded when the region held only small-|V| samples.
      * ``l1_{value,grad,h1}_{switching,rest}`` — region **mean** per-sample
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
        "count_switching": float(int(switching.sum().item())),
        "count_rest": float(int(rest.sum().item())),
    }
    for region, sel in (("switching", switching), ("rest", rest)):
        if not bool(sel.any()):
            metrics.update({f"rel_{kind}_{region}": float("nan") for kind in ("l2", "grad", "h1")})
            metrics.update({f"l1_{kind}_{region}": float("nan") for kind in ("value", "grad", "h1")})
            continue
        errors = relative_errors(v_pred[sel], dv_pred[sel], v_true[sel], dv_true[sel])
        for kind, error in errors.items():
            metrics[f"rel_{kind}_{region}"] = error

    # Mean per-sample L1 (absolute), normalized by global mean ‖true‖ — count-fair
    # across regions and robust to V→0.
    v_err = (v_pred - v_true).abs().reshape(-1)
    g_err = (dv_pred - dv_true).norm(dim=1)
    v_den = v_true.abs().mean().clamp_min(1e-30)
    g_den = dv_true.norm(dim=1).mean().clamp_min(1e-30)
    for region, sel in (("switching", switching), ("rest", rest)):
        if not bool(sel.any()):
            continue
        nv = v_err[sel].mean()
        ng = g_err[sel].mean()
        metrics[f"l1_value_{region}"] = float((nv / v_den).item())
        metrics[f"l1_grad_{region}"] = float((ng / g_den).item())
        metrics[f"l1_h1_{region}"] = float(((nv + ng) / (v_den + g_den)).item())
    return metrics


@torch.no_grad()
def distance_binned_error(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    distance: torch.Tensor,
    n_bins: int = DISTANCE_BINS,
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
        ratio = (per_sample[in_bin].mean() / overall) if bool(in_bin.any()) else per_sample.new_tensor(float("nan"))
        metrics[f"distbin{i + 1}_ratio"] = float(ratio.item())
    return metrics
