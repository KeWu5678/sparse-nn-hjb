#!/usr/bin/env python3
"""Analyse the region_split_pendulum study.

Reads per-run JSON records from ``rawdata/logs/multirun/region_split_pendulum/``
plus the ReLU^p (fractional-exponent penalty) H1 runs from
``rawdata/logs/multirun/penaltypowers/pendulum/`` — same dataset, same
``eval=region_split`` hook, so the region metrics are directly comparable.
The region metrics are computed by the training hook on the live as-fit model over
the **full dataset** (see ``scripts/train.py``); this just aggregates them into
``results.md`` (two tables + the error-vs-distance plot). `near` = lowest 10% of
samples by distance to the switching set; `far` = the rest. See ``README.md`` for
the error-metric rationale. Reproduce with ``make region_split_pendulum`` (and
``make penaltypowers DATA=pendulum`` for the ReLU^p rows).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

EXPERIMENT = "region_split_pendulum"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
# ReLU^p rows (fractional-exponent penalty q = 2/(p+1), gamma=0 by design) come
# from the penaltypowers sweep on the same dataset; only its H1 runs join the
# comparison here (the region_split sweep is H1-only).
RELU_MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "penaltypowers" / "pendulum"
DATASET_STEM = "Pendulum_pmp_value_samples_2000"
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}
_N_BINS = 30
# The Fig. 2 open-loop data panels (value surface, switching set, regions of
# attraction) now live in experiments/00_openloop/pendulum; this analysis only
# scores the region split and plots error-vs-distance.


def _record_row(record: dict[str, Any], *, relu_power: bool) -> dict[str, Any] | None:
    """One table row from a run record; None if it lacks region metrics or scope."""
    cfg = record["config"]
    model = cfg["model"]
    m = record["metrics"][0]["values"]
    if "near_l1_h1" not in m or int(m.get("best_neurons", 0)) == 0:
        return None
    if DATASET_STEM not in cfg["data"]["path"]:
        return None
    loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
    if relu_power:
        # penaltypowers rows: ReLU^p atoms, only the H1 runs join this comparison.
        if model["activation"] != "relu" or loss != "h1":
            return None
        activation = f"relu^{model['power']:g}"
    else:
        activation = model["activation"]
    return {
        "act_name": model["activation"],
        "power": float(model["power"]),
        "data_path": cfg["data"]["path"],
        "kind": model["kind"],
        "insertion": model["insertion"],
        "activation": activation,
        "loss": loss,
        "gamma": float(model["gamma"]),
        "neurons": int(m["best_neurons"]),
        "near_l1": float(m["near_l1_h1"]),
        "far_l1": float(m["far_l1_h1"]),
        "l1_near/far": float(m["near_l1_h1"]) / float(m["far_l1_h1"]) if m["far_l1_h1"] else float("inf"),
        "near_h1": float(m["near_h1"]),
        "far_h1": float(m["far_h1"]),
        "rel_near/far": float(m["near_h1"]) / float(m["far_h1"]) if m["far_h1"] else float("inf"),
        "bins": [m.get(f"distbin{i + 1}_ratio", float("nan")) for i in range(_N_BINS)],
        "cache": cfg.get("eval", {}).get("distance_cache"),
        "result_path": "",  # filled by load_rows (needs the record path)
    }


def load_rows() -> tuple[list[dict[str, Any]], str | None]:
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make region_split_pendulum` first"
        )
    relu_records = sorted(RELU_MULTIRUN_DIR.glob("*/*.json"))
    rows, cache = [], None
    for path, relu_power in [(p, False) for p in records] + [(p, True) for p in relu_records]:
        row = _record_row(json.loads(path.read_text(encoding="utf-8")), relu_power=relu_power)
        if row is None:
            continue
        cache = cache or row.pop("cache")
        row.pop("cache", None)
        row["result_path"] = str(path.parent / f"result_{path.stem}.pkl")
        rows.append(row)
    return rows, cache


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lowest far-region L1 per (kind, insertion, activation, loss).

    For the region_split sweep this selects gamma; for the ReLU^p rows (gamma=0 by
    design — the nonconvexity is the penalty exponent q = 2/(p+1)) it selects alpha.
    """
    def cell(row: dict[str, Any]) -> tuple:
        return (row["kind"], row["insertion"], row["activation"], row["loss"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["far_l1"]))
    return best


def _bin_centers(cache: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Equal-width bin centers, counts, and raw distances from the distance cache.

    Counts depend only on the fixed distance cache (not the model), so they are the
    same for every run; the plot overlays them so under-supported far-tail points
    are visible. Edges match ``distance_binned_error`` (uniform over [min, max],
    last bin inclusive).
    """
    if not cache:
        return None
    d = np.load(DATA_DIR / cache)["distance"]
    edges = np.linspace(d.min(), d.max(), _N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(d, edges)
    return centers, counts, d


def _smooth_distance_curve(values, counts: np.ndarray,
                           *, sigma_bins: float = 1.6, radius: int = 5) -> np.ndarray:
    """Local Gaussian smoothing over equal-width distance bins."""
    y = np.asarray(values, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    out = np.full_like(y, np.nan, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(c) & (c > 0.0)
    for i in range(y.size):
        lo = max(0, i - radius)
        hi = min(y.size, i + radius + 1)
        idx = np.arange(lo, hi)
        m = valid[idx]
        if not np.any(m):
            continue
        dx = idx[m] - i
        w = np.exp(-0.5 * (dx / sigma_bins) ** 2)
        out[i] = float(np.sum(w * y[idx[m]]) / np.sum(w))
    return out


def _smooth_distance_xy(centers: np.ndarray, values, counts: np.ndarray,
                        *, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed distance curve evaluated on a dense x-grid for visual continuity."""
    y = _smooth_distance_curve(values, counts)
    x = np.asarray(centers, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return x[valid], y[valid]
    xs = np.linspace(float(x[valid].min()), float(x[valid].max()), n)
    try:
        from scipy.interpolate import PchipInterpolator
        ys = PchipInterpolator(x[valid], y[valid], extrapolate=False)(xs)
    except Exception:
        ys = np.interp(xs, x[valid], y[valid])
    return xs, ys


def _plot_error_vs_distance(
    best: list[dict[str, Any]], centers: np.ndarray, counts: np.ndarray
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    wanted = set(_MODEL_STYLE)
    rows = [
        r for r in best
        if r["kind"] == "signed" and r["loss"] == "h1" and r["activation"] in wanted
    ]
    rows = sorted(rows, key=lambda r: list(_MODEL_STYLE).index(r["activation"]))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = (centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 0.1
    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=width, color="0.88", zorder=0)
    ax2.set_ylabel("samples per bin", color="0.55")
    ax2.tick_params(axis="y", colors="0.55")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for r in rows:
        color, ls = _MODEL_STYLE[r["activation"]]
        xs, ys = _smooth_distance_xy(centers, r["bins"], counts)
        ax.plot(xs, ys, color=color, ls=ls, lw=2.4,
                label=_DISPLAY[r["activation"]], zorder=3)
    ax.axhline(1.0, color="0.2", lw=0.9, ls="--", zorder=2)
    ax.set_xlabel("distance to switching set")
    ax.set_ylabel("combined per-sample error / model mean")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout(pad=2.0)
    out = OUTPUT_DIR / "figures" / "error_vs_distance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- #
# Model-comparison figures (F3–F7): representative signed H1 models
# ---------------------------------------------------------------------------- #
# Publication house style (CLAUDE.md): palette only, serif + cm mathtext, no
# in-figure titles/annotations (captions carry the identifying info), top/right
# spines hidden, frameless legends, PNG-only at 300 dpi.
PALETTE = {
    "blue_main": "#0F4D92",
    "teal": "#42949E",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "violet": "#9A4D8E",
}
# The compared models (all signed, H1 loss, best run per cell by far L1).
_MODEL_STYLE = {
    "gaussian": (PALETTE["blue_main"], "-"),
    "softplus": (PALETTE["violet"], "-"),
    "relu^2": (PALETTE["red_strong"], "-"),
    "relu^5": (PALETTE["teal"], "-"),
}
_TRUE_COLOR = "0.35"
_DISPLAY = {"relu^2": r"ReLU$^2$", "relu^5": r"ReLU$^5$",
            "gaussian": "gaussian", "softplus": "softplus"}
_SURFACE_MODEL_ORDER = ("gaussian", "softplus", "relu^2", "relu^5")
NEARSWITCH_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "region_split_pendulum_nearswitch"
RELU_NEARSWITCH_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "penaltypowers_nearswitch"
FIG_DIR = OUTPUT_DIR / "figures"


def _apply_publication_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": ["serif"],
        "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "font.size": 12,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "svg.fonttype": "none",
        "text.usetex": False,
    })


def _save_png(fig, name: str, *, tight: bool = True, pad: float = 2.0) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=pad)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"figures/{out.name}"


def select_models(best: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The comparison models out of the best-per-cell rows (signed, H1)."""
    wanted = list(_MODEL_STYLE)
    picked = {r["activation"]: r for r in best
              if r["kind"] == "signed" and r["loss"] == "h1" and r["activation"] in wanted}
    missing = [w for w in wanted if w not in picked]
    if missing:
        raise ValueError(f"missing comparison models in records: {missing}")
    return {w: picked[w] for w in wanted}


def _build_net(row: dict[str, Any]):
    import pickle
    import torch  # noqa: F401
    from src.models.net import ShallowNetwork
    from src.config.activations import get_activation
    from src.plots import _best_iteration_atoms

    with open(row["result_path"], "rb") as f:
        history = pickle.load(f)
    a, b, u = _best_iteration_atoms(history)
    net = ShallowNetwork(
        layer_sizes=[a.shape[1], a.shape[0], 1],
        activation=get_activation(row["act_name"]), p=row["power"],
        inner_weights=a, inner_bias=b, outer_weights=u,
    )
    net.eval()
    return net


def _value_grad_phys(net, x_phys: np.ndarray, norm) -> tuple[np.ndarray, np.ndarray]:
    import torch
    xn = torch.tensor(np.asarray(x_phys, dtype=np.float64) / norm.x_scale,
                      dtype=torch.float64, requires_grad=True)
    val = net(xn)
    (grad,) = torch.autograd.grad(val.sum(), xn)
    return (val.detach().numpy().reshape(-1) * norm.v_scale,
            grad.detach().numpy() * (norm.v_scale / norm.x_scale))


def _load_geometry(best: list[dict[str, Any]]):
    """Dataset, normalizer, switching curve, restricted pool, and tiled raw truth.

    ``pool`` is the branch-restricted in-basin point set (the models' training
    domain — it stops AT the switching curve). The global value function beyond
    the curve is the **lower envelope of the raw (unrestricted) trajectories,
    tiled by 2πk in θ** (the 3k curated dataset covers neither side densely):
    ``rawt`` holds the tiled raw points for envelope queries.
    """
    import pickle
    from src.data import load_value_samples, ValueSampleNormalizer
    from src.OpenLoop.pendulum.nonsmooth import NonsmoothCurve, restrict_trajectory_to_curve

    data_rel = next(r["data_path"] for r in best)
    samples = load_value_samples(data_rel)
    norm = ValueSampleNormalizer.fit(samples)
    data_abs = DATA_DIR / data_rel
    curve = NonsmoothCurve.load_npz(data_abs.with_name(data_abs.stem + "_nonsmooth_curve.npz"))
    raw_pkl = sorted(data_abs.parent.glob("*raw_trajectories*.pkl"))[0]
    with open(raw_pkl, "rb") as f:
        raw = pickle.load(f)
    xs, vs, dvs = [], [], []
    for tr in raw:
        cut, _ = restrict_trajectory_to_curve(tr, curve)
        if cut.state.size:
            xs.append(cut.state); vs.append(cut.value); dvs.append(cut.costate)
    pool = {"x": np.vstack(xs), "v": np.concatenate(vs), "dv": np.vstack(dvs)}
    x_raw = np.vstack([tr.state for tr in raw])
    v_raw = np.concatenate([tr.value for tr in raw])
    dv_raw = np.vstack([tr.costate for tr in raw])
    shifts = (-2, -1, 0, 1, 2)
    rawt = {
        "x": np.vstack([x_raw + np.array([2.0 * np.pi * k, 0.0]) for k in shifts]),
        "v": np.concatenate([v_raw] * len(shifts)),
        "dv": np.vstack([dv_raw] * len(shifts)),
        "tile": np.concatenate([np.full(x_raw.shape[0], k, dtype=int) for k in shifts]),
    }
    return samples, norm, curve, pool, rawt


def _transect_frame(curve_pts: np.ndarray, pool_x: np.ndarray):
    """Anchor point on the switching curve + unit normal, in the densest data region."""
    from scipy.spatial import cKDTree
    counts = cKDTree(pool_x).query_ball_point(curve_pts, r=0.3, return_length=True)
    c0 = curve_pts[int(np.argmax(counts))]
    local = curve_pts[np.linalg.norm(curve_pts - c0, axis=1) < 0.5]
    _, _, vt = np.linalg.svd(local - local.mean(axis=0))
    tang = vt[0] / np.linalg.norm(vt[0])
    nrm = np.array([-tang[1], tang[0]])
    return c0, tang, nrm


def _envelope_tube(rawt, c0, tang, nrm, *, s_max: float, tube: float,
                   n_bins: int = 60, v_slack: float = 0.4):
    """Lower-envelope truth dots in a thin tube around the transect.

    The raw trajectories make multiple (suboptimal) passes over a state; the true
    global V is their pointwise minimum. Per s-bin, keep only points within
    ``v_slack`` of the bin minimum — those lie on the optimal branch, so their
    costates are the true (branch) gradients.
    """
    rel = rawt["x"] - c0
    s_all, t_all = rel @ nrm, rel @ tang
    mask = (np.abs(t_all) <= tube) & (np.abs(s_all) <= s_max)
    s, v, g = s_all[mask], rawt["v"][mask], rawt["dv"][mask] @ nrm
    keep = np.zeros(s.size, dtype=bool)
    edges = np.linspace(-s_max, s_max, n_bins + 1)
    for i in range(n_bins):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.any():
            keep |= m & (v <= v[m].min() + v_slack)
    return s[keep], v[keep], g[keep]


def fig_transect(models: dict[str, dict[str, Any]], nets: dict[str, Any], norm,
                 curve, pool, rawt, *, s_max: float = 0.8, tube: float = 0.10) -> str:
    """F3 — V and n·∇V along the switching-set normal through the densest data region.

    Truth (both sides of the curve) = lower envelope of the tiled raw trajectories;
    the models were trained only on the s<0 side (the restricted basin), so s>0
    shows their smooth continuation against the other branch's true values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    c0, tang, nrm = _transect_frame(curve.points, pool["x"])
    s_true, v_true, g_true = _envelope_tube(rawt, c0, tang, nrm, s_max=s_max, tube=tube)
    # Subsample per side: the data side is ~50x denser than the far side, so a
    # uniform draw would erase the far-side truth entirely.
    rng = np.random.default_rng(0)
    keep = np.zeros(s_true.size, dtype=bool)
    for side_mask, cap in ((s_true <= 0.0, 400), (s_true > 0.0, 400)):
        idx = np.flatnonzero(side_mask)
        if idx.size > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep[idx] = True
    s_true, v_true, g_true = s_true[keep], v_true[keep], g_true[keep]

    s = np.linspace(-s_max, s_max, 401)
    line = c0[None, :] + s[:, None] * nrm[None, :]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    for ax in axes:
        ax.axvline(0.0, color="0.8", lw=1.0, ls="--", zorder=0)
    axes[0].scatter(s_true, v_true, s=12, color=_TRUE_COLOR, alpha=0.6,
                    lw=0, zorder=1)
    axes[1].scatter(s_true, g_true, s=12, color=_TRUE_COLOR, alpha=0.6,
                    lw=0, zorder=1)
    for name in models:
        color, ls = _MODEL_STYLE[name]
        v, g = _value_grad_phys(nets[name], line, norm)
        axes[0].plot(s, v, color=color, ls=ls, lw=2.0, label=_DISPLAY[name], zorder=2)
        axes[1].plot(s, g @ nrm, color=color, ls=ls, lw=2.0, label=_DISPLAY[name], zorder=2)
    pad_v = 0.15 * (v_true.max() - v_true.min())
    axes[0].set_ylim(min(0.0, v_true.min()) - pad_v, v_true.max() + pad_v)
    pad_g = 0.15 * (g_true.max() - g_true.min())
    axes[1].set_ylim(g_true.min() - pad_g, g_true.max() + pad_g)
    axes[0].set_ylabel(r"$V$")
    axes[1].set_ylabel(r"$n\cdot\nabla V$")
    axes[1].set_xlabel(r"signed distance $s$ along the switching-set normal")
    axes[0].legend(loc="upper left", fontsize=10)
    return _save_png(fig, "transect_switching_set")


def fig_transect_split(models: dict[str, dict[str, Any]], nets: dict[str, Any], norm,
                       curve, pool, rawt, *, s_max: float = 0.45,
                       tube: float = 0.10) -> dict[str, str]:
    """Separate V and n·∇V transects with the selected PMP branch shown explicitly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    c0, tang, nrm = _transect_frame(curve.points, pool["x"])
    s = np.linspace(-s_max, s_max, 401)
    line = c0[None, :] + s[:, None] * nrm[None, :]
    v0, g0, _ = _branch_candidate_on_line(rawt, c0, nrm, s, tile=0)
    v1, g1, _ = _branch_candidate_on_line(rawt, c0, nrm, s, tile=1)
    left = s <= 0.0
    v_true = np.where(left, v0, v1)
    g_true = np.where(left, g0, g1)
    # Nearest-neighbour support is sparse in a few places; fall back to the
    # available lower candidate rather than leaving small holes in the diagnostic.
    v_stack = np.vstack([v0, v1])
    g_stack = np.vstack([g0, g1])
    fallback = np.nanargmin(np.where(np.isfinite(v_stack), v_stack, np.inf), axis=0)
    missing = ~np.isfinite(v_true)
    v_true[missing] = v_stack[fallback[missing], np.flatnonzero(missing)]
    g_true[missing] = g_stack[fallback[missing], np.flatnonzero(missing)]

    model_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in models:
        v, g = _value_grad_phys(nets[name], line, norm)
        model_values[name] = (v, g @ nrm)

    out: dict[str, str] = {}
    for key, truth, ylabel, stem in (
        ("value", v_true, r"$V$", "transect_value"),
        ("gradient", g_true, r"$n\cdot\nabla V$", "transect_normal_gradient"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.axvline(0.0, color="0.8", lw=1.0, ls="--", zorder=0)
        ax.plot(s, truth, color=_TRUE_COLOR, ls="--", lw=2.6, label="true PMP",
                zorder=3)
        model_stack = []
        for name in models:
            color, ls = _MODEL_STYLE[name]
            y = model_values[name][0 if key == "value" else 1]
            model_stack.append(y)
            ax.plot(s, y, color=color, ls=ls, lw=2.0, label=_DISPLAY[name], zorder=2)
        all_y = np.concatenate([truth, *model_stack])
        ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        pad = 0.06 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel(r"signed distance $s$ along the switching-set normal")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=10)
        out[key] = _save_png(fig, stem)
    return out


def _branch_candidate_on_line(rawt, c0: np.ndarray, nrm: np.ndarray,
                              s: np.ndarray, tile: int, *, radius: float = 0.18,
                              k_neigh: int = 20):
    """Nearest local candidate branch from one 2π tile of the raw PMP data."""
    from scipy.spatial import cKDTree

    mask = rawt["tile"] == tile
    x = rawt["x"][mask]
    v = rawt["v"][mask]
    g = rawt["dv"][mask] @ nrm
    tree = cKDTree(x)
    line = c0[None, :] + s[:, None] * nrm[None, :]
    dist, idx = tree.query(line, k=k_neigh)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    if dist.shape[0] != s.size:
        dist = dist.T
        idx = idx.T
    vv = np.full(s.size, np.nan)
    gg = np.full(s.size, np.nan)
    dd = np.full(s.size, np.nan)
    for i in range(s.size):
        local = dist[i] <= radius
        if not np.any(local):
            continue
        # Use the closest few points to reduce nearest-neighbour jitter while
        # still staying on this single tiled PMP branch.
        jj = idx[i, local][:5]
        vv[i] = float(np.median(v[jj]))
        gg[i] = float(np.median(g[jj]))
        dd[i] = float(np.median(dist[i, local][:5]))
    return vv, gg, dd


def fig_true_branch_transect(curve, pool, rawt, *, s_max: float = 0.45) -> dict[str, str]:
    """True branch candidates before taking the lower envelope.

    Tile 0 and tile +1 are the two PMP branches that meet at the selected
    switching-set transect. The true value is their pointwise minimum; the true
    gradient jumps from the gradient of the lower branch on one side to the
    gradient of the lower branch on the other side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    c0, _, nrm = _transect_frame(curve.points, pool["x"])
    s = np.linspace(-s_max, s_max, 241)
    v0, g0, _ = _branch_candidate_on_line(rawt, c0, nrm, s, tile=0)
    v1, g1, _ = _branch_candidate_on_line(rawt, c0, nrm, s, tile=1)
    left = s <= 0.0
    v_min = np.where(left, v0, v1)
    g_min = np.where(left, g0, g1)
    vals = np.vstack([v0, v1])
    g_stack = np.vstack([g0, g1])
    fallback = np.nanargmin(np.where(np.isfinite(vals), vals, np.inf), axis=0)
    missing = ~np.isfinite(v_min)
    v_min[missing] = vals[fallback[missing], np.flatnonzero(missing)]
    g_min[missing] = g_stack[fallback[missing], np.flatnonzero(missing)]

    styles = {
        "branch0": (PALETTE["blue_main"], "-", "PMP branch to upright 0"),
        "branch1": (PALETTE["teal"], "-", r"PMP branch to upright $2\pi$"),
        "min": ("0.2", "--", "selected minimum"),
    }
    out: dict[str, str] = {}
    for y0, y1, ymin, ylabel, stem in (
        (v0, v1, v_min, r"$V$", "transect_true_branches_value"),
        (g0, g1, g_min, r"$n\cdot\nabla V$", "transect_true_branches_gradient"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.axvline(0.0, color="0.8", lw=1.0, ls="--", zorder=0)
        ax.plot(s, y0, color=styles["branch0"][0], ls=styles["branch0"][1],
                lw=2.2, label=styles["branch0"][2])
        ax.plot(s, y1, color=styles["branch1"][0], ls=styles["branch1"][1],
                lw=2.2, label=styles["branch1"][2])
        ax.plot(s, ymin, color=styles["min"][0], ls=styles["min"][1],
                lw=2.4, label=styles["min"][2])
        ax.set_xlabel(r"signed distance $s$ along the switching-set normal")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=10)
        out[stem] = _save_png(fig, stem)
    return out


def fig_dumbbell(models: dict[str, dict[str, Any]]) -> str:
    """F4 — near vs far mean per-sample L1 per model (log scale, paired dots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    order = sorted(models, key=lambda n: models[n]["far_l1"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for y, name in enumerate(order):
        r = models[name]
        color, _ = _MODEL_STYLE[name]
        ax.plot([r["far_l1"], r["near_l1"]], [y, y], color=color, lw=2.0, zorder=1)
        ax.scatter([r["far_l1"]], [y], s=70, facecolor="white", edgecolor=color,
                   lw=2.0, zorder=2)
        ax.scatter([r["near_l1"]], [y], s=70, color=color, zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_DISPLAY[n] for n in order])
    ax.set_xlabel(r"mean per-sample $L^1$ error / global mean $\|\mathrm{true}\|$")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="0.3", color="0.3", label="near"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor="0.3", color="0.3", label="far"),
    ], loc="lower left", fontsize=10)
    return _save_png(fig, "near_far_dumbbell")


def fig_error_vs_distance_split(models: dict[str, dict[str, Any]], nets: dict[str, Any],
                                samples, norm, distance: np.ndarray,
                                centers: np.ndarray, counts: np.ndarray) -> dict[str, str]:
    """Separate value and gradient error profiles for the comparison models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    x = np.asarray(samples["x"], dtype=np.float64)
    v_true = np.asarray(samples["v"], dtype=np.float64).reshape(-1)
    g_true = np.asarray(samples["dv"], dtype=np.float64)
    edges = np.linspace(float(distance.min()), float(distance.max()), _N_BINS + 1)
    width = (centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 0.1

    curves: dict[str, dict[str, list[float]]] = {}
    for name in models:
        v_pred, g_pred = _value_grad_phys(nets[name], x, norm)
        v_err_abs = np.abs(v_pred - v_true)
        g_err_abs = np.linalg.norm(g_pred - g_true, axis=1)
        v_den = max(float(np.mean(v_err_abs)), 1e-30)
        g_den = max(float(np.mean(g_err_abs)), 1e-30)
        v_err = v_err_abs / v_den
        g_err = g_err_abs / g_den
        curves[name] = {"value": [], "gradient": []}
        for i in range(_N_BINS):
            lo, hi = edges[i], edges[i + 1]
            mask = (distance >= lo) & ((distance <= hi) if i == _N_BINS - 1 else (distance < hi))
            curves[name]["value"].append(float(np.mean(v_err[mask])) if np.any(mask) else np.nan)
            curves[name]["gradient"].append(float(np.mean(g_err[mask])) if np.any(mask) else np.nan)

    out: dict[str, str] = {}
    for quantity, ylabel, stem in (
        ("value", r"bin mean $|\hat V - V|$ / model global mean $|\hat V - V|$",
         "error_vs_distance_value"),
        ("gradient", r"bin mean $\|\nabla\hat V-\nabla V\|$ / model global mean error",
         "error_vs_distance_gradient"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=width, color="0.88", zorder=0)
        ax2.set_ylabel("samples per bin", color="0.55")
        ax2.tick_params(axis="y", colors="0.55")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        for name in models:
            color, ls = _MODEL_STYLE[name]
            xs, ys = _smooth_distance_xy(centers, curves[name][quantity], counts)
            ax.plot(xs, ys, color=color, ls=ls, lw=2.4,
                    label=_DISPLAY[name], zorder=3)
        ax.set_xlabel("distance to switching set")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=10)
        out[quantity] = _save_png(fig, stem)
    return out


def _best_ratio_by_cell(multirun: Path, *, relu_power: bool) -> dict[str, dict[str, float]]:
    """activation -> {'ratio','near','far'} from the best-per-cell H1 run in a sweep dir."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(multirun.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        model = record["config"]["model"]
        m = record["metrics"][0]["values"]
        if "near_l1_h1" not in m or int(m.get("best_neurons", 0)) == 0:
            continue
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), "")
        if relu_power:
            if model["activation"] != "relu":
                continue
            key = f"relu^{model['power']:g}|{loss}"
        else:
            if model["kind"] != "signed" or loss != "h1":
                continue
            key = f"{model['activation']}|{loss}"
        far = float(m["far_l1_h1"])
        if key not in out or far < out[key]["far"]:
            out[key] = {"far": far, "near": float(m["near_l1_h1"]),
                        "ratio": float(m["near_l1_h1"]) / far if far else float("nan")}
    return out


def fig_sampling_control(models: dict[str, dict[str, Any]]) -> str | None:
    """F5 — (a) near/far ratio baseline → density-balanced per model;
    (b) balanced-data near/far vs ReLU power under L2 and H1."""
    if not NEARSWITCH_DIR.exists() or not RELU_NEARSWITCH_DIR.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    smooth = _best_ratio_by_cell(NEARSWITCH_DIR, relu_power=False)
    relu = _best_ratio_by_cell(RELU_NEARSWITCH_DIR, relu_power=True)
    balanced = {**{k.split("|")[0]: v for k, v in smooth.items()},
                **{k.split("|")[0]: v for k, v in relu.items() if k.endswith("|h1")}}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, row in models.items():
        if name not in balanced:
            continue
        color, _ = _MODEL_STYLE[name]
        axes[0].plot([0, 1], [row["l1_near/far"], balanced[name]["ratio"]],
                     "-o", color=color, lw=2.0, ms=6, label=_DISPLAY[name])
    axes[0].axhline(1.0, color="0.75", lw=1.0, ls="--", zorder=0)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["biased (3k)", "balanced (6k)"])
    axes[0].set_xlim(-0.25, 1.25)
    axes[0].set_ylabel("near/far mean L1 ratio")
    axes[0].legend(loc="upper right", fontsize=10)

    powers = [2.0, 2.01, 3.0, 4.0, 5.0]
    for loss, color, ls in (("l2", PALETTE["red_strong"], "-"),
                            ("h1", PALETTE["blue_main"], "-")):
        ratios = [relu.get(f"relu^{p:g}|{loss}", {}).get("ratio", np.nan) for p in powers]
        axes[1].plot(powers, ratios, "-o", color=color, ls=ls, lw=2.0, ms=6,
                     label={"l2": r"$L^2$ trained", "h1": r"$H^1$ trained"}[loss])
    axes[1].axhline(1.0, color="0.75", lw=1.0, ls="--", zorder=0)
    axes[1].set_xlabel(r"atom power $p$ (penalty $q=2/(p{+}1)$)")
    axes[1].set_ylabel("near/far mean L1 ratio (balanced)")
    axes[1].legend(loc="upper left", fontsize=10)
    return _save_png(fig, "sampling_control")


def fig_sampling_control_split(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Separate sampling-density and ReLU-power controls."""
    if not NEARSWITCH_DIR.exists() or not RELU_NEARSWITCH_DIR.exists():
        return {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    smooth = _best_ratio_by_cell(NEARSWITCH_DIR, relu_power=False)
    relu = _best_ratio_by_cell(RELU_NEARSWITCH_DIR, relu_power=True)
    balanced = {**{k.split("|")[0]: v for k, v in smooth.items()},
                **{k.split("|")[0]: v for k, v in relu.items() if k.endswith("|h1")}}
    out: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, row in models.items():
        if name not in balanced:
            continue
        color, ls = _MODEL_STYLE[name]
        ax.plot([0, 1], [row["l1_near/far"], balanced[name]["ratio"]],
                marker="o", color=color, ls=ls, lw=2.0, ms=6, label=_DISPLAY[name])
    ax.axhline(1.0, color="0.75", lw=1.0, ls="--", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["biased (3k)", "balanced (6k)"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("near/far mean L1 ratio")
    ax.legend(loc="upper right", fontsize=10)
    out["density"] = _save_png(fig, "sampling_density_balance")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    powers = [2.0, 2.01, 3.0, 4.0, 5.0]
    for loss, color in (("l2", PALETTE["red_strong"]), ("h1", PALETTE["blue_main"])):
        ratios = [relu.get(f"relu^{p:g}|{loss}", {}).get("ratio", np.nan) for p in powers]
        ax.plot(powers, ratios, "-o", color=color, lw=2.0, ms=6,
                label={"l2": r"$L^2$ trained", "h1": r"$H^1$ trained"}[loss])
    ax.axhline(1.0, color="0.75", lw=1.0, ls="--", zorder=0)
    ax.set_xlabel(r"atom power $p$ (penalty $q=2/(p{+}1)$)")
    ax.set_ylabel("near/far mean L1 ratio (balanced)")
    ax.legend(loc="upper left", fontsize=10)
    out["power"] = _save_png(fig, "sampling_relu_power")
    return out


CAPACITY_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "region_split_nearswitch_capacity"
# Dataset variants of the fixed-budget oversampling control, in near-share order.
_OVERSAMPLE_VARIANTS = {
    "3k 10% (baseline)": [MULTIRUN_DIR / "7"],
    "6k 10% prop": sorted((CAPACITY_DIR / "near10").glob("[0-9]*")),
    "6k 10% strat": sorted((CAPACITY_DIR / "near10_strat").glob("[0-9]*")),
    "6k 20% strat": sorted((CAPACITY_DIR / "near20_strat").glob("[0-9]*")),
    "6k 40% strat": [NEARSWITCH_DIR / "7"],
}
_NEAR_THRESH = 0.571446   # the shared near-band threshold (10% of the baseline set)


def _common_pool_scores(pool, curve) -> list[dict[str, Any]] | None:
    """Score every oversampling-variant model (signed gaussian γ=1) on the pool.

    Each variant's recorded region metrics use its *own* near percentile and its
    own global denominator, so they are not cross-comparable. Here every fitted
    model is rebuilt (with its own training normalizer) and scored on ONE common
    set — the full restricted pool — with one near band (d ≤ 0.5714) and one
    denominator pair. The reported number is the per-sample mean of
    (|ΔV|/mean|V| + ‖Δ∇V‖/mean‖∇V‖)/2.
    """
    if not CAPACITY_DIR.exists():
        return None
    import pickle
    import torch
    from scipy.spatial import cKDTree
    from src.data import load_value_samples, ValueSampleNormalizer
    from src.models.net import ShallowNetwork
    from src.config.activations import get_activation
    from src.plots import _best_iteration_atoms

    X, V, DV = pool["x"], pool["v"], pool["dv"]
    d = cKDTree(curve.points).query(X)[0]
    bands = {"ultra-near": d <= 0.2, "near": d <= _NEAR_THRESH, "far": d > _NEAR_THRESH}
    v_den = np.abs(V).mean()
    g_den = np.linalg.norm(DV, axis=1).mean()

    norms: dict[str, ValueSampleNormalizer] = {}
    scored = []
    for variant, run_dirs in _OVERSAMPLE_VARIANTS.items():
        for run_dir in run_dirs:
            recs = sorted(Path(run_dir).glob("*.json"))
            if not recs:
                continue
            record = json.loads(recs[0].read_text(encoding="utf-8"))
            model = record["config"]["model"]
            if (model["kind"], model["activation"], model["gamma"]) != ("signed", "gaussian", 1.0):
                continue
            data_path = record["config"]["data"]["path"]
            if data_path not in norms:
                norms[data_path] = ValueSampleNormalizer.fit(load_value_samples(data_path))
            norm = norms[data_path]
            with open(Path(run_dir) / f"result_{recs[0].stem}.pkl", "rb") as f:
                a, b, u = _best_iteration_atoms(pickle.load(f))
            net = ShallowNetwork(layer_sizes=[a.shape[1], a.shape[0], 1],
                                 activation=get_activation(model["activation"]),
                                 p=model["power"], inner_weights=a, inner_bias=b,
                                 outer_weights=u)
            net.eval()
            err = np.empty(len(X))
            for lo in range(0, len(X), 100_000):
                hi = min(lo + 100_000, len(X))
                xn = torch.tensor(X[lo:hi] / norm.x_scale, dtype=torch.float64,
                                  requires_grad=True)
                val = net(xn)
                (grad,) = torch.autograd.grad(val.sum(), xn)
                vp = val.detach().numpy().reshape(-1) * norm.v_scale
                gp = grad.detach().numpy() * (norm.v_scale / norm.x_scale)
                err[lo:hi] = 0.5 * (np.abs(vp - V[lo:hi]) / v_den
                                    + np.linalg.norm(gp - DV[lo:hi], axis=1) / g_den)
            entry = {"variant": variant,
                     "neurons": int(record["metrics"][0]["values"]["best_neurons"])}
            entry.update({band: float(err[sel].mean()) for band, sel in bands.items()})
            scored.append(entry)
    return scored or None


def fig_oversampling_control(scored: list[dict[str, Any]]) -> str:
    """F8 — common-set error per band vs the training near-share ladder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    variants = list(_OVERSAMPLE_VARIANTS)
    series = {"ultra-near": PALETTE["red_strong"], "near": PALETTE["blue_main"],
              "far": PALETTE["teal"]}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for band, color in series.items():
        best = []
        for i, variant in enumerate(variants):
            vals = [r[band] for r in scored if r["variant"] == variant]
            if not vals:
                best.append(np.nan)
                continue
            ax.scatter([i] * len(vals), vals, s=22, color=color, alpha=0.35, lw=0,
                       zorder=2)
            best.append(min(vals))
        ax.plot(range(len(variants)), best, "-o", color=color, lw=2.0, ms=7,
                label=band, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("common-set mean normalized error")
    ax.legend(loc="upper left", fontsize=10)
    return _save_png(fig, "oversampling_control")


def fig_feedback_split(models: dict[str, dict[str, Any]], nets: dict[str, Any], norm,
                       curve, pool, rawt, *, offset: float = 0.25,
                       roll_t: float = 10.0, dt: float = 0.005,
                       u_clip: float = 30.0):
    """Separate phase portraits and off-data control trace for feedback synthesis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem

    _apply_publication_style()
    problem = PendulumSwingUpProblem()
    c0, _, nrm = _transect_frame(curve.points, pool["x"])
    starts = {"A": c0 - offset * nrm, "B": c0 + offset * nrm}
    tree = cKDTree(rawt["x"])

    def true_u(x):
        _, j = tree.query(np.asarray(x, dtype=np.float64).reshape(1, 2), k=40)
        j = np.asarray(j).ravel()
        opt = j[int(np.argmin(rawt["v"][j]))]
        return float(np.ravel(problem.feedback_from_gradient(rawt["dv"][opt]))[0])

    def model_u(net):
        def u(x):
            _, g = _value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
            return float(problem.feedback_from_gradient(g)[0])
        return u

    laws = {"true PMP": (_TRUE_COLOR, "-", true_u)}
    for name in models:
        color, ls = _MODEL_STYLE[name]
        laws[name] = (color, ls, model_u(nets[name]))
    rolled = {(side, name): problem.rk4_rollout(uf, x0, T=roll_t, dt=dt, u_clip=u_clip)
              for side, x0 in starts.items() for name, (_, _, uf) in laws.items()}

    def _reached(xs):
        xf = xs[-1]
        return abs((xf[0] + np.pi) % (2 * np.pi) - np.pi) < 0.4 and abs(xf[1]) < 0.4

    all_xs = np.vstack([xs for (_, xs, _, _) in rolled.values()])
    all_xs = all_xs[np.all(np.abs(all_xs) < 12.0, axis=1)]
    xlo, xhi = all_xs[:, 0].min() - 0.7, all_xs[:, 0].max() + 0.7
    ylo, yhi = all_xs[:, 1].min() - 0.7, all_xs[:, 1].max() + 0.7

    # One small phase panel per feedback law (house subfigure rule): the two
    # starts are the series (A = data side, blue; B = beyond the curve, red), so
    # no trajectory ever hides another — in the composite overplot, ReLU^5's
    # dashed line was drawn underneath four near-identical trajectories, which
    # made it look disconnected from its start.
    out: dict[str, str] = {}
    for name in laws:
        stem = "feedback_" + name.replace("^", "").replace(" ", "_").lower()
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(curve.points[:, 0], curve.points[:, 1], s=3, color="0.1", zorder=3)
        for side, scolor in (("A", PALETTE["blue_main"]), ("B", PALETTE["red_strong"])):
            _, xs, _, _ = rolled[(side, name)]
            ax.plot(xs[:, 0], xs[:, 1], color=scolor, lw=2.2,
                    label=f"start {side}", zorder=2)
            ax.scatter([starts[side][0]], [starts[side][1]], s=80, color=scolor,
                       marker="x", lw=2.2, zorder=4)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.legend(loc="upper right", fontsize=9)
        out[name] = _save_png(fig, stem)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, (color, ls, _) in laws.items():
        t, _, us, _ = rolled[("B", name)]
        lw = 3.0 if name == "true PMP" else 1.8
        ax.plot(t, us, color=color, ls=ls, lw=lw, label=_DISPLAY.get(name, name))
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"feedback control $u(t)$")
    ax.set_xlim(0, roll_t)
    ax.legend(loc="best", fontsize=9)
    out["control_b"] = _save_png(fig, "feedback_control_b")

    table = []
    for name in laws:
        entry = {"model": _DISPLAY.get(name, name).replace("$", "")}
        for side in ("A", "B"):
            _, xs, _, cost = rolled[(side, name)]
            entry[f"cost {side}"] = f"{cost:.1f}"
            entry[f"upright {side}"] = "yes" if _reached(xs) else "no"
        table.append(entry)
    return out, table, starts


def fig_atom_portrait(models: dict[str, dict[str, Any]], nets: dict[str, Any],
                      norm, curve) -> str:
    """F7 — atom lines {a·x+b=0} (physical coords) vs the switching curve,
    relu^2 and gaussian panels; line strength ∝ |outer weight|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()
    lim = 9.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, name in zip(axes, ("relu^2", "gaussian")):
        net = nets[name]
        a = net.hidden.weight.detach().numpy() / norm.x_scale   # physical-coords normals
        b = net.hidden.bias.detach().numpy()
        c = np.abs(net.output.weight.detach().numpy()).ravel()
        w = c / c.max()
        color, _ = _MODEL_STYLE[name]
        for (a1, a2), bi, wi in zip(a, b, w):
            nn = np.hypot(a1, a2)
            if nn < 1e-12:
                continue
            # line a1·x + a2·y + b = 0 through the box
            p0 = -bi * np.array([a1, a2]) / nn**2
            d = np.array([-a2, a1]) / nn
            pts = np.array([p0 - 3 * lim * d, p0 + 3 * lim * d])
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.5 + 1.6 * wi,
                    alpha=0.12 + 0.55 * wi, zorder=1)
        ax.scatter(curve.points[:, 0], curve.points[:, 1], s=3, color="0.15", zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.set_aspect("equal")
    return _save_png(fig, "atom_portrait")


def fig_learned_surfaces(best: list[dict[str, Any]], norm) -> dict[str, str]:
    """Learned value surface, one PNG per signed H1 model in the surface gallery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from src.plots import plot_model_value_surface

    picked = {
        r["activation"]: r for r in best
        if r["kind"] == "signed" and r["loss"] == "h1"
        and r["activation"] in _SURFACE_MODEL_ORDER
    }
    missing = [name for name in _SURFACE_MODEL_ORDER if name not in picked]
    if missing:
        raise ValueError(f"missing learned-surface models in records: {missing}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name in _SURFACE_MODEL_ORDER:
        row = picked[name]
        out_path = FIG_DIR / f"surface_{name.replace('^', '')}.png"
        with mpl.rc_context():
            _apply_publication_style()
            fig = plt.figure(figsize=(4.2, 4.0))
            ax = fig.add_subplot(111, projection="3d")
            plot_model_value_surface(
                row["result_path"],
                activation=row["act_name"],
                power=row["power"],
                x_scale=norm.x_scale,
                v_scale=norm.v_scale,
                grid_n=220,
                x_range=(-8.0, 8.0),
                y_range=(-8.0, 8.0),
                vmax=60.0,
                xticks=(-8.0, 0.0, 8.0),
                yticks=(-8.0, 0.0, 8.0),
                zticks=(0.0, 30.0, 60.0),
                ax=ax,
                show=False,
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        out[name] = f"figures/{out_path.name}"
    return out


def _frontier_trajectory(result_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(neurons, running-best relative H1 validation error) for one insertion run."""
    import pickle

    with open(result_path, "rb") as f:
        history = pickle.load(f)
    pts = []
    for iw, h1 in zip(history.inner_weights, history.err_h1_val):
        n = int(np.asarray(iw["weight"]).shape[0])
        h1 = float(h1)
        if n >= 1 and np.isfinite(h1):
            pts.append((n, h1))
    pts.sort(key=lambda nh: nh[0])
    if not pts:
        return np.array([]), np.array([])
    ns = np.array([n for n, _ in pts])
    h1 = np.minimum.accumulate(np.array([h for _, h in pts], dtype=np.float64))
    return ns, h1


def fig_frontier(best: list[dict[str, Any]]) -> str:
    """Running-best relative H1 validation error vs network size."""
    from src.plots import frontier_penalty_label, plot_neuron_h1_frontier

    picked = {
        r["activation"]: r for r in best
        if r["kind"] == "signed" and r["loss"] == "h1"
        and r["activation"] in _SURFACE_MODEL_ORDER
    }
    markers = {"gaussian": "o", "softplus": "s", "relu^2": "^", "relu^5": "D"}
    labels = {
        "gaussian": frontier_penalty_label(r"\mathrm{gaussian}",
                                           insertion=picked["gaussian"]["insertion"],
                                           subscript=r"\gamma"),
        "softplus": frontier_penalty_label(r"\mathrm{softplus}",
                                           insertion=picked["softplus"]["insertion"],
                                           subscript=r"\gamma"),
        "relu^2": frontier_penalty_label(r"\mathrm{ReLU}^2",
                                         insertion=picked["relu^2"]["insertion"],
                                         subscript="2"),
        "relu^5": frontier_penalty_label(r"\mathrm{ReLU}^5",
                                         insertion=picked["relu^5"]["insertion"],
                                         subscript="5"),
    }
    series = []
    for name in _SURFACE_MODEL_ORDER:
        ns, h1 = _frontier_trajectory(picked[name]["result_path"])
        color, ls = _MODEL_STYLE[name]
        series.append({
            "ns": ns,
            "h1": h1,
            "label": labels[name],
            "color": color,
            "marker": markers[name],
            "ls": ls,
        })
    out = FIG_DIR / "frontier.png"
    plot_neuron_h1_frontier(series, save_path=out)
    return f"figures/{out.name}"


def main() -> int:
    rows, cache = load_rows()
    best = best_per_cell(rows)

    l1 = sorted(best, key=lambda r: (r["kind"], r["insertion"], r["loss"], r["l1_near/far"]))
    l1_table = format_table(
        l1,
        ["kind", "insertion", "activation", "loss", "gamma", "neurons",
         "near_l1", "far_l1", "l1_near/far"],
        headers={"near_l1": "near L1", "far_l1": "far L1", "l1_near/far": "near/far"},
        formats={"gamma": "{:g}", "near_l1": "{:.2e}", "far_l1": "{:.2e}",
                 "l1_near/far": "{:.2f}"},
        title="Mean per-sample L1 over the full dataset — count-fair, robust to V→0",
    )
    rel = sorted(best, key=lambda r: (r["kind"], r["insertion"], r["loss"], r["rel_near/far"]))
    rel_table = format_table(
        rel,
        ["kind", "insertion", "activation", "loss", "gamma", "neurons",
         "near_h1", "far_h1", "rel_near/far"],
        headers={"near_h1": "near H1", "far_h1": "far H1", "rel_near/far": "near/far"},
        formats={"gamma": "{:g}", "near_h1": "{:.2e}", "far_h1": "{:.2e}",
                 "rel_near/far": "{:.2f}"},
        title="Relative H1 (kept for continuity — confounded by the V→0 interior)",
    )

    binned = _bin_centers(cache)
    fig_line = ""
    distance = None
    if binned is not None:
        centers, counts, distance = binned
        fig = _plot_error_vs_distance(best, centers, counts)
        fig_line = f"\n![error vs distance](figures/{fig.name})\n"

    # -- model-comparison figures (F3–F7) --------------------------------------
    models = select_models(best)
    nets = {name: _build_net(row) for name, row in models.items()}
    samples, norm, curve, pool, rawt = _load_geometry(best)
    surface_figs = fig_learned_surfaces(best, norm)
    fig_frontier(best)
    distance_figs = (
        fig_error_vs_distance_split(models, nets, samples, norm, distance, centers, counts)
        if binned is not None and distance is not None else {}
    )
    true_branch_figs = fig_true_branch_transect(curve, pool, rawt)
    f3 = fig_transect(models, nets, norm, curve, pool, rawt)
    f3_split = fig_transect_split(models, nets, norm, curve, pool, rawt)
    f4 = fig_dumbbell(models)
    f5 = fig_sampling_control(models)
    f5_split = fig_sampling_control_split(models)
    f6_figs, cost_rows, starts = fig_feedback_split(models, nets, norm, curve, pool, rawt)
    f7 = fig_atom_portrait(models, nets, norm, curve)
    cost_table = format_table(
        cost_rows,
        ["model", "cost A", "upright A", "cost B", "upright B"],
        title=(f"Closed-loop cost / stabilization from the two straddling starts "
               f"(A = ({starts['A'][0]:.2f}, {starts['A'][1]:.2f}), "
               f"B = ({starts['B'][0]:.2f}, {starts['B'][1]:.2f}); T=10)"),
    )
    scored = _common_pool_scores(pool, curve)
    f8_block = ""
    if scored:
        f8 = fig_oversampling_control(scored)
        best_rows = []
        for variant in _OVERSAMPLE_VARIANTS:
            runs = [r for r in scored if r["variant"] == variant]
            if not runs:
                continue
            bn = min(runs, key=lambda r: r["near"])
            best_rows.append({
                "variant": variant, "runs": len(runs),
                "ultra-near": f"{min(r['ultra-near'] for r in runs):.3f}",
                "near": f"{min(r['near'] for r in runs):.3f}",
                "far": f"{min(r['far'] for r in runs):.3f}",
                "neurons": f"{bn['neurons']}",
            })
        f8_table = format_table(
            best_rows, ["variant", "runs", "ultra-near", "near", "far", "neurons"],
            title=("Best common-set error per variant (min over that variant's runs; "
                   "neurons = size of the near-best run)"),
        )
        f8_block = (
            "### 3.2 Reallocating a fixed budget toward the near band does not help\n\n"
            f"![oversampling control]({f8})\n\n"
            "All signed gaussian (γ=1) models fitted on the oversampling dataset "
            "variants, re-scored on ONE common evaluation set — the full restricted "
            "raw pool (~823k points), one near band (d ≤ 0.571), one denominator — "
            "since each variant's own recorded metrics use its own band and "
            "denominator and are not cross-comparable. Faint dots = individual runs "
            "(capacity levels), lines = the best run per variant; bands: ultra-near "
            "d ≤ 0.2, near d ≤ 0.571, far = rest.\n\n"
            f"{f8_table}\n\n"
            "**Doubling the budget helps; reallocating it does not.** Keeping the "
            "sampling distribution and doubling the count (6k 10% prop) improves "
            "every band — ultra-near 0.97 → 0.17. Raising the near share at a fixed "
            "budget (20%, 40%) makes *every* band worse, including the ultra-near "
            "band the extra samples were spent on: stratifying away the dense "
            "equilibrium band starves the region that anchors the global fit. This "
            "is not a capacity artifact (the 20% variant is bad even at 429 "
            "neurons). Caveat: gaussian γ=1 only, single seed, and the common "
            "evaluation measure is the pool's time-uniform (equilibrium-heavy) "
            "distribution — the near/far *ratio* study (§3.1) used each dataset's own "
            "balanced measure, which is why both statements can hold at once.\n\n"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.md"
    out.write_text(
        f"# {EXPERIMENT} Results\n\n"
        "**Questions.** (1) How well do sparse shallow models fit an optimal value "
        "function whose **gradient jumps across the swing-up switching set**, and "
        "what role do the activation and the nonconvex penalty play? (2) Can a "
        "reliable **feedback law** be synthesized from the fitted value function "
        "near — and across — the switching set?\n\n"
        "**Setup.** Pendulum swing-up value samples (3k, branch-restricted basin; "
        "see `README.md` for data and the error-metric rationale). Two sweeps on "
        "the same dataset and `eval=region_split` hook: smooth activations "
        "(profile insertion, gamma selected per cell) and **ReLU^p atoms with the "
        "fractional-exponent penalty** q = 2/(p+1) (finite_step insertion, gamma=0 "
        "by design, alpha selected per cell) — see `../02_pendulum/"
        "frac_exp_penalty`. `near` = lowest 10% of samples by distance to the "
        "switching set (d ≤ 0.57); `far` = the rest. The model-level studies "
        "(§4–§5) use four representative signed H1 models — gaussian, softplus, "
        "relu², relu⁵; semiconcave models are excluded there (they do not "
        "round-trip through the fit artifact, #19).\n\n"

        "## 1. The target: a value function with a gradient discontinuity\n\n"
        "The regions of attraction of the (periodic) upright equilibria — PMP "
        "characteristics filled by nearest-point classification — are separated by "
        "the nonsmooth switching curves the region split is built on (open-loop "
        "data visualisations are centralised in [`experiments/00_openloop/pendulum`]"
        "(../00_openloop/pendulum)).\n"
        "\n![regions of attraction & switching set](../00_openloop/pendulum/figures/regions_of_attraction.png)\n\n"
        "### 1.1 The kink, seen in the data\n\n"
        "Along a transect normal to the switching curve (through the densest data "
        "region), the two PMP branch values cross; the optimal V is their lower "
        "envelope — continuous, with a concave kink where the branches exchange "
        "optimality, so ∇V jumps (left: V; right: n·∇V):\n\n"
        "| value along the transect | normal gradient along the transect |\n"
        "| --- | --- |\n"
        f"| ![true branches, value]({true_branch_figs['transect_true_branches_value']}) "
        f"| ![true branches, gradient]({true_branch_figs['transect_true_branches_gradient']}) |\n\n"
        "One structural fact controls everything below: **the branch-restricted "
        "training data stop AT the curve** — no arm has samples on both sides, so "
        "the far branch (and the jump itself) is invisible to every model. Ground "
        "truth on both sides is reconstructed as the lower envelope of the raw "
        "(unrestricted) PMP trajectories tiled by 2πk in θ.\n\n"

        "## 2. Symptom: error concentrates near the switching set\n\n"
        "Region mean per-sample L1 (absolute) error / global mean ‖true‖ — "
        "count-fair and robust to the V→0 interior; `near/far` > 1 ⇒ worse at the "
        "switching set. (The naive relative-H1 metric flips this conclusion — see "
        "the Appendix.)\n\n"
        f"{l1_table}\n\n"
        "Every model — both kinds, every activation, every penalty — is **2.2–3.7× "
        "worse in the near band**. The ReLU^p rows sit in the same ratio band as "
        "the smooth activations (2.19–2.94, rising with p) while being an order of "
        "magnitude better in absolute terms: the *relative* near-band penalty is "
        "family-independent, a first hint that it is a property of the "
        "sampling/geometry rather than of the atom class.\n\n"
        "### 2.1 The error profile is a valley\n\n"
        "Per-sample absolute error against distance to the switching set "
        "(equal-width bins; count-weighted smoothing), split into the value and "
        "gradient components:\n\n"
        "| value error vs distance | gradient error vs distance |\n"
        "| --- | --- |\n"
        f"| ![value error]({distance_figs.get('value', '')}) "
        f"| ![gradient error]({distance_figs.get('gradient', '')}) |\n\n"
        "Error peaks right at the switching set, drops to its minimum in the dense "
        "band at distance ≈ 0.7 (where ~2/3 of the samples sit — the upright "
        "equilibrium), then climbs toward the under-sampled outer basin edge. The "
        "near/far ratio stays > 1 because `near` sits on the switching-set peak "
        "while the bulk of `far` is the low-error dense band. The valley tracks "
        "the **sample density**, which motivates the controls in §3.\n\n"

        "## 3. Diagnosis: sampling density, not the kink\n\n"
        "### 3.1 Density-balanced resampling collapses the near/far gap\n\n"
        "Refitting on a density-balanced 6k resample (same spatial band d ≤ 0.57, "
        "`eval.near_percentile` matched) collapses the near/far ratio to ≈ 1 for "
        "every model (left). On the balanced data the residual *intrinsic* "
        "switching-set penalty is small, grows with the ReLU power (more concave "
        "penalty, stiffer atoms), and is visible mainly without gradient "
        "supervision — H1 training absorbs it (right).\n\n"
        "| near/far ratio: biased → balanced | residual vs ReLU power (balanced) |\n"
        "| --- | --- |\n"
        f"| ![density balance]({f5_split.get('density', '')}) "
        f"| ![relu power]({f5_split.get('power', '')}) |\n\n"
        f"{f8_block}"

        "## 4. Which atoms fit the switching-set target best\n\n"
        "### 4.1 Accuracy per model\n\n"
        f"![near/far dumbbell]({f4})\n\n"
        "Mean per-sample L1 (log scale) in the near band (filled) and far region "
        "(open), per model, from the primary table (§2); rows ordered by far L1. "
        "**ReLU² dominates on both bands** — near 4.3e-02 / far 1.9e-02, roughly "
        "8× better than the best smooth activation (gaussian: 3.6e-01 / 1.4e-01) "
        "at a comparable neuron count (107 vs 97).\n\n"
        "### 4.2 Learned value surfaces\n\n"
        "| gaussian | softplus |\n"
        "| --- | --- |\n"
        f"| ![gaussian surface]({surface_figs.get('gaussian', '')}) "
        f"| ![softplus surface]({surface_figs.get('softplus', '')}) |\n\n"
        "| ReLU² | ReLU⁵ |\n"
        "| --- | --- |\n"
        f"| ![relu2 surface]({surface_figs.get('relu^2', '')}) "
        f"| ![relu5 surface]({surface_figs.get('relu^5', '')}) |\n\n"
        "The learned V̂ over the state plane (z clipped at 60): gaussian and the "
        "ReLU powers reproduce the in-basin bowl; softplus — the weakest fit "
        "throughout — flattens it.\n\n"
        "### 4.3 Models on the transect\n\n"
        "The same transect as §1.1, with the fitted models overlaid (grey dots = "
        "lower-envelope truth; the models saw data only on s < 0):\n\n"
        "| value | normal gradient |\n"
        "| --- | --- |\n"
        f"| ![transect value]({f3_split.get('value', '')}) "
        f"| ![transect gradient]({f3_split.get('gradient', '')}) |\n\n"
        "On the data side every model except softplus tracks V and n·∇V up to the "
        "curve. At s = 0 the true n·∇V jumps from ≈ +80 to ≈ −3; **every model "
        "continues smoothly across** — the jump was never in their data. The "
        "near-band error of §2 is therefore a *boundary-layer* fitting effect "
        "(steep one-sided values + thin data), not a failure to represent a seen "
        "discontinuity.\n\n"
        "### 4.4 Mechanism: where the atoms sit\n\n"
        f"![atom portrait]({f7})\n\n"
        "Each atom's active line {a·x + b = 0} in the physical (θ, θ̇) plane (line "
        "strength ∝ |outer weight|), for relu² (left) and gaussian (right), with "
        "the switching curve in black. ReLU²'s strongest atom lines align with the "
        "main switching arm — piecewise low-degree ridges seat the steep one-sided "
        "gradient — while gaussian bumps tile the basin isotropically. This is the "
        "mechanism behind §4.1.\n\n"

        "## 5. Can a reliable feedback law be synthesized?\n\n"
        "Closed-loop rollouts of u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x), one phase panel "
        "per feedback law, from two starts placed symmetrically either side of the "
        "switching curve (× markers): **start A** (blue) on the data side, "
        "**start B** (red) beyond the curve — off-data for every model. Switching "
        "set in black; all panels share the same axes. True PMP feedback = envelope "
        "nearest-neighbour over the tiled raw trajectories (valid on both sides): "
        "from B it swings over the top to the 2π upright, while every model pulls "
        "back through the curve to the θ = 0 upright.\n\n"
        "| true PMP | gaussian | softplus |\n"
        "| --- | --- | --- |\n"
        f"| ![true PMP]({f6_figs['true PMP']}) | ![gaussian]({f6_figs['gaussian']}) "
        f"| ![softplus]({f6_figs['softplus']}) |\n\n"
        "| ReLU² | ReLU⁵ |\n"
        "| --- | --- |\n"
        f"| ![relu2]({f6_figs['relu^2']}) | ![relu5]({f6_figs['relu^5']}) |\n\n"
        "The control signal from the off-data start B, per feedback law (true PMP "
        "pushes positive to swing over; the models brake toward θ = 0; softplus "
        "settles at a spurious equilibrium with u ≈ −5):\n\n"
        f"![control from B]({f6_figs['control_b']})\n\n"
        f"{cost_table}\n\n"
        "From start A every model except softplus matches the true closed-loop "
        "cost (10.5–10.9 vs 10.6) — the feedback is reliable arbitrarily close to "
        "the switching set, *on the training branch*. From the off-data start B "
        "the true law pays 26.3; the models stabilize but on the wrong branch, at "
        "~2× the cost (51–66), because the branch beyond the curve was never in "
        "the data.\n\n"

        "## 6. Conclusions\n\n"
        "- **The switching set is the boundary of the training data, not an "
        "interior kink** (§1.1, §4.3). No curve arm has branch-restricted samples "
        "on both sides, so no model ever faces the gradient jump; each fits a "
        "smooth one-sided target on an irregular domain.\n"
        "- **The near-band accuracy gap is a sampling artifact** (§3.1): "
        "density-balancing collapses near/far from 2.2–3.7 to ≈ 1 for every "
        "model. The residual intrinsic penalty is small, grows with atom "
        "stiffness, and is absorbed by gradient (H1) training.\n"
        "- **But rebalancing a fixed budget is the wrong fix** (§3.2): doubling "
        "the sample count at the natural distribution improves every band ~6×, "
        "while shifting a fixed 6k budget toward the near band degrades every "
        "band — including the near band itself.\n"
        "- **ReLU² + fractional-exponent penalty is the best atom class for this "
        "target** (§4.1, §4.4): ~8× more accurate than any smooth activation on "
        "both bands, by aligning its strongest ridges with the switching arm.\n"
        "- **Feedback synthesis is reliable up to the curve and mis-branches "
        "beyond it** (§5). The limit is **data coverage across the curve**, not "
        "the atoms' ability to fit — two-branch (multi-well or ±2π-tiled) "
        "training data are required if cross-switching feedback is needed.\n\n"

        "## Appendix: relative H1 (confounded)\n\n"
        "The naive region-local relative H1 metric reports models as *better* near "
        "the switching set (`near/far ≈ 0.55–1.15`, < 1 for 14 of 15 rows) — the "
        "V→0 artifact: with a single well, the `far` denominator is dominated by "
        "the near-zero interior at the upright, inflating far relative error. "
        "Kept for continuity; the count-fair absolute mean-L1 of §2 is the "
        "primary metric.\n\n"
        f"{rel_table}\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
