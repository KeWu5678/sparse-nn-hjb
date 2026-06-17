#!/usr/bin/env python3
"""Analyse the region_split_pendulum study.

Reads per-run JSON records from ``rawdata/logs/multirun/region_split_pendulum/``.
The region metrics are computed by the training hook on the live as-fit model over
the **full dataset** (see ``scripts/train.py``); this just aggregates them into
``results.md`` (two tables + the error-vs-distance plot). `near` = lowest 10% of
samples by distance to the switching set; `far` = the rest. See ``README.md`` for
the error-metric rationale. Reproduce with ``make region_split_pendulum``.
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
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}
_N_BINS = 8
_TWO_PI = 2.0 * np.pi
# Shared state-plane extent for the Fig. 2 panels (x[0], x[1]) — the left scatter
# and the right region-of-attraction map use the same range.
_ROA_DOMAIN = (-10.0, 10.0, -8.0, 8.0)
# Raw backward-PMP characteristics (states along each trajectory). They are not a
# curated per-dataset artifact yet — only this debug pickle persists them — so the
# region-of-attraction panel is best-effort and falls back to the switching-set
# curve when it is absent.
_TRAJ_PKL = DATA_DIR / "_debug_raw_trajectories_256.pkl"


def load_rows() -> tuple[list[dict[str, Any]], str | None]:
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make region_split_pendulum` first"
        )
    rows, cache = [], None
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        m = record["metrics"][0]["values"]
        if "near_l1_h1" not in m:
            continue
        cache = cache or cfg.get("eval", {}).get("distance_cache")
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        rows.append({
            "kind": model["kind"],
            "insertion": model["insertion"],
            "activation": model["activation"],
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
        })
    return rows, cache


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lowest far-region L1 per (kind, insertion, activation, loss) — gamma selection."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["kind"], row["insertion"], row["activation"], row["loss"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["far_l1"]))
    return best


def _bin_centers(cache: str | None) -> np.ndarray | None:
    if not cache:
        return None
    d = np.load(DATA_DIR / cache)["distance"]
    edges = np.quantile(d, np.linspace(0.0, 1.0, _N_BINS + 1))
    return 0.5 * (edges[:-1] + edges[1:])


def _plot_error_vs_distance(best: list[dict[str, Any]], centers: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted({r["kind"] for r in best})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for r in sorted(best, key=lambda r: r["activation"]):
            if r["kind"] == kind:
                ax.plot(centers, r["bins"], "-o", ms=3, label=r["activation"])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{kind} / profile")
        ax.set_xlabel("distance to switching set")
        ax.set_ylabel("per-sample abs error / model mean")
        ax.legend(fontsize=8)
    fig.suptitle("Error vs distance to switching set (near → far); >1 = worse than model average")
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / "error_vs_distance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    return out


def _region_of_attraction_inputs(curve_path: Path, *, n_periods: int = 2):
    """Tiled PMP characteristics + branch labels + switching arms for the RoA panel.

    Each backward-PMP trajectory's states fill the region of attraction of the
    upright equilibrium it converges to; tiling them by 2kπ fills the periodic
    neighbours, whose equilibria sit at θ = 2kπ. The region id is the period index
    ``k`` (0-based ``k + n_periods``), so each colour = the basin of one upright
    equilibrium and the switching curves fall on the colour boundaries, as in the
    paper. ``region_labels[id]`` names that equilibrium angle for the legend.
    Returns ``None`` if the raw-trajectory pickle is unavailable.
    """
    if not _TRAJ_PKL.exists():
        return None
    import pickle
    with open(_TRAJ_PKL, "rb") as f:
        trajectories = pickle.load(f)

    points, labels = [], []
    for traj in trajectories:
        s = np.asarray(traj.state)
        for k in range(-n_periods, n_periods + 1):
            shifted = s.copy()
            shifted[:, 0] += _TWO_PI * k
            points.append(shifted)
            labels.append(np.full(s.shape[0], k + n_periods))
    points = np.vstack(points)
    labels = np.concatenate(labels)

    def _angle_label(k: int) -> str:
        if k == 0:
            return "θ = 0"
        m = 2 * k
        return f"θ = {m:+d}π".replace("+", "")  # e.g. "θ = 2π", "θ = -2π"

    region_labels = [_angle_label(k) for k in range(-n_periods, n_periods + 1)]

    # Switching-set arms (4 tracked spiral arms), tiled across the same periods.
    with np.load(curve_path) as cv:
        cp = np.asarray(cv["points"])
    arm_len = cp.shape[0] // 4
    base = ([cp[i * arm_len:(i + 1) * arm_len] for i in range(4)]
            if arm_len >= 2 and cp.shape[0] == 4 * arm_len else [cp])
    arms = []
    for k in range(-n_periods, n_periods + 1):
        for arm in base:
            shifted = arm.copy()
            shifted[:, 0] += _TWO_PI * k
            arms.append(shifted)
    return points, labels, arms, region_labels


def _plot_figure2(cache: str) -> Path | None:
    """Reproduce the left+middle panels of Fig. 2 (Han & Yang, arXiv:2312.17467).

    Left: a 3D scatter of the raw training value samples (no surface, no
    interpolation) on the same state-plane extent as the middle panel. Middle: the
    regions of attraction to the periodic upright equilibria, separated by the
    nonsmooth switching curves (falls back to the basin/switching-curve plot if the
    raw trajectories are unavailable). Data/curve paths share the ``distance_cache``
    base.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plots import (  # noqa: E402
        plot_nonsmooth_curve,
        plot_regions_of_attraction,
        plot_value_scatter3d,
    )

    dataset_path = DATA_DIR / cache.replace("_region_distances.npz", ".npz")
    curve_path = DATA_DIR / cache.replace("_region_distances.npz", "_nonsmooth_curve.npz")
    dataset = np.load(dataset_path)
    distance = np.load(DATA_DIR / cache)["distance"]
    x0, x1, y0, y1 = _ROA_DOMAIN

    fig = plt.figure(figsize=(13, 5.5))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mid = fig.add_subplot(1, 2, 2)
    plot_value_scatter3d(dataset, ax=ax_left, xlim=(x0, x1), ylim=(y0, y1),
                         title="Training value samples V(x)", show=False)
    regions = _region_of_attraction_inputs(curve_path)
    if regions is not None:
        points, labels, arms, region_labels = regions
        plot_regions_of_attraction(points, labels, curve_arms=arms,
                                   region_labels=region_labels, ax=ax_mid,
                                   domain=_ROA_DOMAIN,
                                   title="Regions of attraction & switching set",
                                   show=False)
    else:
        plot_nonsmooth_curve(curve_path, dataset=dataset, distance=distance,
                             near_percentile=10.0, ax=ax_mid,
                             title="Switching set & smooth basin", show=False)
    fig.suptitle(
        "Pendulum swing-up: training value samples (left) and regions of attraction "
        "/ switching set (middle)\nafter Han & Yang (arXiv:2312.17467), Fig. 2 left/middle"
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / "figure2_value_and_switching.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


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

    centers = _bin_centers(cache)
    fig_line = ""
    if centers is not None:
        fig = _plot_error_vs_distance(best, centers)
        fig_line = f"\n![error vs distance](figures/{fig.name})\n"

    fig2_line = ""
    if cache:
        fig2 = _plot_figure2(cache)
        if fig2 is not None:
            fig2_line = f"\n![value samples & regions of attraction](figures/{fig2.name})\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.md"
    out.write_text(
        f"# {EXPERIMENT} Results\n\n"
        "Region split scored over the **full dataset** on the live as-fit model. "
        "`near` = lowest 10% of samples by distance to the switching set; `far` = the "
        "rest. See `README.md` for the error-metric rationale.\n\n"
        "## Value samples & regions of attraction (Fig. 2 left/middle)\n\n"
        "Left: a 3D scatter of the raw training value samples (no interpolation), on "
        "the same state-plane extent as the right panel. Right: the regions of "
        "attraction to the (periodic) upright equilibria — PMP characteristics filled "
        "by nearest-point classification — separated by the nonsmooth switching "
        "curves. After Han & Yang (arXiv:2312.17467), Fig. 2 left/middle.\n"
        f"{fig2_line}\n"
        "## Mean per-sample L1 (primary)\n\n"
        "`near/far` > 1 ⇒ worse at the switching set. Region mean per-sample L1 "
        "(absolute) error / global mean ‖true‖ — count-fair and robust to the V→0 "
        "interior.\n\n"
        f"{l1_table}\n\n"
        "## Error vs distance to switching set (diagnostic)\n"
        f"{fig_line}\n"
        "## Relative H1 (kept for continuity — confounded)\n\n"
        f"{rel_table}\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
