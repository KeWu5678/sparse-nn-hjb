#!/usr/bin/env python3
"""Generate the pendulum swing-up open-loop value-data figures.

Visualises the open-loop training data only — the backward-PMP value samples and
the switching-set geometry they trace out (``src/OpenLoop/pendulum``), no learned
model. Adds a raw sample scatter alongside the three panels of Fig. 2 in Han &
Yang (arXiv:2312.17467).
Titles are intentionally omitted; see ``README.md`` for what each figure is:

    figures/value_scatter.png         raw sample scatter (theta, theta-dot, V)
    figures/value_surface.png         V(x) over the (theta, theta-dot) plane
    figures/trajectories.png          full backward-PMP characteristics, multicolored
    figures/regions_of_attraction.png basins of the periodic uprights + switching set

The trajectory figure follows the paper's method (github_main_nosat.m, lines 42-90): each
backward characteristic is plotted whole as a line ``plot(theta, theta-dot)`` — the curves
spiral into the centers. The regions figure colors each point by the upright it belongs to
(nearest basin-cut characteristic, tiled by 2*pi*k); the boundaries are the switching-set
spirals. The surface is built from the wired 3000-sample training set: the value function is
2*pi-periodic in theta, so all samples are folded into one fundamental cell [-pi, pi],
interpolated once (denser), then tiled by 2*pi across [-8, 8] — a seamless evaluation of the
same periodic V (no per-grid seam artifact).

Run: ``../../../.venv/bin/python generate.py`` (from this folder) or ``make openloop``.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from scipy.interpolate import (  # noqa: E402
    CloughTocher2DInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import cKDTree  # noqa: E402

from src.OpenLoop.pendulum.nonsmooth import (  # noqa: E402
    compute_nonsmooth_curve,
    restrict_trajectory_to_curve,
)
from src.paths import DATA_DIR  # noqa: E402

# Resolve the wired pendulum dataset from the Hydra config so these figures always
# track whatever conf/data/pendulum.yaml points at.
_CFG = yaml.safe_load((REPO_ROOT / "conf" / "data" / "pendulum.yaml").read_text())
SAMPLES = DATA_DIR / _CFG["data"]["path"]
DATASET_DIR = SAMPLES.parent
BASE = SAMPLES.stem
CURVE = DATASET_DIR / f"{BASE}_nonsmooth_curve.npz"


def _raw_trajectory_pickle() -> Path:
    """Ordered backward-PMP paths for the line figures: prefer a raw-trajectory
    pickle co-located with the dataset, else the legacy 256-path debug pickle."""
    cands = sorted(DATASET_DIR.glob("*raw_trajectories*.pkl"))
    return cands[0] if cands else DATA_DIR / "_debug_raw_trajectories_256.pkl"


FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

_TWO_PI = 2.0 * np.pi
_OMEGA_CAP = 7.7                       # basin theta-dot extent
_N_PERIODS = 3                         # +/- periods to tile for the regions plot
# The regions figure needs the switching set tracked deeper than the training cap
# (basin_value_max=50, which only resolves ~half a spiral turn). cap=80 keeps the
# stable assembly while recovering the multi-winding spiral (the paper's Fig. 2 left);
# this is for visualisation only — it does not affect the wired training samples.
_REGIONS_CAP = 80.0

from src.plotstyle import apply_publication_style as _apply_publication_style

# MATLAB default line-color cycle (the paper cycles these per trajectory).
_MATLAB_CYCLE = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
# Soft fills for the regions of attraction (one per tiled upright).
_REGION_COLS = ["#c9b3de", "#f3b0a0", "#a9c8e8", "#f3e0a0", "#a9dca0", "#d7b5e0", "#bfe0c0"]

_PARULA = LinearSegmentedColormap.from_list("parula", [
    (0.2422, 0.1504, 0.6603), (0.2780, 0.3556, 0.9777), (0.1129, 0.5500, 0.8901),
    (0.0488, 0.6981, 0.7327), (0.2161, 0.7843, 0.5923), (0.6473, 0.7456, 0.4188),
    (0.9856, 0.7372, 0.2537), (0.9763, 0.9831, 0.0538)])


def _finalize_figure(fig, out_path, formats=None, dpi: int = 300, close: bool = True,
                     pad: float = 2.0, tight: bool = True, **kwargs) -> list[Path]:
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["png"]
    if tight:
        fig.tight_layout(pad=pad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def _sparse_ticks(lo: float, hi: float) -> list[float]:
    mid = 0.0 if lo < 0.0 < hi else 0.5 * (lo + hi)
    return [round(float(lo), 2), round(float(mid), 2), round(float(hi), 2)]


def _style_3d_axes(ax, *, xlim: tuple[float, float], ylim: tuple[float, float],
                   zlim: tuple[float, float] | None = None,
                   box_aspect: tuple[float, float, float] = (1.0, 1.0, 0.55),
                   axis_linewidth: float = 1.0) -> None:
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks(_sparse_ticks(*xlim)); ax.set_yticks(_sparse_ticks(*ylim))
    if zlim is not None:
        ax.set_zlim(*zlim)
        ax.set_zticks(_sparse_ticks(*zlim))
    ax.view_init(elev=15, azim=-105)
    ax.set_box_aspect(box_aspect)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor((0, 0, 0, 0))
        a._axinfo["grid"].update(color="0.85", linewidth=0.5)
        a.line.set_linewidth(axis_linewidth)
    ax.tick_params(labelsize=9, pad=1, width=axis_linewidth)


def _value_scatter() -> Path:
    """Raw PMP samples as a 3D scatter: no interpolation, no periodic tiling."""
    with mpl.rc_context():
        _apply_publication_style()
        d = np.load(SAMPLES)
        x = np.asarray(d["x"])
        v = np.asarray(d["v"]).reshape(-1)

        fig = plt.figure(figsize=(8.5, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], v, c=v, cmap=_PARULA, s=9.0, alpha=0.86,
                   depthshade=False, edgecolors="none")
        _style_3d_axes(ax, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), zlim=(0.0, 60.0))
        ax.set_zticks([0, 30, 60])
        _finalize_figure(fig, FIG / "value_scatter", formats=["png"],
                         dpi=300, tight=False, bbox_inches="tight")
    return FIG / "value_scatter.png"


def _surface() -> Path:
    """V(x) surface from the 3000-sample set: fold into one period, interp, tile."""
    d = np.load(SAMPLES)
    th, om, v = d["x"][:, 0], d["x"][:, 1], d["v"].reshape(-1)
    keep = np.abs(om) <= _OMEGA_CAP
    th, om, v = th[keep], om[keep], v[keep]

    thf = (th + np.pi) % _TWO_PI - np.pi            # fold theta into the fundamental cell
    seamR, seamL = thf > (np.pi - 0.8), thf < (-np.pi + 0.8)
    TF = np.concatenate([thf, thf[seamR] - _TWO_PI, thf[seamL] + _TWO_PI])
    OM = np.concatenate([om, om[seamR], om[seamL]])
    VV = np.concatenate([v, v[seamR], v[seamL]])    # wrap seam copies -> periodic interp

    pts = np.column_stack([TF, OM])
    lin = CloughTocher2DInterpolator(pts, VV)
    nea = NearestNDInterpolator(np.column_stack([TF, OM]), VV)
    cf = np.linspace(-np.pi, np.pi, 240)
    co = np.linspace(-_OMEGA_CAP, _OMEGA_CAP, 380)
    CF, CO = np.meshgrid(cf, co)
    Zc = lin(CF, CO)
    nanm = ~np.isfinite(Zc)
    Zc[nanm] = nea(CF[nanm], CO[nanm])
    cell = RegularGridInterpolator((co, cf), Zc, bounds_error=False, fill_value=None)

    g = np.linspace(-8, 8, 480)
    GX, GY = np.meshgrid(g, g)
    gf = (GX + np.pi) % _TWO_PI - np.pi
    Z = cell(np.column_stack([np.clip(GY.ravel(), -_OMEGA_CAP, _OMEGA_CAP),
                              gf.ravel()])).reshape(GX.shape)

    with mpl.rc_context():
        _apply_publication_style()
        fig = plt.figure(figsize=(4.2, 4.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(GX, GY, Z, cmap=_PARULA, rcount=480, ccount=480, linewidth=0,
                        antialiased=True, vmin=0.0, vmax=60.0)
        _style_3d_axes(ax, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), zlim=(0.0, 60.0),
                       box_aspect=(1.0, 1.0, 0.5), axis_linewidth=0.6)
        _finalize_figure(fig, FIG / "value_surface", formats=["png"],
                         dpi=300, tight=False, bbox_inches="tight")
    return FIG / "value_surface.png"


def _raw_trajectories() -> list:
    """The ordered backward-PMP trajectory objects (each with a (theta, theta-dot) path)."""
    with open(_raw_trajectory_pickle(), "rb") as f:
        return pickle.load(f)


def _trajectories() -> Path:
    """Full backward-PMP characteristics, multicolored (paper github_main_nosat.m l.42-90).

    Each trajectory is plotted whole — ``plot(theta, theta-dot)`` — spiralling into the
    centers; colors cycle per trajectory (MATLAB default), giving the dense spiral web."""
    raw = _raw_trajectories()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for i, t in enumerate(raw):
        s = np.asarray(t.state)
        ax.plot(s[:, 0], s[:, 1], lw=0.35,
                color=_MATLAB_CYCLE[i % len(_MATLAB_CYCLE)], alpha=0.7)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\dot\theta$")
    ax.set_xlim(-10, 10); ax.set_ylim(-8, 8); ax.set_aspect("equal")
    ax.set_xticks([-10, -5, 0, 5, 10])
    out = FIG / "trajectories.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _regions() -> Path:
    """Regions of attraction to the periodic uprights + switching set.

    Each backward characteristic, cut at the switching set, lies in one upright's basin;
    tiling by 2*pi*k and coloring each grid cell by its nearest cut point gives the basins,
    whose boundaries are the switching-set spirals. The switching set is tracked deep
    (``_REGIONS_CAP``) so the spiral winds several times around the hanging points, as in
    the paper's Fig. 2 (left); the shallow training cap would only show ~half a turn."""
    raw = tuple(_raw_trajectories())
    curve = compute_nonsmooth_curve(raw, 0.1, basin_value_max=_REGIONS_CAP)
    cut = []
    for t in raw:
        c, _ = restrict_trajectory_to_curve(t, curve)
        s = np.asarray(c.state)
        if s.size:
            cut.append(s)
    cutpts = np.vstack(cut)
    if len(cutpts) > 150_000:                       # subsample for the KD-tree
        cutpts = cutpts[np.random.default_rng(0).choice(len(cutpts), 150_000, replace=False)]

    pts, lab = [], []
    for k in range(-_N_PERIODS, _N_PERIODS + 1):
        sh = cutpts.copy(); sh[:, 0] += _TWO_PI * k
        pts.append(sh); lab.append(np.full(len(cutpts), k + _N_PERIODS))
    pts = np.vstack(pts); lab = np.concatenate(lab)

    gx = np.linspace(-12, 12, 760); gy = np.linspace(-8, 8, 560)
    GX, GY = np.meshgrid(gx, gy)
    _, idx = cKDTree(pts).query(np.column_stack([GX.ravel(), GY.ravel()]))
    reg = lab[idx].reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.pcolormesh(GX, GY, reg, cmap=ListedColormap(_REGION_COLS[:2 * _N_PERIODS + 1]),
                  shading="auto", rasterized=True)
    ax.contour(GX, GY, reg, levels=np.arange(0.5, 2 * _N_PERIODS + 0.5),
               colors="k", linewidths=0.7)        # switching-set spiral boundaries
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\dot\theta$")
    ax.set_xlim(-12, 12); ax.set_ylim(-8, 8); ax.set_aspect("equal")
    out = FIG / "regions_of_attraction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    _value_scatter()
    _surface()
    _trajectories()
    _regions()
    print(f"wrote 4 figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
