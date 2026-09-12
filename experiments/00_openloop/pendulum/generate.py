#!/usr/bin/env python3
"""Generate the pendulum swing-up open-loop value-data figures.

Visualises the open-loop training data only — the backward-PMP value samples and
the switching-set geometry they trace out (``src/OpenLoop/pendulum``), no learned
model. Produces only the three reference-data panels included in paper_0805.tex.
Titles are intentionally omitted; see ``README.md`` for what each figure is:

    paper/plot/pendulum_value_scatter.png  raw sample scatter (theta, theta-dot, V)
    paper/plot/pendulum_value_surface.png  V(x) over the state plane
    paper/plot/pendulum_regions.png        periodic upright basins + switching set

The regions figure colors each point by the upright it belongs to
(nearest basin-cut characteristic, tiled by 2*pi*k); the boundaries are the switching-set
spirals. The surface is built from the wired 3000-sample training set: the value function is
2*pi-periodic in theta, so all samples are folded into one fundamental cell [-pi, pi],
interpolated once (denser), then tiled by 2*pi across [-8, 8] — a seamless evaluation of the
same periodic V (no per-grid seam artifact).

Run: ``uv run python experiments/00_openloop/pendulum/generate.py`` from the repo root,
or ``make openloop``.
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
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.interpolate import (  # noqa: E402
    CloughTocher2DInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import cKDTree  # noqa: E402

from src.data import DATA_DIR  # noqa: E402
from src.OpenLoop.pendulum.nonsmooth import (  # noqa: E402
    compute_nonsmooth_curve,
    restrict_trajectory_to_curve,
)
from src.plots import (
    plot_pendulum_reference_scatter,
    plot_pendulum_reference_surface,
    plot_periodic_regions,
    save_figure,
)
from src.plotstyle import apply_publication_style as _apply_publication_style

# Resolve the wired pendulum dataset from the Hydra config so these figures always
# track whatever conf/data/pendulum.yaml points at.
_CFG = yaml.safe_load((REPO_ROOT / "conf" / "data" / "pendulum.yaml").read_text())
SAMPLES = DATA_DIR / _CFG["data"]["path"]
DATASET_DIR = SAMPLES.parent


def _raw_trajectory_pickle() -> Path:
    """Ordered backward-PMP paths for the line figures: prefer a raw-trajectory
    pickle co-located with the dataset, else the legacy 256-path debug pickle."""
    cands = sorted(DATASET_DIR.glob("*raw_trajectories*.pkl"))
    return cands[0] if cands else DATA_DIR / "_debug_raw_trajectories_256.pkl"


FIG = REPO_ROOT / "paper" / "plot"

_TWO_PI = 2.0 * np.pi
_OMEGA_CAP = 7.7                       # basin theta-dot extent
_N_PERIODS = 3                         # +/- periods to tile for the regions plot
# The regions figure needs the switching set tracked deeper than the training cap
# (basin_value_max=50, which only resolves ~half a spiral turn). cap=80 keeps the
# stable assembly while recovering the multi-winding spiral (the paper's Fig. 2 left);
# this is for visualisation only — it does not affect the wired training samples.
_REGIONS_CAP = 80.0

# Soft fills for the regions of attraction (one per tiled upright).
_REGION_COLS = ["#c9b3de", "#f3b0a0", "#a9c8e8", "#f3e0a0", "#a9dca0", "#d7b5e0", "#bfe0c0"]

_PARULA = LinearSegmentedColormap.from_list("parula", [
    (0.2422, 0.1504, 0.6603), (0.2780, 0.3556, 0.9777), (0.1129, 0.5500, 0.8901),
    (0.0488, 0.6981, 0.7327), (0.2161, 0.7843, 0.5923), (0.6473, 0.7456, 0.4188),
    (0.9856, 0.7372, 0.2537), (0.9763, 0.9831, 0.0538)])


def _value_scatter() -> Path:
    """Raw PMP samples as a 3D scatter: no interpolation, no periodic tiling."""
    with mpl.rc_context():
        _apply_publication_style()
        d = np.load(SAMPLES)
        x = np.asarray(d["x"])
        v = np.asarray(d["v"]).reshape(-1)

        fig, _ = plot_pendulum_reference_scatter(x, v, cmap=_PARULA)
        save_figure(fig, FIG / "pendulum_value_scatter", formats=["png"],
                    dpi=300, tight=False, bbox_inches="tight")
    return FIG / "pendulum_value_scatter.png"


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
        fig, _ = plot_pendulum_reference_surface(GX, GY, Z, cmap=_PARULA)
        save_figure(fig, FIG / "pendulum_value_surface", formats=["png"],
                    dpi=300, tight=False, bbox_inches="tight")
    return FIG / "pendulum_value_surface.png"


def _raw_trajectories() -> list:
    """The ordered backward-PMP trajectory objects (each with a (theta, theta-dot) path)."""
    with open(_raw_trajectory_pickle(), "rb") as f:
        return pickle.load(f)


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

    fig, _ = plot_periodic_regions(GX, GY, reg, colors=_REGION_COLS, periods=_N_PERIODS)
    out = FIG / "pendulum_regions.png"
    save_figure(fig, out, dpi=200, tight=False, bbox_inches="tight")
    return out


def main() -> int:
    _value_scatter()
    _surface()
    _regions()
    print(f"wrote 3 figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
