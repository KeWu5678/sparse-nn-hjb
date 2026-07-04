#!/usr/bin/env python3
"""Generate the Van der Pol open-loop value-data figures.

Visualises the open-loop training data only — the value/gradient samples produced
by the backward-characteristics solver (``src/OpenLoop/vdp``), no learned model.
Three figures (titles intentionally omitted; see ``README.md`` for what each is):

    figures/value_scatter.png   3D scatter of the (x, V) samples
    figures/value_gradient.png  state-plane scatter colored by V with ∇V arrows
    figures/value_surface.png   V(x) interpolated to a smooth surface

Run: ``../../../.venv/bin/python generate.py`` (from this folder) or ``make openloop``.
"""
from __future__ import annotations

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
from scipy.interpolate import griddata  # noqa: E402

from src.paths import DATA_DIR  # noqa: E402
from src.plots import (  # noqa: E402
    plot_value_scatter3d,
    plot_vdp_value_with_gradient_arrows2d,
)

DATA = DATA_DIR / "VDP_beta_0.1_grid_30x30.npy"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def _ref_style(ax, x) -> None:
    """Match the reference value-surface style: no axis names, sparse x/y ticks, and
    the shared 0/10/20 vertical-axis range."""
    def _sparse(lo: float, hi: float) -> list[float]:
        mid = 0.0 if lo < 0.0 < hi else round((lo + hi) / 2.0, 2)
        return sorted({round(lo, 2), mid, round(hi, 2)})

    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_title("")
    ax.set_xticks(_sparse(float(x[:, 0].min()), float(x[:, 0].max())))
    ax.set_yticks(_sparse(float(x[:, 1].min()), float(x[:, 1].max())))
    ax.set_zticks([0, 10, 20])
    ax.set_zlim(0.0, 20.0)


def _surface(dataset) -> Path:
    """Smooth V(x) surface, cubic-interpolated from the scattered samples."""
    x = np.asarray(dataset["x"]); v = np.asarray(dataset["v"]).reshape(-1)
    g0 = np.linspace(x[:, 0].min(), x[:, 0].max(), 140)
    g1 = np.linspace(x[:, 1].min(), x[:, 1].max(), 140)
    G0, G1 = np.meshgrid(g0, g1)
    Z = griddata(x, v, (G0, G1), method="cubic")
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(G0, G1, Z, cmap="viridis", linewidth=0, antialiased=True,
                    rcount=140, ccount=140, vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    _ref_style(ax, x)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((0, 0, 0, 0))
        a._axinfo["grid"].update(color="0.85", linewidth=0.5)
    ax.view_init(elev=15, azim=-105)
    out = FIG / "value_surface.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    dataset = np.load(DATA, allow_pickle=True)

    fig, ax = plot_value_scatter3d(dataset, title="", show=False, colorbar=False, azim=-105.0)
    _ref_style(ax, np.asarray(dataset["x"]))
    fig.savefig(str(FIG / "value_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close("all")

    fig, ax = plot_vdp_value_with_gradient_arrows2d(dataset, title="", show=False)
    ax.set_xlabel("")
    fig.savefig(str(FIG / "value_gradient.png"), dpi=300, bbox_inches="tight")
    plt.close("all")

    _surface(dataset)

    print(f"wrote 3 figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
