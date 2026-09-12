#!/usr/bin/env python3
"""Generate the Van der Pol open-loop value-data figures.

Visualises the open-loop training data only — the value/gradient samples produced
by the backward-characteristics solver (``src/OpenLoop/vdp``), no learned model.
Two manuscript figures (titles intentionally omitted; see ``README.md``):

    paper/plot/v.png   3D scatter of the (x, V) samples
    paper/plot/dv.png  state-plane scatter colored by V with ∇V arrows

Run: ``uv run python experiments/00_openloop/vdp/generate.py`` from the repo root,
or ``make openloop``.
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
from src.data import DATA_DIR, load_value_samples  # noqa: E402
from src.plots import (  # noqa: E402
    plot_value_scatter3d,
    plot_vdp_value_with_gradient_arrows2d,
    save_figure,
    style_vdp_reference_axes,
)

DATA = DATA_DIR / "VDP_beta_0.1_grid_30x30.npy"
FIG = REPO_ROOT / "paper" / "plot"


def main() -> int:
    dataset = load_value_samples(DATA)

    fig, ax = plot_value_scatter3d(dataset, title="", show=False, colorbar=False, azim=-105.0)
    style_vdp_reference_axes(ax, np.asarray(dataset["x"]))
    save_figure(fig, FIG / "v.png", tight=False, bbox_inches="tight")

    fig, ax = plot_vdp_value_with_gradient_arrows2d(dataset, title="", show=False)
    ax.set_xlabel("")
    save_figure(fig, FIG / "dv.png", tight=False, bbox_inches="tight")

    print(f"wrote 2 figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
