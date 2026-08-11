#!/usr/bin/env python3
"""Paper-facing alpha-by-beta heatmaps for the moment axis.

The screen's own report (``analysis.py``) writes one composite three-panel
figure per activation and moment order, which is fine on a wide screen but
illegible at paper width. This script writes the same data as *one PNG per
(activation, metric)* so the paper can arrange them as subfigures, each with
its own subcaption -- the house rule for multi-panel results.

House style applies: the palette-derived sequential colormaps replace the
matplotlib default, and no panel carries a title or any identifying text.
Everything that says which activation, which moment order and which metric a
panel shows belongs to the LaTeX subcaption. The cell annotations stay: they
are the measurement itself, not a label, and a 4x5 grid is unreadable without
them at this size.

Run:  MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \\
          experiments/01_vdp/moment_penalty/paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotstyle import PALETTE, apply_publication_style  # noqa: E402

sys.path.insert(0, str(OUTPUT_DIR))
import analysis as screen  # noqa: E402

FIGURE_DIR = OUTPUT_DIR / "figures"

# The two representative nonhomogeneous activations the paper carries, both at
# one moment order so the panels are directly comparable along alpha and beta.
PAPER_ACTIVATIONS = ("gaussian", "softplus")
PAPER_ORDER = 3.0

# Sequential ramps built from the house palette; "low" is a near-white tint of
# the accent so the light end stays legible behind dark annotation text.
_RAMPS = {
    "rel_h1": ("#F7F2F2", PALETTE["red_strong"]),
    "neurons": ("#EEF2F7", PALETTE["blue_main"]),
    "radius_r95": ("#F4EEF3", PALETTE["violet"]),
}

# metric key -> (colorbar label, annotation format, log-scaled colour)
PAPER_METRICS = {
    "rel_h1": (r"relative $H^1$ error", ".2f", True),
    "neurons": ("active neurons", ".0f", False),
    "radius_r95": (r"$R_{95}$", ".2g", True),
}


def _cmap(metric: str) -> LinearSegmentedColormap:
    low, high = _RAMPS[metric]
    return LinearSegmentedColormap.from_list(f"house_{metric}", [low, high])


def _panel(
    index: dict[tuple[Any, ...], dict[str, Any]],
    activation: str,
    order: float,
    metric: str,
) -> Path:
    label, annotation_format, use_log = PAPER_METRICS[metric]
    raw = screen._matrix(index, activation, order, metric)

    if use_log:
        positive = raw[raw > 0.0]
        floor = float(positive.min()) if positive.size else 1.0
        colors = np.log10(np.maximum(raw, floor))
    else:
        colors = raw
    color_label = label

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    image = ax.imshow(colors, aspect="auto", cmap=_cmap(metric))

    # Annotate every cell with the raw value; flip the text to white only once
    # the cell is dark enough for black to lose contrast.
    finite = colors[np.isfinite(colors)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo if hi > lo else 1.0
    else:
        lo, span = 0.0, 1.0
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            shade = (colors[row, column] - lo) / span
            ax.text(
                column,
                row,
                format(raw[row, column], annotation_format),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if shade > 0.62 else "0.15",
            )

    ax.set_xticks(range(len(screen.BETAS)))
    ax.set_xticklabels([screen._tick(beta) for beta in screen.BETAS])
    ax.set_yticks(range(len(screen.ALPHAS)))
    ax.set_yticklabels([screen._tick(alpha) for alpha in screen.ALPHAS])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.tick_params(length=0)

    # A matrix needs its frame, so all four spines stay -- the house rule that
    # hides top and right is for line plots, where the box is redundant.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.4")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(color_label)
    if use_log:
        # The colour scale is log10 but the cells report raw values; relabel the
        # ticks in raw units so the two readings agree, and keep them sparse --
        # the cells carry the numbers, the bar only has to show the direction.
        colorbar.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        colorbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{10.0 ** value:.3g}")
        )
    else:
        colorbar.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("0.4")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"moment_grid_{activation}_{metric}.png"
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> int:
    rows = screen.load_rows()
    index = screen._index(rows)
    written = [
        _panel(index, activation, PAPER_ORDER, metric)
        for activation in PAPER_ACTIVATIONS
        for metric in PAPER_METRICS
    ]
    for path in written:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
