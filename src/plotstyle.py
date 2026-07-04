"""Publication house style — single source for all experiment figures.

The conventions (palette, serif + cm mathtext, hidden top/right spines,
frameless legends, PNG-only at 300 dpi, no in-figure titles) are documented in
CLAUDE.md ("Plotting Style"). Experiment analysis/generate scripts import from
here instead of carrying their own copy.
"""

import matplotlib as mpl

PALETTE = {
    "blue_main": "#0F4D92",
    "teal": "#42949E",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "violet": "#9A4D8E",
}


def style_frontier_axes(ax, *, legend_ncol: int | None = None) -> None:
    """Boxed variant used by every neuron/H1 frontier plot.

    Frontier plots deviate from the default open-spine style: full box,
    dotted grid, and a framed legend row centred below the axes (save with
    ``bbox_inches="tight"`` so the legend is not clipped).
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.grid(True, which="major", ls=":", lw=0.6, color="0.8")
    ax.set_axisbelow(True)
    handles, _ = ax.get_legend_handles_labels()
    ncol = legend_ncol if legend_ncol is not None else min(len(handles), 4)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                       ncol=ncol, frameon=True, fontsize=10.5,
                       handletextpad=0.5, columnspacing=1.2)
    legend.get_frame().set_edgecolor("0.3")
    legend.get_frame().set_linewidth(0.8)


def apply_publication_style(font_size: int = 12, axes_linewidth: float = 1.0) -> None:
    """Set the house-style matplotlib rcParams (call once before plotting)."""
    mpl.rcParams.update({
        "font.family": ["serif"],
        "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "legend.frameon": False,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "svg.fonttype": "none",
        "text.usetex": False,
    })
