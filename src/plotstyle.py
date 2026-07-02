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
