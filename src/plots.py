#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared rendering for value-function experiments and current-paper figures.

Current-paper renderers accept prepared arrays; ``src.results`` owns saved-model
loading and physical prediction. Legacy convenience wrappers remain available.
Renderers are limited to the manuscript's included figures.
Tabular summaries live in ``src/metric.py``.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PLOT_CACHE = os.path.join(_REPO_ROOT, "rawdata", "logs", "matplotlib")
_XDG_CACHE = os.path.join(_REPO_ROOT, "rawdata", "logs", "cache")
os.makedirs(_PLOT_CACHE, exist_ok=True)
os.makedirs(_XDG_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _PLOT_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from .plotstyle import FRONTIER_RC, PALETTE, apply_publication_style, style_frontier_axes
from .results import history_atoms, model_from_history


def save_figure(fig, path, *, dpi=300, tight=True, pad=2.0, close=True,
                formats=None, palette_colors=None, **kwargs):
    """Save a publication figure without changing the arrays used to draw it.

    Layout and bbox choices are explicit so existing figure implementations keep
    their geometry. ``formats`` is retained for legacy callers; new callers use
    a PNG path. Returns the saved paths.
    """
    path = Path(path)
    formats = formats or [path.suffix.lstrip(".") or "png"]
    if tight:
        fig.tight_layout(pad=pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, **kwargs)
        if palette_colors is not None:
            from PIL import Image
            with Image.open(out) as image:
                image.convert("P", palette=Image.Palette.ADAPTIVE,
                              colors=palette_colors).save(out, optimize=True)
        saved.append(out)
    if close:
        plt.close(fig)
    return saved


def plot_activation_curves(x, curves, styles, panels):
    """Value/derivative panels from precomputed activation curves."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for col, (key, ylabel) in enumerate(panels):
        ax = axes[col]
        if key in ("deriv", "curv"):
            ax.axhline(0.0, color=PALETTE["neutral"], lw=1.4, zorder=1)
        for name, values in curves.items():
            label, color, ls = styles[name]
            ax.plot(x, values[col], color=color, ls=ls, lw=2.6,
                    label=label, zorder=3, clip_on=False)
        ax.set_xlabel(r"$x$  (pre-activation)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-4, 4)
        if col == 0:
            ax.legend(loc="upper left")
    return fig, axes


def plot_summary_frontier(series):
    """Preserve the summary frontier's index-spaced markers and 8.5x6 layout."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for s in series:
        ax.plot(s["ns"], s["h1"], color=s["color"], ls=s["ls"], lw=1.6,
                label=s["label"], marker="o", ms=6.0, mec="0.15", mew=0.8,
                markevery=max(1, len(s["ns"]) // 12))
    ax.set_xlabel("number of neurons")
    ax.set_ylabel(r"best relative $H^1$ error")
    ax.set_yscale("log")
    style_frontier_axes(ax, legend_ncol=3)
    return fig, ax


def plot_feedback_trace(series, *, ylabel, time_limit):
    """One VDP feedback panel; inputs already contain the desired quantity."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in series:
        ax.plot(s["t"], s["y"], color=s["color"], ls=s["ls"],
                lw=s["lw"], label=s["label"])
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, time_limit)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    return fig, ax


def plot_normal_cross_section(s, truth, series, *, ylabel):
    """Draw a prepared reference and predictions; no branch selection here."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axvline(0.0, color="0.8", lw=1.0, ls="--", zorder=0)
    ax.plot(s, truth, color="0.0", ls="-", lw=2.6, label="true PMP", zorder=3)
    for item in series:
        ax.plot(s, item["y"], color=item["color"], ls=item["ls"], lw=2.0,
                label=item["label"], zorder=2)
    all_y = np.concatenate([truth, *(item["y"] for item in series)])
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.06 * max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=10, borderaxespad=0.0)
    return fig, ax


def plot_regional_dumbbell(series, *, threshold):
    """Draw caller-ordered physical-coordinate regional scores."""
    from matplotlib.lines import Line2D
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for y, item in enumerate(series):
        color = item["color"]
        ax.plot([item["rest"], item["tube"]], [y, y], color=color, lw=2.0, zorder=1)
        ax.scatter([item["rest"]], [y], s=70, facecolor="white", edgecolor=color,
                   lw=2.0, zorder=2)
        ax.scatter([item["tube"]], [y], s=70, color=color, zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([item["label"] for item in series])
    ax.set_xlabel(r"relative $H^1$ error")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="0.3", color="0.3",
               label=f"switching tube ($d\\leq{threshold:g}$)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor="0.3", color="0.3", label="rest"),
    ], loc="lower left", fontsize=10)
    return fig, ax


def plot_feedback_phase(curve, paths, starts, *, xlim, ylim):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(curve[:, 0], curve[:, 1], s=3, color="0.1", zorder=3)
    for side, color in (("A", PALETTE["blue_main"]), ("B", PALETTE["red_strong"])):
        xs = paths[side]
        ax.plot(xs[:, 0], xs[:, 1], color=color, lw=2.2, label=f"start {side}", zorder=2)
        ax.scatter([starts[side][0]], [starts[side][1]], s=80, color=color,
                   marker="x", lw=2.2, zorder=4)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax


def plot_pendulum_control(reference, series, *, time_limit):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    t_true, us_true = reference
    all_us = np.concatenate([us_true, *(item["y"] for item in series)])
    ylo, yhi = all_us.min() - 1.0, all_us.max() + 1.0
    ax.plot(t_true, us_true, color="0.0", ls="-", lw=3.2, zorder=2, label="true PMP")
    for item in series:
        ax.plot(item["t"], item["y"], color=item["color"], ls=item["ls"],
                lw=1.9, zorder=3, label=item["label"])
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"feedback control $u(t)$")
    ax.set_xlim(0, time_limit)
    ax.set_ylim(ylo, yhi)
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax

# Default colormap for learned value surfaces — the "MATLAB surf" blue→yellow ramp
# shared across all the experiment surface plots (see ``plot_model_value_surface``).
_SURFACE_CMAP = LinearSegmentedColormap.from_list("surface_blue_yellow", [
    (0.2298, 0.2987, 0.7537),
    (0.1050, 0.5090, 0.9180),
    (0.0000, 0.6900, 0.8200),
    (0.3000, 0.7600, 0.5200),
    (0.9100, 0.7400, 0.1800),
    (1.0000, 0.8800, 0.0000),
])

logger = logging.getLogger(__name__)


# ============================================================================ #
# INTERNAL HELPERS
# ============================================================================ #
def _get_field(dataset: Any, name: str) -> np.ndarray:
    """Extract a field from a structured array or dict-like dataset."""
    if isinstance(dataset, np.ndarray) and dataset.dtype.fields is not None:
        if name not in dataset.dtype.fields:
            raise KeyError(f"Dataset is missing field '{name}'. Available: {list(dataset.dtype.fields.keys())}")
        return dataset[name]
    if hasattr(dataset, "keys") and hasattr(dataset, "__getitem__"):
        if name not in dataset:
            raise KeyError(f"Dataset is missing key '{name}'. Available: {list(dataset.keys())}")
        return np.asarray(dataset[name])
    raise TypeError(
        "dataset must be a NumPy structured array (with fields) or a dict-like object containing keys "
        "'x', 'v', 'dv'."
    )


def _best_iteration_atoms(history: Any, run_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-iteration ``(a, b, u)`` from a fit result, robust to its container.

    Accepts a single ``History`` object (attribute access), a dict-like run, or a
    list/tuple of either (selecting ``run_index``). Returns inner weights ``a``
    (n, d), inner bias ``b`` (n,), and outer weights ``u`` (1, n).
    """
    return history_atoms(history, run_index=run_index)


# ---------------------------------------------------------------------------- #
# Neuron / H1 frontier
# ---------------------------------------------------------------------------- #
# Shared publication style for frontier plots (boxed variant — see
# src.plotstyle.style_frontier_axes). Applied via rc_context so the module
# leaves global rcParams untouched.
_FRONTIER_RC = FRONTIER_RC  # compatibility name; typography belongs to plotstyle




def plot_weight_portrait(a, b, u, *, limit=4.0):
    """Raw inner-parameter portrait with the existing signed-size encoding."""
    apply_publication_style()
    lim = limit
    th = np.linspace(0, 2 * np.pi, 200)
    cx, cy, cz = np.cos(th), np.sin(th), np.zeros_like(th)
    u = np.asarray(u).reshape(-1)
    sizes = np.abs(u) / (np.abs(u).max() or 1.0) * 130 + 12
    colors = np.where(u >= 0, "#001BF8", "#FFFD3A")
    fig = plt.figure(figsize=(4.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(cx, cy, cz, color="#4BFE52", lw=1.0)
    ax.scatter(a[:, 0], a[:, 1], b, s=sizes, c=colors,
               alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([-lim, 0, lim]); ax.set_yticks([-lim, 0, lim]); ax.set_zticks([-lim, 0, lim])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    # House 3D style: vertical axis on the left, no grey pane walls, faint grid.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((0, 0, 0, 0))
        axis._axinfo["grid"].update(color="0.85", linewidth=0.5)
    ax.view_init(elev=15, azim=-105)
    return fig, ax


def plot_moment_order_panels(panels, orders, *, ylabel):
    """Moment-order study with the current two-panel, shared-axis geometry."""
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for ax, series in zip(axes, panels):
        for item in series:
            ax.plot(orders, item["y"], color=item["color"], marker=item["marker"],
                    ls=item["ls"], lw=1.4, markersize=4.5,
                    markeredgecolor="0.2", markeredgewidth=0.5)
        ax.set_xticks(orders)
        ax.set_xticklabels([f"{p:g}" for p in orders])
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel(r"moment order $p$")
    return fig, axes


def style_vdp_reference_axes(ax, x) -> None:
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



def _reference_sparse_ticks(lo: float, hi: float) -> list[float]:
    mid = 0.0 if lo < 0.0 < hi else 0.5 * (lo + hi)
    return [round(float(lo), 2), round(float(mid), 2), round(float(hi), 2)]


def style_pendulum_reference_axes(ax, *, xlim: tuple[float, float], ylim: tuple[float, float],
                   zlim: tuple[float, float] | None = None,
                   box_aspect: tuple[float, float, float] = (1.0, 1.0, 0.55),
                   axis_linewidth: float = 1.0) -> None:
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks(_reference_sparse_ticks(*xlim)); ax.set_yticks(_reference_sparse_ticks(*ylim))
    if zlim is not None:
        ax.set_zlim(*zlim)
        ax.set_zticks(_reference_sparse_ticks(*zlim))
    ax.view_init(elev=15, azim=-105)
    ax.set_box_aspect(box_aspect)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor((0, 0, 0, 0))
        a._axinfo["grid"].update(color="0.85", linewidth=0.5)
        a.line.set_linewidth(axis_linewidth)
    ax.tick_params(labelsize=9, pad=1, width=axis_linewidth)



def plot_pendulum_reference_scatter(x, v, *, cmap):
    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], v, c=v, cmap=cmap, s=9.0, alpha=0.86,
               depthshade=False, edgecolors="none")
    style_pendulum_reference_axes(ax, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), zlim=(0.0, 60.0))
    ax.set_zticks([0, 30, 60])
    return fig, ax

def plot_pendulum_reference_surface(GX, GY, Z, *, cmap):
    fig = plt.figure(figsize=(4.2, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(GX, GY, Z, cmap=cmap, rcount=480, ccount=480, linewidth=0,
                    antialiased=True, vmin=0.0, vmax=60.0)
    style_pendulum_reference_axes(ax, xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), zlim=(0.0, 60.0),
                   box_aspect=(1.0, 1.0, 0.5), axis_linewidth=0.6)
    return fig, ax

def plot_periodic_regions(GX, GY, reg, *, colors, periods):
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.pcolormesh(GX, GY, reg, cmap=ListedColormap(colors[:2 * periods + 1]),
                  shading="auto", rasterized=True)
    ax.contour(GX, GY, reg, levels=np.arange(0.5, 2 * periods + 0.5),
               colors="k", linewidths=0.7)        # switching-set spiral boundaries
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\dot\theta$")
    ax.set_xlim(-12, 12); ax.set_ylim(-8, 8); ax.set_aspect("equal")
    return fig, ax


def penalty_symbol(insertion: str) -> str:
    """Greek penalty symbol implied by the insertion rule.

    ``profile`` insertion uses the penalty written ``phi``; ``finite_step``
    (finite) insertion uses ``psi``. Returned as a mathtext token (``\\phi`` /
    ``\\psi``) for embedding inside a ``$...$`` label.
    """
    return r"\phi" if insertion == "profile" else r"\psi"


def frontier_penalty_label(activation_tex: str, *, insertion: str,
                           subscript: str, coeff: str = r"\alpha",
                           symbol: str | None = None) -> str:
    """Legend label ``$<activation> + <coeff> <sym>_<subscript>$`` with the penalty
    symbol (phi/psi) selected from ``insertion`` (see :func:`penalty_symbol`),
    unless ``symbol`` overrides it. The thesis convention keys the symbol to the
    penalty *family* (log -> ``\\phi``, fractional-exponent -> ``\\psi``), which
    for log-penalty series trained with finite-step insertion requires the
    override ``symbol=r"\\phi"``.

    Example: ``frontier_penalty_label(r"\\mathrm{ReLU}^k",
    insertion="finite_step", subscript="k")`` ->
    ``$\\mathrm{ReLU}^k + \\alpha\\,\\psi_{k}$``.
    """
    sym = symbol if symbol is not None else penalty_symbol(insertion)
    return rf"${activation_tex} + {coeff}\,{sym}_{{{subscript}}}$"


# ============================================================================ #
# PLOTTING FUNCTIONS
# ============================================================================ #
def plot_value_scatter3d(
    dataset: Any,
    *,
    ax: Optional[Any] = None,
    title: str = "Value samples V(x)",
    s: float = 8.0,
    alpha: float = 0.85,
    cmap: str = "viridis",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    elev: float = 15.0,
    azim: float = -60.0,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, Any]:
    """3D scatter of dataset samples ``(x[0], x[1], V)``, colored by value.

    Experiment-agnostic: works on any dataset exposing ``x`` (N, 2) and ``v`` (N,)
    fields. Plots the raw samples only — no surface, no interpolation — so it shows
    exactly the training data and its support. ``xlim``/``ylim`` set the state-plane
    extent (e.g. to match a companion phase-plane panel). Pass ``ax`` (a 3D axis)
    to compose into a multi-panel figure.
    """
    x = _get_field(dataset, "x")
    v = _get_field(dataset, "v")

    x = np.asarray(x)
    v = np.asarray(v).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected dataset['x'] shape (N, 2), got {x.shape}")
    if v.shape[0] != x.shape[0]:
        raise ValueError(f"Mismatched lengths: x has {x.shape[0]} rows, v has {v.shape[0]} entries")

    if ax is None:
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    sc = ax.scatter(x[:, 0], x[:, 1], v, c=v, cmap=cmap, s=s, alpha=alpha,
                    depthshade=True)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((0, 0, 0, 0))
        a._axinfo["grid"].update(color="0.85", linewidth=0.5)
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("V(x)")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_zlabel("V(x)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, tight=False, close=False, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_vdp_value_with_gradient_arrows2d(
    dataset: Any,
    *,
    title: str = "VDP dataset: V(x₀) with ∇V arrows",
    grid_size: int = 15,
    point_s: float = 20.0,
    point_alpha: float = 0.6,
    cmap: str = "viridis",
    arrow_color: str = "red",
    arrow_alpha: float = 0.7,
    arrow_scale: float = 0.15,
    head_width: float = 0.05,
    head_length: float = 0.08,
    normalize: bool = True,
    color_arrows_by_magnitude: bool = True,
    magnitude_cmap: str = "magma",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """2D scatter of x colored by V, with ∇V arrows sampled on a regular grid."""
    x = _get_field(dataset, "x")
    v = _get_field(dataset, "v")
    dv = _get_field(dataset, "dv")

    x = np.asarray(x)
    v = np.asarray(v).reshape(-1)
    dv = np.asarray(dv)
    grid_size = int(grid_size)

    x0_0 = x[:, 0]
    x0_1 = x[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(x0_0, x0_1, c=v, cmap=cmap, s=point_s, alpha=point_alpha)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Value function V")

    x_min, x_max = float(np.min(x0_0)), float(np.max(x0_0))
    y_min, y_max = float(np.min(x0_1)), float(np.max(x0_1))
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    # Collect arrows so we can optionally color by ||dv||
    X_list: list[float] = []
    Y_list: list[float] = []
    U_list: list[float] = []
    V_list: list[float] = []
    M_list: list[float] = []

    for i in range(grid_size):
        for j in range(grid_size):
            x_point = float(Xg[i, j])
            y_point = float(Yg[i, j])

            # nearest neighbor in dataset (squared Euclidean distance in x-space)
            d2 = (x0_0 - x_point) ** 2 + (x0_1 - y_point) ** 2
            idx = int(np.argmin(d2))

            dx = float(dv[idx, 0])
            dy = float(dv[idx, 1])
            mag = float(np.sqrt(dx * dx + dy * dy))
            if mag <= 0.0:
                continue

            if normalize:
                # Keep arrows legible; magnitude is shown by color if enabled.
                scale = arrow_scale * (1.0 / (1.0 + 0.5 * mag))
                dx_plot = dx * scale
                dy_plot = dy * scale
            else:
                # Arrow length corresponds to magnitude (may look cluttered).
                dx_plot = dx
                dy_plot = dy

            X_list.append(x_point)
            Y_list.append(y_point)
            U_list.append(dx_plot)
            V_list.append(dy_plot)
            M_list.append(mag)

    if color_arrows_by_magnitude and len(X_list) > 0:
        q = ax.quiver(
            np.array(X_list),
            np.array(Y_list),
            np.array(U_list),
            np.array(V_list),
            np.array(M_list),
            cmap=magnitude_cmap,
            alpha=arrow_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )
        cbar2 = fig.colorbar(q, ax=ax, pad=0.02, fraction=0.046)
        cbar2.set_label(r"$\|\nabla V(x)\|$")
    else:
        for x_point, y_point, dx_plot, dy_plot in zip(X_list, Y_list, U_list, V_list):
            ax.arrow(
                x_point,
                y_point,
                dx_plot,
                dy_plot,
                head_width=head_width,
                head_length=head_length,
                fc=arrow_color,
                ec=arrow_color,
                alpha=arrow_alpha,
                length_includes_head=True,
            )

    ax.set_xlabel("x₀[0]")
    ax.set_ylabel("x₀[1]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, tight=False, close=False, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_model_value_surface(
    history: "Any | str | os.PathLike[str]",
    *,
    activation: "str | Any" = "relu",
    power: float = 1.0,
    x_scale: "Optional[Sequence[float]]" = None,
    v_scale: float = 1.0,
    run_index: int = 0,
    grid_n: int = 120,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    dataset: Any = None,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    cmap: Any = None,
    vmax: Optional[float] = None,
    xticks: "Optional[Sequence[float]]" = None,
    yticks: "Optional[Sequence[float]]" = None,
    zticks: "Optional[Sequence[float]]" = None,
    elev: float = 15.0,
    azim: float = -105.0,
    colorbar: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, Any]:
    """3D surface of the learned V(x) — the fitted network evaluated on a 2D grid.

    Reproduces the left panel of Fig. 2 in Han & Yang (arXiv:2312.17467): the value
    function as a surface, here the model's *learned* approximation (not the data
    samples). Rebuilds a :class:`ShallowNetwork` from a fit result's best-iteration
    atoms — ``history`` may be a ``History`` object, a path to its pickle, or a list
    of runs (``run_index`` selects one). ``activation`` (a name resolved via
    ``src.config.activations.get_activation``, or a callable) and ``power`` must
    match the run's config, as they are not stored on the ``History``.

    If training used max-abs normalization, pass ``x_scale`` (per-dim) and
    ``v_scale`` so the grid is normalized before the forward pass and the output
    rescaled back to physical units — the surface is then drawn over the physical
    state plane. Pass ``ax`` (a 3D axis) to compose into a multi-panel figure.

    This is the single source of truth for learned-value surface plots: callers
    render **one surface per call** and arrange any row/grid in their markdown/LaTeX.
    The uniform style is **no axis names**, **sparse ticks** ({min, 0, max} on x/y;
    {0, mid, max} on z), and **no title** unless one is passed — the surface itself
    stays smooth and precise (fine ``grid_n`` evaluation, no mesh lines drawn).
    ``cmap`` defaults to the shared blue→yellow ``_SURFACE_CMAP``. Values below
    zero are clipped to the display floor. ``vmax`` clips the surface above and
    fixes the z-range (e.g. to the data's value range when the model extrapolates
    wildly off-support); ``None`` auto-scales to the nonnegative surface.
    ``xticks``/``yticks``/``zticks`` override the sparse defaults when a caller wants
    explicit tick positions; pass ``x_range``/``y_range`` to set the displayed extent
    (and re-evaluate the surface over it).

    """
    net = model_from_history(history, activation=activation, power=power,
                             run_index=run_index)
    d = int(net.hidden.weight.shape[1])
    if d != 2:
        raise ValueError(f"Expected 2D input, got d={d}. This function only supports 2D plots.")

    # Determine grid range (physical units).
    if x_range is None or y_range is None:
        if dataset is not None:
            x_data = np.asarray(_get_field(dataset, "x"))
            if x_range is None:
                x_range = (float(x_data[:, 0].min()), float(x_data[:, 0].max()))
            if y_range is None:
                y_range = (float(x_data[:, 1].min()), float(x_data[:, 1].max()))
        else:
            x_range = x_range or (-3.0, 3.0)
            y_range = y_range or (-3.0, 3.0)

    x0 = np.linspace(x_range[0], x_range[1], grid_n)
    x1 = np.linspace(y_range[0], y_range[1], grid_n)
    X0, X1 = np.meshgrid(x0, x1)
    grid_points = np.column_stack([X0.ravel(), X1.ravel()])  # (grid_n^2, 2), physical

    # The network was fit on normalized samples (x / x_scale, v / v_scale): feed
    # normalized grid points and rescale the prediction back to physical units.
    scale = np.ones(d) if x_scale is None else np.asarray(x_scale, dtype=np.float64).reshape(d)
    with torch.no_grad():
        V = net(torch.tensor(grid_points / scale, dtype=torch.float64)).numpy().reshape(grid_n, grid_n)
    V = V * float(v_scale)

    return plot_value_surface(
        X0, X1, V, ax=ax, title=title, cmap=cmap, clip=(0.0, vmax),
        xticks=xticks, yticks=yticks, zticks=zticks, elev=elev, azim=azim,
        colorbar=colorbar, save_path=save_path, show=show,
    )


def plot_value_surface(
    X0, X1, V, *, ax=None, title=None, cmap=None, clip=(0.0, None),
    xticks=None, yticks=None, zticks=None, elev=15.0, azim=-105.0,
    colorbar=False, save_path=None, show=False,
):
    """Draw a prepared physical-coordinate value grid; clipping is display-only.

    The supplied V array is never modified. Model loading, checkpoint selection
    and physical-coordinate evaluation belong to src.results.
    """
    X0, X1, V = np.asarray(X0), np.asarray(X1), np.asarray(V)
    if X0.shape != X1.shape or X0.shape != V.shape or V.ndim != 2:
        raise ValueError("surface coordinates and values must have matching 2D shapes")
    floor, vmax = clip
    if floor != 0.0:
        raise ValueError("value-surface display clipping uses a zero floor")
    x_range = (float(np.min(X0)), float(np.max(X0)))
    y_range = (float(np.min(X1)), float(np.max(X1)))
    if ax is None:
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Clip to the displayed nonnegative value range. The model can extrapolate
    # wildly off-support; the surface plot is a visual comparison, not a signed
    # residual diagnostic.
    if vmax is not None:
        V = np.clip(V, floor, vmax)
    else:
        V = np.maximum(V, floor)
    zmax = float(vmax) if vmax is not None else float(np.nanmax(V))
    cmap = cmap if cmap is not None else _SURFACE_CMAP
    surf = ax.plot_surface(X0, X1, V, cmap=cmap, vmin=0.0, vmax=zmax, alpha=0.95,
                           edgecolor="none", rcount=X0.shape[0], ccount=X0.shape[1])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((0, 0, 0, 0))
        a._axinfo["grid"].update(color="0.85", linewidth=0.5)
    if colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("V(x)")

    # Uniform style: no axis names, sparse ticks (overridable), title only if given.
    def _sparse(lo: float, hi: float) -> list[float]:
        mid = 0.0 if lo < 0.0 < hi else round((lo + hi) / 2.0, 2)
        return sorted({round(lo, 2), mid, round(hi, 2)})

    ax.set_xticks(list(xticks) if xticks is not None else _sparse(*x_range))
    ax.set_yticks(list(yticks) if yticks is not None else _sparse(*y_range))
    ax.set_zticks(list(zticks) if zticks is not None
                  else sorted({0.0, round(zmax / 2.0, 2), round(zmax, 2)}))
    ax.set_zlim(0.0, zmax)
    if title:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path, tight=False, close=False, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_neuron_h1_frontier(
    series: Sequence[dict[str, Any]],
    *,
    xlabel: str = "number of neurons",
    ylabel: str = r"best relative $H^1$ error",
    save_path: str | os.PathLike[str] | None = None,
    show_plot: bool = False,
):
    """Render the neuron / H1 lower-envelope frontier (log-y) in house style.

    Each entry of ``series`` is a dict with:
      ``ns``     -- 1-D array of neuron counts (x),
      ``h1``     -- 1-D array of relative H1 errors (y),
      ``label``  -- legend label (build penalty labels via
                    :func:`frontier_penalty_label` for the insertion-aware
                    phi/psi convention),
      ``color``  -- line/marker color,
      ``marker`` -- marker shape,
      ``ls``     -- (optional) line style, default ``"-."``.

    Empty series (``ns`` of length 0) are skipped. Returns ``(fig, ax)``.

    Markers are *subsampled*, not drawn at every point.  A frontier traced by
    single-atom insertion has one point per neuron, so marking every point
    would put well over a hundred glyphs on a curve: they overlap into a solid
    band and hide the line they are meant to label.  ``markevery`` is given as
    a fraction of the axes diagonal, which spaces the glyphs evenly in display
    space rather than in data space -- the frontiers are dense at small neuron
    counts, where even spacing in ``ns`` would still pile them up.  Each series
    gets a different phase so glyphs from different curves do not align into
    false columns.
    """
    _MARKER_STRIDE = 0.09          # ~11 markers per curve along its own length
    with plt.rc_context(_FRONTIER_RC):
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        drawn = 0
        n_series = max(len(series), 1)
        for index, s in enumerate(series):
            ns = np.asarray(s["ns"])
            h1 = np.asarray(s["h1"])
            if ns.size == 0:
                logger.warning("frontier: no points for %r — skipping", s.get("label"))
                continue
            phase = _MARKER_STRIDE * index / n_series
            ax.plot(ns, h1, ls=s.get("ls", "-."), lw=1.6, color=s["color"],
                    marker=s["marker"], ms=7.5, mfc=s["color"], mec="0.15",
                    mew=0.8, markevery=(phase, _MARKER_STRIDE),
                    label=s["label"], zorder=3)
            drawn += 1
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        from src.plotstyle import style_frontier_axes
        style_frontier_axes(ax)
        fig.tight_layout(pad=2.0)
        if save_path is not None:
            save_path = os.fspath(save_path)
            save_figure(fig, save_path, tight=False, close=False, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    return fig, ax
