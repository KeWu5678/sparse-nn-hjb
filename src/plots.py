#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized plotting helpers for value-function / experiment visualization.

All figure-producing helpers live here. Tabular summaries live in
``src/metric.py``; the shared result loader ``_load_results`` is imported from
there to avoid duplication.
"""

import logging
import os
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

from .metric import _load_results

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
def _repel_labels(ax, xs, ys, labels, *, fontsize=7, max_iter=300, pad=1.5,
                  max_radius=48.0, spring=0.08):
    """Annotate each (x, y) with its label, then iteratively push the text boxes
    apart in display space so they don't overlap. A thin leader line tethers each
    label back to its point. Each label is clamped within ``max_radius`` points of
    its anchor (so dense clusters can't push labels off to infinity) and pulled
    back toward a baseline offset by a weak ``spring``. Pure matplotlib."""
    base = (0.0, 9.0)
    anns = [
        ax.annotate(
            text, xy=(x, y), xytext=base, textcoords="offset points",
            fontsize=fontsize, ha="center", va="center", zorder=4,
            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.6", shrinkA=0, shrinkB=2),
        )
        for x, y, text in zip(xs, ys, labels)
    ]
    fig = ax.figure
    fig.canvas.draw()
    if len(anns) < 2:
        return anns
    pt_per_px = 72.0 / fig.dpi
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iter):
        bboxes = [a.get_window_extent(renderer) for a in anns]
        shifts = [[0.0, 0.0] for _ in anns]
        overlap = False
        for i in range(len(anns)):
            bi = bboxes[i]
            cix, ciy = (bi.x0 + bi.x1) / 2, (bi.y0 + bi.y1) / 2
            for j in range(i + 1, len(anns)):
                bj = bboxes[j]
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + 2 * pad
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + 2 * pad
                if ox > 0 and oy > 0:
                    overlap = True
                    cjx, cjy = (bj.x0 + bj.x1) / 2, (bj.y0 + bj.y1) / 2
                    ddx, ddy = cix - cjx, ciy - cjy
                    if ddx == 0 and ddy == 0:
                        ddy = 1.0
                    dist = (ddx ** 2 + ddy ** 2) ** 0.5 or 1.0
                    mag = min(ox, oy) / 2.0
                    ux, uy = ddx / dist, ddy / dist
                    shifts[i][0] += ux * mag; shifts[i][1] += uy * mag
                    shifts[j][0] -= ux * mag; shifts[j][1] -= uy * mag
        if not overlap:
            break
        for a, (sx, sy) in zip(anns, shifts):
            px, py = a.get_position()
            # repulsion (px -> pt) + weak spring back toward the baseline offset
            nx = px + sx * pt_per_px + spring * (base[0] - px)
            ny = py + sy * pt_per_px + spring * (base[1] - py)
            # hard clamp so a crowded cluster can never push a label to infinity
            r = (nx ** 2 + ny ** 2) ** 0.5
            if r > max_radius:
                nx, ny = nx * max_radius / r, ny * max_radius / r
            a.set_position((nx, ny))
    return anns


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


def _extract_active_weights(run: dict, u_thresh: float = 1e-4) -> dict:
    """Return active (a, b, u) at the best iteration of one run.

    Active means |u| > u_thresh.  Returns a dict with keys:
        'a'     : np.ndarray (n_active, d)
        'b'     : np.ndarray (n_active,)
        'u'     : np.ndarray (n_active,)
        'gamma' : float
    """
    it = run["best_iteration"]
    iw = run["inner_weights"][it]
    a = np.asarray(iw["weight"])               # (n, d)
    b = np.asarray(iw["bias"])                 # (n,)
    u = np.asarray(run["outer_weights"][it]).flatten()  # (n,)
    mask = np.abs(u) > u_thresh
    return {"a": a[mask], "b": b[mask], "u": u[mask], "gamma": run["gamma"]}


def _best_iteration_atoms(history: Any, run_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-iteration ``(a, b, u)`` from a fit result, robust to its container.

    Accepts a single ``History`` object (attribute access), a dict-like run, or a
    list/tuple of either (selecting ``run_index``). Returns inner weights ``a``
    (n, d), inner bias ``b`` (n,), and outer weights ``u`` (1, n).
    """
    if isinstance(history, (list, tuple)):
        history = history[run_index]

    def field(name: str) -> Any:
        if hasattr(history, name):
            return getattr(history, name)
        if hasattr(history, "__getitem__"):
            try:
                return history[name]
            except (KeyError, TypeError, IndexError):
                pass
        raise AttributeError(f"fit result exposes no '{name}' (got {type(history).__name__})")

    best_it = int(field("best_iteration"))
    iw = field("inner_weights")[best_it]
    a = np.asarray(iw["weight"])                       # (n, d)
    b = np.asarray(iw["bias"])                         # (n,)
    u = np.asarray(field("outer_weights")[best_it])    # (1, n)
    return a, b, u


# ---------------------------------------------------------------------------- #
# Neuron / H1 frontier
# ---------------------------------------------------------------------------- #
# Shared publication style for frontier plots. Applied via rc_context so the
# module leaves global rcParams untouched.
_FRONTIER_RC = {
    "font.family": ["serif"],
    "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
}


def penalty_symbol(insertion: str) -> str:
    """Greek penalty symbol implied by the insertion rule.

    ``profile`` insertion uses the penalty written ``phi``; ``finite_step``
    (finite) insertion uses ``psi``. Returned as a mathtext token (``\\phi`` /
    ``\\psi``) for embedding inside a ``$...$`` label.
    """
    return r"\phi" if insertion == "profile" else r"\psi"


def frontier_penalty_label(activation_tex: str, *, insertion: str,
                           subscript: str, coeff: str = r"\alpha") -> str:
    """Legend label ``$<activation> + <coeff> <sym>_<subscript>$`` with the penalty
    symbol (phi/psi) selected from ``insertion`` (see :func:`penalty_symbol`).

    Example: ``frontier_penalty_label(r"\\mathrm{ReLU}^k",
    insertion="finite_step", subscript="k")`` ->
    ``$\\mathrm{ReLU}^k + \\alpha\\,\\psi_{k}$``.
    """
    sym = penalty_symbol(insertion)
    return rf"${activation_tex} + {coeff}\,{sym}_{{{subscript}}}$"


# ============================================================================ #
# PLOTTING FUNCTIONS
# ============================================================================ #
def plot_score_tradeoff(
    rows: Sequence[dict[str, Any]],
    *,
    x: str,
    y: str,
    label: str,
    color: str | None = None,
    title: str = "Score tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | os.PathLike[str] | None = None,
    show_plot: bool = False,
):
    """Scatter of experiment summary rows (x vs y), labeled and optionally colored by a third column."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if color:
        color_values = [str(row.get(color, "")) for row in rows]
        categories = list(dict.fromkeys(color_values))
        cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
        palette = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}
        colors = [palette[value] for value in color_values]
    else:
        categories = []
        palette = {}
        colors = "#4c78a8"

    xs = [float(row[x]) for row in rows]
    ys = [float(row[y]) for row in rows]
    ax.scatter(xs, ys, s=52, c=colors, edgecolor="white", linewidth=0.7,
               alpha=0.9, zorder=3)
    _repel_labels(ax, xs, ys, [str(row.get(label, "")) for row in rows])
    if categories:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[cat],
                       markeredgecolor="white", markersize=7, label=f"{color}={cat}")
            for cat in categories
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    if not categories:
        # tight_layout can't account for a legend placed outside the axes;
        # bbox_inches="tight" at savefig handles that case instead.
        fig.tight_layout()
    if save_path is not None:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_nonsmooth_curve(
    curve: "str | os.PathLike[str] | dict[str, Any]",
    *,
    dataset: Any = None,
    distance: Optional[np.ndarray] = None,
    near_percentile: float = 10.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Switching set & smooth basin",
    curve_color: str = "k",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """2D switching-set arms + smooth basin over the state plane.

    Reproduces the middle panel of Fig. 2 in Han & Yang (arXiv:2312.17467) — the
    nonsmooth curves of the pendulum value function and the smooth region of
    attraction to the upright equilibrium. ``curve`` is a path to (or mapping of) a
    nonsmooth-curve ``npz`` exposing ``points`` (n, 2) spiral-arm samples, optional
    ``value_levels`` (n,), and ``basin`` (m, 2) boundary ring (see
    ``src.OpenLoop.pendulum.nonsmooth.NonsmoothCurve``).

    If ``dataset`` (and optionally per-sample ``distance`` to the switching set) is
    given, the value samples are overlaid: the lowest ``near_percentile``% by
    distance are highlighted as the near region, the rest drawn faintly. Pass
    ``ax`` to compose into a multi-panel figure.
    """
    if hasattr(curve, "keys"):
        points = np.asarray(curve["points"])
        value_levels = np.asarray(curve["value_levels"]) if "value_levels" in curve else None
        basin = np.asarray(curve["basin"]) if "basin" in curve else None
    else:
        with np.load(os.fspath(curve)) as data:
            points = np.asarray(data["points"])
            value_levels = np.asarray(data["value_levels"]) if "value_levels" in data.files else None  # noqa: F841
            basin = np.asarray(data["basin"]) if "basin" in data.files else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure

    # Optional sample overlay, split near/far by distance to the switching set.
    if dataset is not None:
        x = np.asarray(_get_field(dataset, "x"))
        if distance is not None:
            d = np.asarray(distance).reshape(-1)
            thresh = float(np.percentile(d, near_percentile))
            near = d <= thresh
            ax.scatter(x[~near, 0], x[~near, 1], s=4, c="0.8", label="far samples", zorder=1)
            ax.scatter(x[near, 0], x[near, 1], s=7, c="#d62728",
                       label=f"near (≤{near_percentile:g}%)", zorder=2)
        else:
            ax.scatter(x[:, 0], x[:, 1], s=4, c="0.8", label="samples", zorder=1)

    # Smooth basin boundary (closed ring).
    if basin is not None and basin.shape[0] >= 3:
        ring = np.vstack([basin, basin[:1]])
        ax.plot(ring[:, 0], ring[:, 1], "-", color="#1f77b4", lw=1.8,
                label="smooth basin", zorder=3)

    # Switching-set spiral arms drawn as connected curves (the nonsmooth curves the
    # value function is non-differentiable across). The arms are stored as 4 equal
    # blocks tracked across value levels (see NonsmoothCurve), so connect each block
    # in stored order; fall back to a single polyline if the layout differs.
    if points.size:
        n = points.shape[0]
        arm_len = n // 4
        arms = ([points[i * arm_len:(i + 1) * arm_len] for i in range(4)]
                if arm_len >= 2 and n == 4 * arm_len else [points])
        for k, arm in enumerate(arms):
            ax.plot(arm[:, 0], arm[:, 1], "-", color=curve_color, lw=2.2, zorder=4,
                    label="switching set" if k == 0 else None)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_inner_weight_3d_scatter(
    results: "Sequence[dict] | str | os.PathLike[str]",
    *,
    u_thresh: float = 1e-4,
    elev: float = 25.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, list]:
    """3D scatter of active inner weights (a₁, a₂, b) per gamma; size ∝ |u|, color = u."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    result_list = _load_results(results)
    n = len(result_list)
    fig = plt.figure(figsize=(4 * n, 4.5))
    axes: list = []

    for col, run in enumerate(result_list):
        rec = _extract_active_weights(run, u_thresh)
        a, b, u, gamma = rec["a"], rec["b"], rec["u"], rec["gamma"]

        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        axes.append(ax)

        if len(u) == 0:
            ax.set_title(f"gamma={gamma:g}\n(no active neurons)")
            continue

        sizes = (np.abs(u) / np.abs(u).max()) * 120 + 10
        sc = ax.scatter(
            a[:, 0], a[:, 1], b,
            s=sizes, c=u, cmap="RdBu_r",
            vmin=-np.abs(u).max(), vmax=np.abs(u).max(),
            alpha=0.85, edgecolors="k", linewidths=0.3,
        )
        fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.55, label="$u$")
        ax.set_xlabel("$a_1$", labelpad=1)
        ax.set_ylabel("$a_2$", labelpad=1)
        ax.set_zlabel("$b$",   labelpad=1)
        ax.set_title(f"$\\gamma={gamma:g}$\n({len(u)} active)", fontsize=9)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(
        "Inner weights $(a_1, a_2, b)$ — 3D scatter\n"
        "(size $\\propto |u|$, color $= u$)",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_inner_weight_pairwise_distance(
    results: "Sequence[dict] | str | os.PathLike[str]",
    *,
    u_thresh: float = 1e-4,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, list]:
    """Heatmap of pairwise ‖wᵢ - wⱼ‖ among active neurons (sorted by |u|), one subplot per gamma."""
    result_list = _load_results(results)
    n = len(result_list)
    fig, axes_list = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes_list = [axes_list]

    for ax, run in zip(axes_list, result_list):
        rec = _extract_active_weights(run, u_thresh)
        a, b, u, gamma = rec["a"], rec["b"], rec["u"], rec["gamma"]

        if len(u) == 0:
            ax.set_title(f"gamma={gamma:g}\n(no active neurons)")
            continue

        # sort by descending |u| so the most important neurons come first
        order = np.argsort(-np.abs(u))
        a_s, b_s, u_s = a[order], b[order], u[order]

        w = np.column_stack([a_s, b_s])                          # (n, 3)
        D = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=-1)  # (n, n)

        im = ax.imshow(D, cmap="viridis_r", aspect="auto",
                       vmin=0, vmax=D.max())
        fig.colorbar(im, ax=ax, shrink=0.85, label="$\\|w_i - w_j\\|_2$")

        n_act = len(u)
        ax.set_xticks(range(n_act))
        ax.set_yticks(range(n_act))
        if n_act <= 20:
            ax.set_xticklabels([f"{v:.2f}" for v in u_s], rotation=90, fontsize=6)
            ax.set_yticklabels([f"{v:.2f}" for v in u_s], fontsize=6)
        else:
            ax.tick_params(labelsize=6)
        ax.set_xlabel("neuron index (sorted by $|u|$ desc)")
        ax.set_ylabel("neuron index")
        ax.set_title(f"$\\gamma={gamma:g}$  ({n_act} active)", fontsize=9)

    fig.suptitle(
        "Pairwise Euclidean distance $\\|w_i - w_j\\|_2$  in $(a_1, a_2, b)$ space\n"
        "(neurons sorted by $|u|$ descending — dark = close)",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes_list


def plot_regions_of_attraction(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    curve_arms: "Optional[Sequence[np.ndarray]]" = None,
    region_labels: "Optional[Sequence[str]]" = None,
    curve_label: str = "switching set",
    ax: Optional[plt.Axes] = None,
    domain: Tuple[float, float, float, float] = (-10.0, 10.0, -8.0, 8.0),
    grid_n: int = 400,
    region_colors: Sequence[str] = ("#c9b3de", "#f3b0a0", "#a9c8e8", "#f3e0a0", "#a9dca0", "#f0c0e0"),
    curve_color: str = "k",
    curve_lw: float = 0.9,
    title: str = "Regions of attraction & switching set",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Filled regions of attraction (nearest-point fill) + switching curves.

    Reproduces the middle panel of Fig. 2 in Han & Yang (arXiv:2312.17467): the
    colored regions of attraction to the (periodic) equilibria, separated by the
    nonsmooth switching curves. ``points`` (N, 2) are scattered samples — e.g. PMP
    trajectory states tiled across periods — and ``labels`` (N,) their integer
    region id; each grid cell is colored by its nearest point's label, a Voronoi
    fill that turns the characteristics into solid basins. ``curve_arms`` is an
    optional iterable of (m, 2) polylines (the switching-set arms, tiled) drawn on
    top. ``region_labels[i]`` names region id ``i`` in a legend (with the switching
    set, ``curve_label``); omit for no legend. Pass ``ax`` to compose into a
    multi-panel figure.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    x0, x1, y0, y1 = domain

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    nx = int(grid_n)
    ny = max(2, int(round(grid_n * (y1 - y0) / (x1 - x0))))
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    _, idx = cKDTree(points).query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    region = labels[idx].reshape(GX.shape)

    n_regions = int(labels.max()) + 1 if labels.size else 1
    colors = [region_colors[i % len(region_colors)] for i in range(n_regions)]
    ax.pcolormesh(GX, GY, region, cmap=ListedColormap(colors), shading="auto",
                  vmin=-0.5, vmax=n_regions - 0.5)

    drew_curve = False
    for arm in curve_arms or ():
        arm = np.asarray(arm)
        if arm.shape[0] >= 2:
            ax.plot(arm[:, 0], arm[:, 1], "-", color=curve_color, lw=curve_lw, zorder=4)
            drew_curve = True

    if region_labels is not None:
        handles = [Patch(facecolor=colors[i], edgecolor="none", label=region_labels[i])
                   for i in range(min(n_regions, len(region_labels)))]
        if drew_curve:
            handles.append(Line2D([0], [0], color=curve_color, lw=curve_lw, label=curve_label))
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9,
                  title="region of attraction →")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

    Note: ``semiconcave`` models do not round-trip through ``History`` faithfully
    (the lossy atom record drops structure — issue #19); use a ``signed`` run here.
    """
    import pickle

    from src.models.net import ShallowNetwork

    if isinstance(history, (str, os.PathLike)):
        with open(os.fspath(history), "rb") as f:
            history = pickle.load(f)

    a, b, u = _best_iteration_atoms(history, run_index)
    n_neurons, d = a.shape
    if d != 2:
        raise ValueError(f"Expected 2D input, got d={d}. This function only supports 2D plots.")

    if isinstance(activation, str):
        from src.config.activations import get_activation
        activation = get_activation(activation)

    net = ShallowNetwork(
        layer_sizes=[d, n_neurons, 1],
        activation=activation,
        p=power,
        inner_weights=a,
        inner_bias=b,
        outer_weights=u,
    )
    net.eval()

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

    if ax is None:
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Clip to the displayed nonnegative value range. The model can extrapolate
    # wildly off-support; the surface plot is a visual comparison, not a signed
    # residual diagnostic.
    if vmax is not None:
        V = np.clip(V, 0.0, vmax)
    else:
        V = np.maximum(V, 0.0)
    zmax = float(vmax) if vmax is not None else float(np.nanmax(V))
    cmap = cmap if cmap is not None else _SURFACE_CMAP
    surf = ax.plot_surface(X0, X1, V, cmap=cmap, vmin=0.0, vmax=zmax, alpha=0.95,
                           edgecolor="none", rcount=grid_n, ccount=grid_n)
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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
    """
    with plt.rc_context(_FRONTIER_RC):
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        drawn = 0
        for s in series:
            ns = np.asarray(s["ns"])
            h1 = np.asarray(s["h1"])
            if ns.size == 0:
                logger.warning("frontier: no points for %r — skipping", s.get("label"))
                continue
            ax.plot(ns, h1, ls=s.get("ls", "-"), lw=2.2, color=s["color"],
                    marker=s["marker"], ms=4.8, mfc=s["color"], mec=s["color"],
                    mew=0.8, label=s["label"], clip_on=False, zorder=3)
            drawn += 1
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.set_xlim(left=0)
        ax.legend(loc="upper right", fontsize=9.5, ncol=2, handletextpad=0.45,
                  columnspacing=0.9)
        fig.tight_layout(pad=2.0)
        if save_path is not None:
            save_path = os.fspath(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    return fig, ax
