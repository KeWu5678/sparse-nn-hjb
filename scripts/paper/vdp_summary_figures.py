#!/usr/bin/env python3
"""Cross-algorithm rendering helpers for the current-paper VDP evidence."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "01_vdp" / "paper_log_penalty"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.models.net import ShallowNetwork
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem
from src.plots import _best_iteration_atoms, frontier_penalty_label

FIG = OUTPUT_DIR / "figures"
PROBLEM = VdpOptimalControlProblem()

from src.plotstyle import PALETTE, style_frontier_axes
from src.plotstyle import apply_publication_style as _apply_publication_style


def _finalize(fig, stem: str, *, tight: bool = True, **kw) -> str:
    if tight:
        fig.tight_layout(pad=2.0)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, **kw)
    plt.close(fig)
    return f"figures/{stem}.png"


# --------------------------------------------------------------------------- #
# Champion selection — one run per method at the fixed operating point.
# --------------------------------------------------------------------------- #
# key -> (label, color, linestyle, sweep, insertion, activation, power, gamma)
ALGO1 = {
    "tanh":     (r"$\tanh$",              PALETTE["teal"],       "--", "tanh"),
    "softplus": (r"$\mathrm{softplus}$",  PALETTE["blue_main"],  "-",  "softplus"),
    "gaussian": (r"$e^{-x^2}$",           PALETTE["red_strong"], ":",  "gaussian"),
}
# Fifth series color, local to this summary (the shared PALETTE has five
# entries and four are already taken by the other curves).
_GOLD = "#C77F0A"
ALGO2 = {
    "relu2": (r"$\mathrm{ReLU}^2$", PALETTE["violet"], "-.", 2.0),
    "relu5": (r"$\mathrm{ReLU}^5$", _GOLD,             (0, (3, 1, 1, 1)), 5.0),
}


def _atoms(result_path: str):
    with open(result_path, "rb") as f:
        history = pickle.load(f)
    return _best_iteration_atoms(history)   # a (n, d), b (n,), u (1, n)


def _build_net(result_path: str, activation: str, power: float) -> ShallowNetwork:
    a, b, u = _atoms(result_path)
    net = ShallowNetwork(layer_sizes=[a.shape[1], a.shape[0], 1],
                         activation=get_activation(activation), p=power,
                         inner_weights=a, inner_bias=b, outer_weights=u)
    net.eval()
    return net


# --------------------------------------------------------------------------- #
# Figure 1 — neuron / rel-H1 frontier (growth trajectory per champion).
# --------------------------------------------------------------------------- #
def _trajectory(result_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(neurons, cumulative-min rel-H1-val) over the run's insertion iterations."""
    with open(result_path, "rb") as f:
        history = pickle.load(f)
    pts = []
    for iw, h1 in zip(history.inner_weights, history.err_h1_val):
        n = int(np.asarray(iw["weight"]).shape[0])
        if n >= 1 and np.isfinite(float(h1)):
            pts.append((n, float(h1)))
    pts.sort(key=lambda nh: nh[0])
    ns = np.array([n for n, _ in pts])
    h1 = np.minimum.accumulate(np.array([h for _, h in pts]))
    return ns, h1


# Frontier legend tags each curve with its penalty symbol (phi for the log penalty
# / profile insertion, psi for the power penalty / finite_step) — the convention of
# the introduction frontier (fig:neuron_h1_frontier).
_FRONTIER_LABEL = {
    "tanh":     frontier_penalty_label(r"\tanh", insertion="profile", subscript="1"),
    "softplus": frontier_penalty_label(r"\mathrm{softplus}", insertion="profile", subscript="1"),
    "gaussian": frontier_penalty_label(r"e^{-x^2}", insertion="profile", subscript="1"),
    "relu2":    frontier_penalty_label(r"\mathrm{ReLU}^2", insertion="finite_step", subscript="2"),
    "relu5":    frontier_penalty_label(r"\mathrm{ReLU}^5", insertion="finite_step", subscript="5"),
}


def plot_frontier(ch) -> str:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    order = [("tanh", ALGO1), ("softplus", ALGO1), ("gaussian", ALGO1),
             ("relu2", ALGO2), ("relu5", ALGO2)]
    for key, spec in order:
        label, color, ls = _FRONTIER_LABEL[key], spec[key][1], spec[key][2]
        ns, h1 = _trajectory(ch[key]["result_path"])
        ax.plot(ns, h1, color=color, ls=ls, lw=1.6, label=label,
                marker="o", ms=6.0, mec="0.15", mew=0.8,
                markevery=max(1, len(ns) // 12))
    ax.set_xlabel("number of neurons")
    ax.set_ylabel(r"best relative $H^1$ error")
    ax.set_yscale("log")
    style_frontier_axes(ax, legend_ncol=3)
    return _finalize(fig, "frontier", bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Figure 2 — closed-loop feedback (softplus, gaussian, ReLU^5, true).
# --------------------------------------------------------------------------- #
_X0 = (2.0, 1.0)
_ROLL_T, _ROLL_DT, _U_CLIP, _PLOT_T = 12.0, 0.01, 50.0, 3.0
_FEEDBACK_KEYS = ["softplus", "gaussian", "relu5"]


def _model_feedback(ch, key, norm):
    spec = ALGO1[key] if key in ALGO1 else ALGO2[key]
    power = 1.0 if key in ALGO1 else spec[3]
    act = spec[3] if key in ALGO1 else "relu"
    net = _build_net(ch[key]["result_path"], act, power)
    dtype = net.hidden.weight.dtype

    def u(x):
        xn = torch.tensor(np.asarray(x).reshape(1, 2) / norm.x_scale, dtype=dtype,
                          requires_grad=True)
        val = net(xn)
        (g,) = torch.autograd.grad(val.sum(), xn)
        g_phys = g.detach().numpy()[0] * (norm.v_scale / norm.x_scale)
        return float(PROBLEM.feedback_from_gradient(g_phys))
    return u


def plot_feedback(ch, samples, norm) -> tuple[str, str, list[dict]]:
    _apply_publication_style()
    runs = {"true": (r"true", PALETTE["neutral"], "-", PROBLEM.true_feedback(samples))}
    for key in _FEEDBACK_KEYS:
        spec = ALGO1[key] if key in ALGO1 else ALGO2[key]
        runs[key] = (_FRONTIER_LABEL[key], spec[1], spec[2], _model_feedback(ch, key, norm))
    rolled = {k: PROBLEM.rk4_rollout(uf, _X0, T=_ROLL_T, dt=_ROLL_DT, u_clip=_U_CLIP)
              for k, (_, _, _, uf) in runs.items()}

    def _panel(stem, ycol, ylabel):
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, (label, color, ls, _) in runs.items():
            t, xs, us, _ = rolled[k]
            y = np.linalg.norm(xs, axis=1) if ycol == "y" else np.abs(us)
            lw = 3.0 if k == "true" else 2.2
            ax.plot(t, y, color=color, ls=ls, lw=lw, label=label)
        ax.set_xlabel(r"time $t$"); ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, _PLOT_T); ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper right")
        return _finalize(fig, stem)

    fig_state = _panel("feedback_state", "y", r"$\|y(t)\|$")
    fig_ctrl = _panel("feedback_control", "u", r"$|u(t)|$")

    cost_rows = []
    for k in ["true"] + _FEEDBACK_KEYS:
        _, xs, _, cost = rolled[k]
        n = "—" if k == "true" else _atoms(ch[k]["result_path"])[0].shape[0]
        cost_rows.append({"controller": k, "neurons": n,
                          "stabilizes": "yes" if np.linalg.norm(xs[-1]) < 0.2 else "no",
                          "cost": cost})
    return fig_state, fig_ctrl, cost_rows


# --------------------------------------------------------------------------- #
# Figure 3 — weight portraits. Two variants (keep one). Choices: gaussian, softplus, ReLU^5.
# --------------------------------------------------------------------------- #
_WEIGHT_KEYS = ["gaussian", "softplus", "relu5"]
# Bright blue / yellow (positive / negative outer weight), sampled from the reference figure.
_SIGN_POS, _SIGN_NEG = "#001BF8", "#FFFD3A"


def _atom_arrays(ch, key):
    a, b, u = _atoms(ch[key]["result_path"])
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    return a, b, u


def _sizes(u):
    m = np.abs(u).max() or 1.0
    return (np.abs(u) / m) * 130 + 12


def _sign_colors(u):
    return np.where(u >= 0, _SIGN_POS, _SIGN_NEG)


def plot_weights_stereographic(ch) -> dict[str, str]:
    """Variant A — radially project each atom's (a, b) onto S^2, then stereographic-project
    from the south pole (0,0,-1). Green circle = image of the equator."""
    _apply_publication_style()

    def _project(key):
        a, b, u = _atom_arrays(ch, key)
        P = np.column_stack([a, b.reshape(-1, 1)])
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        denom = np.where(np.abs(1.0 + z) < 1e-9, 1e-9, 1.0 + z)
        return x / denom, y / denom, u

    proj = {key: _project(key) for key in _WEIGHT_KEYS}
    # Shared limit across the three panels so the spread contrast reads honestly.
    allv = np.concatenate([np.abs(np.concatenate([X, Y])) for X, Y, _ in proj.values()])
    lim = float(max(np.nanpercentile(allv, 97) * 1.15, 1.3))

    figs = {}
    for key in _WEIGHT_KEYS:
        X, Y, u = proj[key]
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="#2E8B57", lw=1.4, zorder=1)
        ax.scatter(X, Y, s=_sizes(u), c=_sign_colors(u), alpha=0.85,
                   edgecolors="k", linewidths=0.3, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$")
        figs[key] = _finalize(fig, f"weights_stereo_{key}")
    return figs


def plot_weights_raw3d(ch) -> dict[str, str]:
    """Variant B — raw (a1, a2, b) 3D scatter with the unit equator circle (light green,
    at b=0) for reference. ReLU atoms sit on the unit sphere; Algo-1 atoms scatter off it
    (a few clipped by the fixed range)."""
    _apply_publication_style()
    figs = {}
    th = np.linspace(0, 2 * np.pi, 200)
    cx, cy, cz = np.cos(th), np.sin(th), np.zeros_like(th)   # unit equator at b = 0
    lim = 4.0
    for key in _WEIGHT_KEYS:
        a, b, u = _atom_arrays(ch, key)
        fig = plt.figure(figsize=(4.6, 4.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(cx, cy, cz, color="#4BFE52", lw=1.0)
        ax.scatter(a[:, 0], a[:, 1], b, s=_sizes(u), c=_sign_colors(u),
                   alpha=0.85, edgecolors="k", linewidths=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xticks([-lim, 0, lim]); ax.set_yticks([-lim, 0, lim]); ax.set_zticks([-lim, 0, lim])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        # House 3D style: vertical axis on the left, no grey pane walls, faint grid.
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor((0, 0, 0, 0))
            axis._axinfo["grid"].update(color="0.85", linewidth=0.5)
        ax.view_init(elev=15, azim=-105)
        figs[key] = _finalize(fig, f"weights_raw3d_{key}", tight=False,
                              bbox_inches="tight")
    return figs
