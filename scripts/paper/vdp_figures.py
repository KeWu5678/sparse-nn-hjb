#!/usr/bin/env python3
"""Rendering helpers for the current-paper Van der Pol evidence."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "01_vdp" / "paper_log_penalty"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.data import ValueSampleNormalizer
from src.models.net import ShallowNetwork
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem
from src.plots import _best_iteration_atoms, plot_model_value_surface

EXPERIMENT = "paper_log_penalty"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "paper_log_penalty" / "sequential"
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}

# Uniform operating point for the value-fit comparison (Key finding), matching the
# thesis tables — surfaces/metrics/control are all read off this single (alpha, gamma).
_FIXED_ALPHA, _FIXED_GAMMA = 1e-4, 10.0

from src.plotstyle import PALETTE
from src.plotstyle import apply_publication_style as _apply_publication_style


def _create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


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


# Three representatives spanning the smooth-fitting spectrum (all signed):
# broad monotone ridge (sparse) / saturating S (weak) / localized RBF (accurate).
REPS = ["softplus", "tanh", "gaussian"]
_REP_STYLE = {
    "softplus": (r"$\mathrm{softplus}$", PALETTE["blue_main"], "-"),
    "tanh": (r"$\tanh$", PALETTE["teal"], "--"),
    "gaussian": (r"$e^{-x^2}$", PALETTE["red_strong"], ":"),
}

# The VDP OCP the dataset was generated from (Azmi-Kalise-Kunisch): control-affine
# with g = [0, 1]^T, cost beta*u^2, so the value induces u(x) = -d_y2 V / (2 beta).
PROBLEM = VdpOptimalControlProblem()


# ---------------------------------------------------------------------------- #
# Records (global metrics; VDP is smooth, no region split)
# ---------------------------------------------------------------------------- #
def load_rows() -> list[dict[str, Any]]:
    records = sorted(MULTIRUN_DIR.glob("**/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR}"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        if "pendulum" in cfg["data"]["path"].lower():     # this doc is VDP only
            continue
        model = cfg["model"]
        v = record["metrics"][0]["values"]
        neurons = int(v["best_neurons"])
        rel_h1 = float(v["rel_h1_val"])
        rows.append({
            "kind": model["kind"],
            "insertion": model["insertion"],
            "activation": model["activation"],
            "loss": _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"])),
            "gamma": float(model["gamma"]),
            "alpha": float(model["alpha"]),
            "seed": int(cfg["env"]["seed"]),
            "neurons": neurons,
            "rel_h1": rel_h1,
            "rel_l2": float(v["rel_l2_val"]),
            "score": rel_h1 * max(neurons, 1),            # sparsity-aware score
            "data_file": cfg["data"]["path"],
            "result_path": str(_result_pkl(path)),
        })
    return rows


def _result_pkl(json_path: Path) -> Path:
    return json_path.parent / f"result_{json_path.stem}.pkl"


def signed_h1_fixed(rows: list[dict[str, Any]], activation: str) -> dict[str, Any]:
    """Return one signed H1 run at the configured shared operating point."""
    cand = [r for r in rows if r["kind"] == "signed" and r["insertion"] == "profile"
            and r["loss"] == "h1" and r["activation"] == activation
            and abs(r["alpha"] - _FIXED_ALPHA) < 1e-12
            and abs(r["gamma"] - _FIXED_GAMMA) < 1e-12]
    if not cand:
        raise ValueError(
            f"no signed h1 profile run for {activation!r} at "
            f"alpha={_FIXED_ALPHA}, gamma={_FIXED_GAMMA}")
    return min(cand, key=lambda r: r["rel_h1"])


def _build_net(result_path: str, activation: str, power: float = 1.0) -> ShallowNetwork:
    with open(result_path, "rb") as f:
        history = pickle.load(f)
    a, b, u = _best_iteration_atoms(history)
    net = ShallowNetwork(
        layer_sizes=[a.shape[1], a.shape[0], 1],
        activation=get_activation(activation), p=power,
        inner_weights=a, inner_bias=b, outer_weights=u,
    )
    net.eval()
    return net


def _value_grad_phys(net: ShallowNetwork, x_phys: np.ndarray,
                     norm: ValueSampleNormalizer) -> tuple[np.ndarray, np.ndarray]:
    """Physical V̂ and grad V̂ at physical states x_phys (N, 2) via autograd."""
    dtype = net.hidden.weight.dtype
    xn = torch.tensor(np.asarray(x_phys) / norm.x_scale, dtype=dtype, requires_grad=True)
    val = net(xn)
    (grad,) = torch.autograd.grad(val.sum(), xn)
    v_phys = val.detach().numpy().reshape(-1) * norm.v_scale
    g_phys = grad.detach().numpy() * (norm.v_scale / norm.x_scale)
    return v_phys, g_phys


# ---------------------------------------------------------------------------- #
# Key finding — figures
# ---------------------------------------------------------------------------- #
_SHAPE_PANELS = [
    ("value", r"$\sigma(x)$"),
    ("deriv", r"$\sigma'(x)$"),
    ("curv", r"$\sigma''(x)$"),
]
_SHAPE_FIG = Path("figures") / "shape_softplus_tanh_gaussian.png"
_CTRL_FIG = Path("figures") / "control_synthesis.png"
_TRADEOFF_FIG = Path("figures") / "alpha_gamma_tradeoff.png"

# Panel order for the gradient-kernel diagnostic: most saturated → least.
_DERIV_ORDER = ["tanh", "softplus", "gaussian"]
_DERIV_NEAR_ZERO = 0.05          # |σ'| below this counts as a "dead" gradient column


def _value_deriv_curv(fn, x: torch.Tensor):
    xr = x.clone().detach().requires_grad_(True)
    y = fn(xr)
    (dy,) = torch.autograd.grad(y.sum(), xr, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), xr, create_graph=True)
    return y.detach(), dy.detach(), d2y.detach()


def plot_activation_shapes() -> str:
    """Value / first derivative / curvature of the three representatives."""
    _apply_publication_style()
    x = torch.linspace(-4.0, 4.0, 800)
    curves = {name: _value_deriv_curv(get_activation(name), x) for name in REPS}
    xn = x.numpy()

    fig, axes = _create_subplots(1, 3, figsize=(15, 4.6))
    for col, (key, ylabel) in enumerate(_SHAPE_PANELS):
        ax = axes[col]
        if key in ("deriv", "curv"):
            ax.axhline(0.0, color=PALETTE["neutral"], lw=1.4, zorder=1)
        for name in REPS:
            label, color, ls = _REP_STYLE[name]
            y = curves[name][col].numpy()
            ax.plot(xn, y, color=color, ls=ls, lw=2.6, label=label, zorder=3, clip_on=False)
        ax.set_xlabel(r"$x$  (pre-activation)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-4, 4)
        if col == 0:
            ax.legend(loc="upper left")
    _finalize_figure(fig, OUTPUT_DIR / _SHAPE_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _SHAPE_FIG.as_posix()


def _sigma_prime(name: str, z: np.ndarray) -> np.ndarray:
    """σ'(z) of the activation, evaluated elementwise via autograd."""
    zt = torch.tensor(np.asarray(z), dtype=torch.double, requires_grad=True)
    y = get_activation(name)(zt)
    (d,) = torch.autograd.grad(y.sum(), zt)
    return d.detach().numpy()


def plot_derivative_distribution(rows, samples, norm) -> dict[str, Any]:
    """One single-panel σ'(z) histogram PER representative, each to its own file (like
    ``plot_value_surfaces``) so the markdown/LaTeX tiles them and gives each its own
    *subcaption*. The gradient-kernel column for neuron n at x_m is σ'(a_n·x_m + b_n)·a_n;
    the fraction of near-zero σ' measures how many columns are 'dead'. No in-figure title
    or annotation — the subcaption is the information source (house style). Axis names
    are kept. Returns {figures:{act: path}, near_zero:{act: pct}, neurons:{act: N}}."""
    _apply_publication_style()
    x_norm = np.asarray(samples["x"]) / norm.x_scale        # (K, 2) normalized states
    figs: dict[str, str] = {}
    near_zero: dict[str, float] = {}
    neurons: dict[str, int] = {}
    for name in _DERIV_ORDER:
        run = signed_h1_fixed(rows, name)
        net = _build_net(run["result_path"], name)
        a = net.hidden.weight.detach().numpy()              # (N, 2) inner weights
        b = net.hidden.bias.detach().numpy()                # (N,)   inner bias
        z = (x_norm @ a.T) + b[None, :]                     # (K, N) pre-activations
        dprime = _sigma_prime(name, z.ravel())
        near_zero[name] = float((np.abs(dprime) < _DERIV_NEAR_ZERO).mean())
        neurons[name] = run["neurons"]

        fig, axes = _create_subplots(1, 1, figsize=(5.0, 4.0))
        ax = axes[0]
        ax.hist(dprime, bins=60, density=True, color=_REP_STYLE[name][1], alpha=0.85)
        ax.set_xlabel(r"$\sigma'(z)$")
        ax.set_ylabel("Density")
        rel = Path("figures") / f"derivative_distribution_{name}.png"
        _finalize_figure(fig, OUTPUT_DIR / rel.with_suffix(""), formats=["png"], dpi=300)
        figs[name] = rel.as_posix()
    return {"figures": figs, "near_zero": near_zero, "neurons": neurons}


def plot_value_surfaces(rows, samples, norm) -> dict[str, str]:
    """One single V̂(x) surface plot PER representative, each saved to its own file via
    the shared ``plot_model_value_surface`` API (no axis names, sparse ticks, no title —
    the markdown/LaTeX arranges them in a row and titles them). Returns
    {activation: figure-path-relative-to-OUTPUT_DIR}."""
    import matplotlib.pyplot as plt

    _apply_publication_style()
    paths: dict[str, str] = {}
    for name in REPS:
        run = signed_h1_fixed(rows, name)
        fig = plt.figure(figsize=(4.2, 4.0), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_model_value_surface(
            run["result_path"], activation=name, power=1.0,
            x_scale=norm.x_scale, v_scale=norm.v_scale, dataset=samples,
            ax=ax, show=False, vmax=20.0, zticks=[0, 10, 20],
        )
        rel = Path("figures") / f"value_surface_{name}.png"
        _finalize_figure(fig, OUTPUT_DIR / rel.with_suffix(""),
                         formats=["png"], dpi=300, tight=False,
                         bbox_inches="tight", pad_inches=0.05)
        paths[name] = rel.as_posix()
    return paths


# Closed-loop control: synthesize u(x) = -d_y2 V̂ / (2β) and roll it out in the true
# VDP dynamics — the stabilization-to-origin counterpart of the pendulum swing-up.
_U_CLIP = 50.0
_X0 = (2.0, 1.0)          # initial state, matching the paper
_ROLL_T, _ROLL_DT = 12.0, 0.01
_PLOT_T = 3.0             # zoom the time axis to the transient (paper horizon)


def _model_feedback(rows, name, norm):
    net = _build_net(signed_h1_fixed(rows, name)["result_path"], name)

    def u(x):
        _, g = _value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
        return float(PROBLEM.feedback_from_gradient(g[0]))
    return u


def plot_control_synthesis(rows, samples, norm):
    """‖y(t)‖ and u(t) of the closed loop under each synthesized feedback, beside the
    true control, from a common initial state — the stabilization analogue of Han &
    Yang Fig. 3. Returns (figure path, closed-loop cost table rows)."""
    _apply_publication_style()
    runs = {"true": (r"true", PALETTE["neutral"], "-", PROBLEM.true_feedback(samples))}
    for name in REPS:
        label, color, ls = _REP_STYLE[name]
        runs[name] = (label, color, ls, _model_feedback(rows, name, norm))
    rolled = {k: PROBLEM.rk4_rollout(uf, _X0, T=_ROLL_T, dt=_ROLL_DT, u_clip=_U_CLIP)
              for k, (_, _, _, uf) in runs.items()}

    fig, axes = _create_subplots(1, 2, figsize=(14, 5))
    for k, (label, color, ls, _) in runs.items():
        t, xs, us, _ = rolled[k]
        lw = 3.2 if k == "true" else 2.4
        axes[0].plot(t, np.linalg.norm(xs, axis=1), color=color, ls=ls, lw=lw, label=label)
        axes[1].plot(t, np.abs(us), color=color, ls=ls, lw=lw, label=label)
    axes[0].set_xlabel(r"time $t$"); axes[0].set_ylabel(r"$\|y(t)\|$")
    axes[0].set_xlim(0.0, _PLOT_T); axes[0].set_ylim(bottom=0.0); axes[0].legend(loc="upper right")
    axes[1].set_xlabel(r"time $t$"); axes[1].set_ylabel(r"$|u(t)|$")
    axes[1].set_xlim(0.0, _PLOT_T); axes[1].set_ylim(bottom=0.0)
    _finalize_figure(fig, OUTPUT_DIR / _CTRL_FIG.with_suffix(""),
                     formats=["png"], dpi=300)

    cost_rows = []
    for k in ["true"] + REPS:
        _, xs, _, cost = rolled[k]
        cost_rows.append({
            "controller": "true" if k == "true" else k,
            "neurons": "—" if k == "true" else signed_h1_fixed(rows, k)["neurons"],
            "stabilizes": "yes" if np.linalg.norm(xs[-1]) < 0.2 else "no",
            "cost": cost,
        })
    return _CTRL_FIG.as_posix(), cost_rows


# ---------------------------------------------------------------------------- #
# Parameter discussion (alpha, gamma) — tables + tradeoff scatter
# ---------------------------------------------------------------------------- #
