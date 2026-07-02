#!/usr/bin/env python3
"""activationsearch — pendulum / discontinuous-gradient case.

Writes ``results.md`` as the central, Key-Finding-first report for the pendulum
(switching-set) sweep. Method / sweep axes / activation list live in
``README.md``; this script produces the findings:

  Key finding   — activation shape (leaky_relu/softplus/gaussian) → fitted value
                  surfaces (signed) → best metrics → synthesized control vs the
                  true PMP feedback law.
  Parameter discussion — impact of the nonconvex-penalty parameters alpha, gamma.
  Full result   — the detailed region-split absolute-L1 tables.

    python pendulum/analysis.py        # run from experiments/02_pendulum/log_penalty
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT_DIR = Path(__file__).resolve().parent          # experiments/02_pendulum/log_penalty
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.data import ValueSampleNormalizer, load_value_samples
from src.metric import format_table
from src.models.net import ShallowNetwork
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.plots import _best_iteration_atoms, plot_model_value_surface

EXPERIMENT = "activationsearch"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}

from src.plotstyle import PALETTE
from src.plotstyle import apply_publication_style as _apply_publication_style


def _create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def _finalize_figure(fig, out_path, formats=None, dpi: int = 300, close: bool = True,
                     pad: float = 2.0, tight: bool = True, **kwargs) -> list[Path]:
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "png"]
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


# Three representatives spanning the sigma'-regularity ladder (all signed):
# leaky kink (seats the jump) / smooth ridge (smears) / localized RBF (tiles).
# relu itself is excluded — it is the L1 convex baseline (see experiments/02_pendulum/baseline),
# not an H1 candidate; leaky_relu is the relu-like kink that survives gradient training.
REPS = ["leaky_relu", "softplus", "gaussian"]
_REP_STYLE = {                       # (display label, colour, linestyle)
    "leaky_relu": (r"$\mathrm{leaky\,ReLU}$", PALETTE["blue_main"], "-"),
    "softplus": (r"$\mathrm{softplus}$", PALETTE["teal"], "--"),
    "gaussian": (r"$e^{-x^2}$", PALETTE["red_strong"], ":"),
}

# The pendulum OCP the dataset was generated from (Han & Yang). The defaults match
# src/OpenLoop/pendulum: m=l=1, b=0.1, g=9.8, Q=(1,1), R=1, no control saturation.
PROBLEM = PendulumSwingUpProblem()


# ---------------------------------------------------------------------------- #
# Records (region-split absolute L1 + relative H1)
# ---------------------------------------------------------------------------- #
def load_rows() -> list[dict[str, Any]]:
    records = sorted(MULTIRUN_DIR.glob("**/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make activationsearch DATA=pendulum`"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        if "pendulum" not in cfg["data"]["path"].lower():
            continue
        model = cfg["model"]
        v = record["metrics"][0]["values"]
        neurons = int(v["best_neurons"])
        if neurons == 0:                       # degenerate non-fit (predicts ~0)
            continue
        far_lv, near_lv = float(v["far_l1_value"]), float(v["near_l1_value"])
        far_lg, near_lg = float(v["far_l1_grad"]), float(v["near_l1_grad"])
        rows.append({
            "kind": model["kind"],
            "activation": model["activation"],
            "loss": _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"])),
            "gamma": float(model["gamma"]),
            "alpha": float(model["alpha"]),
            "seed": int(cfg["env"]["seed"]),
            "neurons": neurons,
            "rel_h1": float(v["rel_h1_val"]),
            "far_lv": far_lv, "near_lv": near_lv,
            "nf_v": near_lv / far_lv if far_lv else float("nan"),
            "far_lg": far_lg,
            "nf_g": near_lg / far_lg if far_lg else float("nan"),
            "data_file": cfg["data"]["path"],
            "result_path": str(_result_pkl(path)),
        })
    return rows


def _result_pkl(json_path: Path) -> Path:
    return json_path.parent / f"result_{json_path.stem}.pkl"


def best_signed_h1(rows: list[dict[str, Any]], activation: str) -> dict[str, Any]:
    """The most accurate signed H1 run for an activation (lowest relative H1)."""
    cand = [r for r in rows if r["kind"] == "signed"
            and r["loss"] == "h1" and r["activation"] == activation]
    if not cand:
        raise ValueError(f"no signed h1 run for {activation!r}")
    return min(cand, key=lambda r: r["rel_h1"])


# ---------------------------------------------------------------------------- #
# Model reconstruction + physical value/gradient (signed runs only; semiconcave
# does not round-trip through the lossy History — issue #19, src/plots.py)
# ---------------------------------------------------------------------------- #
def _normalizer(data_file: str) -> ValueSampleNormalizer:
    return ValueSampleNormalizer.fit(load_value_samples(data_file))


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
    """Physical V_hat and grad V_hat at physical states x_phys (N, 2) via autograd.

    The net is trained on normalized samples, so evaluate at x/x_scale and undo the
    scaling: V_phys = v_scale * V_norm, grad_phys = grad_norm * (v_scale / x_scale)
    (same chain rule as ValueSampleNormalizer.denormalize_prediction).
    """
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
_SHAPE_FIG = Path("figures") / "shape_leakyrelu_softplus_gaussian.png"
_CTRL_FIG = Path("figures") / "control_synthesis.png"
_TRADEOFF_FIG = Path("figures") / "alpha_gamma_tradeoff.png"


def _sigma(name: str):
    return get_activation(name)


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
    curves = {name: _value_deriv_curv(_sigma(name), x) for name in REPS}
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
                     formats=["png", "pdf"], dpi=300)
    return _SHAPE_FIG.as_posix()


# Per-surface z-axis max (the value scale each fitted V̂ is shown on) and the common
# state-plane extent; the API ticks x/y at {-10, 0, 10} and z at {0, mid, max}.
_SURFACE_XYRANGE = (-10.0, 10.0)
_SURFACE_VMAX = {"leaky_relu": 200.0, "softplus": 200.0, "gaussian": 1000.0}


def plot_value_surfaces(rows: list[dict[str, Any]], samples, norm) -> dict[str, str]:
    """One single V̂(x) surface plot PER representative, each saved to its own file via
    the shared ``plot_model_value_surface`` API (no axis names, sparse ticks, no title —
    the markdown/LaTeX arranges them in a row and titles them). Returns
    {activation: figure-path-relative-to-OUTPUT_DIR}."""
    import matplotlib.pyplot as plt

    _apply_publication_style()
    paths: dict[str, str] = {}
    for name in REPS:
        run = best_signed_h1(rows, name)
        fig = plt.figure(figsize=(4.6, 4.2), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_model_value_surface(
            run["result_path"], activation=name, power=1.0,
            x_scale=norm.x_scale, v_scale=norm.v_scale, dataset=samples,
            ax=ax, show=False,
            x_range=_SURFACE_XYRANGE, y_range=_SURFACE_XYRANGE, vmax=_SURFACE_VMAX[name],
        )
        rel = Path("figures") / f"value_surface_{name}.png"
        _finalize_figure(fig, OUTPUT_DIR / rel.with_suffix(""),
                         formats=["png", "pdf"], dpi=300, tight=False,
                         bbox_inches="tight", pad_inches=0.05)
        paths[name] = rel.as_posix()
    return paths


# Closed-loop control: synthesize u(x) = feedback_from_gradient(∇V̂) and roll it out in
# the true dynamics (after Han & Yang Fig. 3 — the induced controller over time).
_U_CLIP = 30.0            # numerical guard; the dataset is unsaturated (no u_max)
_ROLL_T, _ROLL_DT = 10.0, 0.01


def _deepest_supported_start(samples) -> np.ndarray:
    """The supported state with the largest cost-to-go (argmax V) — the hardest start the
    basin data actually covers. The basin (faithful restriction, issue #18) has NO data at
    the hanging switching set θ=π, so a swing-up-from-hanging is off-data; we instead start
    from the deepest *supported* state and drive to the nearest upright copy."""
    return samples["x"][int(np.argmax(samples["v"].ravel()))].copy()


def _model_feedback(rows, name, norm):
    net = _build_net(best_signed_h1(rows, name)["result_path"], name)

    def u(x):
        _, g = _value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
        return float(PROBLEM.feedback_from_gradient(g)[0])
    return u


def _stabilizes(xf) -> bool:
    return abs((xf[0] + np.pi) % (2 * np.pi) - np.pi) < 0.4 and abs(xf[1]) < 0.4


def plot_control_synthesis(rows, samples, norm):
    """Left: angle relative to upright θ(t)−2kπ. Right: the synthesized feedback law u(t).
    Each fitted V̂ induces u(x) = feedback_from_gradient(∇V̂(x)); we roll it out from the deepest
    supported state (highest cost-to-go in the basin) and drive to the nearest upright copy,
    beside the true PMP feedback (Han & Yang Fig. 3 single-IC comparison). Returns
    (figure path, cost/stabilization rows, start state, goal angle)."""
    _apply_publication_style()
    x0 = _deepest_supported_start(samples)
    goal = 2.0 * np.pi * round(x0[0] / (2.0 * np.pi))     # nearest upright copy 2kπ
    runs = {"true": (r"true PMP", PALETTE["neutral"], "-", PROBLEM.true_feedback(samples))}
    for name in REPS:
        label, color, ls = _REP_STYLE[name]
        runs[name] = (label, color, ls, _model_feedback(rows, name, norm))
    rolled = {k: PROBLEM.rk4_rollout(uf, x0, T=_ROLL_T, dt=_ROLL_DT, u_clip=_U_CLIP)
              for k, (_, _, _, uf) in runs.items()}

    fig, axes = _create_subplots(1, 2, figsize=(14, 5))
    lo, hi = 0.0, 0.0
    for k, (label, color, ls, _) in runs.items():
        t, xs, us, _ = rolled[k]
        blown = np.flatnonzero(np.abs(xs[:, 1]) > 10.0)   # truncate display once diverged
        end = (blown[0] + 1) if blown.size else len(t)
        th = xs[:end, 0] - goal                           # angle relative to the upright copy
        lo, hi = min(lo, float(th.min())), max(hi, float(th.max()))
        lw = 3.0 if k == "true" else 2.4
        axes[0].plot(t[:end], th, color=color, ls=ls, lw=lw, label=label)
        axes[1].plot(t[:end], us[:end], color=color, ls=ls, lw=lw, label=label)
    axes[0].axhline(0.0, color="0.6", lw=1.0, ls="--", zorder=0)
    axes[0].text(_ROLL_T * 0.985, 0.18, "upright", fontsize=10, color="0.4", ha="right")
    axes[0].scatter([0.0], [x0[0] - goal], s=45, color="0.4", zorder=5)
    axes[0].text(0.12, x0[0] - goal, "start", fontsize=10, color="0.4", va="center")
    axes[0].set_xlabel(r"time $t$")
    axes[0].set_ylabel(r"angle from upright $\;\theta(t)-2k\pi$")
    axes[0].set_xlim(0, _ROLL_T)
    axes[0].set_ylim(max(-8.5, lo - 0.6), min(8.5, hi + 0.6))
    axes[0].legend(loc="best")
    axes[1].axhline(0.0, color="0.85", lw=0.8, zorder=0)
    axes[1].set_xlabel(r"time $t$"); axes[1].set_ylabel(r"feedback control $u(t)$")
    axes[1].set_xlim(0, _ROLL_T); axes[1].set_ylim(-_U_CLIP - 2, _U_CLIP + 2)
    axes[1].legend(loc="upper right")
    _finalize_figure(fig, OUTPUT_DIR / _CTRL_FIG.with_suffix(""),
                     formats=["png", "pdf"], dpi=300)

    cost_rows = []
    for k in ["true"] + REPS:
        _, xs, _, cost = rolled[k]
        cost_rows.append({
            "controller": "true PMP" if k == "true" else k,
            "neurons": "—" if k == "true" else best_signed_h1(rows, k)["neurons"],
            "reaches_up": "yes" if _stabilizes(xs[-1]) else "no",
            "cost": cost,
        })
    return _CTRL_FIG.as_posix(), cost_rows, x0, goal


# ---------------------------------------------------------------------------- #
# Parameter discussion (alpha, gamma) — tables + tradeoff scatter
# ---------------------------------------------------------------------------- #
def _gamma_table(rows: list[dict[str, Any]]) -> str:
    """For each signed rep: best (lowest rel_h1) per gamma, over alpha."""
    out = []
    for name in REPS:
        cand = [r for r in rows if r["kind"] == "signed" and r["loss"] == "h1"
                and r["activation"] == name]
        for g in sorted({r["gamma"] for r in cand}):
            sub = [r for r in cand if r["gamma"] == g]
            b = min(sub, key=lambda r: r["rel_h1"])
            out.append({"activation": name, "gamma": g, "alpha": b["alpha"],
                        "neurons": b["neurons"], "rel_h1": b["rel_h1"]})
    return format_table(
        out, ["activation", "gamma", "alpha", "neurons", "rel_h1"],
        headers={"rel_h1": "rel H1"},
        formats={"gamma": "{:g}", "alpha": "{:g}", "rel_h1": "{:.3f}"},
        title="Effect of gamma (best alpha per gamma), signed H1",
    )


def _alpha_table(rows: list[dict[str, Any]]) -> str:
    """For each signed rep: best (lowest rel_h1) per alpha, over gamma."""
    out = []
    for name in REPS:
        cand = [r for r in rows if r["kind"] == "signed" and r["loss"] == "h1"
                and r["activation"] == name]
        for a in sorted({r["alpha"] for r in cand}):
            sub = [r for r in cand if r["alpha"] == a]
            b = min(sub, key=lambda r: r["rel_h1"])
            out.append({"activation": name, "alpha": a, "gamma": b["gamma"],
                        "neurons": b["neurons"], "rel_h1": b["rel_h1"]})
    return format_table(
        out, ["activation", "alpha", "gamma", "neurons", "rel_h1"],
        headers={"rel_h1": "rel H1"},
        formats={"alpha": "{:g}", "gamma": "{:g}", "rel_h1": "{:.3f}"},
        title="Effect of alpha (best gamma per alpha), signed H1",
    )


def plot_alpha_gamma_tradeoff(rows: list[dict[str, Any]]) -> str:
    """Neurons vs relative H1 for the signed reps, marker = activation, colour = gamma."""

    _apply_publication_style()
    gammas = sorted({r["gamma"] for r in rows
                     if r["kind"] == "signed" and r["loss"] == "h1"})
    gcolor = {g: c for g, c in zip(
        gammas, [PALETTE["blue_main"], PALETTE["teal"], PALETTE["red_strong"], PALETTE["violet"]])}
    markers = {"leaky_relu": "o", "softplus": "s", "gaussian": "^"}

    fig, axes = _create_subplots(1, 1, figsize=(8.5, 6))
    ax = axes[0]
    for name in REPS:
        for g in gammas:
            pts = [r for r in rows if r["kind"] == "signed" and r["loss"] == "h1"
                   and r["activation"] == name and r["gamma"] == g]
            if not pts:
                continue
            ax.scatter([p["neurons"] for p in pts], [p["rel_h1"] for p in pts],
                       marker=markers[name], color=gcolor[g], s=70,
                       edgecolor="white", linewidth=0.6, zorder=3)
    # Two legends: activation (marker) and gamma (colour).
    from matplotlib.lines import Line2D
    act_handles = [Line2D([], [], marker=markers[n], color="0.3", ls="",
                          markersize=9, label=_REP_STYLE[n][0]) for n in REPS]
    g_handles = [Line2D([], [], marker="o", color=gcolor[g], ls="", markersize=9,
                        label=fr"$\gamma={g:g}$") for g in gammas]
    leg1 = ax.legend(handles=act_handles, title="activation", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=g_handles, title="penalty $\\gamma$", loc="lower left")
    ax.set_xlabel("Neurons"); ax.set_ylabel(r"Relative $H^1$")
    _finalize_figure(fig, OUTPUT_DIR / _TRADEOFF_FIG.with_suffix(""),
                     formats=["png", "pdf"], dpi=300)
    return _TRADEOFF_FIG.as_posix()


# ---------------------------------------------------------------------------- #
# Full result — region-split absolute L1 tables (both kinds, all activations)
# ---------------------------------------------------------------------------- #
def best_per_cell(rows: list[dict[str, Any]], loss: str) -> list[dict[str, Any]]:
    sel = [r for r in rows if r["loss"] == loss]

    def cell(r: dict[str, Any]) -> tuple:
        return (r["kind"], r["activation"])

    best = []
    for _, group in itertools.groupby(sorted(sel, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["far_lv"]))
    best.sort(key=lambda r: r["far_lv"])
    return best


def _fit_table(rows: list[dict[str, Any]], loss: str, title: str) -> str:
    return format_table(
        best_per_cell(rows, loss),
        ["kind", "activation", "gamma", "alpha", "neurons",
         "far_lv", "near_lv", "nf_v", "far_lg", "nf_g"],
        headers={"far_lv": "far Lv", "near_lv": "near Lv", "nf_v": "near/far V",
                 "far_lg": "far Lg", "nf_g": "near/far G"},
        formats={"gamma": "{:g}", "alpha": "{:g}", "far_lv": "{:.3f}",
                 "near_lv": "{:.3f}", "nf_v": "{:.2f}", "far_lg": "{:.3f}",
                 "nf_g": "{:.2f}"},
        title=title,
    )


# ---------------------------------------------------------------------------- #
# Markdown assembly
# ---------------------------------------------------------------------------- #
def _metrics_table(rows: list[dict[str, Any]]) -> str:
    out = [dict(activation=n, **{k: best_signed_h1(rows, n)[k]
                                 for k in ("neurons", "rel_h1", "far_lv", "far_lg")})
           for n in REPS]
    return format_table(
        out, ["activation", "neurons", "rel_h1", "far_lv", "far_lg"],
        headers={"rel_h1": "rel H1", "far_lv": "far Lv", "far_lg": "far Lg"},
        formats={"rel_h1": "{:.3f}", "far_lv": "{:.3f}", "far_lg": "{:.3f}"},
        title="Best signed H1 fit per representative",
    )


def _cost_table(cost_rows, x0) -> str:
    return format_table(
        cost_rows, ["controller", "neurons", "reaches_up", "cost"],
        headers={"reaches_up": "reaches upright?", "cost": "closed-loop cost"},
        formats={"cost": "{:.1f}"},
        title=f"Stabilize to upright from deepest supported start x0=[{x0[0]:.2f}, {x0[1]:.2f}], T={_ROLL_T:g}",
    )


def results_markdown(rows: list[dict[str, Any]]) -> str:
    samples = load_value_samples(rows[0]["data_file"])
    norm = ValueSampleNormalizer.fit(samples)

    shape_fig = plot_activation_shapes()
    surf_figs = plot_value_surfaces(rows, samples, norm)
    ctrl_fig, cost_rows, x0, goal = plot_control_synthesis(rows, samples, norm)
    tradeoff_fig = plot_alpha_gamma_tradeoff(rows)

    def _surf_row() -> str:
        head, sep, imgs = [], [], []
        for name in REPS:
            run = best_signed_h1(rows, name)
            head.append(f"{_REP_STYLE[name][0]} · {run['neurons']} neurons · rel $H^1$={run['rel_h1']:.2f}")
            sep.append("---")
            imgs.append(f"![{name}]({surf_figs[name]})")
        return ("| " + " | ".join(head) + " |\n"
                + "| " + " | ".join(sep) + " |\n"
                + "| " + " | ".join(imgs) + " |")

    return f"""# activationsearch — pendulum (discontinuous gradient)

Which activations fit a value function whose gradient **jumps across a switching
set** (pendulum swing-up). Method, sweep axes, and the activation list are in
`README.md`; this file reports the findings. Three representatives span the
derivative-regularity ladder — `leaky_relu` (kink), `softplus` (smooth ridge),
`gaussian` (localized RBF) — all under the `signed` model (semiconcave models do
not round-trip through the saved result, issue #19; plain `relu` is the convex L1
baseline, not an H1 candidate — see `../../baseline`).

## Key finding

Under the gradient-augmented (H1) loss each neuron contributes σ to V and
**w·σ′ to ∇V**, so the activation *derivative* σ′ is the basis that reconstructs
the gradient field. On the **faithful** basin data (issue #18: wider basin, V≲57,
θ̇±7.7), the result **reverses** the old narrow-basin (cap-35) finding: the sampled
region is dominated by the **smooth interior** of the upright basin — the switching
set is only its sparse boundary — so the activation that tiles a smooth field most
economically wins. That is the **localized RBF (`gaussian`)**, best on every error
band *and* in closed loop; the kink (`leaky_relu`) is a competent, uniquely
sparse-robust runner-up, and the smooth ridge (`softplus`) is worst.

### Activation shape

![activation shape]({shape_fig})

`leaky_relu′` is a step (slope `a`→`1` across its hyperplane — it can seat a finite
gradient jump, and unlike plain `relu` has no dead side); `softplus′` is a smooth
sigmoid (it can only smear a jump over width ≈ 1/β); the Gaussian derivative is a
sign-changing bump (localized, non-monotone). The kink can *localize* a switching-set
discontinuity, but most of the basin is smooth — where the localized bump tiles best.

### Fitted value surfaces

The learned value function V̂(x) of each representative, plotted as a surface over
the state plane (after Han & Yang Fig. 2 left). `gaussian` reproduces the value bowl
most faithfully; `leaky_relu` is competent; `softplus` collapses.

{_surf_row()}

### Best metrics

{_metrics_table(rows)}

These are the runs shown in the surface and control panels (the lowest-rel-H1
signed fit per activation). `rel H1` is the relative H1 error on the validation
split; `far Lv`/`far Lg` are the absolute far-field value/gradient L1 (the robust
metric the rest of the doc uses). On the faithful basin data the two metrics now
**agree**:

- **`gaussian` is best on every column** — lowest rel H1 (0.049), lowest far value
  L1 (0.064) *and* lowest far gradient L1 (0.040). On the old cap-35 basin rel H1
  *inverted* (gaussian posted a low rel H1 but a large absolute error); here the
  V→0 confound does not flip the ranking, and the robust L1 confirms it.
- **`leaky_relu` is the competent runner-up** (rel H1 0.211, far Lv 0.095) — it
  seats the boundary kink but pays on the gradient band (far Lg 0.246, ~6× gaussian).
- **`softplus` is worst on all three** (far Lv 0.521): its smooth, single-signed σ′
  cannot localize structure, so its surface collapses (see the panel).

### Synthesized control vs true feedback

The pendulum is control-affine with cost `r·u²`, so a value function induces the
feedback `u(x) = −(1/(2r·ml²)) ∂_θ̇ V(x)` (`PendulumSwingUpProblem.feedback_from_gradient`).
We synthesize û from each fitted V̂ and **roll it out in the true dynamics** from the
deepest *supported* state (the basin has no data at hanging θ=π, so a swing-up from
hanging is off-data; we start from the highest cost-to-go sample and drive to the
nearest upright copy), beside the true PMP feedback (nearest-neighbour interpolated
from the dataset's costate samples) — the closed-loop test of Han & Yang Fig. 3:
does the induced controller reach upright, and at what cost.

![control synthesis]({ctrl_fig})

{_cost_table(cost_rows, x0)}

Left is the angle from upright θ(t)−2kπ; right is the feedback law u(t); cost is the
accumulated running cost. From a supported start the closed loop **agrees with the
accuracy ranking**:

- **`gaussian` ≈ true PMP** — it reaches upright at cost 57.5, essentially matching
  the true feedback (57.6) and the best learned controller. Its low-error, smooth
  surface induces a benign global field.
- **`leaky_relu` reaches upright** at a slightly higher cost (59.4) — competent, as
  its piecewise-linear extrapolation stays bounded.
- **`softplus` fails** (cost 494) — its collapsed value surface gives a feedback that
  never reaches upright.

So on the faithful basin data the localized RBF wins *both* on-data accuracy and
closed-loop control; the kink is a safe runner-up; the smooth ridge is unusable.
This **reverses** the old cap-35 conclusion (where the kink was the only stabilizing
controller and gaussian diverged) — with the corrected wider basin, the closed loop
no longer punishes the smooth surface, because the data now covers the region the
controlled trajectory actually traverses.

## Parameter discussion (α, γ)

The nonconvex penalty `α·Σ φ(|c|)` has two knobs: **α** scales the penalty (the
sparsity lever) and **γ** controls the log-term nonconvexity (γ=0 turns it off;
larger γ prunes redundant clustered atoms). The tables take the three signed reps.

{_alpha_table(rows)}

**α is the dominant sparsity lever**: raising it from 1e-5 to 1e-2 cuts the neuron
count sharply (leaky_relu 148→30, gaussian 126→32). What that pruning costs depends on
the activation, and the dependence is itself a finding:

- `gaussian` is **most accurate but pruning-sensitive** — rel H1 0.049 at α=1e-5
  (126 neurons) degrades to 0.470 by α=1e-2 (32 neurons). Its low error is bought by
  *tiling* the smooth field with many localized bumps; remove them and it cannot fit.
- `leaky_relu` **stays competent when sparse** — rel H1 holds ≈ 0.21–0.34 all the way
  down to 30 neurons. A kink atom carries irreducible structure, so few are needed —
  this sparse-robustness is its remaining edge over the RBF.
- `softplus` never fits (rel H1 ≈ 0.63–0.80 at every α).

{_gamma_table(rows)}

**γ only refines**: the best γ improves `leaky_relu` modestly (0.211 at γ=10 vs 0.257
at γ=0) and `gaussian` slightly (0.049 at γ=1 vs 0.070 at γ=0); it does not change the
ranking.

![alpha/gamma tradeoff]({tradeoff_fig})

The scatter places every signed-H1 run on the neurons-vs-accuracy plane (marker =
activation, colour = γ). `gaussian` reaches the lowest error but only at high neuron
count, rising steeply as it is pruned; `leaky_relu` is a low, flat band — competent
across the whole sparsity range; `softplus` sits high throughout. The penalty moves a
model *along* its frontier, but where that frontier sits — and whether accuracy
survives sparsity — is set by the activation.

## Full result

Region-split **mean per-sample L1**, normalized by the global mean ‖true‖ — robust
to the V→0 interior. `far` = smooth region, `near` = 10% closest to the switching
set, `near/far` = how much worse the fit is at the switching set. Best (α, γ) per
(model, activation) by far value-L1; ranked best-first; both model kinds, all seven
activations.

### H1 (gradient-augmented) loss

{_fit_table(rows, 'h1', 'Pendulum H1 fit — best far value-L1 per model/activation')}

### L2 (value-only) loss

{_fit_table(rows, 'l2', 'Pendulum L2 fit — best far value-L1 per model/activation')}
"""


def main() -> int:
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.md"
    out.write_text(results_markdown(rows), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
