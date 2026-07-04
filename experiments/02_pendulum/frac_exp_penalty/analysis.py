#!/usr/bin/env python3
"""penaltypowers — pendulum / switching-set case.

Key-Finding-first report for the ReLU-power / fractional-penalty sweep on the
pendulum swing-up value function, the counterpart of
``../../01_vdp/frac_exp_penalty`` (the smooth case). Method and sweep axes live
in ``README.md``; this file reports the findings.

The swept axis is the **power** ``p`` of the atom ``σ(z)^p``, which sets the nonconvex
coefficient penalty exponent ``q = 2/(p+1)``. Representatives are ReLU at powers
{2, 3, 5}. On this switching-set target the power knob **reverses** vs smooth VDP:
power 2 (the ReLU² atom) is best, and higher power degrades the fit.

    make penaltypowers DATA=pendulum   # sweep + this analysis
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

OUTPUT_DIR = Path(__file__).resolve().parent          # experiments/02_pendulum/frac_exp_penalty
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.data import ValueSampleNormalizer, load_value_samples
from src.metric import format_table
from src.models.net import ShallowNetwork
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.plots import _best_iteration_atoms, plot_model_value_surface

EXPERIMENT = "penaltypowers"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "frac_exp_penalty"
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}

from src.plotstyle import PALETTE
from src.plotstyle import apply_publication_style as _apply_publication_style

POWERS = [2.0, 3.0, 5.0]
ACT = "relu"
_POWER_STYLE = {
    2.0: (r"$p=2$", PALETTE["blue_main"], "-"),
    3.0: (r"$p=3$", PALETTE["teal"], "--"),
    5.0: (r"$p=5$", PALETTE["red_strong"], ":"),
}
PROBLEM = PendulumSwingUpProblem()


def _q(p: float) -> float:
    return 2.0 / (p + 1.0)


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


# ---------------------------------------------------------------------------- #
# Records (region-split absolute L1 metrics)
# ---------------------------------------------------------------------------- #
def load_rows() -> list[dict[str, Any]]:
    records = sorted(MULTIRUN_DIR.glob("**/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make sweep EXPERIMENT=pendulum/frac_exp_penalty`"
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
        if neurons == 0:
            continue
        far_lv, near_lv = float(v["far_l1_value"]), float(v["near_l1_value"])
        far_lg, near_lg = float(v["far_l1_grad"]), float(v["near_l1_grad"])
        rows.append({
            "kind": model["kind"],
            "activation": model["activation"],
            "power": float(model["power"]),
            "loss": _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"])),
            "gamma": float(model["gamma"]),
            "alpha": float(model["alpha"]),
            "seed": int(cfg["env"]["seed"]),
            "neurons": neurons,
            "rel_h1": float(v["rel_h1_val"]),
            "far_lv": far_lv, "near_lv": near_lv,
            "nf_v": near_lv / far_lv if far_lv else float("nan"),
            "far_lg": far_lg, "near_lg": near_lg,
            "nf_g": near_lg / far_lg if far_lg else float("nan"),
            "data_file": cfg["data"]["path"],
            "result_path": str(_result_pkl(path)),
        })
    return rows


def _result_pkl(json_path: Path) -> Path:
    return json_path.parent / f"result_{json_path.stem}.pkl"


def best_relu_at_power(rows, power: float) -> dict[str, Any]:
    """Best ReLU H1 run at a given power (lowest far value-L1 over the α sweep)."""
    cand = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
            and r["loss"] == "h1" and r["power"] == power]
    if not cand:
        raise ValueError(f"no signed relu h1 run at power {power}")
    return min(cand, key=lambda r: r["far_lv"])


# ---------------------------------------------------------------------------- #
# Model reconstruction
# ---------------------------------------------------------------------------- #
def _build_net(result_path: str, activation: str, power: float) -> ShallowNetwork:
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


def _value_grad_phys(net, x_phys, norm):
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
_SHAPE_FIG = Path("figures") / "penalty_shape.png"
_CTRL_FIG = Path("figures") / "control_synthesis.png"
_TRADEOFF_FIG = Path("figures") / "power_alpha_tradeoff.png"


def plot_penalty_shape() -> str:
    """Left: ReLU^p atom σ(z)=ReLU(z)^p. Right: penalty φ(c)=|c|^q, q=2/(p+1)."""
    _apply_publication_style()
    z = np.linspace(-2.0, 2.0, 400)
    c = np.linspace(0.0, 2.0, 400)
    fig, axes = _create_subplots(1, 2, figsize=(11, 4.4))
    for p in POWERS:
        label, color, ls = _POWER_STYLE[p]
        axes[0].plot(z, np.maximum(z, 0.0) ** p, color=color, ls=ls, lw=2.6, label=label)
        axes[1].plot(c, c ** _q(p), color=color, ls=ls, lw=2.6, label=fr"$q={_q(p):.2f}$")
    axes[0].set_xlabel(r"$z$  (pre-activation)")
    axes[0].set_ylabel(r"$\sigma(z)=\mathrm{ReLU}(z)^p$")
    axes[0].set_xlim(-2, 2); axes[0].legend(loc="upper left")
    axes[1].set_xlabel(r"$|c|$  (coefficient)")
    axes[1].set_ylabel(r"penalty $\varphi(c)=|c|^{\,q},\; q=2/(p{+}1)$")
    axes[1].set_xlim(0, 2); axes[1].legend(loc="lower right")
    _finalize_figure(fig, OUTPUT_DIR / _SHAPE_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _SHAPE_FIG.as_posix()


# Per-power z-axis max (the value scale each fitted ReLU^p V̂ is shown on, rising with the
# off-basin extrapolation) and the common state-plane extent; the API ticks x/y at
# {-10, 0, 10} and z at {0, mid, max}.
_SURFACE_XYRANGE = (-10.0, 10.0)
_SURFACE_VMAX = {2.0: 400.0, 3.0: 500.0, 5.0: 40000.0}


def plot_value_surfaces(rows, samples, norm) -> dict[float, str]:
    """One ReLU^p value surface per power, each its own title-less file, via the shared
    ``plot_model_value_surface`` API (no axis names, sparse ticks, no title; per-power z
    range so the off-basin extrapolation is shown rather than flattened)."""
    import matplotlib.pyplot as plt

    _apply_publication_style()
    paths = {}
    for p in POWERS:
        run = best_relu_at_power(rows, p)
        fig = plt.figure(figsize=(4.6, 4.2), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_model_value_surface(
            run["result_path"], activation=ACT, power=p,
            x_scale=norm.x_scale, v_scale=norm.v_scale, dataset=samples,
            ax=ax, show=False,
            x_range=_SURFACE_XYRANGE, y_range=_SURFACE_XYRANGE, vmax=_SURFACE_VMAX[p],
        )
        rel = Path("figures") / f"value_surface_p{int(p)}.png"
        _finalize_figure(fig, OUTPUT_DIR / rel.with_suffix(""),
                         formats=["png"], dpi=300, tight=False,
                         bbox_inches="tight", pad_inches=0.05)
        paths[p] = rel.as_posix()
    return paths


# Closed-loop control: synthesize the feedback law u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x) from each
# fitted V̂ and roll a single stabilization out, beside the true PMP feedback (Han & Yang
# Fig. 3, single-IC comparison). NOTE on the start: the PMP dataset is the *upright smooth
# basin* (faithful basin restriction, issue #18) — it covers θ̇∈[−7.7,7.7] and V≲57 around
# each upright, with NO data at hanging-down θ=π. So a swing-up-from-hanging is unsupported; we start
# from the deepest *supported* state (highest cost-to-go in the basin) and drive to upright.
_U_CLIP = 30.0
_ROLL_T, _ROLL_DT = 8.0, 0.01


def _deepest_supported_start(samples) -> np.ndarray:
    """The supported state with the largest cost-to-go (argmax V) — the hardest start the
    basin data actually covers. Principled (worst-case in-data), independent of outcome."""
    return samples["x"][int(np.argmax(samples["v"].ravel()))].copy()


def _model_feedback(rows, power, norm):
    """The feedback law induced by the fitted ReLU^p value V̂: u(x) = feedback_from_gradient(∇V̂(x))."""
    net = _build_net(best_relu_at_power(rows, power)["result_path"], ACT, power)

    def u(x):
        _, g = _value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
        return float(PROBLEM.feedback_from_gradient(g)[0])
    return u


def _stabilizes(xf) -> bool:
    return abs((xf[0] + np.pi) % (2 * np.pi) - np.pi) < 0.4 and abs(xf[1]) < 0.4


def plot_control_synthesis(rows, samples, norm):
    """Left: the angle relative to upright, θ(t)−2kπ. Right: the synthesized feedback law u(t).
    Each fitted ReLU^p V̂ induces u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x); we roll it out from the deepest
    supported state (highest cost-to-go in the basin) and drive to upright, beside the true PMP
    feedback (Han & Yang Fig. 3 single-IC comparison)."""
    _apply_publication_style()
    x0 = _deepest_supported_start(samples)
    goal = 2.0 * np.pi * round(x0[0] / (2.0 * np.pi))     # nearest upright copy 2kπ
    runs = {"true": (r"true PMP", PALETTE["neutral"], "-", PROBLEM.true_feedback(samples))}
    for p in POWERS:
        label, color, ls = _POWER_STYLE[p]
        runs[p] = (label, color, ls, _model_feedback(rows, p, norm))
    rolled = {k: PROBLEM.rk4_rollout(uf, x0, T=_ROLL_T, dt=_ROLL_DT, u_clip=_U_CLIP)
              for k, (_, _, _, uf) in runs.items()}

    fig, axes = _create_subplots(1, 2, figsize=(14, 5))
    lo, hi = 0.0, 0.0
    for k, (label, color, ls, _) in runs.items():
        t, xs, us, _ = rolled[k]
        # truncate display once a controller has clearly diverged (|θ̇| ≫ stabilization speeds),
        # so a failed run reads as "runs off and stops" rather than re-entering the frame
        blown = np.flatnonzero(np.abs(xs[:, 1]) > 10.0)
        end = (blown[0] + 1) if blown.size else len(t)
        th = xs[:end, 0] - goal                          # angle relative to the upright copy
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
                     formats=["png"], dpi=300)

    cost_rows = []
    for k in ["true"] + POWERS:
        _, xs, _, cost = rolled[k]
        cost_rows.append({
            "controller": "true PMP" if k == "true" else f"ReLU p={int(k)}",
            "neurons": "—" if k == "true" else best_relu_at_power(rows, k)["neurons"],
            "reaches_up": "yes" if _stabilizes(xs[-1]) else "no",
            "cost": cost,
        })
    return _CTRL_FIG.as_posix(), cost_rows, x0, goal


# ---------------------------------------------------------------------------- #
# Parameter discussion (power, alpha)
# ---------------------------------------------------------------------------- #
def _alpha_table(rows) -> str:
    out = []
    for p in POWERS:
        cand = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
                and r["loss"] == "h1" and r["power"] == p]
        for a in sorted({r["alpha"] for r in cand}):
            b = min((r for r in cand if r["alpha"] == a), key=lambda r: r["far_lv"])
            out.append({"power": p, "alpha": a, "neurons": b["neurons"],
                        "far_lv": b["far_lv"], "far_lg": b["far_lg"]})
    return format_table(
        out, ["power", "alpha", "neurons", "far_lv", "far_lg"],
        headers={"alpha": "α", "far_lv": "far Lv", "far_lg": "far Lg"},
        formats={"power": "{:g}", "alpha": "{:.0e}", "far_lv": "{:.3f}", "far_lg": "{:.3f}"},
        title="ReLU H1: effect of α at each power (best far Lv per cell)",
    )


def plot_power_alpha_tradeoff(rows) -> str:
    _apply_publication_style()
    sel = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
           and r["loss"] == "h1" and r["power"] in POWERS]
    alphas = sorted({r["alpha"] for r in sel})
    cmap = plt.cm.viridis(np.linspace(0.12, 0.9, len(alphas)))
    acolor = {a: c for a, c in zip(alphas, cmap)}
    markers = {2.0: "o", 3.0: "s", 5.0: "^"}

    fig, axes = _create_subplots(1, 1, figsize=(8.5, 6))
    ax = axes[0]
    for p in POWERS:
        for a in alphas:
            pts = [r for r in sel if r["power"] == p and r["alpha"] == a]
            if pts:
                ax.scatter([q["neurons"] for q in pts], [q["far_lv"] for q in pts],
                           marker=markers[p], color=acolor[a], s=70,
                           edgecolor="white", linewidth=0.6, zorder=3)
    from matplotlib.lines import Line2D
    p_handles = [Line2D([], [], marker=markers[p], color="0.3", ls="", markersize=9,
                        label=_POWER_STYLE[p][0]) for p in POWERS]
    a_handles = [Line2D([], [], marker="o", color=acolor[a], ls="", markersize=9,
                        label=fr"$\alpha={a:g}$") for a in alphas]
    leg1 = ax.legend(handles=p_handles, title="power", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=a_handles, title=r"penalty $\alpha$", loc="upper right")
    ax.set_xlabel("Neurons"); ax.set_ylabel(r"far value-L1")
    _finalize_figure(fig, OUTPUT_DIR / _TRADEOFF_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _TRADEOFF_FIG.as_posix()


# ---------------------------------------------------------------------------- #
# Full result — region-split absolute L1, ReLU powers × alpha
# ---------------------------------------------------------------------------- #
def best_per_cell(rows, loss: str) -> list[dict[str, Any]]:
    sel = [r for r in rows if r["loss"] == loss]

    def cell(r):
        return (r["activation"], r["power"])

    best = []
    for _, group in itertools.groupby(sorted(sel, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["far_lv"]))
    best.sort(key=lambda r: (r["activation"], r["power"]))
    return best


def _fit_table(rows, loss: str, title: str) -> str:
    return format_table(
        best_per_cell(rows, loss),
        ["power", "alpha", "neurons", "far_lv", "nf_v", "far_lg", "nf_g"],
        headers={"alpha": "α", "far_lv": "far Lv", "nf_v": "near/far V",
                 "far_lg": "far Lg", "nf_g": "near/far G"},
        formats={"power": "{:g}", "alpha": "{:.0e}", "far_lv": "{:.3f}", "nf_v": "{:.2f}",
                 "far_lg": "{:.3f}", "nf_g": "{:.2f}"},
        title=title,
    )


# ---------------------------------------------------------------------------- #
# Markdown assembly
# ---------------------------------------------------------------------------- #
def _metrics_table(rows) -> str:
    out = []
    for p in POWERS:
        r = best_relu_at_power(rows, p)
        out.append({"power": p, "q": _q(p), "alpha": r["alpha"], "neurons": r["neurons"],
                    "rel_h1": r["rel_h1"],
                    "near_lv": r["near_lv"], "far_lv": r["far_lv"],
                    "near_lg": r["near_lg"], "far_lg": r["far_lg"]})
    return format_table(
        out, ["power", "q", "alpha", "neurons", "rel_h1",
              "near_lv", "far_lv", "near_lg", "far_lg"],
        headers={"q": "q=2/(p+1)", "alpha": "α", "rel_h1": "rel H1",
                 "near_lv": "near Lv", "far_lv": "far Lv",
                 "near_lg": "near Lg", "far_lg": "far Lg"},
        formats={"power": "{:g}", "q": "{:.2f}", "alpha": "{:.0e}", "rel_h1": "{:.3f}",
                 "near_lv": "{:.3f}", "far_lv": "{:.3f}",
                 "near_lg": "{:.3f}", "far_lg": "{:.3f}"},
        title="Best ReLU^p H1 fit per power (rel H1 + region-split absolute L1)",
    )


def _cost_table(cost_rows, x0) -> str:
    return format_table(
        cost_rows, ["controller", "neurons", "reaches_up", "cost"],
        headers={"reaches_up": "reaches upright?", "cost": "closed-loop cost"},
        formats={"cost": "{:.1f}"},
        title=f"Stabilize to upright from deepest supported start x0=[{x0[0]:.2f}, {x0[1]:.2f}], T={_ROLL_T:g}",
    )


def results_markdown(rows) -> str:
    samples = load_value_samples(rows[0]["data_file"])
    norm = ValueSampleNormalizer.fit(samples)

    shape_fig = plot_penalty_shape()
    surf_figs = plot_value_surfaces(rows, samples, norm)
    ctrl_fig, cost_rows, x0, goal = plot_control_synthesis(rows, samples, norm)
    tradeoff_fig = plot_power_alpha_tradeoff(rows)

    # Dynamic figures for the prose (kept in sync with the regenerated sweep).
    far_lv_by_p = {p: best_relu_at_power(rows, p)["far_lv"] for p in POWERS}
    lo_p, hi_p = min(POWERS), max(POWERS)
    cost_by = {r["controller"]: r["cost"] for r in cost_rows}
    cost_phrase = "; ".join(
        [f"true PMP {cost_by['true PMP']:.1f}"]
        + [f"power {int(p)} {cost_by[f'ReLU p={int(p)}']:.1f}" for p in POWERS]
    )

    def _surf_row() -> str:
        head, sep, imgs = [], [], []
        for p in POWERS:
            r = best_relu_at_power(rows, p)
            head.append(f"ReLU $p={int(p)}$ · {r['neurons']} neurons · far Lv={r['far_lv']:.2f}")
            sep.append("---")
            imgs.append(f"![p{int(p)}]({surf_figs[p]})")
        return ("| " + " | ".join(head) + " |\n"
                + "| " + " | ".join(sep) + " |\n"
                + "| " + " | ".join(imgs) + " |")

    return f"""# penaltypowers — pendulum (switching set)

How the **power** of the atom `σ(z)^p` — equivalently the nonconvex penalty exponent
`q = 2/(p+1)` — behaves on a value function with a **gradient jump across the
switching set** (pendulum swing-up). Method and sweep axes are in `README.md`;
this file reports the findings, with ReLU at powers {{2, 3, 5}} as
representatives. This is the switching-set counterpart of `../../01_vdp/frac_exp_penalty`:
there higher power was free sparsity; **here it is the opposite** — the reversal is the point.

## Key finding

The coefficient penalty is `α·Σ |c|^q`, `q = 2/(power+1)`: higher power ⇒ smaller
`q` ⇒ more aggressive nonconvex pruning. On a smooth target that was free, but a
switching-set value function needs **more, lower-degree** atoms to seat the gradient
discontinuity — so raising the power both over-smooths each atom and over-prunes,
and the fit degrades sharply. On the current **two-sided** data (the pad/collar band
puts the gradient jump in-sample) the reversal is starker than on the earlier
one-sided basin: the band dominates the H1 objective, and only low-power atoms can
spend that error mass usefully.

### Penalty & atom shape

![penalty shape]({shape_fig})

Left: the atom `σ(z)=ReLU(z)^p` sharpens with `p`. Right: the penalty `φ(c)=|c|^q`
grows more concave as `q=2/(p+1)` shrinks. The mildest nonconvex penalty, `p=2`
(`q=0.67`, the ReLU² atom), is the sweet spot here.

### Fitted value surfaces

The learned `V̂(x)` of the best ReLU fit at each power (shared `plot_model_value_surface`
renderer, z **unclipped**). The reconstruction degrades visibly as the power rises: the
multi-well landscape and its switching walls are shaped at `p=2`, but a high-power atom
`σ(z)^p` extrapolates as a degree-`p` polynomial off the data, so by `p=5` the surface
is dominated by a ~10⁴ off-support spike and the true value range (≲65) is squashed
flat — itself a picture of why high power over-fits the boundary and loses the interior.

{_surf_row()}

### Best metrics

{_metrics_table(rows)}

`rel H1` is the global relative H1 loss (total value+gradient); `near`/`far` are the
region-split **mean per-sample absolute L1** (`near` = lowest-10% distance to the
switching set, `far` = the smooth rest — absolute, so robust to the V→0 upright
interior). The reversal: **power 2 (ReLU², `q=0.67`) is best on every representative
column** — total H1, both value bands, and both gradient bands worsen as the power
is raised to `p=3` and `p=5`. The `near` columns are the harder
switching-set band and stay above `far` throughout. The aggressive nonconvex penalty
that was free on smooth VDP is *harmful* here, because the kinked target cannot be
tiled by a few high-power atoms.

### Synthesized control vs true feedback

The pendulum is control-affine with cost `r·u²`, so the value induces the **feedback
law** `u(x) = −(1/(2r·ml²)) ∂_θ̇ V(x)` (Han & Yang Eq. 15,
`PendulumSwingUpProblem.feedback_from_gradient`). We synthesize û from each fitted ReLU^p
`V̂` and roll it out in the true dynamics, beside the true PMP feedback. **Start.** The
samples cover the upright basin plus the two-sided switching band (`θ̇∈[−7.7,7.7]`,
`V≲65`); the hanging configuration `θ=π` itself remains at the edge of support (band
samples sit on the switching spiral around it, not at it), so we start from the
**deepest supported state** (the highest cost-to-go sample, here
`x0≈[{x0[0]:.2f}, {x0[1]:.2f}]`, a fast-moving edge-of-basin state) and drive to the
nearest upright copy. Left is the angle from upright `θ(t)−2kπ`; right is the feedback
law `u(t)`.

![control synthesis]({ctrl_fig})

{_cost_table(cost_rows, x0)}

At `t=0` all four controllers sit at the same supported state, and their controls
**agree in sign and rough magnitude** — the feedback law is synthesized correctly (an
off-data hanging start, by contrast, gives sign-flipped garbage because no sample
constrains `∇V̂` there). The closed loop now amplifies the accuracy gap ({cost_phrase}):
**powers 2 and 3 track the true swing to the 2π upright almost exactly, while power 5
overshoots and never settles** — its oscillating control keeps the pendulum orbiting
past the upright, the closed-loop face of the same off-support polynomial growth seen
in its value surface. On the two-sided data the high-power controller failure is back
(it had vanished in the one-sided interlude, where the smooth interior dominated the
objective). (Caveats: a single initial condition, and closed-loop outcomes are
sensitive to the start because `∇V̂` is pinned only near the data. The robust,
data-level statement is the accuracy reversal itself: **higher power degrades the
fit**, far Lv rising from {far_lv_by_p[lo_p]:.2f} at `p={int(lo_p)}` to {far_lv_by_p[hi_p]:.2f} at `p={int(hi_p)}`.)

## Parameter discussion (power, α)

The **power** is the headline lever above; the penalty strength **α** only refines
within a power (the sweep fixes `γ=0`, so the penalty is the pure `α·Σ|c|^q`).

{_alpha_table(rows)}

![power/alpha tradeoff]({tradeoff_fig})

The scatter places every signed ReLU-H1 run on the neurons-vs-(far value-L1) plane
(marker = power, colour = α): power 2 occupies the accurate region; higher powers sit
at larger error regardless of α. α moves a model along its frontier, but the power
sets which frontier — and at a switching set, low power wins.

## Full result

Region-split **mean per-sample L1**, normalized by the global mean ‖true‖. `far` =
smooth region, `near/far` = how many times worse the switching set is. Best α per
power by far value-L1 (ReLU, `γ=0`).

### H1 (gradient-augmented) loss

{_fit_table(rows, 'h1', 'Pendulum H1 fit — best far value-L1 per power (α swept)')}

### L2 (value-only) loss

{_fit_table(rows, 'l2', 'Pendulum L2 fit — best far value-L1 per power (α swept)')}
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
