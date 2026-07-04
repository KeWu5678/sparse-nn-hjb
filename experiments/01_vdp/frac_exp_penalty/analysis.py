#!/usr/bin/env python3
"""penaltypowers — VDP / smooth case.

Key-Finding-first report for the penalty-power sweep on the smooth Van der Pol
dataset, the counterpart of ``../../02_pendulum/frac_exp_penalty`` (the switching-set case). Method, sweep
axes, and the activation list live in ``README.md``; this file reports the findings.

The swept axes are the **power** ``p`` of the ReLU^p atom (which sets the nonconvex
penalty exponent ``q = 2/(p+1)``) and the penalty weight **alpha**.  The penalty is
the pure power penalty ``alpha·Sum |c|^q`` — the Algorithm-1 log term is off, so
``gamma`` is not a variable here (``load_rows`` keeps only the ``gamma = 0`` runs).
The value-fit comparison is read at one fixed operating point (``alpha = 1e-5``);
alpha is swept only for the sparsity–accuracy frontier.  Representatives are ReLU at
powers {2, 3, 5}.

    python vdp/analysis.py        # run from experiments/01_vdp/frac_exp_penalty
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT_DIR = Path(__file__).resolve().parent          # experiments/01_vdp/frac_exp_penalty
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.data import ValueSampleNormalizer, load_value_samples
from src.metric import format_table
from src.models.net import ShallowNetwork
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem
from src.plots import _best_iteration_atoms, plot_model_value_surface

EXPERIMENT = "penaltypowers"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "frac_exp_penalty"
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}

# Fixed operating point for the value-fit comparison (matches the thesis tables):
# powers are read off at one alpha so the comparison is apples-to-apples. alpha is
# swept only for the frontier in the parameter discussion.
_FIXED_ALPHA = 1e-5

from src.plotstyle import PALETTE
from src.plotstyle import apply_publication_style as _apply_publication_style

# All swept powers (tables) and the three representatives (figures).
POWERS_ALL = [2.0, 2.01, 3.0, 4.0, 5.0]
POWERS = [2.0, 3.0, 5.0]
ACT = "relu"
_POWER_STYLE = {
    2.0: (r"$p=2$", PALETTE["blue_main"], "-"),
    3.0: (r"$p=3$", PALETTE["teal"], "--"),
    5.0: (r"$p=5$", PALETTE["red_strong"], ":"),
}
PROBLEM = VdpOptimalControlProblem()


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
# Records (global metrics; VDP is smooth, no region split)
# ---------------------------------------------------------------------------- #
def load_rows() -> list[dict[str, Any]]:
    records = sorted(MULTIRUN_DIR.glob("**/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make sweep EXPERIMENT=vdp/frac_exp_penalty`"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        if "pendulum" in cfg["data"]["path"].lower():     # this doc is VDP only
            continue
        model = cfg["model"]
        # Algorithm 2 is the pure power penalty: gamma must be 0. A positive gamma
        # mixes in the Algorithm-1 log-nonconvexity (a different penalty) — skip it.
        if float(model["gamma"]) != 0.0:
            continue
        v = record["metrics"][0]["values"]
        rows.append({
            "kind": model["kind"],
            "activation": model["activation"],
            "power": float(model["power"]),
            "loss": _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"])),
            "alpha": float(model["alpha"]),
            "seed": int(cfg["env"]["seed"]),
            "neurons": int(v["best_neurons"]),
            "rel_h1": float(v["rel_h1_val"]),
            "rel_l2": float(v["rel_l2_val"]),
            "data_file": cfg["data"]["path"],
            "result_path": str(_result_pkl(path)),
        })
    return rows


def _result_pkl(json_path: Path) -> Path:
    return json_path.parent / f"result_{json_path.stem}.pkl"


def relu_fixed(rows, power: float, loss: str = "h1") -> dict[str, Any]:
    """The relu run at the fixed operating point (alpha=1e-5) for a given power and
    loss.  A single deterministic run — *not* a min over any sweep axis, so it can
    never collapse onto a degenerate near-empty model the way a ``rel_h1 × neurons``
    score does."""
    cand = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
            and r["loss"] == loss and r["power"] == power
            and abs(r["alpha"] - _FIXED_ALPHA) < 1e-12]
    if not cand:
        raise ValueError(f"no signed relu {loss} run at power {power}, alpha={_FIXED_ALPHA}")
    return min(cand, key=lambda r: r["rel_h1"])   # unique in practice (one seed)


# ---------------------------------------------------------------------------- #
# Model reconstruction + physical value/gradient (signed runs round-trip cleanly)
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
_SURFACES_FIG = Path("figures") / "H1_power.png"
_CTRL_FIG = Path("figures") / "control_synthesis.png"
_TRADEOFF_FIG = Path("figures") / "power_alpha_tradeoff.png"


def plot_penalty_shape() -> str:
    """Left: the ReLU^p atom σ(z)=ReLU(z)^p. Right: the nonconvex coefficient penalty
    φ(c)=|c|^q with q=2/(p+1) — higher power ⇒ smaller q ⇒ more concave ⇒ sparser."""
    _apply_publication_style()
    z = np.linspace(-2.0, 2.0, 400)
    c = np.linspace(0.0, 2.0, 400)

    fig, axes = _create_subplots(1, 2, figsize=(11, 4.4))
    for p in POWERS:
        label, color, ls = _POWER_STYLE[p]
        axes[0].plot(z, np.maximum(z, 0.0) ** p, color=color, ls=ls, lw=2.6, label=label)
        axes[1].plot(c, c ** _q(p), color=color, ls=ls, lw=2.6,
                     label=fr"$q={_q(p):.2f}$")
    axes[0].set_xlabel(r"$z$  (pre-activation)")
    axes[0].set_ylabel(r"$\sigma(z)=\mathrm{ReLU}(z)^p$")
    axes[0].set_xlim(-2, 2); axes[0].legend(loc="upper left")
    axes[1].set_xlabel(r"$|c|$  (coefficient)")
    axes[1].set_ylabel(r"penalty $\varphi(c)=|c|^{\,q},\; q=2/(p{+}1)$")
    axes[1].set_xlim(0, 2); axes[1].legend(loc="lower right")
    _finalize_figure(fig, OUTPUT_DIR / _SHAPE_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _SHAPE_FIG.as_posix()


def _surface_onto(ax, rows, power, samples, norm) -> dict[str, Any]:
    run = relu_fixed(rows, power, "h1")
    plot_model_value_surface(
        run["result_path"], activation=ACT, power=power,
        x_scale=norm.x_scale, v_scale=norm.v_scale, dataset=samples,
        ax=ax, show=False, vmax=20.0, zticks=[0, 10, 20],
    )
    return run


def plot_value_surfaces(rows, samples, norm) -> dict[float, str]:
    """One ReLU^p value surface per representative power (fixed operating point),
    each its own title-less file for the results-md image row."""
    _apply_publication_style()
    paths = {}
    for p in POWERS:
        fig = plt.figure(figsize=(4.2, 4.0), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _surface_onto(ax, rows, p, samples, norm)
        rel = Path("figures") / f"value_surface_p{int(p)}.png"
        _finalize_figure(fig, OUTPUT_DIR / rel.with_suffix(""),
                         formats=["png"], dpi=300, tight=False,
                         bbox_inches="tight", pad_inches=0.05)
        paths[p] = rel.as_posix()
    return paths


def plot_power_surfaces_row(rows, samples, norm) -> str:
    """The three H1-trained value surfaces (p=2,3,5) in one row — the combined
    ``H1_power`` panel used by the thesis, rendered from the fixed-point fits."""
    _apply_publication_style()
    fig = plt.figure(figsize=(12.6, 4.0), dpi=300)
    for i, p in enumerate(POWERS):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _surface_onto(ax, rows, p, samples, norm)
    _finalize_figure(fig, OUTPUT_DIR / _SURFACES_FIG.with_suffix(""),
                     formats=["png"], dpi=300, tight=False,
                     bbox_inches="tight", pad_inches=0.1)
    return _SURFACES_FIG.as_posix()


# Closed-loop control rollout (VDP stabilization to the origin from x=(2,1)).
_U_CLIP = 50.0
_X0 = (2.0, 1.0)
_ROLL_T, _ROLL_DT, _PLOT_T = 12.0, 0.01, 3.0


def _model_feedback(rows, power, norm):
    net = _build_net(relu_fixed(rows, power, "h1")["result_path"], ACT, power)

    def u(x):
        _, g = _value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
        return float(PROBLEM.feedback_from_gradient(g[0]))
    return u


def plot_control_synthesis(rows, samples, norm):
    _apply_publication_style()
    runs = {"true": (r"true", PALETTE["neutral"], "-", PROBLEM.true_feedback(samples))}
    for p in POWERS:
        label, color, ls = _POWER_STYLE[p]
        runs[p] = (label, color, ls, _model_feedback(rows, p, norm))
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
    for k in ["true"] + POWERS:
        _, xs, _, cost = rolled[k]
        cost_rows.append({
            "controller": "true" if k == "true" else f"ReLU p={int(k)}",
            "neurons": "—" if k == "true" else relu_fixed(rows, k, "h1")["neurons"],
            "stabilizes": "yes" if np.linalg.norm(xs[-1]) < 0.2 else "no",
            "cost": cost,
        })
    return _CTRL_FIG.as_posix(), cost_rows


# ---------------------------------------------------------------------------- #
# Parameter discussion (power, alpha) — the sparsity-accuracy frontier
# ---------------------------------------------------------------------------- #
def _relu_grid(rows, loss: str) -> list[dict[str, Any]]:
    """Every relu run for a loss, one row per (power, alpha)."""
    sel = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
           and r["loss"] == loss]
    sel.sort(key=lambda r: (r["power"], -r["alpha"]))
    return sel


def _alpha_table(rows) -> str:
    grid = _relu_grid(rows, "h1")
    out = [{"power": r["power"], "alpha": r["alpha"], "neurons": r["neurons"],
            "rel_h1": r["rel_h1"]} for r in grid]
    return format_table(
        out, ["power", "alpha", "neurons", "rel_h1"],
        headers={"rel_h1": "rel H1"},
        formats={"power": "{:g}", "alpha": "{:g}", "rel_h1": "{:.3f}"},
        title="ReLU H1: sparsity-accuracy frontier over alpha, per power",
    )


def plot_power_alpha_tradeoff(rows) -> str:
    """Neurons vs relative H1: each representative power's frontier as alpha varies.
    Higher power sits at fewer neurons for the same accuracy — the free-sparsity story."""
    _apply_publication_style()
    fig, axes = _create_subplots(1, 1, figsize=(8.5, 6))
    ax = axes[0]
    for p in POWERS:
        pts = [r for r in rows if r["kind"] == "signed" and r["activation"] == ACT
               and r["loss"] == "h1" and r["power"] == p and r["neurons"] > 0]
        pts.sort(key=lambda r: r["neurons"])
        label, color, ls = _POWER_STYLE[p]
        ax.plot([q["neurons"] for q in pts], [q["rel_h1"] for q in pts],
                color=color, ls=ls, lw=2.2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
    ax.set_xlabel("Neurons"); ax.set_ylabel(r"Relative $H^1$")
    ax.legend(title="power", loc="upper right")
    _finalize_figure(fig, OUTPUT_DIR / _TRADEOFF_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _TRADEOFF_FIG.as_posix()


# ---------------------------------------------------------------------------- #
# Full result — the complete relu power x alpha grid, both losses
# ---------------------------------------------------------------------------- #
def _fit_table(rows, loss: str, title: str) -> str:
    grid = _relu_grid(rows, loss)
    out = [{"power": r["power"], "alpha": r["alpha"], "neurons": r["neurons"],
            "rel_l2": r["rel_l2"], "rel_h1": r["rel_h1"]} for r in grid]
    return format_table(
        out, ["power", "alpha", "neurons", "rel_l2", "rel_h1"],
        headers={"rel_h1": "rel H1", "rel_l2": "rel L2"},
        formats={"power": "{:g}", "alpha": "{:g}", "rel_l2": "{:.3f}", "rel_h1": "{:.3f}"},
        title=title,
    )


# ---------------------------------------------------------------------------- #
# Markdown assembly
# ---------------------------------------------------------------------------- #
def _metrics_table(rows) -> str:
    """Per-power value-fit at the fixed operating point (alpha=1e-5)."""
    out = []
    for p in POWERS_ALL:
        r = relu_fixed(rows, p, "h1")
        out.append({"power": p, "q": _q(p), "neurons": r["neurons"],
                    "rel_l2": r["rel_l2"], "rel_h1": r["rel_h1"]})
    return format_table(
        out, ["power", "q", "neurons", "rel_l2", "rel_h1"],
        headers={"rel_h1": "rel H1", "rel_l2": "rel L2", "q": "q=2/(p+1)"},
        formats={"power": "{:g}", "q": "{:.2f}", "rel_l2": "{:.3f}", "rel_h1": "{:.3f}"},
        title="ReLU^p H1 fit per power at the fixed operating point (alpha=1e-5)",
    )


def _cost_table(cost_rows) -> str:
    return format_table(
        cost_rows, ["controller", "neurons", "stabilizes", "cost"],
        headers={"stabilizes": "stabilizes?", "cost": "closed-loop cost"},
        formats={"cost": "{:.2f}"},
        title="Closed-loop stabilization from y₀=(2, 1)",
    )


def results_markdown(rows) -> str:
    samples = load_value_samples(rows[0]["data_file"])
    norm = ValueSampleNormalizer.fit(samples)

    shape_fig = plot_penalty_shape()
    surf_figs = plot_value_surfaces(rows, samples, norm)
    plot_power_surfaces_row(rows, samples, norm)     # combined H1_power panel (thesis)
    ctrl_fig, cost_rows = plot_control_synthesis(rows, samples, norm)
    tradeoff_fig = plot_power_alpha_tradeoff(rows)

    def _surf_row() -> str:
        head, sep, imgs = [], [], []
        for p in POWERS:
            r = relu_fixed(rows, p, "h1")
            head.append(f"ReLU $p={int(p)}$ · {r['neurons']} neurons · rel $H^1$={r['rel_h1']:.2f}")
            sep.append("---")
            imgs.append(f"![p{int(p)}]({surf_figs[p]})")
        return ("| " + " | ".join(head) + " |\n"
                + "| " + " | ".join(sep) + " |\n"
                + "| " + " | ".join(imgs) + " |")

    return f"""# penaltypowers — vdp (smooth)

How the **power** of the ReLU^p atom — equivalently the nonconvex penalty exponent
`q = 2/(p+1)` — trades accuracy against sparsity on the **smooth** Van der Pol value
function. Method and sweep axes are in `README.md`; this file reports the findings,
with ReLU at powers {{2, 3, 5}} as representatives. The penalty is the *pure* power
penalty `α·Σ|c|^q` (the Algorithm-1 log term is off). The value-fit is compared at one
fixed operating point (**alpha = 1e-5**); alpha is swept only for the frontier. This is
the smooth counterpart of `../../02_pendulum/frac_exp_penalty`, where the same power knob behaves in the
**opposite** way — the reversal is the point.

## Key finding

The coefficient penalty is `α·Σ |c|^q` with `q = 2/(power+1)`: raising the atom
power lowers `q`, making the penalty more concave and the selection more
aggressively sparse. On a smooth target this is **free sparsity** — at a fixed
`alpha` higher power keeps the same accuracy with far fewer neurons.

### Penalty & atom shape

![penalty shape]({shape_fig})

Left: the atom `σ(z)=ReLU(z)^p` sharpens at the origin as `p` grows. Right: the
coefficient penalty `φ(c)=|c|^q` becomes more concave as `q=2/(p+1)` shrinks
(`q = 0.67, 0.50, 0.33` for `p = 2, 3, 5`) — a stronger sparsity prior.

### Fitted value surfaces

The learned `V̂(x)` at the fixed operating point (alpha=1e-5) for each power. On the
smooth VDP bowl all powers reconstruct the value well; they differ in neuron count.

{_surf_row()}

### Value-fit at the fixed operating point (alpha=1e-5)

{_metrics_table(rows)}

At a fixed penalty weight the gradient accuracy is essentially flat across powers
(`rel H1 ≈ 0.10`), while the neuron count drops sharply with power: `p=2` needs ~41
atoms, `p=3`/`p=4`/`p=5` only ~20. Raising the power buys sparsity for free on this
smooth target — no `p=3` "sweet spot" and no high-power collapse; the ordering is
monotone in `p`.

### Synthesized feedback vs true control

The VDP value induces the static feedback `u(x) = −∂_{{x₂}}V(x)/(2β)` (`g=[0,1]ᵀ`,
cost `β u²` — Azmi–Kalise–Kunisch). We synthesize û from each fitted ReLU^p `V̂` and
roll it out in the true dynamics from x=(2, 1), beside the true control (smooth C¹
interpolant of the costate). Plots ‖y(t)‖ and |u(t)| after Azmi–Kalise–Kunisch Fig. 8.

![control synthesis]({ctrl_fig})

{_cost_table(cost_rows)}

Every power yields a working stabilizing feedback at essentially the true optimal
cost — on a smooth problem the power knob is purely a **sparsity** lever, not a
control-viability one. (Contrast `../../02_pendulum/frac_exp_penalty`, where high power destroys the fit and
the controller with it.)

## Parameter discussion (power, alpha)

The **power** is the headline lever above; **alpha** moves each power along its own
sparsity–accuracy frontier.

{_alpha_table(rows)}

![power/alpha tradeoff]({tradeoff_fig})

Each power's curve is its frontier as `alpha` varies (smaller `alpha` → more neurons,
lower error). Higher power sits **below and to the left** — fewer neurons at the same
accuracy — so the power knob shifts the whole frontier rather than moving along it.

## Full result

The complete relu × power × alpha grid, both losses. `rel L2`/`rel H1` are global
relative errors; no score is used to pick a "best" run — the frontier is shown in full
so the sparsity–accuracy trade is explicit.

### H1 (gradient-augmented) loss

{_fit_table(rows, 'h1', 'VDP H1 fit — relu power x alpha')}

### L2 (value-only) loss

{_fit_table(rows, 'l2', 'VDP L2 fit — relu power x alpha')}
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
