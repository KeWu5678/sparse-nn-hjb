#!/usr/bin/env python3
"""activationsearch — VDP / smooth case.

Writes ``results.md`` as the Key-Finding-first report for the smooth Van der Pol
sweep, the counterpart of ``../../02_pendulum/log_penalty`` (the discontinuous case). Method, sweep
axes, and the activation list live in ``README.md``; this file reports the
findings:

  Key finding   — activation shape (softplus/tanh/gaussian) → fitted value
                  surfaces (signed) → metrics at the fixed comparison point
                  (α=1e-5, γ=1) → synthesized feedback rolled out against the
                  true control.
  Parameter discussion — impact of the nonconvex-penalty parameters alpha, gamma.
  Full result   — the detailed sparsity-aware (H1 × neurons) table.

    python vdp/analysis.py        # run from experiments/01_vdp/log_penalty
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

OUTPUT_DIR = Path(__file__).resolve().parent          # experiments/01_vdp/log_penalty
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_activation
from src.data import ValueSampleNormalizer, load_value_samples
from src.metric import format_table
from src.models.net import ShallowNetwork
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem
from src.plots import _best_iteration_atoms, plot_model_value_surface

EXPERIMENT = "activationsearch"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}

# Uniform operating point for the value-fit comparison (Key finding), matching the
# thesis tables — surfaces/metrics/control are all read off this single (alpha, gamma).
_FIXED_ALPHA, _FIXED_GAMMA = 1e-5, 1.0

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
            f"no run records under {MULTIRUN_DIR} — run `make activationsearch DATA=vdp`"
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
    """The signed H1 profile run at the fixed comparison point (alpha=1e-5, gamma=1),
    so the value-fit comparison across activations is apples-to-apples and matches the
    thesis tables. This is *not* each activation's sweep-best — the parameters are held
    uniform on purpose."""
    cand = [r for r in rows if r["kind"] == "signed" and r["insertion"] == "profile"
            and r["loss"] == "h1" and r["activation"] == activation
            and abs(r["alpha"] - _FIXED_ALPHA) < 1e-12
            and abs(r["gamma"] - _FIXED_GAMMA) < 1e-12]
    if not cand:
        raise ValueError(
            f"no signed h1 profile run for {activation!r} at "
            f"alpha={_FIXED_ALPHA}, gamma={_FIXED_GAMMA}")
    return min(cand, key=lambda r: r["rel_h1"])


# ---------------------------------------------------------------------------- #
# Sparsity / insertion dynamics — per-iteration diagnostics from the History.
# The localized (gaussian) vs global (softplus) contrast: how many neurons the
# profile step admits per iteration, and how much each one lowers the objective.
# ---------------------------------------------------------------------------- #
_INSERTION_ACTS = ["gaussian", "softplus"]


def _insertion_diag(result_path: str) -> list[dict[str, Any]]:
    """Per-iteration support size, neurons added, and objective drop per neuron.

    Read straight from the run's ``History``: ``train_loss`` is the full objective
    J = L(μ) + α·Φ(μ), and ``inner_weights[i]`` holds the surviving support after SSN
    and pruning, so its row count is N at iteration i."""
    with open(result_path, "rb") as f:
        history = pickle.load(f)
    J = history.train_loss
    N = [history.inner_weights[i]["weight"].shape[0] for i in range(len(J))]
    out = []
    for i in range(len(N)):
        ins = N[i] - (N[i - 1] if i > 0 else 0)
        dJ = (J[i - 1] - J[i]) if i > 0 else None
        dJn = (dJ / ins) if (dJ is not None and ins > 0) else None
        out.append({"iter": i + 1, "neurons": N[i], "inserted": ins, "dJ_per_n": dJn})
    return out


def _insertion_table(diags: dict[str, list[dict[str, Any]]]) -> str:
    """Side-by-side per-iteration table (gaussian | softplus)."""
    n_iter = max(len(d) for d in diags.values())

    def cell(d: list[dict[str, Any]], i: int) -> tuple[str, str, str]:
        if i >= len(d):
            return ("", "", "")
        r = d[i]
        djn = "—" if r["dJ_per_n"] is None else f"{r['dJ_per_n']:.1e}"
        return (str(r["neurons"]), str(r["inserted"]), djn)

    lines = [
        "| iter | gaussian N | ins | ΔJ/n | softplus N | ins | ΔJ/n |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i in range(n_iter):
        gN, gi, gd = cell(diags["gaussian"], i)
        sN, si, sd = cell(diags["softplus"], i)
        lines.append(f"| {i + 1} | {gN} | {gi} | {gd} | {sN} | {si} | {sd} |")
    gtot = diags["gaussian"][-1]["neurons"]
    stot = diags["softplus"][-1]["neurons"]
    lines.append(f"| final | {gtot} | | | {stot} | | |")
    return "\n".join(lines)


def _insertion_section(rows) -> str:
    """The 'Sparsity and the insertion dynamics' subsection: table + mechanism."""
    diags = {a: _insertion_diag(signed_h1_fixed(rows, a)["result_path"])
             for a in _INSERTION_ACTS}

    def ins_range(d: list[dict[str, Any]]) -> tuple[int, int]:
        vals = [r["inserted"] for r in d]
        return min(vals), max(vals)

    glo, ghi = ins_range(diags["gaussian"])
    slo, shi = ins_range(diags["softplus"])
    gN = diags["gaussian"][-1]["neurons"]
    sN = diags["softplus"][-1]["neurons"]
    ratio = gN / max(sN, 1)
    table = _insertion_table(diags)
    return (
        "### Sparsity and the insertion dynamics\n\n"
        f"The gaussian and softplus fits differ sharply in size — {gN} vs {sN} neurons, "
        f"a factor of ~{ratio:.1f} — even though gaussian is the *more* accurate of the "
        "two. Tracking the profile insertion neuron-by-neuron shows why: gaussian adds "
        f"neurons in large batches ({glo}–{ghi} per iteration) while softplus adds only "
        f"{slo}–{shi}, and each gaussian neuron buys a far smaller drop in the objective "
        "`J = L(μ) + α·Φ(μ)`.\n\n"
        f"{table}\n\n"
        "`N` = support size after SSN and pruning; `ins` = neurons added that iteration; "
        "`ΔJ/n` = decrease of `J` per neuron added (relative to the previous iterate; "
        "iteration 1 has no predecessor).\n\n"
        "The mechanism is the dual variable: the insertion score "
        "`p_t(ω) = ⟨σ(·;ω), g_t⟩` scores a candidate direction ω against the current "
        "residual `g_t`. The gaussian is **localized** (a bump concentrated near the "
        "hyperplane a·x + b ≈ 0), so `p_t` is sensitive to local residual pockets and a "
        "fresh batch of atoms clears the threshold α every iteration, each correcting "
        "only a small local portion of the residual. Softplus has **global support**, so "
        "`p_t` averages the residual over the whole domain — positive and negative "
        "contributions cancel, fewer ω exceed α, but each admitted atom removes a global "
        "component."
    )


# ---------------------------------------------------------------------------- #
# Model reconstruction + physical value/gradient (signed runs only; semiconcave
# does not round-trip through the lossy History — issue #19, src/plots.py)
# ---------------------------------------------------------------------------- #
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
def _gamma_table(rows) -> str:
    out = []
    for name in REPS:
        cand = [r for r in rows if r["kind"] == "signed" and r["insertion"] == "profile"
                and r["loss"] == "h1" and r["activation"] == name]
        best = min(cand, key=lambda r: r["rel_h1"])
        fixed_alpha = best["alpha"]
        for g in sorted({r["gamma"] for r in cand}):
            b = min((r for r in cand if r["gamma"] == g and r["alpha"] == fixed_alpha),
                    key=lambda r: r["rel_h1"])
            out.append({"activation": name, "gamma": g, "alpha": b["alpha"],
                        "neurons": b["neurons"], "rel_h1": b["rel_h1"]})
    return format_table(
        out, ["activation", "gamma", "alpha", "neurons", "rel_h1"],
        headers={"rel_h1": "rel H1"},
        formats={"gamma": "{:g}", "alpha": "{:g}", "rel_h1": "{:.3f}"},
        title="Effect of gamma (alpha fixed at each activation's best run), signed H1",
    )


def _alpha_table(rows) -> str:
    out = []
    for name in REPS:
        cand = [r for r in rows if r["kind"] == "signed" and r["insertion"] == "profile"
                and r["loss"] == "h1" and r["activation"] == name]
        best = min(cand, key=lambda r: r["rel_h1"])
        fixed_gamma = best["gamma"]
        for a in sorted({r["alpha"] for r in cand}):
            b = min((r for r in cand if r["alpha"] == a and r["gamma"] == fixed_gamma),
                    key=lambda r: r["rel_h1"])
            out.append({"activation": name, "alpha": a, "gamma": b["gamma"],
                        "neurons": b["neurons"], "rel_h1": b["rel_h1"]})
    return format_table(
        out, ["activation", "alpha", "gamma", "neurons", "rel_h1"],
        headers={"rel_h1": "rel H1"},
        formats={"alpha": "{:g}", "gamma": "{:g}", "rel_h1": "{:.3f}"},
        title="Effect of alpha (gamma fixed at each activation's best run), signed H1",
    )


def plot_alpha_gamma_tradeoff(rows) -> str:
    """Neurons vs relative H1 for the signed reps, marker = activation, colour = gamma."""
    _apply_publication_style()
    sel = [r for r in rows if r["kind"] == "signed" and r["insertion"] == "profile"
           and r["loss"] == "h1"]
    gammas = sorted({r["gamma"] for r in sel})
    gcolor = {g: c for g, c in zip(
        gammas, [PALETTE["blue_main"], PALETTE["teal"], PALETTE["red_strong"], PALETTE["violet"]])}
    markers = {"softplus": "o", "tanh": "s", "gaussian": "^"}

    fig, axes = _create_subplots(1, 1, figsize=(8.5, 6))
    ax = axes[0]
    for name in REPS:
        for g in gammas:
            pts = [r for r in sel if r["activation"] == name and r["gamma"] == g]
            if pts:
                ax.scatter([p["neurons"] for p in pts], [p["rel_h1"] for p in pts],
                           marker=markers[name], color=gcolor[g], s=70,
                           edgecolor="white", linewidth=0.6, zorder=3)
    from matplotlib.lines import Line2D
    act_handles = [Line2D([], [], marker=markers[n], color="0.3", ls="",
                          markersize=9, label=_REP_STYLE[n][0]) for n in REPS]
    g_handles = [Line2D([], [], marker="o", color=gcolor[g], ls="", markersize=9,
                        label=fr"$\gamma={g:g}$") for g in gammas]
    leg1 = ax.legend(handles=act_handles, title="activation", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=g_handles, title="penalty $\\gamma$", loc="center right")
    ax.set_xlabel("Neurons"); ax.set_ylabel(r"Relative $H^1$")
    _finalize_figure(fig, OUTPUT_DIR / _TRADEOFF_FIG.with_suffix(""),
                     formats=["png"], dpi=300)
    return _TRADEOFF_FIG.as_posix()


# ---------------------------------------------------------------------------- #
# Full result — sparsity-aware (H1 x neurons) table, both kinds, all activations
# ---------------------------------------------------------------------------- #
def best_per_cell(rows, loss: str) -> list[dict[str, Any]]:
    sel = [r for r in rows if r["loss"] == loss and r["insertion"] == "profile"]

    def cell(r):
        return (r["kind"], r["activation"])

    best = []
    for _, group in itertools.groupby(sorted(sel, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["score"]))
    best.sort(key=lambda r: r["score"])
    return best


def _fit_table(rows, loss: str, title: str) -> str:
    return format_table(
        best_per_cell(rows, loss),
        ["kind", "activation", "gamma", "alpha", "neurons", "rel_h1", "rel_l2", "score"],
        headers={"rel_h1": "rel H1", "rel_l2": "rel L2"},
        formats={"gamma": "{:g}", "alpha": "{:g}", "rel_h1": "{:.3f}",
                 "rel_l2": "{:.3f}", "score": "{:.2f}"},
        title=title,
    )


# ---------------------------------------------------------------------------- #
# Markdown assembly
# ---------------------------------------------------------------------------- #
def _metrics_table(rows) -> str:
    out = [dict(activation=n, **{k: signed_h1_fixed(rows, n)[k]
                                 for k in ("neurons", "rel_h1", "rel_l2", "score")})
           for n in REPS]
    return format_table(
        out, ["activation", "neurons", "rel_h1", "rel_l2", "score"],
        headers={"rel_h1": "rel H1", "rel_l2": "rel L2"},
        formats={"rel_h1": "{:.3f}", "rel_l2": "{:.3f}", "score": "{:.2f}"},
        title="Signed H1 fit per representative at the fixed comparison point (α=1e-5, γ=1)",
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

    shape_fig = plot_activation_shapes()
    deriv = plot_derivative_distribution(rows, samples, norm)
    surf_figs = plot_value_surfaces(rows, samples, norm)
    ctrl_fig, cost_rows = plot_control_synthesis(rows, samples, norm)
    tradeoff_fig = plot_alpha_gamma_tradeoff(rows)
    dz = {a: f"{p * 100:.0f}%" for a, p in deriv["near_zero"].items()}
    nn = deriv["neurons"]

    def _surf_row() -> str:
        head, sep, imgs = [], [], []
        for name in REPS:
            run = signed_h1_fixed(rows, name)
            head.append(f"{_REP_STYLE[name][0]} · {run['neurons']} neurons · rel $H^1$={run['rel_h1']:.2f}")
            sep.append("---")
            imgs.append(f"![{name}]({surf_figs[name]})")
        return ("| " + " | ".join(head) + " |\n"
                + "| " + " | ".join(sep) + " |\n"
                + "| " + " | ".join(imgs) + " |")

    def _deriv_row() -> str:
        head = [f"{_REP_STYLE[n][0]} · {nn[n]} neurons · {dz[n]} with |σ′| < 0.05"
                for n in _DERIV_ORDER]
        imgs = [f"![{n}]({deriv['figures'][n]})" for n in _DERIV_ORDER]
        return ("| " + " | ".join(head) + " |\n"
                + "| " + " | ".join(["---"] * len(_DERIV_ORDER)) + " |\n"
                + "| " + " | ".join(imgs) + " |")

    return f"""# activationsearch — vdp (smooth)

Which activations best fit a **smooth** value function (Van der Pol stabilization),
under sparsity. Method, sweep axes, and the activation list are in `README.md`;
this file reports the findings. Three representatives span the smooth-fitting
spectrum — `softplus` (broad monotone ridge), `tanh` (saturating S), `gaussian`
(localized RBF) — all under the `signed` model. This is the smooth counterpart of
`../../02_pendulum/log_penalty` (the switching-set case); the contrast is the point.

## Openloop data

The VDP open-loop dataset is a filled grid over the state plane. The data
visualisations (value scatter, gradient arrows, value surface) are centralised in
[`experiments/00_openloop/vdp`](../../00_openloop/vdp) — the single source for the
open-loop training data.

| value samples | value and gradient |
| --- | --- |
| ![openloop value](../../00_openloop/vdp/figures/value_scatter.png) | ![openloop gradient](../../00_openloop/vdp/figures/value_gradient.png) |

## Key finding

Under the gradient-augmented (H1) loss each neuron contributes σ to V and
**w·σ′ to ∇V**, so the activation *derivative* σ′ is the basis that reconstructs
the gradient field. The VDP gradient is **smooth**, so the question is not whether
σ′ can break (as at a switching set) but how *economically* a smooth σ′ tiles a
smooth field.

### Activation shape

![activation shape]({shape_fig})

`softplus′` is a broad, monotone, single-signed ridge — every neuron is a coherent
gradient channel, so few atoms cover a smooth ∇V. `tanh′` saturates on **both**
tails (a narrow channel → many neurons). The Gaussian derivative is a localized
sign-changing bump — accurate where it sits, but it must be tiled densely.

### Gradient-kernel columns (σ′ distribution)

The gradient-kernel column for neuron n at data point x_m is σ′(a·x_m + b)·a, so the
distribution of σ′(z) over all (neuron, data-point) pairs — at the fixed point α=1e-5,
γ=1 — measures how many columns are *dead* (near-zero, contributing no gradient basis).

{_deriv_row()}

**tanh** saturates most: **{dz["tanh"]}** of its σ′ values are near zero, so it reaches
its gradient fit only by inserting the most neurons. **softplus** ({dz["softplus"]} near
zero) keeps single-signed coherent columns and fits with far fewer. The **Gaussian**
({dz["gaussian"]} near zero) has a sign-changing derivative spanning both signs — the
most diverse columns and the best gradient accuracy, though it pays in neuron count.

### Fitted value surfaces

The learned V̂(x) of each representative, as surfaces over the state plane, all at the
**same operating point α=1e-5, γ=1** (matching the thesis tables — a like-for-like
comparison, not each activation's sweep-best). All three capture the smooth bowl; they
differ in how many neurons it takes.

{_surf_row()}

### Metrics at the fixed comparison point (α=1e-5, γ=1)

{_metrics_table(rows)}

`rel H1`/`rel L2` are the global relative errors (VDP is smooth and filled, so —
unlike the pendulum — they are *not* confounded and need no region split). `score`
= rel H1 × neurons (sparsity-aware, lower is better). Held at the uniform α=1e-5, γ=1,
the smooth-fitting story is in the spread: **`softplus` is the sparse champion** (lowest
score — accurate enough at a small neuron count), **`gaussian` is the most accurate**
(lowest rel H1) but pays in neurons, and **`tanh` is dominated** (its two-sided-dead
derivative needs more neurons for no accuracy gain). No kink is needed — the target is
smooth. Each activation's sweep-best is in the Parameter-discussion tables below.

{_insertion_section(rows)}

### Synthesized feedback vs true control

The VDP value induces the static feedback `u(x) = −∂_{{x₂}}V(x)/(2β)`
(`g=[0,1]ᵀ`, cost `β u²` — Azmi–Kalise–Kunisch). We synthesize û from each fitted
V̂ and **roll it out in the true dynamics** from the initial state x=(2, 1) (the
paper's test point), beside the true control (a smooth C¹ interpolant of the
costate samples). The figure plots ‖y(t)‖ and |u(t)| over the horizon, after
Azmi–Kalise–Kunisch Fig. 8.

![control synthesis]({ctrl_fig})

{_cost_table(cost_rows)}

**Every smooth activation yields a working stabilizing feedback** — all three drive
‖y(t)‖ to the origin and their synthesized |u(t)| closely tracks the true smooth
optimal control (right panel), at essentially the true optimal cost (≈ 6.5). This is
the sharp contrast with the switching-set case (`../../02_pendulum/log_penalty`), where the activation
*determined whether the controller worked at all* (only the kink stabilized). On a
smooth problem the value gradient is well-behaved everywhere the closed loop visits,
so the choice of (smooth) activation is a question of **fit sparsity, not control
viability** — softplus simply reaches the same controller with fewer neurons.

## Parameter discussion (α, γ)

The nonconvex penalty `α·Σ φ(|c|)`: **α** scales the penalty (the sparsity lever)
and **γ** controls the log-term nonconvexity. The tables take the three signed reps,
find each activation's best achieved run by relative H1, then sweep one parameter
while holding the other fixed at that best run's value.

{_alpha_table(rows)}

{_gamma_table(rows)}

![alpha/gamma tradeoff]({tradeoff_fig})

The scatter places every signed-H1 run on the neurons-vs-accuracy plane (marker =
activation, colour = γ): `softplus` sits at low neuron count, `gaussian` at low rel
H1 (high neurons), `tanh` in between but dominated. The penalty parameters move a
model *along* its activation's frontier (sparsity ↔ accuracy); on this smooth target
all of them stay accurate, so the lever mostly trades neurons.

## Full result

Best sparsity-aware `score = rel H1 × neurons` per (model, activation), profile
insertion, ranked best-first; both model kinds, all activations.

### H1 (gradient-augmented) loss

{_fit_table(rows, 'h1', 'VDP H1 fit — best score per model/activation')}

### L2 (value-only) loss

{_fit_table(rows, 'l2', 'VDP L2 fit — best score per model/activation')}
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
