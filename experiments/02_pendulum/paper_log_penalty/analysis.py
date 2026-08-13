#!/usr/bin/env python3
"""Algorithm 1 log-penalty study — pendulum / switching-set case.

Writes ``results.md`` for the current normalized-measure Algorithm 1.
Method and sweep axes live in ``README.md``; this file reports:

  Key finding   — batch versus sequential insertion on the neurons-vs-error
                  frontier.
  Termination   — how often the paper loop stopped on its own rule (no candidate
                  clears the threshold) rather than exhausting T_out.
  Full result   — the sparsity-aware (H1 × neurons) table per activation and mode.

    python analysis.py        # run from this directory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table
from src.plotstyle import PALETTE, style_frontier_axes
from src.plotstyle import apply_publication_style as _apply_publication_style

PROBLEM = "pendulum"                 # vdp | pendulum
FAMILY = "paper_log_penalty"
SWEEP_ROOT = REPO_ROOT / "rawdata" / "logs" / "multirun" / PROBLEM / FAMILY
FIGURES = OUTPUT_DIR / "figures"
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}
# The axis this family varies across its rows, beside the shared alpha/gamma grid.
_VARIANT_KEY = "activation"        # activation | power
_VARIANT_LABEL = "activation"


def _save_png(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# Records
# ---------------------------------------------------------------------------- #
def _rows_under(root: Path, mode: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/*.json")):
        # Sweep trees also hold non-record JSON (eval pools, caches, Hydra state),
        # so a file that is not a run record is skipped rather than fatal.
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            cfg = record["config"]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            continue
        if record.get("status") not in (None, "ok", "completed", "success"):
            continue
        metrics = record.get("metrics")
        if not metrics:
            continue
        values = metrics[0]["values"]
        model, training = cfg["model"], cfg["training"]
        neurons = int(values["best_neurons"])
        rel_h1 = float(values["rel_h1_val"])
        rows.append({
            "mode": mode or training.get("insert_mode", "batch"),
            "activation": model["activation"],
            "power": float(model["power"]),
            "alpha": float(model["alpha"]),
            "gamma": float(model["gamma"]),
            "moment_order": float(model.get("moment_order", float("nan"))),
            "loss": _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"])),
            "neurons": neurons,
            "rel_h1": rel_h1,
            "rel_l2": float(values["rel_l2_val"]),
            "score": rel_h1 * max(neurons, 1),
            "iterations": int(values.get("iterations", 0) or 0),
            "requested_iterations": int(training["num_iterations"]),
            "elapsed_s": float(record.get("elapsed_s") or 0.0),
        })
    return rows


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _rows_under(SWEEP_ROOT / "batch", "batch") + _rows_under(
        SWEEP_ROOT / "sequential", "sequential"
    )
    if not rows:
        raise FileNotFoundError(
            f"no run records under {SWEEP_ROOT} — run "
            f"`make paper-sweep EXPERIMENT={PROBLEM}/{FAMILY}`"
        )
    p_rows = _rows_under(SWEEP_ROOT / "p_study_sequential", "sequential")
    return rows, p_rows


def _best_by(rows: list[dict[str, Any]], key: str, value: Any, mode: str) -> dict | None:
    cand = [r for r in rows if r[key] == value and r["mode"] == mode and r["loss"] == "h1"]
    return min(cand, key=lambda r: r["score"]) if cand else None


# ---------------------------------------------------------------------------- #
# Figure: the neurons-vs-error frontier, batch against sequential
# ---------------------------------------------------------------------------- #
def frontier_figure(rows: list[dict[str, Any]]) -> Path:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    styles = {
        "batch": dict(color=PALETTE["blue_main"], marker="o", ls="-."),
        "sequential": dict(color=PALETTE["red_strong"], marker="s", ls="--"),
    }
    for mode, style in styles.items():
        pts = sorted(
            ((r["neurons"], r["rel_h1"]) for r in rows if r["mode"] == mode and r["loss"] == "h1"),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        # Running-best frontier: the lowest error achieved at or below each width.
        xs, ys, best = [], [], np.inf
        for n, e in pts:
            best = min(best, e)
            xs.append(n)
            ys.append(best)
        ax.plot(xs, ys, label=mode, markersize=4, markeredgecolor="0.2",
                markeredgewidth=0.6, lw=1.4, **style)
    ax.set_xlabel("neurons")
    ax.set_ylabel(r"relative $H^1$ validation error")
    ax.set_yscale("log")
    style_frontier_axes(ax)
    return _save_png(fig, "frontier_modes.png")


# ---------------------------------------------------------------------------- #
# results.md
# ---------------------------------------------------------------------------- #
def main() -> None:
    rows, p_rows = load_rows()
    variants = sorted({r[_VARIANT_KEY] for r in rows})
    modes = sorted({r["mode"] for r in rows})

    figure = frontier_figure(rows)

    summary = []
    for variant in variants:
        entry: dict[str, Any] = {_VARIANT_KEY: variant}
        for mode in ("batch", "sequential"):
            best = _best_by(rows, _VARIANT_KEY, variant, mode)
            entry[f"{mode}_n"] = best["neurons"] if best else float("nan")
            entry[f"{mode}_h1"] = best["rel_h1"] if best else float("nan")
            entry[f"{mode}_score"] = best["score"] if best else float("nan")
        summary.append(entry)
    summary.sort(key=lambda e: (np.isnan(e["batch_score"]), e["batch_score"]))

    # Termination: the paper loop stops when no candidate clears the threshold.
    by_mode = {}
    for mode in modes:
        of_mode = [r for r in rows if r["mode"] == mode]
        early = [r for r in of_mode if 0 < r["iterations"] < r["requested_iterations"]]
        iters = sorted(r["iterations"] for r in of_mode)
        by_mode[mode] = (
            len(early), len(of_mode),
            float(np.median(iters)) if iters else float("nan"),
            max(iters) if iters else 0,
            of_mode[0]["requested_iterations"] if of_mode else 0,
        )

    full = sorted(
        (r for r in rows if r["loss"] == "h1"),
        key=lambda r: (r[_VARIANT_KEY], r["mode"], r["alpha"], r["gamma"]),
    )

    lines: list[str] = []
    lines.append(f"# Results — {PROBLEM} / {FAMILY}\n")
    lines.append(
        "Generated by `analysis.py`. Scope, sweep axes, and fixed settings are "
        "documented in `README.md`.\n"
    )

    lines.append("## Key finding\n")
    best_overall = {m: min((r for r in rows if r["mode"] == m and r["loss"] == "h1"),
                           key=lambda r: r["score"], default=None) for m in modes}
    most_accurate = {m: min((r for r in rows if r["mode"] == m and r["loss"] == "h1"),
                            key=lambda r: r["rel_h1"], default=None) for m in modes}
    for mode, best in best_overall.items():
        if best is None:
            continue
        acc = most_accurate[mode]
        lines.append(
            f"- **{mode}**: sparsest good fit is "
            f"`{_VARIANT_LABEL}={best[_VARIANT_KEY]}`, "
            f"α={best['alpha']:g}, γ={best['gamma']:g} — "
            f"{best['neurons']} neurons at relative H¹ {best['rel_h1']:.4f} "
            f"(score {best['score']:.3f}). Most accurate fit is "
            f"`{_VARIANT_LABEL}={acc[_VARIANT_KEY]}`, α={acc['alpha']:g}, "
            f"γ={acc['gamma']:g} — relative H¹ {acc['rel_h1']:.4f} at "
            f"{acc['neurons']} neurons."
        )
    lines.append("")
    lines.append(
        "The score `H¹ × neurons` rewards very sparse fits, so it and the accuracy "
        "envelope pick different cells; both are reported rather than one standing "
        "in for the other.\n"
    )
    lines.append(
        f"![frontier](figures/{figure.name})\n\n"
        "*Running best relative H¹ validation error against neuron count, for the "
        "two insertion modes. Batch admits up to `N_ins = 15` atoms per outer "
        "iteration against one frozen residual; sequential admits one — the "
        "maximizer of `|P_p|` — and runs at `T_out = 150` for a matched neuron "
        "budget. All cells at seed 42.*\n"
    )

    lines.append("## Termination\n")
    lines.append(
        "The paper loop stops as soon as no candidate clears the insertion "
        "threshold; a run that stops early converged in the algorithm's own terms, "
        "one that reaches `T_out` was still finding candidates.\n"
    )
    for mode, (early, total, med, mx, req) in sorted(by_mode.items()):
        lines.append(
            f"- **{mode}**: {early}/{total} cells terminated before `T_out = {req}`; "
            f"median {med:g} iterations, max {mx}."
        )
    lines.append("")

    lines.append("## Best cell per variant (H¹ loss)\n")
    lines.append(format_table(
        summary,
        [_VARIANT_KEY, "batch_n", "batch_h1", "batch_score",
         "sequential_n", "sequential_h1", "sequential_score"],
        headers={
            _VARIANT_KEY: _VARIANT_LABEL,
            "batch_n": "batch N", "batch_h1": "batch H¹", "batch_score": "batch score",
            "sequential_n": "seq N", "sequential_h1": "seq H¹", "sequential_score": "seq score",
        },
        formats={
            "batch_h1": "{:.4f}", "sequential_h1": "{:.4f}",
            "batch_score": "{:.3f}", "sequential_score": "{:.3f}",
            "batch_n": "{:.0f}", "sequential_n": "{:.0f}",
        },
    ))
    lines.append("")

    if p_rows:
        lines.append("## Moment order p\n")
        lines.append(
            "`p` sets the weight `w_p(ω) = 1 + |ω|^p` inside the penalty and drives the theorem "
            "search radius through the exponent `1/(p − s₁)`; on this data the "
            "radius binds inside the `exp(5)` clamp only at `p = 4` "
            "(see `docs/adr/0006`).\n"
        )
        p_summary = []
        for variant in sorted({r["activation"] for r in p_rows}):
            for p in sorted({r["moment_order"] for r in p_rows}):
                cand = [r for r in p_rows
                        if r["activation"] == variant and r["moment_order"] == p]
                if not cand:
                    continue
                best = min(cand, key=lambda r: r["score"])
                p_summary.append({
                    "activation": variant, "p": p, "neurons": best["neurons"],
                    "rel_h1": best["rel_h1"], "score": best["score"],
                })
        lines.append(format_table(
            p_summary, ["activation", "p", "neurons", "rel_h1", "score"],
            headers={"rel_h1": "rel H¹", "p": "p"},
            formats={"p": "{:g}", "rel_h1": "{:.4f}", "score": "{:.3f}", "neurons": "{:.0f}"},
        ))
        lines.append("")

    lines.append("## Full result (H¹ loss)\n")
    lines.append(format_table(
        full,
        [_VARIANT_KEY, "mode", "alpha", "gamma", "neurons", "rel_l2", "rel_h1", "score"],
        headers={_VARIANT_KEY: _VARIANT_LABEL, "alpha": "α", "gamma": "γ",
                 "rel_l2": "rel L²", "rel_h1": "rel H¹"},
        formats={"alpha": "{:g}", "gamma": "{:g}", "rel_l2": "{:.4f}",
                 "rel_h1": "{:.4f}", "score": "{:.3f}", "neurons": "{:.0f}"},
    ))
    lines.append("")

    (OUTPUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'results.md'} ({len(rows)} cells, {len(p_rows)} p-study cells)")


if __name__ == "__main__":
    main()
