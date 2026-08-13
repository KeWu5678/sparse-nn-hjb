#!/usr/bin/env python3
"""Plot the effect of the moment order on Algorithm 1's fitted support.

The weight ``w_p(omega) = 1 + |omega|^p`` appears inside the scalar penalty, so
the moment order changes both the fitted objective and the theorem radius.

The plotted statistic is ``radius_max`` = max|omega|, the **support radius**,
because that is the quantity the support theorem bounds. ``R_0.95`` is a
total-variation quantile and is plotted separately in ``p_study_r95.png``.
The script prints monotonicity counts from the records instead of embedding a
claim from a particular run in this source file.

Panels share the p axis, one per benchmark; per house style unrelated panels
would be separate PNGs.

Run:  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \\
          experiments/01_vdp/paper_log_penalty/p_study_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotstyle import PALETTE
from src.plotstyle import apply_publication_style as _apply_publication_style

MULTIRUN = REPO_ROOT / "rawdata" / "logs" / "multirun"
PROBLEMS = ("vdp", "pendulum")
ORDERS = (2.01, 2.5, 3.0, 4.0)
# Colour carries the activation; the two alphas are distinguished by line style,
# so the alpha-dependence of the trend is visible rather than hidden by a choice.
STYLE = {
    "softplus": (PALETTE["blue_main"], "o"),
    "tanh": (PALETTE["teal"], "s"),
    "gaussian": (PALETTE["red_strong"], "^"),
    "gelu_squared": (PALETTE["violet"], "D"),
}
ALPHA_STYLE = {1e-5: "-", 1e-4: "--"}
METRICS = {"radius_max": "max_radius", "radius_r95": "r95"}


def load(records: Path) -> dict[tuple[str, float, float], dict[str, float]]:
    out: dict[tuple[str, float, float], dict[str, float]] = {}
    for path in sorted(records.glob("**/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            model = record["config"]["model"]
            values = record["metrics"][0]["values"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        key = (model["activation"], float(model["moment_order"]), float(model["alpha"]))
        out[key] = {
            "max_radius": float(values["radius_max"]),
            "r95": float(values["radius_r95"]),
            "neurons": float(values["best_neurons"]),
            "rel_h1": float(values["rel_h1_val"]),
        }
    if not out:
        raise FileNotFoundError(f"no run records under {records}")
    return out


def _panel(ax, data: dict, metric: str) -> None:
    for activation, (color, marker) in STYLE.items():
        for alpha, linestyle in ALPHA_STYLE.items():
            ys = [data.get((activation, p, alpha), {}).get(metric) for p in ORDERS]
            if any(y is None for y in ys):
                continue
            ax.plot(ORDERS, ys, color=color, marker=marker, ls=linestyle,
                    lw=1.4, markersize=4.5, markeredgecolor="0.2",
                    markeredgewidth=0.5)
    ax.set_xticks(ORDERS)
    ax.set_xticklabels([f"{p:g}" for p in ORDERS])


def _figure(per_problem: dict[str, dict], metric: str, ylabel: str, path: Path) -> None:
    _apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for ax, problem in zip(axes, PROBLEMS):
        _panel(ax, per_problem[problem], metric)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel(r"moment order $p$")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-vdp", type=Path,
                        default=MULTIRUN / "vdp" / "paper_log_penalty" / "p_study_sequential")
    parser.add_argument("--records-pendulum", type=Path,
                        default=MULTIRUN / "pendulum" / "paper_log_penalty" / "p_study_sequential")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "figures")
    args = parser.parse_args(argv)

    per_problem = {
        "vdp": load(args.records_vdp),
        "pendulum": load(args.records_pendulum),
    }

    _figure(per_problem, "max_radius", r"$\max|\omega|$", args.out / "p_study.png")
    _figure(per_problem, "r95", r"$R_{0.95}$", args.out / "p_study_r95.png")

    # Numbers behind the figure, so the caption can quote them without re-deriving.
    def monotone(xs) -> bool:
        return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))

    counts = {"max_radius": 0, "r95": 0, "neurons": 0}
    total = 0
    for problem in PROBLEMS:
        data = per_problem[problem]
        for activation in STYLE:
            for alpha in ALPHA_STYLE:
                rows = [data.get((activation, p, alpha)) for p in ORDERS]
                if any(r is None for r in rows):
                    continue
                total += 1
                for metric in counts:
                    counts[metric] += monotone([r[metric] for r in rows])
                series = " -> ".join(f"{r['max_radius']:.2f}" for r in rows)
                print(f"  {problem:9s} {activation:13s} alpha={alpha:g}  max|w| {series}")
    print(f"\nmonotone decreasing in p, out of {total} rows:")
    for metric, hits in counts.items():
        print(f"  {metric:11s} {hits}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
