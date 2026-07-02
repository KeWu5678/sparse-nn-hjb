#!/usr/bin/env python3
"""Neuron / H1 frontier (VDP): best relative H1 error achievable at a given
network size, for four fixed model + penalty choices (signed model, alpha=1e-5).

Every PDAP run grows the network atom by atom, and ``History`` stores the
per-iteration relative H1 error and the atom set. So each run is already a
trajectory of (neurons, H1) points; pooling those points per series and taking
the cumulative minimum gives "best H1 achievable with <= n neurons" — the
lower-envelope frontier plotted here.

All four series are signed model, H1 loss, and the SAME alpha = 1e-5. The two
log-penalty series use gamma = 10. Legend penalty symbols follow the insertion
of each series' data (profile -> phi, finite_step -> psi); see
``src.plots.plot_neuron_h1_frontier``. Pass the dataset as argv[1]
(``vdp`` default, or ``pendulum``):

  1. baseline   relu,    power=1, gamma=0    -> convex L1 penalty
  2. ReLU^k     relu,    power=K, gamma=0    -> fractional penalty (psi_k)
  3. softplus   softplus, gamma=10           -> non-convex log penalty (phi/psi_gamma)
  4. gaussian   gaussian, gamma=10           -> non-convex log penalty (phi/psi_gamma)

The ReLU^k exponent is the module constant ``K`` (currently 3).

For VDP all four sweeps are finite_step; for pendulum the softplus/gaussian
(activationsearch) sweep is profile and the relu^k (penaltypowers) sweep is
finite_step. Rendering + legend convention live in ``src.plots``.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plots import frontier_penalty_label, plot_neuron_h1_frontier

MULTIRUN = REPO_ROOT / "rawdata" / "logs" / "multirun"
OUTPUT_DIR = Path(__file__).resolve().parent

_H1_LOSS = [1.0, 1.0]
ALPHA = 1e-5   # shared regularization strength for all four series
GAMMA = 10.0   # log-penalty non-convexity for the softplus/gaussian series
               # (best precision at alpha=1e-5 for both)
K = 3          # ReLU^k exponent for the fractional-penalty series (penaltypowers
               # supports k in {2, 2.01, 3, 4, 5}); described in the thesis text


def _run_points(json_path: Path) -> list[tuple[int, float]]:
    """Per-iteration (neurons, rel-H1-val) points from one run's History pkl."""
    pkl = json_path.parent / f"result_{json_path.stem}.pkl"
    if not pkl.exists():
        return []
    history = pickle.load(open(pkl, "rb"))
    points = []
    for iw, h1 in zip(history.inner_weights, history.err_h1_val):
        n = int(np.asarray(iw["weight"]).shape[0])
        h1 = float(h1)
        if n >= 1 and np.isfinite(h1):
            points.append((n, h1))
    return points


def collect(multirun_name: str | tuple[str, ...], keep: Callable[[dict[str, Any]], bool],
            dataset: str = "vdp") -> list[tuple[int, float]]:
    """Pool (neurons, H1) points over every <dataset> / signed / H1 / alpha=1e-5
    run of a sweep that also matches ``keep``."""
    points: list[tuple[int, float]] = []
    runs = 0
    multirun_names = (multirun_name,) if isinstance(multirun_name, str) else multirun_name
    for name in multirun_names:
        for json_path in sorted((MULTIRUN / name).glob("*/*.json")):
            cfg = json.loads(json_path.read_text(encoding="utf-8"))["config"]
            model = cfg["model"]
            if not (dataset in cfg["data"]["path"].lower()
                    and model["kind"] == "signed"
                    and list(model["loss_weights"]) == _H1_LOSS
                    and abs(float(model["alpha"]) - ALPHA) < 1e-12
                    and keep(model)):
                continue
            points.extend(_run_points(json_path))
            runs += 1
    print(f"  {', '.join(multirun_names)}: {runs} runs, {len(points)} points")
    return points


def envelope(points: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Running-best frontier: best (min) H1 achievable with <= n neurons.

    Order the per-iteration ``(neurons, H1)`` points by neuron count (keeping every
    node — duplicates and post-insertion regressions are NOT dropped) and report
    the cumulative minimum H1. So when an insertion step makes the loss stagnate or
    increase, that node is still shown, held at the best error reached so far rather
    than deleted; the curve stays monotone non-increasing.
    """
    if not points:
        return np.array([]), np.array([])
    pts = sorted(points, key=lambda nh: nh[0])  # stable: equal-n keep iteration order
    ns = np.array([n for n, _ in pts])
    h1 = np.array([h for _, h in pts], dtype=float)
    return ns, np.minimum.accumulate(h1)


def build_series(dataset: str) -> list[dict[str, Any]]:
    """The four fixed model/penalty series (all signed, H1 loss, alpha=1e-5) for a
    dataset, as plot-ready dicts (label, color, marker, ns, h1) for
    :func:`src.plots.plot_neuron_h1_frontier`.

    VDP and pendulum live in parallel ``<sweep>/<dataset>`` subdirs. The penalty
    symbol in each legend label follows the insertion of that series' data
    (profile -> phi, finite_step -> psi): the relu^k (penaltypowers) sweep is
    finite_step for both datasets, while the softplus/gaussian (activationsearch)
    sweep is finite_step for VDP and profile for pendulum.
    """
    if dataset == "vdp":
        relu_l1_sweep, powers_sweep, act_sweep = (
            ("frontier_relu_l1/vdp", "frontier_relu_l1"),
            "penaltypowers/vdp",
            "activationsearch/vdp",
        )
        act_insertion = "finite_step"
    elif dataset == "pendulum":
        relu_l1_sweep, powers_sweep, act_sweep = (
            "frontier_relu_l1/pendulum",
            "penaltypowers/pendulum",
            "activationsearch/pendulum",
        )
        act_insertion = "profile"  # pendulum activationsearch is profile-only
    else:
        raise SystemExit(f"unknown dataset {dataset!r} (expected vdp|pendulum)")

    def act_keep(activation: str) -> Callable[[dict[str, Any]], bool]:
        return lambda m: (m["activation"] == activation
                          and float(m["gamma"]) == GAMMA
                          and m["insertion"] == act_insertion)

    # (label, color, marker, points, description) — markers ▽ △ □ ○ so curves stay
    # distinguishable in grayscale / colorblind print, not by color alone.
    specs = [
        (r"baseline: $\mathrm{ReLU} + L^1$ penalty", "#808080", "v",
         collect(relu_l1_sweep,
                 lambda m: m["activation"] == "relu" and float(m["power"]) == 1
                 and float(m["gamma"]) == 0, dataset),
         "**baseline: ReLU + L1** — relu, power=1, gamma=0 (convex L1)."),
        (frontier_penalty_label(r"\mathrm{ReLU}^k", insertion="finite_step",
                                subscript="k"), "#1A4FE6", "^",
         collect(powers_sweep,
                 lambda m: m["activation"] == "relu" and float(m["power"]) == K
                 and float(m["gamma"]) == 0, dataset),
         f"**ReLU^k + alpha*psi_k** — penaltypowers, relu, power=k (k={K}), "
         "gamma=0 (fractional penalty, finite_step)."),
        (frontier_penalty_label(r"\mathrm{softplus}", insertion=act_insertion,
                                subscript=r"\gamma"), "#E8000B", "s",
         collect(act_sweep, act_keep("softplus"), dataset),
         "**softplus + alpha*phi_gamma** — activationsearch, softplus, gamma=10 "
         "(non-convex log); best classical/monotone activation."),
        (frontier_penalty_label(r"\mathrm{gaussian}", insertion=act_insertion,
                                subscript=r"\gamma"), "#9C27B0", "o",
         collect(act_sweep, act_keep("gaussian"), dataset),
         "**gaussian + alpha*phi_gamma** — activationsearch, gaussian, gamma=10 "
         "(non-convex log); a radial kernel activation (same family as the "
         "thesis's Matern 5/2)."),
    ]

    series = []
    for label, color, marker, points, description in specs:
        ns, h1 = (envelope(points) if points
                  else (np.array([]), np.array([])))
        series.append({"label": label, "color": color, "marker": marker,
                       "ns": ns, "h1": h1, "description": description})
    return series


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "vdp"
    print(f"collecting frontier points ({dataset}, signed, H1 loss, "
          f"alpha={ALPHA:g}):")
    series = build_series(dataset)

    fig_name = f"neuron_h1_frontier_{dataset}.png"
    fig_path = OUTPUT_DIR / "figures" / fig_name
    plot_neuron_h1_frontier(series, save_path=fig_path)

    insertion_note = ("**finite_step** insertion" if dataset == "vdp"
                      else "**profile** insertion (the relu^k series uses "
                           "finite_step; pendulum activationsearch is profile-only)")
    available = [s for s in series if len(s["ns"]) > 0]
    missing = [s for s in series if len(s["ns"]) == 0]
    available_lines = "\n".join(
        f"{i}. {s['description']}" for i, s in enumerate(available, start=1)
    )
    missing_note = (
        ""
        if not missing
        else "\n\nNo matching run records were found at the fixed filter for: "
        + ", ".join(s["description"].split(" — ", maxsplit=1)[0] for s in missing)
        + "."
    )
    results_name = f"results_{dataset}.md" if dataset != "vdp" else "results.md"
    (OUTPUT_DIR / results_name).write_text(
        f"# neuron_h1_frontier Results ({dataset})\n\n"
        "Best achievable relative H1 error vs. network size (number of neurons), "
        f"{dataset} data, **signed** model, H1 loss, {insertion_note}, fixed "
        "**alpha = 1e-5** for all series. The two log-penalty series use "
        "**gamma = 10**. Each curve is the per-run growth trajectory reduced to its "
        "lower envelope (cumulative-min H1 per neuron count). The available curves "
        f"in the current local run records are:\n\n{available_lines}{missing_note}\n\n"
        f"![neuron/H1 frontier ({dataset})](figures/{fig_name})\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_DIR / results_name} and {fig_path.relative_to(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
