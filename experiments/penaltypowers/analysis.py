#!/usr/bin/env python3
"""Analyse the penaltypowers experiment.

Reads the per-run JSON records written by ``scripts/train.py`` under the Hydra
multirun directory and produces a Markdown results table and figure under
``experiments/penaltypowers/``. Selection rule: the best (lowest-score)
gamma per (data, activation, power, loss, seed).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import ACTIVATIONS
from src.data import load_value_samples, ValueSampleNormalizer
from src.metric import format_table
from src.plots import plot_model_value_surface, plot_score_tradeoff

EXPERIMENT = "penaltypowers"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}


def _result_pkl(json_path: Path) -> Path:
    """The fit-result pickle (a ``History``) sits beside each run's JSON record."""
    return json_path.parent / f"result_{json_path.stem}.pkl"


def _dataset_label(path: str) -> str:
    p = path.lower()
    if "pendulum" in p:
        return "pendulum"
    if "vdp" in p or "van_der_pol" in p:
        return "vdp"
    return Path(path).stem


def homogeneous_activation_names() -> list[str]:
    return [name for name, (_, use_sphere) in ACTIVATIONS.items() if use_sphere]


def load_rows() -> list[dict[str, Any]]:
    """Read Hydra multirun records: axes from config, metrics from the run record."""
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} - run `make penaltypowers` first"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        metrics = record["metrics"][0]["values"]
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        neurons = int(metrics["best_neurons"])
        val_h1 = float(metrics["rel_h1_val"])
        rows.append({
            "data": _dataset_label(cfg["data"]["path"]),
            "activation": model["activation"],
            "power": float(model["power"]), "loss": loss,
            "gamma": float(model["gamma"]), "seed": int(record["config"]["env"]["seed"]),
            "neurons": neurons, "val_l2": float(metrics["rel_l2_val"]),
            "val_h1": val_h1, "score": val_h1 * max(neurons, 1),
            "kind": model.get("kind", ""), "data_file": cfg["data"]["path"],
            "result_path": str(_result_pkl(path)),
        })
    return rows


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest-score row for each (data, activation, power, loss, seed)."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["data"], row["activation"], row["power"], row["loss"], row["seed"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda row: row["score"]))
    return best


def coverage_summary(rows: list[dict[str, Any]]) -> str:
    expected_data = ["vdp", "pendulum"]
    expected_activations = homogeneous_activation_names()
    observed_data = sorted({row["data"] for row in rows})
    observed_activations = sorted({row["activation"] for row in rows})
    missing_data = [value for value in expected_data if value not in observed_data]
    missing_activations = [
        value for value in expected_activations if value not in observed_activations
    ]

    def line(label: str, values: list[str]) -> str:
        return f"- {label}: {', '.join(values) if values else 'none'}"

    return "\n".join(
        [
            "## Sweep coverage",
            "",
            line("Configured data", expected_data),
            line("Observed data", observed_data),
            line("Missing configured data", missing_data),
            line("Configured homogeneous activations", expected_activations),
            line("Observed activations", observed_activations),
            line("Missing configured activations", missing_activations),
        ]
    )


def _plot_relu_surfaces(relu_rows: list[dict[str, Any]], data_name: str) -> str:
    """Grid of learned V(x) surfaces, one panel per ReLU run. Returns the figure
    path relative to OUTPUT_DIR for embedding in the Markdown."""
    # max-abs scales are recomputed from the dataset (not stored per-run); fitting
    # on the full samples matches train.py, which normalizes before the split.
    samples = load_value_samples(relu_rows[0]["data_file"])
    normalizer = ValueSampleNormalizer.fit(samples)

    n = len(relu_rows)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))
    for i, row in enumerate(relu_rows):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_model_value_surface(
            row["result_path"], activation="relu", power=row["power"],
            x_scale=normalizer.x_scale, v_scale=normalizer.v_scale,
            dataset=samples, ax=ax, colorbar=False, show=False,
            title=f"p={row['power']:g}  ·  {row['neurons']} neurons  ·  H1={row['val_h1']:.2e}",
        )
    fig.suptitle(f"penaltypowers ({data_name}): learned ReLU value surfaces",
                 fontsize=13, y=1.02)
    rel_path = Path("figures") / f"relu_surfaces_{data_name}.png"
    out_path = OUTPUT_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rel_path.as_posix()


def relu_section(best: list[dict[str, Any]]) -> str:
    """A ReLU-only section: best-gamma-per-power table plus the learned V(x)
    surface for every ReLU run (signed model, H1 loss), per dataset."""
    blocks = []
    for data_name in sorted({row["data"] for row in best}):
        relu_rows = sorted(
            [
                row for row in best
                if row["data"] == data_name
                and row["activation"] == "relu"
                and row["loss"] == "h1"
            ],
            key=lambda row: row["power"],
        )
        if not relu_rows:
            continue
        table = format_table(
            relu_rows,
            ["power", "gamma", "neurons", "val_h1", "score"],
            headers={"val_h1": "Val H1"},
            formats={"power": "{:g}", "gamma": "{:g}",
                     "val_h1": "{:.2e}", "score": "{:.2e}"},
            title=f"ReLU on {data_name} (H1 loss) — best gamma per power",
        )
        fig_rel = _plot_relu_surfaces(relu_rows, data_name)
        blocks.append(
            f"### {data_name}\n\n{table}\n\n"
            f"![learned ReLU value surfaces]({fig_rel})\n"
        )
    if not blocks:
        return ""
    return (
        "## ReLU activation function\n\n"
        "Dedicated view of the `relu` runs (signed model, H1 loss): the best-gamma "
        "row per penalty power, and the learned value surface V(x) for each — the "
        "fitted network evaluated on the physical state plane.\n\n"
        + "\n".join(blocks)
    )


def main() -> int:
    rows = load_rows()
    best = best_per_cell(rows)
    best.sort(key=lambda row: (
        row["data"], row["activation"], row["power"], row["loss"], row["score"],
    ))

    def loss_table(loss: str) -> str:
        return format_table(
            [row for row in best if row["loss"] == loss],
            ["data", "activation", "power", "seed", "gamma", "neurons", "val_h1", "score"],
            headers={"val_h1": "Val H1"},
            formats={"power": "{:g}", "gamma": "{:g}",
                     "val_h1": "{:.2e}", "score": "{:.2e}"},
            title=f"Best gamma per data/activation/power/seed ({loss} loss)",
        )

    tables = (
        "## All activations — best gamma per cell\n\n"
        f"### H1 loss\n\n{loss_table('h1')}\n\n"
        f"### L2 loss\n\n{loss_table('l2')}\n"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    relu_md = relu_section(best)
    (OUTPUT_DIR / "results.md").write_text(
        f"# {EXPERIMENT} Results\n\n{coverage_summary(rows)}\n\n{relu_md}\n\n{tables}\n",
        encoding="utf-8",
    )

    plot_activations = {"relu"}
    for data_name in sorted({row["data"] for row in best}):
        subset = [
            row for row in best
            if row["data"] == data_name
            and row["loss"] == "h1"
            and row["activation"] in plot_activations
        ]
        if not subset:
            continue
        plot_score_tradeoff(
            [dict(row, power_label=f"p={row['power']:g}") for row in subset],
            x="neurons", y="val_h1", label="power_label", color="activation",
            title=f"penaltypowers ({data_name}): sparsity/accuracy tradeoff",
            xlabel="Neurons", ylabel="Validation H1 (h1 loss)",
            save_path=OUTPUT_DIR / "figures" / f"tradeoff_{data_name}.png",
        )
    print(f"wrote {OUTPUT_DIR / 'results.md'} and figures/tradeoff_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
