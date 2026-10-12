#!/usr/bin/env python3
"""Report the Allen-Cahn studies using saved physical-coordinate predictions."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig
from src.metric import format_table
from src.plots import plot_neuron_h1_frontier
from src.plotstyle import PALETTE, apply_publication_style
from src.results import evaluate_history, load_run, validation_samples

DATASETS = {"allen_cahn6": 6, "allen_cahn10": 10}
EXPERIMENTS = {
    "higher_dim_log_penalty": ("profile", "Algorithm 1", "blue_main", "o"),
    "higher_dim_frac_exp_penalty": ("finite_step", "Algorithm 2", "teal", "s"),
    "higher_dim_relu_l1_baseline": ("profile", "ReLU-L1 baseline", "red_strong", "^"),
}


def scientific_signature(config: dict) -> str:
    """Identify a scientific config point, excluding presentation/runtime policy."""
    point = {key: copy.deepcopy(config[key]) for key in ("model", "training", "data", "eval")}
    point["seed"] = config["env"]["seed"]
    point["training"].pop("display_every", None)

    def numeric(value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, dict):
            return {key: numeric(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [numeric(item) for item in value]
        return value

    return json.dumps(numeric(point), sort_keys=True, allow_nan=False)


def study_grid() -> tuple[dict, dict]:
    """Read expected cells from Hydra; never hardcode the sweep's cell counts."""
    ConfigStore.instance().store(name="config_schema", node=ExperimentConfig)
    parser = OverridesParser.create()
    expected, data_paths = {}, {}
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        for dataset, experiment in itertools.product(DATASETS, EXPERIMENTS):
            model = EXPERIMENTS[experiment][0]
            config = compose(
                config_name="config",
                overrides=[f"+experiment={experiment}", f"+model={model}", f"+data={dataset}"],
                return_hydra_config=True,
            )
            data_paths[dataset] = config.data.path
            axes = {}
            for key, value in config.hydra.sweeper.params.items():
                if key in ("+data", "+model"):
                    continue
                override = parser.parse_override(f"{key}={value}")
                axes[key] = (list(override.sweep_iterator()) if override.is_sweep_override()
                             else [override.value()])
            base = OmegaConf.to_container(config, resolve=False)
            cells = set()
            for values in itertools.product(*axes.values()):
                point = copy.deepcopy(base)
                for key, value in zip(axes, values):
                    section, field = key.split(".")
                    point[section][field] = value
                cells.add(scientific_signature(point))
            expected[dataset, experiment] = cells
    return expected, data_paths


def lower_envelope(neurons, errors) -> tuple[np.ndarray, np.ndarray]:
    """Minimum observed error with at most N neurons, across all supplied rows."""
    neurons, errors = np.asarray(neurons), np.asarray(errors)
    if neurons.shape != errors.shape or not np.all(np.isfinite(errors) & (errors >= 0)):
        raise ValueError("frontier inputs must have matching shapes and finite nonnegative errors")
    counts = np.unique(neurons)
    best = np.asarray([np.min(errors[neurons == count]) for count in counts])
    return counts, np.minimum.accumulate(best)


def collect(records: Path, expected: dict, data_paths: dict) -> tuple[list, dict, dict]:
    """Validate provenance, then rescore every checkpoint on its recorded holdout."""
    rows, curves = [], defaultdict(list)
    found = {key: set() for key in expected}
    seen, holdouts = set(), {}
    for path in sorted(records.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        choices = record.get("hydra", {}).get("runtime", {}).get("choices", {})
        dataset, experiment = choices.get("data"), choices.get("experiment")
        if (dataset, experiment) not in expected:
            continue
        config = record["config"]
        if config["data"]["path"] != data_paths[dataset]:
            raise ValueError(f"record does not use the currently configured dataset: {path}")
        if not record.get("data_sha256") or record.get("metric_coordinates") != "physical":
            raise ValueError(f"record needs a dataset hash and physical-coordinate metrics: {path}")
        signature = scientific_signature(config)
        identity = dataset, experiment, signature
        if identity in seen:
            raise ValueError(f"duplicate scientific config point: {path}")
        seen.add(identity)
        split = config["env"]["seed"], config["data"]["train_fraction"]
        if dataset in holdouts and split != holdouts[dataset]:
            raise ValueError(f"different validation splits for {dataset}; report them separately")
        holdouts[dataset] = split
        run = load_run(path)  # Verifies the input-byte hash and recorded normalization.
        if run.model.input_dim != DATASETS[dataset]:
            raise ValueError(f"expected {DATASETS[dataset]} state variables: {path}")
        samples = validation_samples(run)
        history = evaluate_history(run, samples)
        if not all(np.all(np.isfinite(history[key])) for key in ("rel_l2", "rel_grad", "rel_h1")):
            raise ValueError(f"nonfinite physical-coordinate validation errors: {path}")
        study = signature in expected[dataset, experiment]
        if study:
            found[dataset, experiment].add(signature)
        kind = "configured" if study else "pilot/override"
        model, training = config["model"], config["training"]
        loss = ",".join(f"{weight:g}" for weight in model["loss_weights"])
        row = {
            "dataset": dataset, "dimension": run.model.input_dim,
            "experiment": experiment, "kind": kind,
            "activation": model["activation"], "power": model["power"],
            "alpha": model["alpha"], "gamma": model["gamma"], "loss_weights": loss,
            "iterations": training["num_iterations"], "candidates": training["num_insertion"],
            "seed": split[0], "train_fraction": split[1], "validation_count": len(samples[0]),
            "best_iteration": run.iteration, "neurons": int(history["neurons"][run.iteration]),
            "rel_l2": float(history["rel_l2"][run.iteration]),
            "rel_grad": float(history["rel_grad"][run.iteration]),
            "rel_h1": float(history["rel_h1"][run.iteration]),
            "elapsed_s": record.get("elapsed_s"), "run_id": record["run_id"],
            "data_sha256": record["data_sha256"], "record": str(path.resolve()),
        }
        rows.append(row)
        curves[dataset, loss, kind, experiment].append(history)
    if not rows:
        raise ValueError(f"no completed higher-dimensional Run Records found under {records}")
    return rows, dict(curves), found


def write_report(output: Path, rows: list, curves: dict, found: dict, expected: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = [{"dataset": dataset, "experiment": experiment,
                 "completed": len(found[dataset, experiment]),
                 "expected": len(expected[dataset, experiment])}
                for dataset, experiment in expected]
    complete = all(item["completed"] == item["expected"] for item in coverage)
    lines = ["# Higher-dimensional Allen-Cahn results", "",
             f"Study status: **{'complete configured grid' if complete else 'partial study / pilot'}**.",
             "", format_table(coverage, ["dataset", "experiment", "completed", "expected"]), "",
             "Only exact configured scientific settings count toward completion; reduced budgets "
             "or other overrides are labeled `pilot/override`. Duplicate cells, changed datasets, "
             "nonphysical records, and mixed validation splits are rejected.", "",
             "Errors below are recomputed in physical coordinates. The saved checkpoint minimizes "
             "the training objective. For each dataset, loss, run kind, and algorithm, this table "
             "selects the run with the smallest validation H1 error; this is a tuning diagnostic, "
             "not an independent test estimate. All runs and their provenance are in [runs.csv](runs.csv).", ""]
    best = {}
    for row in rows:
        group = row["dataset"], row["loss_weights"], row["kind"], row["experiment"]
        if group not in best or row["rel_h1"] < best[group]["rel_h1"]:
            best[group] = row
    columns = ["dataset", "experiment", "kind", "loss_weights", "activation", "power", "alpha",
               "gamma", "neurons", "rel_l2", "rel_grad", "rel_h1", "iterations", "candidates"]
    lines += [format_table([best[key] for key in sorted(best)], columns), "",
              "Each frontier is the smallest observed validation H1 error with at most N neurons, "
              "pooled across all checkpoints and parameter choices within one algorithm. Pilot "
              "and configured runs appear in separate panels. These envelopes are descriptive "
              "and do not change the training-objective checkpoint selection.", ""]
    apply_publication_style()
    for dataset, loss, kind in sorted({key[:3] for key in curves}):
        series, arrays = [], {}
        for experiment, (_, label, color, marker) in EXPERIMENTS.items():
            histories = curves.get((dataset, loss, kind, experiment), [])
            if not histories:
                continue
            counts, errors = lower_envelope(
                np.concatenate([item["neurons"] for item in histories]),
                np.concatenate([item["rel_h1"] for item in histories]),
            )
            arrays[f"{experiment}_neurons"], arrays[f"{experiment}_h1"] = counts, errors
            series.append({"ns": counts, "h1": np.maximum(errors, np.finfo(float).tiny),
                           "label": label, "color": PALETTE[color], "marker": marker})
        stem = f"{dataset}_loss_{loss.replace(',', '_')}_{kind.replace('/', '_')}"
        figure = output / f"{stem}.png"
        plot_neuron_h1_frontier(series, save_path=figure)
        np.savez_compressed(output / f"{stem}.npz", **arrays)
        lines += [f"![Neuron/H1 frontier]({figure.name})", "",
                  f"{dataset}: loss weights ({loss}), {kind} runs. "
                  "Exact frontier arrays are saved beside the PNG as NPZ.", ""]
    path = output / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (output / "coverage.json").write_text(
        json.dumps({"complete": complete, "cells": coverage}, indent=2) + "\n", encoding="utf-8",
    )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "rawdata/logs/multirun")
    parser.add_argument("--output", type=Path, default=ROOT / "rawdata/reports/allen_cahn")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail before writing unless every configured sweep cell is present")
    args = parser.parse_args(argv)
    expected, paths = study_grid()
    rows, curves, found = collect(args.records, expected, paths)
    if args.require_complete and any(found[key] != expected[key] for key in expected):
        raise ValueError("the configured higher-dimensional study is incomplete")
    print(write_report(args.output, rows, curves, found, expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
