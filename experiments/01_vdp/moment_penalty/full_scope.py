#!/usr/bin/env python3
"""Regenerate the original Van der Pol evidence scope for the moment model.

The moment sweep owns model selection.  This script restores the downstream
tests from the original paper using the selected positive-beta checkpoints:
activation surfaces, derivative diagnostics, insertion dynamics, Algorithm 1
feedback, and the Algorithm 1/Algorithm 2 summary figures.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
MOMENT_ROOT = ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "moment_penalty"
HOMOGENEOUS_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "frac_exp_penalty"
)
TRADITIONAL_ROOT = ROOT / "rawdata" / "logs" / "multirun" / "frontier_relu_l1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ValueSampleNormalizer, load_value_samples

REPRESENTATIVES = {
    "tanh": {"alpha": 1e-5, "beta": 1e-10, "order": 2.01},
    "softplus": {"alpha": 1e-5, "beta": 1e-10, "order": 2.01},
    "gaussian": {"alpha": 1e-5, "beta": 1e-10, "order": 3.0},
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load analysis module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)


def _artifact_path(record: dict[str, Any], record_path: Path) -> Path:
    artifacts = {
        artifact["name"]: Path(artifact["path"])
        for artifact in record.get("artifacts", [])
    }
    path = artifacts.get("fit_history")
    local_path = record_path.parent / f"result_{record_path.stem}.pkl"
    if path is None or not path.exists():
        path = local_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _row(record: dict[str, Any], path: Path) -> dict[str, Any]:
    cfg = record["config"]
    model = cfg["model"]
    values = record["metrics"][-1]["values"]
    weights = tuple(float(value) for value in model["loss_weights"])
    loss = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}[weights]
    return {
        "kind": model["kind"],
        "insertion": model["insertion"],
        "activation": model["activation"],
        "loss": loss,
        "gamma": float(model["gamma"]),
        "alpha": float(model["alpha"]),
        "beta": float(model["moment_beta"]),
        "order": float(model["moment_order"]),
        "seed": int(cfg["env"]["seed"]),
        "neurons": int(values["best_neurons"]),
        "rel_h1": float(values["rel_h1_val"]),
        "rel_l2": float(values["rel_l2_val"]),
        "radius_r95": float(values["radius_r95"]),
        "score": float(values["rel_h1_val"])
        * max(int(values["best_neurons"]), 1),
        "data_file": cfg["data"]["path"],
        "result_path": str(_artifact_path(record, path)),
    }


def load_representative_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(MOMENT_ROOT.glob("**/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        model = record["config"]["model"]
        selection = REPRESENTATIVES.get(str(model["activation"]))
        if selection is None:
            continue
        if not (
            _close(float(model["alpha"]), selection["alpha"])
            and _close(float(model["moment_beta"]), selection["beta"])
            and _close(float(model["moment_order"]), selection["order"])
        ):
            continue
        rows.append(_row(record, path))

    expected = {
        (activation, loss, gamma)
        for activation in REPRESENTATIVES
        for loss in ("l2", "h1")
        for gamma in (0.0, 0.1, 1.0, 10.0)
    }
    actual = {
        (row["activation"], row["loss"], row["gamma"])
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            "representative follow-up is incomplete or duplicated: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, rows={len(rows)}"
        )
    return rows


def _selected(
    rows: list[dict[str, Any]],
    activation: str,
    *,
    loss: str = "h1",
    gamma: float = 1.0,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["activation"] == activation
        and row["loss"] == loss
        and _close(row["gamma"], gamma)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {activation}/{loss}/gamma={gamma} row, got {len(matches)}"
        )
    return matches[0]


def _homogeneous_champion(power: float) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for path in sorted(HOMOGENEOUS_ROOT.glob("**/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        if not (
            record.get("status") == "completed"
            and model["kind"] == "signed"
            and model["insertion"] == "finite_step"
            and model["activation"] == "relu"
            and tuple(float(value) for value in model["loss_weights"])
            == (1.0, 1.0)
            and _close(float(model["power"]), power)
            and _close(float(model["alpha"]), 1e-5)
            and _close(float(model["gamma"]), 0.0)
            and int(cfg["env"]["seed"]) == 42
        ):
            continue
        values = record["metrics"][-1]["values"]
        matches.append(
            {
                "rel_h1": float(values["rel_h1_val"]),
                "result_path": str(_artifact_path(record, path)),
                "activation": "relu",
                "power": power,
                "data_file": cfg["data"]["path"],
            }
        )
    if len(matches) != 1:
        raise ValueError(f"expected one ReLU^{power:g} champion, got {len(matches)}")
    return matches[0]


def _traditional_relu_champion(data_file: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for path in sorted(TRADITIONAL_ROOT.glob("**/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or "config" not in record:
            continue
        cfg = record["config"]
        model = cfg["model"]
        if not (
            cfg["data"]["path"] == data_file
            and model["kind"] == "signed"
            and model["activation"] == "relu"
            and tuple(float(value) for value in model["loss_weights"])
            == (1.0, 1.0)
            and _close(float(model["power"]), 1.0)
            and _close(float(model["alpha"]), 1e-5)
            and _close(float(model["gamma"]), 0.0)
            and int(cfg["env"]["seed"]) == 42
        ):
            continue
        matches.append(
            {
                "result_path": str(_artifact_path(record, path)),
                "activation": "relu",
                "power": 1.0,
                "data_file": data_file,
            }
        )
    if len(matches) != 1:
        raise ValueError(f"expected one traditional ReLU fit, got {len(matches)}")
    return matches[0]


def _plot_intro_frontier(
    summary: ModuleType,
    champions: dict[str, dict[str, Any]],
    data_file: str,
) -> str:
    from src.plots import plot_neuron_h1_frontier
    from src.plotstyle import PALETTE

    selected = {
        "traditional": _traditional_relu_champion(data_file),
        "softplus": champions["softplus"],
        "gaussian": champions["gaussian"],
        "relu2": champions["relu2"],
    }
    styles = {
        "traditional": (r"ReLU $+\ell^1$", PALETTE["neutral"], "v", "--"),
        "softplus": ("softplus", PALETTE["blue_main"], "s", "-"),
        "gaussian": ("Gaussian", PALETTE["red_strong"], "o", ":"),
        "relu2": (r"ReLU$^2$", PALETTE["violet"], "^", "-."),
    }
    series = []
    for name in ("traditional", "softplus", "gaussian", "relu2"):
        ns, h1 = summary._trajectory(selected[name]["result_path"])
        label, color, marker, linestyle = styles[name]
        series.append(
            {
                "ns": ns,
                "h1": h1,
                "label": label,
                "color": color,
                "marker": marker,
                "ls": linestyle,
            }
        )
    path = OUTPUT_DIR / "figures" / "frontier_intro.png"
    plot_neuron_h1_frontier(series, save_path=path)
    return f"figures/{path.name}"


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _loss_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| activation | loss | gamma | rel L2 | rel H1 | N | R95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for activation in REPRESENTATIVES:
        for loss in ("l2", "h1"):
            row = _selected(rows, activation, loss=loss, gamma=1.0)
            lines.append(
                f"| {activation} | {loss} | 1 | {_format_float(row['rel_l2'])} | "
                f"{_format_float(row['rel_h1'])} | {row['neurons']} | "
                f"{row['radius_r95']:.3g} |"
            )
    return "\n".join(lines)


def _cost_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| controller | N | stabilizes | closed-loop cost |",
        "|---|---:|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['controller']} | {row['neurons']} | {row['stabilizes']} | "
            f"{row['cost']:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    rows = load_representative_rows()
    legacy = _load_module(
        "vdp_log_penalty_scope",
        ROOT / "experiments" / "01_vdp" / "log_penalty" / "analysis.py",
    )
    legacy.OUTPUT_DIR = OUTPUT_DIR

    data_file = _selected(rows, "softplus")["data_file"]
    samples = load_value_samples(data_file)
    norm = ValueSampleNormalizer.fit(samples)

    activation_shape = legacy.plot_activation_shapes()
    surfaces = legacy.plot_value_surfaces(rows, samples, norm)
    derivatives = legacy.plot_derivative_distribution(rows, samples, norm)
    algorithm1_feedback, algorithm1_costs = legacy.plot_control_synthesis(
        rows, samples, norm
    )
    insertion = legacy._insertion_section(rows)
    insertion = (
        insertion.replace(
            "`J = L(μ) + α·Φ(μ)`",
            "`J = L(μ) + α·Φ(μ) + β·Ψ_p(μ)`",
        )
        .replace(
            "clears the threshold α",
            "clears the weighted insertion threshold",
        )
        .replace(
            "fewer ω exceed α",
            "fewer ω clear the weighted insertion threshold",
        )
    )

    summary = _load_module(
        "vdp_summary_scope",
        ROOT / "experiments" / "01_vdp" / "summary" / "analysis.py",
    )
    summary.OUTPUT_DIR = OUTPUT_DIR
    summary.FIG = OUTPUT_DIR / "figures"
    summary._FRONTIER_LABEL.update(
        {
            "tanh": r"$\tanh$",
            "softplus": "softplus",
            "gaussian": "Gaussian",
            "relu2": r"ReLU$^2$",
            "relu5": r"ReLU$^5$",
        }
    )
    champions = {
        activation: {
            "rel_h1": _selected(rows, activation)["rel_h1"],
            "result_path": _selected(rows, activation)["result_path"],
            "activation": activation,
            "power": 1.0,
            "data_file": data_file,
        }
        for activation in REPRESENTATIVES
    }
    champions["relu2"] = _homogeneous_champion(2.0)
    champions["relu5"] = _homogeneous_champion(5.0)

    _plot_intro_frontier(summary, champions, data_file)
    frontier = summary.plot_frontier(champions)
    feedback_state, feedback_control, summary_costs = summary.plot_feedback(
        champions, samples, norm
    )
    raw_weights = summary.plot_weights_raw3d(champions)

    surface_order = ("softplus", "tanh", "gaussian")
    surface_row = " | ".join(
        f"![{activation}]({surfaces[activation]})"
        for activation in surface_order
    )
    derivative_row = " | ".join(
        f"![{activation}]({derivatives['figures'][activation]})"
        for activation in REPRESENTATIVES
    )
    raw_weight_row = " | ".join(
        f"![{key}]({raw_weights[key]})"
        for key in ("gaussian", "softplus", "relu5")
    )
    report = f"""# Van der Pol full evidence scope with a parameter moment

All Algorithm 1 rows use selected interior positive-beta configurations.
Algorithm 2 rows reuse the unchanged homogeneous experiment.

## Algorithm 1: gradient augmentation

{_loss_table(rows)}

![representative activation shapes]({activation_shape})

| softplus | tanh | gaussian |
|---|---|---|
| {surface_row} |

| tanh | softplus | gaussian |
|---|---|---|
| {derivative_row} |

{insertion}

## Algorithm 1: synthesized feedback

![Algorithm 1 feedback]({algorithm1_feedback})

{_cost_table(algorithm1_costs)}

## Algorithm 1 versus Algorithm 2

![error-support frontier]({frontier})

| state norm | control magnitude |
|---|---|
| ![state]({feedback_state}) | ![control]({feedback_control}) |

{_cost_table(summary_costs)}

| gaussian | softplus | ReLU5 |
|---|---|---|
| {raw_weight_row} |
"""
    output = OUTPUT_DIR / "full_scope.md"
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")
    print(f"Algorithm 1 feedback: {algorithm1_feedback}")
    for row in algorithm1_costs:
        print(
            f"  {row['controller']}: N={row['neurons']}, "
            f"stabilizes={row['stabilizes']}, cost={row['cost']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
