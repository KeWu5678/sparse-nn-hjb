#!/usr/bin/env python3
"""Generate the current-paper Van der Pol evidence for both algorithms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "01_vdp" / "paper_log_penalty"
ALGORITHM1_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "paper_log_penalty" / "sequential"
)
HOMOGENEOUS_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "paper_frac_exp_penalty" / "sequential"
)
TRADITIONAL_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "vdp" / "paper_frac_exp_penalty" / "relu_l1"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ValueSampleNormalizer, load_value_samples

REPRESENTATIVES = {
    "tanh": {"alpha": 1e-4, "order": 2.01},
    "softplus": {"alpha": 1e-4, "order": 2.01},
    "gaussian": {"alpha": 1e-4, "order": 2.01},
}

# Shared current-paper operating point for the cross-activation figures.
SELECTED_GAMMA = 10.0
# None => pick the lowest-H1 run at each power instead of pinning alpha.
HOMOGENEOUS_ALPHA: float | None = 1e-5


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
    for path in sorted(ALGORITHM1_ROOT.glob("**/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        model = record["config"]["model"]
        selection = REPRESENTATIVES.get(str(model["activation"]))
        if selection is None:
            continue
        if not (
            _close(float(model["alpha"]), selection["alpha"])
            and _close(float(model["moment_order"]), selection["order"])
            and model.get("objective") == "normalized_moment"
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
    gamma: float | None = None,
) -> dict[str, Any]:
    if gamma is None:
        gamma = SELECTED_GAMMA
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
            and (HOMOGENEOUS_ALPHA is None
                 or _close(float(model["alpha"]), HOMOGENEOUS_ALPHA))
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
    if not matches:
        raise ValueError(f"no ReLU^{power:g} champion found under {HOMOGENEOUS_ROOT}")
    if HOMOGENEOUS_ALPHA is not None and len(matches) != 1:
        raise ValueError(f"expected one ReLU^{power:g} champion, got {len(matches)}")
    # With alpha unpinned, the champion is the lowest-H1 run at this power.
    return min(matches, key=lambda m: m["rel_h1"])


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
            row = _selected(rows, activation, loss=loss)
            lines.append(
                f"| {activation} | {loss} | {SELECTED_GAMMA:g} | "
                f"{_format_float(row['rel_l2'])} | "
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=None,
                        help="Algorithm 1 sequential run-record root")
    parser.add_argument("--homogeneous-records", type=Path, default=None,
                        help="Algorithm 2 sequential run-record root")
    parser.add_argument("--traditional-records", type=Path, default=None,
                        help="ReLU plus l1 run-record root")
    parser.add_argument("--out", type=Path, default=None,
                        help="study directory the figures are written under")
    parser.add_argument("--alpha", type=float, default=None,
                        help="shared alpha for the cross-activation figures")
    parser.add_argument("--gamma", type=float, default=None,
                        help="shared gamma for the cross-activation figures")
    parser.add_argument("--order", type=float, default=None,
                        help="shared moment order p")
    parser.add_argument("--homogeneous-alpha", type=float, default=None,
                        help="pin Algorithm 2's alpha; omit with --records to take "
                             "the lowest-H1 run at each power")
    parser.add_argument("--free-homogeneous-alpha", action="store_true",
                        help="choose Algorithm 2 runs by lowest H1 instead of a pinned alpha")
    return parser.parse_args(argv)


def _apply_overrides(args: argparse.Namespace) -> None:
    """Repoint the module-level roots/selection; globals resolve at call time."""
    global ALGORITHM1_ROOT, HOMOGENEOUS_ROOT, TRADITIONAL_ROOT, OUTPUT_DIR, REPRESENTATIVES
    global SELECTED_GAMMA, HOMOGENEOUS_ALPHA

    if args.records is not None:
        ALGORITHM1_ROOT = args.records.resolve()
    if args.homogeneous_records is not None:
        HOMOGENEOUS_ROOT = args.homogeneous_records.resolve()
    if args.traditional_records is not None:
        TRADITIONAL_ROOT = args.traditional_records.resolve()
    if args.out is not None:
        OUTPUT_DIR = args.out.resolve()
    if args.gamma is not None:
        SELECTED_GAMMA = args.gamma
    if args.free_homogeneous_alpha:
        HOMOGENEOUS_ALPHA = None
    elif args.homogeneous_alpha is not None:
        HOMOGENEOUS_ALPHA = args.homogeneous_alpha

    if any(v is not None for v in (args.alpha, args.order)):
        REPRESENTATIVES = {
            activation: {
                "alpha": args.alpha if args.alpha is not None else spec["alpha"],
                "order": args.order if args.order is not None else spec["order"],
            }
            for activation, spec in REPRESENTATIVES.items()
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _apply_overrides(args)

    rows = load_representative_rows()
    figures = _load_module(
        "vdp_paper_figures",
        ROOT / "scripts" / "paper" / "vdp_figures.py",
    )
    figures.OUTPUT_DIR = OUTPUT_DIR
    # The loaded module reads its own fixed parameters when selecting the
    # run behind each surface/feedback figure; keep it in step with ours.
    figures._FIXED_ALPHA = REPRESENTATIVES["softplus"]["alpha"]
    figures._FIXED_GAMMA = SELECTED_GAMMA
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    data_file = _selected(rows, "softplus")["data_file"]
    samples = load_value_samples(data_file)
    norm = ValueSampleNormalizer.fit(samples)

    activation_shape = figures.plot_activation_shapes()
    surfaces = figures.plot_value_surfaces(rows, samples, norm)
    derivatives = figures.plot_derivative_distribution(rows, samples, norm)
    algorithm1_feedback, algorithm1_costs = figures.plot_control_synthesis(
        rows, samples, norm
    )
    gaussian = _selected(rows, "gaussian")
    softplus = _selected(rows, "softplus")
    insertion = (
        "### Sequential insertion and pruning\n\n"
        "Each outer iteration retains at most one candidate satisfying "
        "`|P_p(ω)| > α L_φ`. The guarded coefficient correction and pruning "
        "then determine the recorded support, so its size may increase by one, "
        "stay unchanged, or decrease; a negative change is pruning, not a "
        "negative insertion. At the shared operating point, Gaussian uses "
        f"{gaussian['neurons']} atoms and softplus {softplus['neurons']} atoms "
        f"at relative H1 {gaussian['rel_h1']:.4f} and "
        f"{softplus['rel_h1']:.4f}, respectively."
    )

    summary = _load_module(
        "vdp_summary_scope",
        ROOT / "scripts" / "paper" / "vdp_summary_figures.py",
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
    spec = next(iter(REPRESENTATIVES.values()))
    selection_note = (
        "All Algorithm 1 rows use shared parameters "
        f"(alpha={spec['alpha']:.0e}, gamma={SELECTED_GAMMA:g}, "
        f"p={spec['order']:g}), so the cross-activation comparison is not "
        "confounded with per-activation tuning."
    )
    report = f"""# Van der Pol full evidence scope

{selection_note}
Algorithm 2 rows use the sphere formulation.

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
