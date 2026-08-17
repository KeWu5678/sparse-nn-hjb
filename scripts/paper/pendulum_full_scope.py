#!/usr/bin/env python3
"""Generate the current-paper pendulum evidence for both algorithms."""

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
OUTPUT_DIR = ROOT / "experiments" / "02_pendulum" / "paper_log_penalty"
HOMOGENEOUS_OUTPUT_DIR = (
    ROOT / "experiments" / "02_pendulum" / "paper_frac_exp_penalty"
)
ALGORITHM1_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "paper_log_penalty" / "sequential"
)
HOMOGENEOUS_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "paper_frac_exp_penalty" / "sequential"
)
TRADITIONAL_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "paper_frac_exp_penalty" / "relu_l1"
)
OVERSAMPLE_ALGORITHM1_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "paper_log_penalty" / "oversampling"
)
OVERSAMPLE_ALGORITHM2_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "paper_frac_exp_penalty" / "oversampling"
)
OPERATING_POINT = {"alpha": 1e-4, "gamma": 10.0, "order": 2.01}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import format_table
from src.plotstyle import PALETTE

REPRESENTATIVES = ("tanh", "softplus", "gaussian")


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


def _region_values(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return physical-coordinate region metrics; the sidecar is mandatory."""
    sidecar = path.parent / f"region_rescored_{record.get('run_id', path.stem)}.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"missing physical-coordinate region sidecar for {path}: {sidecar}"
        )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _algorithm1_row(record: dict[str, Any], path: Path) -> dict[str, Any]:
    cfg = record["config"]
    model = cfg["model"]
    values = record["metrics"][-1]["values"]
    region = _region_values(record, path)
    near_l1 = float(region["switching_l1_h1"])
    far_l1 = float(region["rest_l1_h1"])
    near_h1 = float(region["switching_h1"])
    far_h1 = float(region["rest_h1"])
    return {
        "act_name": str(model["activation"]),
        "power": float(model["power"]),
        "data_path": str(cfg["data"]["path"]),
        "kind": str(model["kind"]),
        "insertion": str(model["insertion"]),
        "activation": str(model["activation"]),
        "loss": "h1",
        "gamma": float(model["gamma"]),
        "alpha": float(model["alpha"]),
        "order": float(model["moment_order"]),
        "neurons": int(values["best_neurons"]),
        "near_l1": near_l1,
        "far_l1": far_l1,
        "l1_near/far": near_l1 / far_l1,
        "near_h1": near_h1,
        "far_h1": far_h1,
        "rel_near/far": near_h1 / far_h1,
        "bins": [
            values.get(f"distbin{index + 1}_ratio", float("nan"))
            for index in range(30)
        ],
        "cache": cfg.get("eval", {}).get("distance_cache"),
        "result_path": str(_artifact_path(record, path)),
    }


def _matches_selection(model: dict[str, Any], activation: str) -> bool:
    """Does this record supply the Algorithm 1 checkpoint for ``activation``?"""
    return (
        activation in REPRESENTATIVES
        and _close(float(model["alpha"]), OPERATING_POINT["alpha"])
        and _close(float(model["gamma"]), OPERATING_POINT["gamma"])
        and _close(float(model["moment_order"]), OPERATING_POINT["order"])
        and model.get("objective") == "normalized_moment"
    )


def load_algorithm1_rows() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ALGORITHM1_ROOT.glob("**/*.json")):
        if path.name.startswith("region_rescored_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        model = record["config"]["model"]
        activation = str(model["activation"])
        if activation not in REPRESENTATIVES:
            continue
        if not (
            model["kind"] == "signed"
            and model["insertion"] == "profile"
            and tuple(float(value) for value in model["loss_weights"])
            == (1.0, 1.0)
            and _matches_selection(model, activation)
        ):
            continue
        rows.append(_algorithm1_row(record, path))

    counts = {
        activation: sum(row["activation"] == activation for row in rows)
        for activation in REPRESENTATIVES
    }
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"expected one Algorithm 1 checkpoint per activation: {counts}")
    cache = next(row["cache"] for row in rows if row["cache"])
    for row in rows:
        row.pop("cache", None)
    return rows, str(cache)


def load_algorithm2_rows(figures: ModuleType) -> list[dict[str, Any]]:
    rows, _ = figures.load_rows()
    homogeneous = [
        row
        for row in figures.best_per_cell(rows)
        if row["activation"] in {"relu^2", "relu^3"}
        and row["kind"] == "signed"
        and row["loss"] == "h1"
    ]
    counts = {
        activation: sum(row["activation"] == activation for row in homogeneous)
        for activation in ("relu^2", "relu^3")
    }
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"expected one Algorithm 2 checkpoint per power: {counts}")
    return homogeneous


def _traditional_relu_row(data_file: str) -> dict[str, Any]:
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
                "activation": "traditional",
                "result_path": str(_artifact_path(record, path)),
            }
        )
    if len(matches) != 1:
        raise ValueError(f"expected one traditional ReLU fit, got {len(matches)}")
    return matches[0]


def _plot_frontier(
    figures: ModuleType,
    rows: list[dict[str, Any]],
    order: tuple[str, ...],
    stem: str,
) -> str:
    from src.plots import plot_neuron_h1_frontier

    labels = {
        "traditional": r"ReLU $+\ell^1$",
        "gaussian": "Gaussian",
        "softplus": "softplus",
        "tanh": r"$\tanh$",
        "relu^2": r"ReLU$^2$",
        "relu^3": r"ReLU$^3$",
    }
    markers = {
        "traditional": "v",
        "gaussian": "o",
        "softplus": "s",
        "tanh": "v",
        "relu^2": "^",
        "relu^3": "P",
    }
    styles = dict(figures._MODEL_STYLE)
    styles["traditional"] = (PALETTE["neutral"], "--")
    selected = {row["activation"]: row for row in rows}
    series = []
    for name in order:
        ns, h1 = figures._frontier_trajectory(selected[name]["result_path"])
        color, linestyle = styles[name]
        series.append(
            {
                "ns": ns,
                "h1": h1,
                "label": labels[name],
                "color": color,
                "marker": markers[name],
                "ls": linestyle,
            }
        )
    path = OUTPUT_DIR / "figures" / f"{stem}.png"
    plot_neuron_h1_frontier(series, save_path=path)
    return f"figures/{path.name}"


def _feedback_table(
    rows: list[dict[str, Any]], members: set[str], title: str
) -> str:
    def normalized(label: str) -> str:
        return (
            label.replace("\\", "")
            .replace("$", "")
            .replace("^", "")
            .lower()
        )

    selected = [
        row
        for row in rows
        if row["model"] == "true PMP"
        or normalized(row["model"]) in members
    ]
    return format_table(
        selected,
        ["model", "cost A", "upright A", "cost B", "upright B"],
        title=title,
    )


def _fit_table(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            {
                "activation": row["activation"],
                "alpha": f"{row['alpha']:.0e}",
                "p": f"{row['order']:g}",
                "gamma": f"{row['gamma']:g}",
                "N": row["neurons"],
                "switching H1": f"{row['near_h1']:.3f}",
                "rest H1": f"{row['far_h1']:.3f}",
            }
        )
    columns = ["activation", "alpha", "p", "gamma", "N", "switching H1", "rest H1"]
    return format_table(rendered, columns, title="Selected Algorithm 1 fits")


def _algorithm2_fit_table(rows: list[dict[str, Any]]) -> str:
    rendered = [
        {
            "activation": row["activation"],
            "alpha": f"{row['alpha']:.0e}",
            "N": row["neurons"],
            "switching H1": f"{row['near_h1']:.3f}",
            "rest H1": f"{row['far_h1']:.3f}",
        }
        for row in sorted(rows, key=lambda item: item["power"])
    ]
    return format_table(
        rendered,
        ["activation", "alpha", "N", "switching H1", "rest H1"],
        title="Selected Algorithm 2 fits",
    )


def _oversampling_table(scored: list[dict[str, Any]], figures: ModuleType) -> str:
    rendered = []
    for family in figures._OVERSAMPLE_FAMILIES:
        for variant in figures._OVERSAMPLE_VARIANTS:
            candidates = [
                row
                for row in scored
                if row["family"] == family and row["variant"] == variant
            ]
            if not candidates:
                continue
            switching_best = min(candidates, key=lambda row: row["switching"])
            rendered.append(
                {
                    "family": figures._DISPLAY[family].replace("$", ""),
                    "variant": variant,
                    "runs": len(candidates),
                    "alpha": f"{switching_best['alpha']:.0e}",
                    "switching": f"{switching_best['switching']:.3f}",
                    "rest": f"{switching_best['rest']:.3f}",
                    "N": switching_best["neurons"],
                }
            )
    return format_table(
        rendered,
        ["family", "variant", "runs", "alpha", "switching", "rest", "N"],
        title=(
            "Common-set relative H1 error for the switching-best run "
            "(three alpha values per variant; all entries come from one run)"
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-alg1", type=Path, default=None,
                        help="Algorithm 1 sequential run-record root")
    parser.add_argument(
        "--traditional-records", type=Path, default=None,
        help="record directory for the ReLU plus l1 curve",
    )
    parser.add_argument(
        "--records-alg2", type=Path, default=None,
        help="Algorithm 2 sequential run-record root",
    )
    parser.add_argument(
        "--oversampling-alg1-records", type=Path, default=None,
        help="Algorithm 1 switching-set oversampling root",
    )
    parser.add_argument(
        "--oversampling-alg2-records", type=Path, default=None,
        help="Algorithm 2 switching-set oversampling root",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="study directory to write figures/ and full_scope.md into",
    )
    parser.add_argument(
        "--operating-point", type=str, default=None, metavar="ALPHA,GAMMA,P",
        help="pin one shared (alpha, gamma, moment_order) triple for every Algorithm 1 "
             "activation",
    )
    return parser.parse_args(argv)


def _apply_args(args: argparse.Namespace) -> None:
    """Rebind the module-level roots so one code path serves both studies."""
    global TRADITIONAL_ROOT
    global ALGORITHM1_ROOT, HOMOGENEOUS_ROOT, OUTPUT_DIR, OPERATING_POINT
    global OVERSAMPLE_ALGORITHM1_ROOT, OVERSAMPLE_ALGORITHM2_ROOT
    if args.records_alg1 is not None:
        ALGORITHM1_ROOT = args.records_alg1.resolve()
    if args.oversampling_alg1_records is not None:
        OVERSAMPLE_ALGORITHM1_ROOT = args.oversampling_alg1_records.resolve()
    if args.oversampling_alg2_records is not None:
        OVERSAMPLE_ALGORITHM2_ROOT = args.oversampling_alg2_records.resolve()
    if args.records_alg2 is not None:
        HOMOGENEOUS_ROOT = args.records_alg2.resolve()
    if args.traditional_records is not None:
        TRADITIONAL_ROOT = args.traditional_records.resolve()
    if args.out is not None:
        OUTPUT_DIR = args.out.resolve()
    if args.operating_point is not None:
        alpha, gamma, order = (float(v) for v in args.operating_point.split(","))
        OPERATING_POINT = {"alpha": alpha, "gamma": gamma, "order": order}


def main(argv: list[str] | None = None) -> int:
    _apply_args(_parse_args(argv))
    figures = _load_module(
        "pendulum_paper_figures",
        ROOT / "scripts" / "paper" / "pendulum_figures.py",
    )
    figures.OUTPUT_DIR = OUTPUT_DIR
    figures.FIG_DIR = OUTPUT_DIR / "figures"
    figures.OVERSAMPLE_ALGORITHM1_ROOT = OVERSAMPLE_ALGORITHM1_ROOT
    figures.OVERSAMPLE_ALGORITHM2_ROOT = OVERSAMPLE_ALGORITHM2_ROOT
    figures.ALGORITHM1_ROOT = ALGORITHM1_ROOT
    figures.ALGORITHM2_ROOT = HOMOGENEOUS_ROOT
    figures._MODEL_STYLE = {
        "gaussian": (PALETTE["blue_main"], "-"),
        "softplus": (PALETTE["violet"], "-"),
        "tanh": ("0.25", "-."),
        "relu^2": (PALETTE["red_strong"], "-"),
        "relu^3": (PALETTE["neutral"], "--"),
    }
    figures._DISPLAY = {
        "gaussian": "Gaussian",
        "softplus": "softplus",
        "tanh": "tanh",
        "relu^2": r"ReLU$^2$",
        "relu^3": r"ReLU$^3$",
    }
    figures._SURFACE_MODEL_ORDER = (
        "gaussian",
        "softplus",
        "tanh",
        "relu^2",
        "relu^3",
    )

    algorithm1, cache = load_algorithm1_rows()
    algorithm2 = load_algorithm2_rows(figures)
    rows = algorithm1 + algorithm2
    models = figures.select_models(rows)
    nets = {name: figures._build_net(row) for name, row in models.items()}
    samples, norm, curve, pool, rawt, _ = figures._load_geometry(rows)

    surfaces = figures.fig_learned_surfaces(rows, norm)
    frontier = _plot_frontier(
        figures,
        rows,
        ("gaussian", "softplus", "tanh", "relu^2", "relu^3"),
        "frontier",
    )
    _plot_frontier(
        figures,
        [
            _traditional_relu_row(algorithm1[0]["data_path"]),
            next(row for row in algorithm1 if row["activation"] == "softplus"),
            next(row for row in algorithm1 if row["activation"] == "gaussian"),
            next(row for row in algorithm2 if row["activation"] == "relu^2"),
        ],
        ("traditional", "softplus", "gaussian", "relu^2"),
        "frontier_intro",
    )
    true_branches = figures.fig_true_branch_transect(curve, pool, rawt)
    transects = figures.fig_transect_split(models, nets, norm, curve, pool, rawt)
    dumbbell = figures.fig_dumbbell(models)
    feedback, costs, starts = figures.fig_feedback_split(
        models, nets, norm, curve, pool, rawt
    )
    atoms = figures.fig_atom_portrait(models, nets, norm, curve)

    binned = figures._bin_centers(cache)
    distance_figs: dict[str, str] = {}
    if binned is not None:
        centers, counts, distance = binned
        distance_figs = figures.fig_error_vs_distance_split(
            models, nets, samples, norm, distance, centers, counts
        )

    scored = figures._common_pool_scores(algorithm1[0]["data_path"])
    if not scored:
        raise ValueError("oversampling records produced no common-pool scores")
    oversampling_figure = figures.fig_oversampling_control(scored)
    oversampling_table = _oversampling_table(scored, figures)

    algorithm1_feedback = _feedback_table(
        costs,
        {"gaussian", "softplus", "tanh"},
        (
            "Algorithm 1 feedback: closed-loop cost and stabilization "
            f"from A=({starts['A'][0]:.2f}, {starts['A'][1]:.2f}) and "
            f"B=({starts['B'][0]:.2f}, {starts['B'][1]:.2f})"
        ),
    )
    algorithm2_feedback = _feedback_table(
        costs,
        {"relu2", "relu3"},
        (
            "Algorithm 2 feedback: closed-loop cost and stabilization "
            f"from A=({starts['A'][0]:.2f}, {starts['A'][1]:.2f}) and "
            f"B=({starts['B'][0]:.2f}, {starts['B'][1]:.2f})"
        ),
    )

    surface_rows = "\n".join(
        f"- {name}: `{path}`" for name, path in surfaces.items()
    )
    feedback_rows = "\n".join(
        f"- {name}: `{path}`" for name, path in feedback.items()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "full_scope.md"
    out.write_text(
        "# Pendulum full experimental scope\n\n"
        "## Algorithm 1: normalized-measure nonhomogeneous model\n\n"
        f"{_fit_table(algorithm1)}\n\n"
        "### Synthesized feedback law\n\n"
        f"{algorithm1_feedback}\n\n"
        f"Control trace: `{feedback['control_b_log']}`\n\n"
        "## Algorithm 2: homogeneous ReLU model\n\n"
        f"{_algorithm2_fit_table(algorithm2)}\n\n"
        "### Synthesized feedback law\n\n"
        f"{algorithm2_feedback}\n\n"
        f"Control trace: `{feedback['control_b_relu']}`\n\n"
        "## Cross-model diagnostics\n\n"
        f"- insertion frontier: `{frontier}`\n"
        f"- switching/rest comparison: `{dumbbell}`\n"
        f"- atom portrait: `{atoms}`\n"
        f"- value transect: `{transects['value']}`\n"
        f"- gradient transect: `{transects['gradient']}`\n"
        f"- true branch value: `{true_branches['transect_true_branches_value']}`\n"
        f"- true branch gradient: `{true_branches['transect_true_branches_gradient']}`\n"
        f"- error/distance value: `{distance_figs.get('value', '')}`\n"
        f"- error/distance gradient: `{distance_figs.get('gradient', '')}`\n\n"
        "### Learned surfaces\n\n"
        f"{surface_rows}\n\n"
        "### Feedback phase portraits\n\n"
        f"{feedback_rows}\n\n"
        "## Oversampling control\n\n"
        f"{oversampling_table}\n\n"
        f"Figure: `{oversampling_figure}`\n",
        encoding="utf-8",
    )
    HOMOGENEOUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    homogeneous_out = HOMOGENEOUS_OUTPUT_DIR / "results.md"
    homogeneous_out.write_text(
        "# Algorithm 2 — pendulum swing-up\n\n"
        "Fresh sequential runs using the actual one-atom increment and the "
        "global-prox warm-start-scaled correction.\n\n"
        f"{_algorithm2_fit_table(algorithm2)}\n\n"
        "## Synthesized feedback law\n\n"
        f"{algorithm2_feedback}\n\n"
        f"Control trace: `{feedback['control_b_relu']}`\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"wrote {homogeneous_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
