"""Shared analysis for the two adaptive moment-penalty refinements."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.plotstyle import PALETTE, apply_publication_style

ALL_BETAS = (
    0.0,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
)
SCALE_CEILING = math.exp(5.0)


@dataclass(frozen=True)
class CurveSpec:
    """One fixed-(activation, alpha, p) beta slice."""

    label: str
    activation: str
    alpha: float
    order: float
    refinement_betas: tuple[float, ...]


@dataclass(frozen=True)
class ProblemSpec:
    """Problem-dependent paths, metric, and selected beta slices."""

    title: str
    record_root: Path
    output_dir: Path
    error_key: str
    error_label: str
    curves: tuple[CurveSpec, ...]


def _tick(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.0e}".replace("e-0", "e-")


def _record_paths(spec: ProblemSpec) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for stage in ("baseline", "screen"):
        paths.extend(
            (stage, path)
            for path in sorted((spec.record_root / stage).glob("*/*.json"))
        )
    paths.extend(
        (f"refine/{path.parent.parent.name}", path)
        for path in sorted((spec.record_root / "refine").glob("*/*/*.json"))
    )
    return paths


def _validate_objective(row: dict[str, Any]) -> None:
    reconstructed = (
        row["data_value_term_train"]
        + row["data_gradient_term_train"]
        + row["alpha_phi_1"]
        + row["beta_psi_p"]
    )
    if not math.isclose(
        row["objective_train"], reconstructed, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(f"objective decomposition does not close for {row['path']}")


def load_record(stage: str, path: Path, error_key: str) -> dict[str, Any]:
    """Parse and validate one completed run record."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"incomplete record: {path}")
    cfg = record["config"]
    model = cfg["model"]
    events = record.get("metrics", [])
    if not events:
        raise ValueError(f"record {path} has no metrics")
    values = events[-1]["values"]
    row = {
        "stage": stage,
        "path": path,
        "activation": str(model["activation"]),
        "alpha": float(model["alpha"]),
        "beta": float(model["moment_beta"]),
        "p": float(model["moment_order"]),
        "seed": int(cfg["env"]["seed"]),
        "gamma": float(model["gamma"]),
        "loss_weights": tuple(float(x) for x in model["loss_weights"]),
        "error": float(values[error_key]),
        "rel_h1": float(values["rel_h1_val"]),
        "neurons": int(values["best_neurons"]),
        "radius_r95": float(values["radius_r95"]),
        "radius_max": float(values["radius_max"]),
        "phi_1": float(values["phi_1"]),
        "psi_p": float(values["psi_p"]),
        "alpha_phi_1": float(values["alpha_phi_1"]),
        "beta_psi_p": float(values["beta_psi_p"]),
        "objective_train": float(values["objective_train"]),
        "data_value_term_train": float(values["data_value_term_train"]),
        "data_gradient_term_train": float(values["data_gradient_term_train"]),
    }
    _validate_objective(row)
    return row


def load_rows(spec: ProblemSpec) -> list[dict[str, Any]]:
    return [
        load_record(stage, path, spec.error_key)
        for stage, path in _record_paths(spec)
    ]


def _key(row: dict[str, Any]) -> tuple[str, float, float, float]:
    return row["activation"], row["alpha"], row["beta"], row["p"]


def validate_rows(spec: ProblemSpec, rows: list[dict[str, Any]]) -> None:
    baseline = [row for row in rows if row["stage"] == "baseline"]
    screen = [row for row in rows if row["stage"] == "screen"]
    refinement = [row for row in rows if row["stage"].startswith("refine/")]
    if len(baseline) != 16 or len(screen) != 256:
        raise ValueError(
            f"first-pass records changed: baseline={len(baseline)}, screen={len(screen)}"
        )

    expected_refinement = {
        (
            curve.activation,
            curve.alpha,
            beta,
            2.01 if beta == 0.0 else curve.order,
        )
        for curve in spec.curves
        for beta in curve.refinement_betas
    }
    actual_refinement = {_key(row) for row in refinement}
    if len(actual_refinement) != len(refinement):
        raise ValueError("duplicate refinement configuration")
    if actual_refinement != expected_refinement:
        missing = sorted(expected_refinement - actual_refinement)
        unexpected = sorted(actual_refinement - expected_refinement)
        raise ValueError(
            f"refinement does not match protocol; missing={missing}, "
            f"unexpected={unexpected}"
        )

    for row in rows:
        if row["seed"] != 42 or row["gamma"] != 1.0:
            raise ValueError(f"unexpected protocol setting in {row['path']}")
        if row["loss_weights"] != (1.0, 1.0):
            raise ValueError(f"unexpected loss weights in {row['path']}")


def _curve_rows(rows: list[dict[str, Any]], curve: CurveSpec) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["activation"] == curve.activation
        and row["alpha"] == curve.alpha
        and (row["beta"] == 0.0 or row["p"] == curve.order)
    ]
    order = {beta: index for index, beta in enumerate(ALL_BETAS)}
    return sorted(selected, key=lambda row: order[row["beta"]])


def _interior_positive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["beta"] > 0.0 and row["radius_r95"] < 0.99 * SCALE_CEILING
    ]


def _best_by_activation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = _interior_positive(rows)
    activations = ("tanh", "softplus", "gaussian", "gelu_squared", "matern52")
    return [
        min(
            (row for row in candidates if row["activation"] == activation),
            key=lambda row: row["error"],
        )
        for activation in activations
    ]


@dataclass(frozen=True)
class PanelSeries:
    """One line across the error / support / radius panels."""

    label: str
    x: list[int]
    error: list[float]
    neurons: list[float]
    radius: list[float]


def series_style(index: int) -> dict[str, Any]:
    """House-palette line style for series ``index``.

    ``PALETTE`` carries four line-strength colours (``neutral`` is a light grey
    for de-emphasis), so runs longer than four are separated by dash pattern as
    well as hue.
    """
    hues = (
        PALETTE["blue_main"],
        PALETTE["teal"],
        PALETTE["red_strong"],
        PALETTE["violet"],
    )
    dashes = ("-", "--", "-.")
    return {
        "color": hues[index % len(hues)],
        "linestyle": dashes[(index // len(hues)) % len(dashes)],
        "marker": "o",
        "linewidth": 1.4,
        "markersize": 4.5,
    }


def plot_metric_panels(
    series: list[PanelSeries],
    *,
    tick_labels: list[str],
    x_label: str,
    error_label: str,
    path: Path,
    legend_ncol: int = 1,
) -> Path:
    """Draw the shared three-panel sweep figure and save it to ``path``.

    The panels are one composite rather than three files because they share a
    single x axis -- the sweep variable and its tick positions are identical, so
    reading a run means reading one abscissa across all three.  Per the house
    style there are no in-figure titles: the caption identifies the study.
    """
    apply_publication_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(11.4, 10.0), sharex=True
    )

    for index, line in enumerate(series):
        style = series_style(index) | {"label": line.label}
        axes[0].plot(line.x, line.error, **style)
        axes[1].plot(line.x, line.neurons, **style)
        axes[2].plot(line.x, line.radius, **style)

    axes[0].set_ylabel(error_label)
    axes[0].set_yscale("log")
    axes[1].set_ylabel("active support")
    axes[2].set_ylabel("amplitude-mass R95")
    axes[2].set_yscale("symlog", linthresh=0.1)
    ceiling = axes[2].axhline(
        SCALE_CEILING,
        color=PALETTE["neutral"],
        linestyle="--",
        linewidth=1.0,
        label=r"radial ceiling $e^5$",
    )
    for ax in axes:
        ax.set_xticks(range(len(tick_labels)), tick_labels)
    axes[2].set_xlabel(x_label)
    # The series are named once, on the top panel; the bottom panel's legend
    # explains only the ceiling rule it adds.
    axes[0].legend(ncol=legend_ncol, fontsize=7.5)
    axes[2].legend(handles=[ceiling], fontsize=7.5)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_refinement(spec: ProblemSpec, rows: list[dict[str, Any]]) -> Path:
    positions = {beta: index for index, beta in enumerate(ALL_BETAS)}
    series = []
    for curve in spec.curves:
        selected = _curve_rows(rows, curve)
        series.append(
            PanelSeries(
                label=curve.label,
                x=[positions[row["beta"]] for row in selected],
                error=[row["error"] for row in selected],
                neurons=[row["neurons"] for row in selected],
                radius=[row["radius_r95"] for row in selected],
            )
        )

    return plot_metric_panels(
        series,
        tick_labels=[_tick(beta) for beta in ALL_BETAS],
        x_label=r"$\beta$",
        error_label=spec.error_label,
        path=spec.output_dir / "figures" / "refinement.png",
        legend_ncol=2,
    )


def _table(rows: list[dict[str, Any]], include_stage: bool = False) -> str:
    stage_header = " stage |" if include_stage else ""
    stage_rule = "---|" if include_stage else ""
    lines = [
        f"|{stage_header} activation | p | alpha | beta | error | N | R95 |",
        f"|{stage_rule}---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        order = "—" if row["beta"] == 0.0 else f"{row['p']:g}"
        stage = f" {row['stage']} |" if include_stage else ""
        lines.append(
            f"|{stage} {row['activation']} | {order} | {_tick(row['alpha'])} | "
            f"{_tick(row['beta'])} | {row['error']:.4f} | "
            f"{row['neurons']} | {row['radius_r95']:.3g} |"
        )
    return "\n".join(lines)


def write_report(
    spec: ProblemSpec, rows: list[dict[str, Any]], figure_path: Path
) -> Path:
    refinement = [row for row in rows if row["stage"].startswith("refine/")]
    best = _best_by_activation(rows)
    best_overall = min(best, key=lambda row: row["error"])
    curve_best = [
        min(_interior_positive(_curve_rows(rows, curve)), key=lambda row: row["error"])
        for curve in spec.curves
    ]
    boundary_refinement = sum(
        row["beta"] > 0.0 and row["radius_r95"] >= 0.99 * SCALE_CEILING
        for row in refinement
    )

    refinement_sorted = sorted(
        refinement,
        key=lambda row: (
            row["activation"],
            row["alpha"],
            row["p"],
            row["beta"],
        ),
    )
    text = "\n".join(
        [
            f"# {spec.title} adaptive moment refinement",
            "",
            f"**Status: complete.** {len(refinement)} new seed-42 records fill only "
            "the unresolved beta decades and add Matern-5/2. The 272 first-pass "
            "records remain unchanged.",
            "",
            "## Headline",
            "",
            f"The lowest positive-beta {spec.error_label} away from the radial "
            f"search ceiling is {best_overall['error']:.4f} for "
            f"`{best_overall['activation']}` at alpha={_tick(best_overall['alpha'])}, "
            f"beta={_tick(best_overall['beta'])}, p={best_overall['p']:g}, with "
            f"N={best_overall['neurons']} and R95={best_overall['radius_r95']:.3g}. "
            "Beta=0 remains a plotted reference, not a candidate for the modified "
            "narrow-convergence model.",
            "",
            "## Best interior positive-beta cell by activation",
            "",
            _table(best),
            "",
            "## Best point on each selected beta slice",
            "",
            _table(curve_best),
            "",
            "## Refined beta slices",
            "",
            "The curves retain alpha and p fixed. Missing markers correspond to "
            "decades that the first pass did not identify as part of that row's "
            "transition.",
            "",
            f"![adaptive beta refinement]({figure_path.relative_to(spec.output_dir).as_posix()})",
            "",
            f"{boundary_refinement} of the new positive-beta records place R95 at "
            "the radial search ceiling and are censored for model selection.",
            "",
            "## New record manifest",
            "",
            _table(refinement_sorted, include_stage=True),
        ]
    )
    path = spec.output_dir / "refinement.md"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def run(spec: ProblemSpec) -> None:
    rows = load_rows(spec)
    validate_rows(spec, rows)
    figure = plot_refinement(spec, rows)
    report = write_report(spec, rows, figure)
    refinement_count = sum(
        row["stage"].startswith("refine/") for row in rows
    )
    print(
        f"analyzed {len(rows)} total records ({refinement_count} refinement); "
        f"wrote {figure} and {report}"
    )
