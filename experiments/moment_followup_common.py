"""Shared analysis for the gamma and loss-channel moment follow-ups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from experiments.moment_refinement_common import load_record
from src.plotstyle import apply_publication_style

GAMMAS = (0.0, 0.1, 1.0, 10.0)
H1_WEIGHTS = (1.0, 1.0)
L2_WEIGHTS = (1.0, 0.0)
SCALE_CEILING = math.exp(5.0)


@dataclass(frozen=True)
class Selection:
    """One fixed positive-beta configuration selected before the follow-up."""

    slug: str
    label: str
    activation: str
    alpha: float
    beta: float
    order: float


@dataclass(frozen=True)
class FollowupSpec:
    """Problem-dependent record paths, metric, and selected configurations."""

    title: str
    record_root: Path
    output_dir: Path
    error_key: str
    error_label: str
    selections: tuple[Selection, ...]


def _tick(value: float) -> str:
    return "0" if value == 0.0 else f"{value:g}"


def _source_paths(spec: FollowupSpec) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for stage in ("screen",):
        paths.extend(
            (stage, path)
            for path in sorted((spec.record_root / stage).glob("*/*.json"))
        )
    paths.extend(
        (f"refine/{path.parent.parent.name}", path)
        for path in sorted((spec.record_root / "refine").glob("*/*/*.json"))
    )
    return paths


def _followup_paths(spec: FollowupSpec) -> list[tuple[str, Path]]:
    return [
        (f"followup/{path.parent.parent.name}", path)
        for path in sorted((spec.record_root / "followup").glob("*/*/*.json"))
    ]


def _matches_selection(row: dict[str, Any], selection: Selection) -> bool:
    return (
        row["activation"] == selection.activation
        and row["alpha"] == selection.alpha
        and row["beta"] == selection.beta
        and row["p"] == selection.order
    )


def _protocol_key(
    row: dict[str, Any],
) -> tuple[str, float, float, float, tuple[float, float], float]:
    return (
        row["activation"],
        row["alpha"],
        row["beta"],
        row["p"],
        row["loss_weights"],
        row["gamma"],
    )


def load_and_validate(spec: FollowupSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = [
        load_record(stage, path, spec.error_key)
        for stage, path in _source_paths(spec)
    ]
    anchors: list[dict[str, Any]] = []
    for selection in spec.selections:
        matches = [
            row
            for row in source_rows
            if _matches_selection(row, selection)
            and row["loss_weights"] == H1_WEIGHTS
            and row["gamma"] == 1.0
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one gamma=1 H1 anchor for {selection.slug}; "
                f"found {len(matches)}"
            )
        anchors.append(matches[0])

    followup = [
        load_record(stage, path, spec.error_key)
        for stage, path in _followup_paths(spec)
    ]
    expected = {
        (
            selection.activation,
            selection.alpha,
            selection.beta,
            selection.order,
            weights,
            gamma,
        )
        for selection in spec.selections
        for weights, gammas in (
            (H1_WEIGHTS, (0.0, 0.1, 10.0)),
            (L2_WEIGHTS, GAMMAS),
        )
        for gamma in gammas
    }
    actual = {_protocol_key(row) for row in followup}
    if len(actual) != len(followup):
        raise ValueError("duplicate follow-up configuration")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"follow-up does not match protocol; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for row in anchors + followup:
        if row["seed"] != 42:
            raise ValueError(f"unexpected seed in {row['path']}")

    return anchors + followup, followup


def _selection_rows(
    rows: list[dict[str, Any]], selection: Selection
) -> list[dict[str, Any]]:
    return [row for row in rows if _matches_selection(row, selection)]


def _loss_label(weights: tuple[float, float]) -> str:
    return "value + gradient" if weights == H1_WEIGHTS else "value only"


def plot_selection(
    spec: FollowupSpec, rows: list[dict[str, Any]], selection: Selection
) -> Path:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    positions = {gamma: index for index, gamma in enumerate(GAMMAS)}
    colors = ("#31688e", "#35b779")

    selected = _selection_rows(rows, selection)
    for color, weights in zip(colors, (L2_WEIGHTS, H1_WEIGHTS)):
        curve = sorted(
            (row for row in selected if row["loss_weights"] == weights),
            key=lambda row: positions[row["gamma"]],
        )
        x = [positions[row["gamma"]] for row in curve]
        style = {
            "marker": "o",
            "linewidth": 1.6,
            "color": color,
            "label": _loss_label(weights),
        }
        axes[0].plot(x, [row["error"] for row in curve], **style)
        axes[1].plot(x, [row["neurons"] for row in curve], **style)
        axes[2].plot(x, [row["radius_r95"] for row in curve], **style)

    axes[0].set_ylabel(spec.error_label)
    axes[0].set_yscale("log")
    axes[1].set_ylabel("active support")
    axes[2].set_ylabel("amplitude-mass R95")
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[2].axhline(
        SCALE_CEILING,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"radial ceiling $e^5$",
    )
    for ax in axes:
        ax.set_xticks(range(len(GAMMAS)), [_tick(gamma) for gamma in GAMMAS])
        ax.set_xlabel(r"$\gamma$")
        ax.grid(True, alpha=0.18)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(selection.label)

    figure_dir = spec.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"followup_{selection.slug}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _table(rows: list[dict[str, Any]], include_stage: bool = False) -> str:
    stage_header = " stage |" if include_stage else ""
    stage_rule = "---|" if include_stage else ""
    lines = [
        f"|{stage_header} activation | loss | gamma | error | N | R95 |",
        f"|{stage_rule}---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        stage = f" {row['stage']} |" if include_stage else ""
        lines.append(
            f"|{stage} {row['activation']} | {_loss_label(row['loss_weights'])} | "
            f"{_tick(row['gamma'])} | {row['error']:.4f} | "
            f"{row['neurons']} | {row['radius_r95']:.3g} |"
        )
    return "\n".join(lines)


def write_report(
    spec: FollowupSpec,
    rows: list[dict[str, Any]],
    followup: list[dict[str, Any]],
    figures: dict[str, Path],
) -> Path:
    best: list[dict[str, Any]] = []
    for selection in spec.selections:
        interior = [
            row
            for row in _selection_rows(rows, selection)
            if row["radius_r95"] < 0.99 * SCALE_CEILING
        ]
        best.append(min(interior, key=lambda row: row["error"]))
    best_overall = min(best, key=lambda row: row["error"])
    interior_rows = [
        row for row in rows if row["radius_r95"] < 0.99 * SCALE_CEILING
    ]
    best_h1 = min(
        (row for row in interior_rows if row["loss_weights"] == H1_WEIGHTS),
        key=lambda row: row["error"],
    )
    best_l2 = min(
        (row for row in interior_rows if row["loss_weights"] == L2_WEIGHTS),
        key=lambda row: row["error"],
    )
    h1_wins = sum(row["loss_weights"] == H1_WEIGHTS for row in best)
    gamma_summary = ", ".join(
        f"{row['activation']}={_tick(row['gamma'])}" for row in best
    )
    boundary_count = sum(
        row["radius_r95"] >= 0.99 * SCALE_CEILING for row in followup
    )

    figure_lines: list[str] = []
    for selection in spec.selections:
        figure_lines.extend(
            [
                f"### {selection.activation}",
                "",
                f"![gamma and loss comparison]({figures[selection.slug].relative_to(spec.output_dir).as_posix()})",
                "",
            ]
        )

    manifest = sorted(
        followup,
        key=lambda row: (
            row["activation"],
            row["loss_weights"],
            row["gamma"],
        ),
    )
    text = "\n".join(
        [
            f"# {spec.title} gamma and loss-channel follow-up",
            "",
            f"**Status: complete.** {len(followup)} new seed-42 records vary gamma "
            "and the loss channels at five preselected interior positive-beta "
            "configurations. Five existing gamma=1 H1 records are reused.",
            "",
            "## Headline",
            "",
            f"The lowest {spec.error_label} away from the radial search ceiling is "
            f"{best_overall['error']:.4f} for `{best_overall['activation']}` with "
            f"{_loss_label(best_overall['loss_weights'])} loss and "
            f"gamma={_tick(best_overall['gamma'])}; N={best_overall['neurons']} "
            f"and R95={best_overall['radius_r95']:.3g}.",
            "",
            "## Follow-up observations",
            "",
            f"- Gradient augmentation wins at {h1_wins} of "
            f"{len(spec.selections)} selected activations. The best H1-trained "
            f"error is {best_h1['error']:.4f}, compared with "
            f"{best_l2['error']:.4f} for the best value-only fit.",
            f"- The error-minimizing gamma values by activation are: "
            f"{gamma_summary}. Gamma is therefore retained as an independent "
            "hyperparameter rather than fixed universally.",
            f"- {boundary_count} new records place R95 at the radial search "
            "ceiling and are censored for selection.",
            "",
            "## Best gamma/loss choice by activation",
            "",
            _table(best),
            "",
            "## Per-activation comparisons",
            "",
            *figure_lines,
            "## New record manifest",
            "",
            _table(manifest, include_stage=True),
        ]
    )
    path = spec.output_dir / "followup.md"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def run(spec: FollowupSpec) -> None:
    rows, followup = load_and_validate(spec)
    figures = {
        selection.slug: plot_selection(spec, rows, selection)
        for selection in spec.selections
    }
    report = write_report(spec, rows, followup, figures)
    print(
        f"analyzed {len(followup)} new records plus {len(spec.selections)} anchors; "
        f"wrote {len(figures)} figures and {report}"
    )
