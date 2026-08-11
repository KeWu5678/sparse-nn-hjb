#!/usr/bin/env python3
"""Isolate beta at the previous study's softplus operating point.

The moment-penalty representatives were selected on their own grid, so they
differ from the previous softplus fit in alpha, gamma, and beta at once. This
stage holds alpha=1e-5, gamma=10, p=2.01 fixed -- the configuration the earlier
log-penalty study selected -- and varies only beta, so the feedback comparison
attributes its difference to the moment term rather than to reselection.

Reads only ``rawdata/logs/multirun/pendulum/moment_penalty/control`` and writes
``control.md``. Rollouts use the same protocol as the full-scope feedback
figure: starts A and B at +/- 0.25 along the switching-curve normal, RK4 with
T=10, dt=0.005, controls clipped to |u| <= 30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = (
    ROOT / "rawdata" / "logs" / "multirun" / "pendulum" / "moment_penalty" / "control"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.metric import format_table  # noqa: E402

_full_scope = None
_legacy = None


def _modules():
    """Load the full-scope helpers and the region-split analysis they delegate to."""
    global _full_scope, _legacy
    if _full_scope is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "moment_full_scope", OUTPUT_DIR / "full_scope.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot load full_scope.py")
        _full_scope = importlib.util.module_from_spec(spec)
        sys.modules["moment_full_scope"] = _full_scope
        spec.loader.exec_module(_full_scope)
        _legacy = _full_scope._load_module(
            "region_split_analysis",
            ROOT / "experiments" / "02_pendulum" / "region_split" / "analysis.py",
        )
    return _full_scope, _legacy


def load_rows() -> list[dict[str, Any]]:
    """One row per control record, ordered by beta."""
    full_scope, _ = _modules()
    rows: list[dict[str, Any]] = []
    for path in sorted(CONTROL_ROOT.glob("**/*.json")):
        with open(path) as handle:
            record = json.load(handle)
        if record.get("status") != "completed":
            continue
        row = full_scope._moment_row(record, path)
        values = record["metrics"][-1]["values"]
        row["rel_h1"] = float(values["rel_h1_val"])
        row["r95"] = float(values["radius_r95"])
        row["r_max"] = float(values["radius_max"])
        row["phi_1"] = float(values["phi_1"])
        row["psi_p"] = float(values["psi_p"])
        rows.append(row)
    rows.sort(key=lambda r: r["beta"])
    return rows


def rollout_table(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Closed-loop cost and stabilization from A and B for every beta."""
    _, legacy = _modules()
    from scipy.spatial import cKDTree

    from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem

    problem = PendulumSwingUpProblem()
    samples, norm, curve, pool, rawt, _ = legacy._load_geometry(rows)
    c0, _, nrm = legacy._transect_frame(curve.points, pool["x"])
    starts = {"A": c0 - 0.25 * nrm, "B": c0 + 0.25 * nrm}
    tree = cKDTree(rawt["x"])

    def true_u(x):
        _, j = tree.query(np.asarray(x, dtype=np.float64).reshape(1, 2), k=40)
        j = np.asarray(j).ravel()
        opt = j[int(np.argmin(rawt["v"][j]))]
        return float(np.ravel(problem.feedback_from_gradient(rawt["dv"][opt]))[0])

    def model_u(net):
        def u(x):
            _, g = legacy._value_grad_phys(net, np.asarray(x).reshape(1, 2), norm)
            return float(problem.feedback_from_gradient(g)[0])

        return u

    def reached(xs) -> bool:
        xf = xs[-1]
        return abs((xf[0] + np.pi) % (2 * np.pi) - np.pi) < 0.4 and abs(xf[1]) < 0.4

    laws: list[tuple[str, Any]] = [("true PMP", true_u)]
    for row in rows:
        laws.append((f"beta = {row['beta']:.0e}", model_u(legacy._build_net(row))))

    body = []
    results: dict[str, Any] = {}
    for label, law in laws:
        cells: dict[str, Any] = {"law": label}
        for side in ("A", "B"):
            _, xs, _, cost = problem.rk4_rollout(
                law, starts[side], T=10.0, dt=0.005, u_clip=30.0
            )
            up = reached(xs)
            results[f"{label}|{side}"] = (cost, up)
            cells[f"cost {side}"] = f"{cost:.1f}"
            cells[f"upright {side}"] = "yes" if up else "no"
        body.append(cells)

    table = format_table(
        body, ["law", "cost A", "upright A", "cost B", "upright B"]
    )
    return table, results


def fit_table(rows: list[dict[str, Any]]) -> str:
    body = [
        {
            "beta": f"{row['beta']:.0e}",
            "N": str(row["neurons"]),
            "val H1": f"{row['rel_h1']:.4f}",
            "switching H1": f"{row['near_h1']:.4f}",
            "rest H1": f"{row['far_h1']:.4f}",
            "R95": f"{row['r95']:.3g}",
            "R max": f"{row['r_max']:.3g}",
            "Phi_1": f"{row['phi_1']:.3g}",
            "Psi_p": f"{row['psi_p']:.3g}",
        }
        for row in rows
    ]
    return format_table(
        body,
        ["beta", "N", "val H1", "switching H1", "rest H1", "R95", "R max",
         "Phi_1", "Psi_p"],
    )


def main() -> int:
    rows = load_rows()
    if not rows:
        print(f"no completed control records under {CONTROL_ROOT}", file=sys.stderr)
        return 1

    feedback, _ = rollout_table(rows)
    report = (
        "# Pendulum moment control: beta at a fixed operating point\n\n"
        "**Status: complete.** Four seed-42 records hold `softplus`, "
        "alpha=1e-5, gamma=10, p=2.01, H1 loss `[1,1]` fixed and vary only "
        "beta. The beta=0 row reproduces the configuration the earlier "
        "log-penalty study selected, so the rows differ in the moment weight "
        "alone.\n\n"
        "## Fits\n\n"
        f"{fit_table(rows)}\n\n"
        "## Synthesized feedback\n\n"
        "Closed-loop cost and stabilization from A and B, the two sides of the "
        "switching curve, under the full-scope rollout protocol (RK4, T=10, "
        "dt=0.005, |u| <= 30).\n\n"
        f"{feedback}\n"
    )
    (OUTPUT_DIR / "control.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
