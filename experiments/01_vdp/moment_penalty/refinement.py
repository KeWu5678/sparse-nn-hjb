#!/usr/bin/env python3
"""Analyze the selected Van der Pol beta refinements and Matern addition."""

from __future__ import annotations

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.moment_refinement_common import CurveSpec, ProblemSpec, run

EARLY = (1e-9, 1e-8, 1e-7, 1e-6)
LATE = (1e-4, 1e-3)
MATERN = (
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


if __name__ == "__main__":
    run(
        ProblemSpec(
            title="Van der Pol",
            record_root=REPO_ROOT
            / "rawdata"
            / "logs"
            / "multirun"
            / "vdp"
            / "moment_penalty",
            output_dir=OUTPUT_DIR,
            error_key="rel_h1_val",
            error_label="validation H1",
            curves=(
                CurveSpec(
                    r"tanh, $\alpha=10^{-5}$, $p=2.01$",
                    "tanh",
                    1e-5,
                    2.01,
                    EARLY,
                ),
                CurveSpec(
                    r"softplus, $\alpha=10^{-5}$, $p=2.01$",
                    "softplus",
                    1e-5,
                    2.01,
                    EARLY,
                ),
                CurveSpec(
                    r"Gaussian, $\alpha=10^{-5}$, $p=3$",
                    "gaussian",
                    1e-5,
                    3.0,
                    LATE,
                ),
                CurveSpec(
                    r"squared GELU, $\alpha=10^{-3}$, $p=2.01$",
                    "gelu_squared",
                    1e-3,
                    2.01,
                    LATE,
                ),
                CurveSpec(
                    r"Matern-5/2, $\alpha=10^{-4}$",
                    "matern52",
                    1e-4,
                    2.01,
                    MATERN,
                ),
                CurveSpec(
                    r"Matern-5/2, $\alpha=10^{-5}$",
                    "matern52",
                    1e-5,
                    2.01,
                    MATERN,
                ),
            ),
        )
    )
