#!/usr/bin/env python3
"""Analyze the Van der Pol gamma and loss-channel follow-up."""

from __future__ import annotations

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.moment_followup_common import (
    FollowupSpec,
    Selection,
    run,
)

if __name__ == "__main__":
    run(
        FollowupSpec(
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
            selections=(
                Selection(
                    "tanh",
                    r"tanh: $\alpha=10^{-5}$, $\beta=10^{-10}$, $p=2.01$",
                    "tanh",
                    1e-5,
                    1e-10,
                    2.01,
                ),
                Selection(
                    "softplus",
                    r"softplus: $\alpha=10^{-5}$, $\beta=10^{-10}$, $p=2.01$",
                    "softplus",
                    1e-5,
                    1e-10,
                    2.01,
                ),
                Selection(
                    "gaussian",
                    r"Gaussian: $\alpha=10^{-5}$, $\beta=10^{-10}$, $p=3$",
                    "gaussian",
                    1e-5,
                    1e-10,
                    3.0,
                ),
                Selection(
                    "gelu_squared",
                    r"squared GELU: $\alpha=10^{-3}$, $\beta=10^{-10}$, $p=3$",
                    "gelu_squared",
                    1e-3,
                    1e-10,
                    3.0,
                ),
                Selection(
                    "matern52",
                    r"Matern-5/2: $\alpha=10^{-5}$, $\beta=10^{-7}$, $p=2.01$",
                    "matern52",
                    1e-5,
                    1e-7,
                    2.01,
                ),
            ),
        )
    )
