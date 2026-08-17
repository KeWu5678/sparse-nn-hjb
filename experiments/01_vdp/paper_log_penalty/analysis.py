#!/usr/bin/env python3
"""Generate the Van der Pol Algorithm 1 study report."""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    from scripts.paper.log_penalty_analysis import generate

    generate("vdp", OUTPUT_DIR)
