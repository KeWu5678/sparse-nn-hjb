#!/usr/bin/env python3
"""Generate dated, internally consistent Van der Pol value/costate samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR
from src.OpenLoop.vdp import (
    VdpOpenLoopSolver,
    VdpOpenLoopSolverConfig,
    VdpOptimalControlProblem,
    grid_initial_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    states = grid_initial_states(args.grid_size, args.grid_size, (-3, 3), (-3, 3))
    solver = VdpOpenLoopSolver(
        VdpOptimalControlProblem(T_final=3.0),
        VdpOpenLoopSolverConfig(profile="pmp", num_time_points=151),
    )

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"VDP samples: {done}/{total}", flush=True)

    solution = solver.solve(states, progress=progress)
    if solution.value_samples.size != len(states):
        raise RuntimeError(f"incomplete VDP grid: {len(solution.failed_initial_states)} failures")
    paths = solution.save_dataset(
        args.output_dir, grid_shape=(args.grid_size, args.grid_size), date_tag=args.tag,
    )
    metadata = json.loads(paths["meta"].read_text())
    metadata["data_sha256"] = hashlib.sha256(paths["data"].read_bytes()).hexdigest()
    metadata["collocation_residuals"] = [r.collocation_residual for r in solution.sample_results]
    metadata["generator_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), ROOT / "src/OpenLoop/vdp/problem.py",
                     ROOT / "src/OpenLoop/vdp/solver.py")
    }
    paths["meta"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"data: {paths['data']}", flush=True)
    print(f"metadata: {paths['meta']}", flush=True)


if __name__ == "__main__":
    main()
