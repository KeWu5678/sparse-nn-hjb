#!/usr/bin/env python3
"""Generate dated, internally consistent Van der Pol value/costate samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR
from src.OpenLoop.collocation import CollocationConfig, CollocationSolution, CollocationSolver
from src.OpenLoop.sampling import grid_initial_states
from src.OpenLoop.vdp import VdpOptimalControlProblem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    states = grid_initial_states(args.grid_size, args.grid_size, (-3, 3), (-3, 3))
    solver = CollocationSolver(
        VdpOptimalControlProblem(T_final=3.0),
        CollocationConfig(
            collocation_tol=1e-8, collocation_max_nodes=20000,
            initial_mesh_points=151, quadrature_order=3,
        ),
    )

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"VDP samples: {done}/{total}", flush=True)

    # An incomplete grid raises BatchError before anything is written.
    solution = solver.solve(states, progress=progress)
    date = datetime.now().strftime("%Y%m%d") if args.tag is None else args.tag
    paths = save_dataset(solution, args.output_dir, args.grid_size, date)
    print(f"data: {paths['data']}", flush=True)
    print(f"metadata: {paths['meta']}", flush=True)


def save_dataset(
    solution: CollocationSolution, output_dir: Path, grid_size: int, date: str,
) -> dict[str, Path]:
    """Write one complete dataset and its provenance into a new run folder."""
    run_dir = Path(output_dir) / f"VDP_{date}_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    stem = f"grid_{grid_size}x{grid_size}_{date}"
    data_path = solution.value_samples.save_npz(run_dir / f"VDP_pmp_{stem}.npz")
    meta_path = run_dir / f"VDP_pmp_meta_{stem}.json"
    metadata = {
        "problem": asdict(solution.problem),
        "solver": asdict(solution.config),
        "requested_samples": len(solution.sample_results),
        "retained_samples": solution.value_samples.size,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "collocation_residuals": [
            result.diagnostics.collocation_residual for result in solution.sample_results
        ],
        "generator_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "src/OpenLoop/vdp/problem.py",
                         ROOT / "src/OpenLoop/collocation.py")
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"run_dir": run_dir, "data": data_path, "meta": meta_path}


if __name__ == "__main__":
    main()
