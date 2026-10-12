#!/usr/bin/env python3
"""Generate complete, audited Allen–Cahn value/costate datasets at 6 or 10 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Small collocation systems should not spawn a BLAS thread team in every worker.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from src.data import DATA_DIR
from src.OpenLoop.allen_cahn import (
    AllenCahnOpenLoopSolver,
    AllenCahnProblem,
    AllenCahnSolverConfig,
)
from src.OpenLoop.allen_cahn.sampling import sobol_initial_states
from src.OpenLoop.value_samples import ValueSamples


def _solve_one(payload):
    problem, config, initial = payload
    return AllenCahnOpenLoopSolver(problem, config).solve_sample(initial)


def _diagnostic(result) -> dict:
    return {
        name: getattr(result, name)
        for name in ("converged", "message", "iterations", "collocation_residual",
                     "boundary_residual", "nodes", "reduced_gradient_norm")
    }


def _gradient_checks(solver, states, results) -> list[dict]:
    """Compare initial costates to independently re-solved directional differences."""
    rng = np.random.default_rng(1234)
    checks = []
    for index in np.linspace(0, len(states) - 1, min(3, len(states)), dtype=int):
        direction = rng.normal(size=states.shape[1])
        direction /= np.linalg.norm(direction)
        epsilon = 1e-4
        plus = solver.solve_sample(states[index] + epsilon * direction)
        minus = solver.solve_sample(states[index] - epsilon * direction)
        if not plus.converged or not minus.converged:
            raise RuntimeError(f"finite-difference validation failed to converge at row {index}")
        difference = (plus.value - minus.value) / (2 * epsilon)
        costate = float(results[index].gradient @ direction)
        error = float(abs(difference - costate))
        tolerance = 2e-5 + 2e-4 * abs(difference)
        checks.append({
            "row": int(index), "direction": direction.tolist(), "epsilon": epsilon,
            "finite_difference": float(difference), "costate_derivative": costate,
            "absolute_error": error, "tolerance": tolerance,
        })
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(f"costate finite-difference check failed: {checks[-1]}")
    return checks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(6, 10), required=True)
    parser.add_argument("--samples", type=int, default=2048, help="power of two")
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--diffusion", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--mesh-points", type=int, default=81)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.tag or Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("--tag must be a single directory-name component")
    problem = AllenCahnProblem(
        dimension=args.dimension, nu=args.diffusion, beta=args.beta, T_final=args.horizon,
    )
    config = AllenCahnSolverConfig(
        collocation_tol=args.tol, collocation_max_nodes=args.max_nodes,
        initial_mesh_points=args.mesh_points,
    )
    states = sobol_initial_states(args.samples, args.dimension, bound=args.bound, seed=args.seed)
    destination = args.output_dir / f"AllenCahn{args.dimension}_{args.tag}"
    destination.mkdir(parents=True, exist_ok=False)
    # Preserve requested states even if solving fails; never silently omit rows.
    np.savez(destination / "requested_states.npz", x=states)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://jmlr.org/papers/volume24/22-1010/22-1010.pdf",
        "source_section": "9.3, equation (58)",
        "adaptation": (
            "Uniform cell-centered finite volumes with exact cell-averaged actuators, "
            "6/10 states instead of 19 Chebyshev degrees of freedom; "
            "The control-cost coefficient is a repository choice; scrambled Sobol cube sampling."
        ),
        "target": "V_T(x)=V(0,x); free terminal state and zero terminal cost",
        "reference_scope": "least-cost converged PMP stationary solution; no global certificate",
        "problem": asdict(problem), "solver": asdict(config),
        "sampling": {"kind": "scrambled_sobol", "count": args.samples,
                     "bound": args.bound, "seed": args.seed},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "requested_samples": args.samples, "complete": False,
        "generator_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "src/OpenLoop/value_samples.py",
                         *sorted((ROOT / "src/OpenLoop/allen_cahn").glob("*.py"))]
        },
    }
    meta_path = destination / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    payloads = ((problem, config, initial) for initial in states)
    results = []

    def collect(iterator):
        for result in iterator:
            results.append(result)
            if len(results) % 64 == 0 or len(results) == len(states):
                print(f"Allen–Cahn {args.dimension}D: {len(results)}/{len(states)}", flush=True)

    if args.workers == 1:
        collect(map(_solve_one, payloads))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            collect(executor.map(_solve_one, payloads, chunksize=4))
    diagnostics = [_diagnostic(result) for result in results]
    (destination / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    failures = [dict(row=index, initial_state=states[index].tolist(), **diagnostics[index])
                for index, result in enumerate(results) if not result.converged]
    (destination / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    metadata["failed_samples"] = len(failures)
    metadata["retained_samples"] = len(results) - len(failures)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} failed solves; no training dataset saved in {destination}")
    metadata["gradient_checks"] = _gradient_checks(AllenCahnOpenLoopSolver(problem, config), states, results)
    samples = ValueSamples(
        x=states, v=np.array([result.value for result in results]),
        dv=np.stack([result.gradient for result in results]),
    )
    if not all(np.isfinite(array).all() for array in (samples.x, samples.v, samples.dv)):
        raise RuntimeError("non-finite training samples; dataset not saved")
    path = samples.save_npz(destination / "value_samples.npz")
    metadata.update(
        complete=True, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        max_collocation_residual=max(result.collocation_residual for result in results),
        max_boundary_residual=max(result.boundary_residual for result in results),
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"data: {path}\nmetadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
