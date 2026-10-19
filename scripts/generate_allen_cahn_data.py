#!/usr/bin/env python3
"""Generate complete, audited Allen–Cahn value/costate datasets at 6 or 10 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
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
from src.OpenLoop.allen_cahn import AllenCahnProblem
from src.OpenLoop.collocation import (
    BatchError,
    CollocationConfig,
    CollocationSolver,
    SampleFailure,
)
from src.OpenLoop.sampling import sobol_initial_states


def main(argv=None) -> None:
    args = _parse_args(argv)
    problem = AllenCahnProblem(
        dimension=args.dimension, nu=args.diffusion, beta=args.beta, T_final=args.horizon,
    )
    config = CollocationConfig(
        collocation_tol=args.tol, collocation_max_nodes=args.max_nodes,
        initial_mesh_points=args.mesh_points,
    )
    states = sobol_initial_states(args.samples, args.dimension, bound=args.bound, seed=args.seed)
    destination = args.output_dir / f"AllenCahn{args.dimension}_{args.tag}"
    metadata = _generation_metadata(problem, config, args)
    generate_dataset(destination, CollocationSolver(problem, config), states,
                     metadata, workers=args.workers)
    print(f"data: {destination / 'value_samples.npz'}\n"
          f"metadata: {destination / 'metadata.json'}", flush=True)


def _parse_args(argv):
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
    return args


def _generation_metadata(problem, config, args) -> dict:
    sources = [Path(__file__), ROOT / "src/data.py",
               ROOT / "src/OpenLoop/collocation.py", ROOT / "src/OpenLoop/sampling.py",
               *sorted((ROOT / "src/OpenLoop/allen_cahn").glob("*.py"))]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://jmlr.org/papers/volume24/22-1010/22-1010.pdf",
        "source_section": "9.3, equation (58)",
        "adaptation": (
            "Uniform cell-centered finite volumes with exact cell-averaged actuators, "
            "6/10 states instead of 19 Chebyshev degrees of freedom; "
            "the control-cost coefficient is a repository choice; scrambled Sobol cube sampling."
        ),
        "target": "V_T(x)=V(0,x); free terminal state and zero terminal cost",
        "reference_scope": "least-cost converged PMP stationary solution; no global certificate",
        "problem": asdict(problem), "solver": asdict(config),
        "sampling": {"kind": "scrambled_sobol", "count": args.samples,
                     "bound": args.bound, "seed": args.seed},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "requested_samples": args.samples, "retained_samples": 0, "complete": False,
        "generator_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


def generate_dataset(destination, solver, states, metadata, *, workers=1) -> None:
    """Persist a complete validated dataset, or leave diagnostics and raise.

    ``destination`` must not exist. The supplied metadata is copied; callers
    need not observe mutation to learn whether generation succeeded.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata)
    np.savez(destination / "requested_states.npz", x=states)
    _write_json(destination / "metadata.json", metadata)
    try:
        solution = solver.solve(states, _progress(states.shape[1]), workers=workers)
        _write_solver_records(destination, solution.sample_results)
        metadata["gradient_checks"] = check_value_gradients(solver, solution.value_samples)
        _save_validated_samples(destination, solution, metadata)
    except BatchError as error:
        _write_solver_records(destination, error.sample_results)
        metadata.update(failed_samples=len(error.failed_initial_states), error=str(error))
        raise
    except Exception as error:
        # Retain the reason for an interrupted solve/validation, then propagate it.
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_json(destination / "metadata.json", metadata)


def _progress(dimension):
    def report(done, total):
        if done and (done % 64 == 0 or done == total):
            print(f"Allen–Cahn {dimension}D: {done}/{total}", flush=True)
    return report


def _write_solver_records(destination, results) -> None:
    diagnostics = [asdict(result.diagnostics) for result in results]
    failures = [dict(row=index, initial_state=result.initial_state.tolist(), **diagnostics[index])
                for index, result in enumerate(results) if isinstance(result, SampleFailure)]
    _write_json(destination / "diagnostics.json", diagnostics)
    _write_json(destination / "failures.json", failures)


def check_value_gradients(solver, samples) -> list[dict]:
    """Check three stored costates against re-solved directional differences.

    This checks sensitivity consistency of the same collocation method. It is
    not a comparison with an independent optimizer or a global-optimality test.
    """
    rng = np.random.default_rng(1234)
    checks = []
    for index in np.linspace(0, samples.size - 1, min(3, samples.size), dtype=int):
        direction = rng.normal(size=samples.x.shape[1])
        direction /= np.linalg.norm(direction)
        epsilon = 1e-4
        plus = solver.solve_sample(samples.x[index] + epsilon * direction)
        minus = solver.solve_sample(samples.x[index] - epsilon * direction)
        difference = (plus.value - minus.value) / (2 * epsilon)
        costate = float(samples.dv[index] @ direction)
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


def _save_validated_samples(destination, solution, metadata) -> None:
    samples = solution.value_samples
    if not samples.size:
        raise ValueError("a training dataset must contain at least one sample")
    if not all(np.isfinite(array).all() for array in (samples.x, samples.v, samples.dv)):
        raise RuntimeError("non-finite training samples; dataset not saved")
    path = samples.save_npz(destination / "value_samples.npz")
    metadata.update(
        complete=True, retained_samples=samples.size, failed_samples=0,
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        max_collocation_residual=max(r.diagnostics.collocation_residual for r in solution.sample_results),
        max_boundary_residual=max(r.diagnostics.boundary_residual for r in solution.sample_results),
    )


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
