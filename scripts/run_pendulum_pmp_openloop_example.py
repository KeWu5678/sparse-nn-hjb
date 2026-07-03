#!/usr/bin/env python3
"""Generate Han-Yang infinite-horizon pendulum value samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum import (  # noqa: E402
    PendulumPmpSolver,
    PendulumPmpSolverConfig,
    PendulumSwingUpProblem,
)
from src.OpenLoop.value_samples import ValueSamples  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=2e-4)
    parser.add_argument("--value-max", type=float, default=100.0)
    parser.add_argument("--t-final", type=float, default=50.0)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--control-limit", type=float, default=None)
    parser.add_argument("--uniform-boundary-sampling", action="store_true")
    parser.add_argument("--reference-value", type=float, default=None)
    parser.add_argument("--boundary-distance-power", type=float, default=0.8)
    parser.add_argument("--contour-delta", type=float, default=1.0)
    parser.add_argument("--basin-value-max", type=float, default=50.0,
                        help="value cap on the basin arms (reference ~57; 35 is too small) — issue #18")
    parser.add_argument("--periodic-copies", type=int, default=0)
    parser.add_argument("--collar-width", type=float, default=0.5,
                        help="switching-set band half-width around the tiled ridge (0 disables)")
    parser.add_argument("--collar-fraction", type=float, default=1.0 / 6.0,
                        help="share of --level-set-samples drawn from the far-side collar")
    parser.add_argument("--pad-fraction", type=float, default=1.0 / 12.0,
                        help="share of --level-set-samples drawn from the near-side pad")
    parser.add_argument("--level-set-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def progress_printer(quiet: bool):
    last_bucket = {"value": -1}

    def _progress(done: int, total: int) -> None:
        if quiet or total == 0:
            return
        bucket = int(10 * done / total)
        if bucket != last_bucket["value"]:
            last_bucket["value"] = bucket
            print(f"processed {done}/{total}")

    return _progress


def thin_value_samples(samples: ValueSamples, target_size: int) -> ValueSamples:
    if target_size <= 0:
        raise ValueError("level-set-samples must be positive")
    if samples.size <= target_size:
        return samples
    indices = np.linspace(0, samples.size - 1, target_size, dtype=np.int64)
    return ValueSamples(
        x=samples.x[indices],
        v=samples.v[indices],
        dv=samples.dv[indices],
    )


def main() -> None:
    args = parse_args()
    date_tag = args.tag or time.strftime("%Y%m%d")
    problem = PendulumSwingUpProblem(control_limit=args.control_limit)
    config = PendulumPmpSolverConfig(
        epsilon=args.epsilon,
        value_max=args.value_max,
        t_final=args.t_final,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        num_trajectories=args.num_trajectories,
        adaptive_sampling=not args.uniform_boundary_sampling,
        reference_value=args.reference_value,
        boundary_distance_power=args.boundary_distance_power,
        contour_delta=args.contour_delta,
        basin_value_max=args.basin_value_max,
        periodic_copies=args.periodic_copies,
        collar_width=args.collar_width,
    )
    if not 0.0 <= args.collar_fraction + args.pad_fraction < 1.0:
        raise ValueError("collar-fraction + pad-fraction must be in [0, 1)")
    solver = PendulumPmpSolver(problem=problem, config=config)

    started = time.perf_counter()
    solution = solver.solve(progress=progress_printer(args.quiet))
    elapsed = time.perf_counter() - started
    if solution.nonsmooth_curve.basin.shape[0] < 3:
        # Known issue-#18 instability: the auto-assembled basin can come out empty
        # (e.g. at 2000 paths), silently disabling BOTH the restriction cut and the
        # collar/pad harvest — the emitted samples would be the raw chaotic region.
        raise RuntimeError(
            "basin assembly produced an empty ring: restriction and switching-band "
            "harvest are no-ops. Re-run the post-processing with a validated basin "
            "injected (issue #18 workaround; see the 20260630 dataset's basin_source)."
        )
    raw_sample_count = solution.value_samples.size
    collar_pool = solution.collar_samples or ValueSamples.concatenate([])
    pad_pool = solution.pad_samples or ValueSamples.concatenate([])
    # Thin the in-basin body, near-side pad, and far-side collar pools separately
    # so the emitted dataset hits the requested near-switch shares (uniform
    # thinning would give the band only its pool proportion).
    collar_target = min(round(args.collar_fraction * args.level_set_samples), collar_pool.size)
    pad_target = min(round(args.pad_fraction * args.level_set_samples), pad_pool.size)
    basin_samples = thin_value_samples(
        solution.value_samples, args.level_set_samples - collar_target - pad_target
    )
    collar_samples = (
        thin_value_samples(collar_pool, collar_target)
        if collar_target > 0
        else ValueSamples.concatenate([])
    )
    pad_samples = (
        thin_value_samples(pad_pool, pad_target)
        if pad_target > 0
        else ValueSamples.concatenate([])
    )
    value_samples = ValueSamples.concatenate([basin_samples, pad_samples, collar_samples])
    solution = replace(
        solution,
        value_samples=value_samples,
        diagnostics=replace(solution.diagnostics, retained_points=value_samples.size),
    )

    paths = solution.save_dataset(args.output_dir, date_tag=date_tag)
    metadata = {
        "source_paper": "https://arxiv.org/pdf/2312.17467",
        "description": "Backward-PMP infinite-horizon pendulum ValueSamples.",
        "data_path": str(paths["data"]),
        "curve_path": str(paths["curve"]),
        "failed_path": str(paths["failed"]),
        "run_dir": str(paths["run_dir"]),
        "elapsed_seconds": elapsed,
        "raw_retained_points": raw_sample_count,
        "level_set_samples": args.level_set_samples,
        "collar_pool_points": collar_pool.size,
        "collar_samples": collar_samples.size,
        "collar_fraction": args.collar_fraction,
        "pad_pool_points": pad_pool.size,
        "pad_samples": pad_samples.size,
        "pad_fraction": args.pad_fraction,
        "problem": {
            "mass": problem.mass,
            "length": problem.length,
            "damping": problem.damping,
            "gravity": problem.gravity,
            "state_weights": list(problem.state_weights),
            "control_weight": problem.control_weight,
            "control_limit": problem.control_limit,
        },
        "config": config.__dict__,
        "diagnostics": solution.diagnostics.__dict__,
    }
    paths["meta"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"saved run dir: {paths['run_dir']}")
        print(f"saved data: {paths['data']}")
        print(f"saved metadata: {paths['meta']}")
        print(f"saved failures: {paths['failed']}")
        print(f"samples: {solution.value_samples.size}, seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
