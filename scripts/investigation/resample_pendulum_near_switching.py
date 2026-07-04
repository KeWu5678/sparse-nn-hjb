#!/usr/bin/env python3
"""Create a pendulum dataset oversampled near the switching set.

This reuses an existing raw-trajectory pickle and switching-curve artifact. It
does not reintegrate the PMP ODEs. The output is a normal ValueSamples ``.npz``
plus an aligned ``*_region_distances.npz`` cache, so it can be used with the
existing ``eval=region_split`` path by overriding ``data.path`` and
``eval.distance_cache``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.nonsmooth import (  # noqa: E402
    NonsmoothCurve,
    restrict_trajectory_to_curve,
)
from src.OpenLoop.value_samples import ValueSamples  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402


def _current_pendulum_dataset() -> Path:
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "data" / "pendulum.yaml").read_text())
    return DATA_DIR / cfg["data"]["path"]


def _default_raw_pickle(dataset: Path) -> Path:
    candidates = sorted(dataset.parent.glob("*raw_trajectories*.pkl"))
    if candidates:
        return candidates[0]
    return DATA_DIR / "_debug_raw_trajectories_256.pkl"


def parse_args() -> argparse.Namespace:
    dataset = _current_pendulum_dataset()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trajectories", type=Path, default=_default_raw_pickle(dataset))
    parser.add_argument(
        "--curve",
        type=Path,
        default=dataset.with_name(dataset.stem + "_nonsmooth_curve.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--target-size", type=int, default=6000)
    parser.add_argument(
        "--near-percentile",
        type=float,
        default=10.0,
        help="Distance percentile of the full retained pool that defines the near source band.",
    )
    parser.add_argument(
        "--near-fraction",
        type=float,
        default=0.40,
        help="Fraction of output samples drawn from the near source band.",
    )
    parser.add_argument("--distance-bins", type=int, default=20)
    parser.add_argument(
        "--far-sampling",
        choices=("proportional", "stratified"),
        default="proportional",
        help="How to draw the non-near samples. Proportional preserves the raw far distribution.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else DATA_DIR / path


def _restricted_pool(raw_path: Path, curve: NonsmoothCurve) -> ValueSamples:
    with raw_path.open("rb") as file:
        raw = tuple(pickle.load(file))

    chunks: list[ValueSamples] = []
    for trajectory in raw:
        cut, _ = restrict_trajectory_to_curve(trajectory, curve)
        if cut.state.size:
            chunks.append(ValueSamples(x=cut.state, v=cut.value, dv=cut.costate))
    return ValueSamples.concatenate(chunks)


def _stratified_choice(
    indices: np.ndarray,
    distance: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    n_bins: int,
) -> np.ndarray:
    if count <= 0 or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    if count >= indices.size:
        return indices.copy()

    local_distance = distance[indices]
    lo = float(local_distance.min())
    hi = float(local_distance.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return rng.choice(indices, size=count, replace=False)

    edges = np.linspace(lo, hi, n_bins + 1)
    groups: list[np.ndarray] = []
    for bin_idx in range(n_bins):
        left, right = edges[bin_idx], edges[bin_idx + 1]
        mask = (
            (local_distance >= left) & (local_distance <= right)
            if bin_idx == n_bins - 1
            else (local_distance >= left) & (local_distance < right)
        )
        group = indices[mask]
        if group.size:
            groups.append(group)

    if not groups:
        return rng.choice(indices, size=count, replace=False)

    base = count // len(groups)
    remainder = count % len(groups)
    selected: list[np.ndarray] = []
    selected_set: set[int] = set()
    for rank, group in enumerate(groups):
        quota = base + (1 if rank < remainder else 0)
        if quota <= 0:
            continue
        take = min(quota, group.size)
        choice = rng.choice(group, size=take, replace=False)
        selected.append(choice)
        selected_set.update(int(i) for i in choice)

    chosen = np.concatenate(selected) if selected else np.empty((0,), dtype=np.int64)
    if chosen.size < count:
        remaining = np.asarray([int(i) for i in indices if int(i) not in selected_set], dtype=np.int64)
        fill = rng.choice(remaining, size=count - chosen.size, replace=False)
        chosen = np.concatenate([chosen, fill])
    return chosen


def _random_choice(
    indices: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    if count >= indices.size:
        return indices.copy()
    return rng.choice(indices, size=count, replace=False)


def _median_nn_spacing(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    nn, _ = cKDTree(x).query(x, k=2)
    return float(np.median(nn[:, 1]))


def main() -> int:
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive")
    if not 0.0 <= args.near_fraction <= 1.0:
        raise ValueError("--near-fraction must be in [0, 1]")
    if not 0.0 < args.near_percentile < 100.0:
        raise ValueError("--near-percentile must be in (0, 100)")
    if args.distance_bins <= 0:
        raise ValueError("--distance-bins must be positive")

    raw_path = _resolve(args.raw_trajectories)
    curve_path = _resolve(args.curve)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not curve_path.exists():
        raise FileNotFoundError(curve_path)

    curve = NonsmoothCurve.load_npz(curve_path)
    if curve.is_empty:
        raise ValueError(f"switching curve is empty: {curve_path}")

    pool = _restricted_pool(raw_path, curve)
    if pool.size < args.target_size:
        raise ValueError(f"restricted pool has only {pool.size} samples")

    distance, _ = cKDTree(curve.points).query(pool.x, k=1)
    threshold = float(np.percentile(distance, args.near_percentile))
    near_idx = np.flatnonzero(distance <= threshold)
    far_idx = np.flatnonzero(distance > threshold)

    n_near = int(round(args.target_size * args.near_fraction))
    n_near = min(n_near, near_idx.size, args.target_size)
    n_far = args.target_size - n_near
    if n_far > far_idx.size:
        n_far = far_idx.size
        n_near = args.target_size - n_far

    rng = np.random.default_rng(args.seed)
    chosen_near = _stratified_choice(
        near_idx, distance, n_near, rng=rng, n_bins=args.distance_bins
    )
    if args.far_sampling == "stratified":
        chosen_far = _stratified_choice(
            far_idx, distance, n_far, rng=rng, n_bins=args.distance_bins
        )
    else:
        chosen_far = _random_choice(far_idx, n_far, rng=rng)
    chosen = np.concatenate([chosen_near, chosen_far])
    rng.shuffle(chosen)

    samples = ValueSamples(x=pool.x[chosen], v=pool.v[chosen], dv=pool.dv[chosen])
    selected_distance = distance[chosen].astype(np.float64)

    date = args.tag or datetime.now().strftime("%Y%m%d")
    run_dir = args.output_dir / f"Pendulum_nearswitch_{date}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    stem = f"Pendulum_pmp_value_samples_nearswitch_{args.target_size}_{date}"
    data_path = samples.save_npz(run_dir / f"{stem}.npz")
    distance_path = run_dir / f"{stem}_region_distances.npz"
    np.savez(distance_path, distance=selected_distance, h=_median_nn_spacing(samples.x))
    curve_copy = run_dir / f"{stem}_nonsmooth_curve.npz"
    shutil.copy2(curve_path, curve_copy)

    metadata = {
        "description": "Pendulum ValueSamples resampled to oversample the switching-set band.",
        "data_path": str(data_path),
        "curve_path": str(curve_copy),
        "distance_cache": str(distance_path),
        "source_raw_trajectories": str(raw_path),
        "source_curve": str(curve_path),
        "restricted_pool_samples": int(pool.size),
        "target_size": int(samples.size),
        "seed": int(args.seed),
        "near_source_percentile": float(args.near_percentile),
        "near_source_threshold": threshold,
        "near_fraction": float(args.near_fraction),
        "far_sampling": args.far_sampling,
        "near_samples": int(chosen_near.size),
        "far_samples": int(chosen_far.size),
        "distance_min_median_max": [
            float(selected_distance.min()),
            float(np.median(selected_distance)),
            float(selected_distance.max()),
        ],
    }
    meta_path = run_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"wrote data: {data_path}")
    print(f"wrote distance cache: {distance_path}")
    print(f"near source band: distance <= {threshold:.6g}")
    print(f"selected near/far: {chosen_near.size}/{chosen_far.size}")
    print(f"distance min/median/max: {metadata['distance_min_median_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
