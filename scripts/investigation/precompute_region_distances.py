#!/usr/bin/env python3
"""Precompute per-sample distance to the pendulum switching set.

Reads a pendulum value-sample ``.npz`` (``x, v, dv``) and its sibling
``..._nonsmooth_curve.npz`` (the persisted switching curve), and writes a parameter-free
distance cache ``..._region_distances.npz`` aligned to the dataset's sample
order, with:

  * ``distance`` (N,)  -- Euclidean distance from each sample's physical state
    (theta, omega) to the nearest switching-curve point.
  * ``h``        ()    -- the dataset's median nearest-neighbor spacing (kept for
    reference; not used by the eval, which is percentile-based — h is dominated by
    dense along-trajectory spacing, not the transverse scale).

The consolidated region eval scores the switching tube on the region-eval pool
(see ``build_region_eval_pool.py``); this dataset-aligned cache remains for the
distance-binned error-profile diagnostic.

Distance is to the nearest switching-curve *point* (KD-tree over the curve points tiled by
2πk in theta, k = -1, 0, 1), not the polyline, so it is robust to the curve
having multiple arcs and makes no assumption about the stored point ordering.
Tiling matters: the stored curve covers one period (the +theta arm and spiral),
while the basin's left boundary is the same arm shifted by -2π — without the
tiles, samples hugging the left arm would be mislabeled far.

    python scripts/investigation/precompute_region_distances.py \
        --data Pendulum_<run>/Pendulum_pmp_value_samples_256_<date>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.nonsmooth import NonsmoothCurve  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="value-sample .npz, bare filename under DATA_DIR or absolute path",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default=None,
        help="switching-curve .npz; defaults to the dataset's sibling ..._nonsmooth_curve.npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output .npz; defaults to the dataset's sibling ..._region_distances.npz",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    return DATA_DIR / path


def main() -> int:
    args = parse_args()
    data_path = _resolve(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    curve_path = (
        _resolve(args.curve)
        if args.curve is not None
        else data_path.with_name(data_path.stem + "_nonsmooth_curve.npz")
    )
    if not curve_path.exists():
        raise FileNotFoundError(
            f"switching curve not found: {curve_path}\n"
            "regenerate the dataset with the curve-persisting solver first"
        )

    output_path = (
        _resolve(args.output)
        if args.output is not None
        else data_path.with_name(data_path.stem + "_region_distances.npz")
    )

    with np.load(data_path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
    curve = NonsmoothCurve.load_npz(curve_path)
    if curve.is_empty:
        raise ValueError(f"switching curve {curve_path} is empty; cannot label regions")

    # Distance from every sample to the nearest switching-curve point
    # (periodically tiled: the stored curve covers one period, but its -2π copy
    # bounds the basin too).
    curve_tiled = np.vstack(
        [curve.points + np.array([2.0 * np.pi * k, 0.0]) for k in (-1, 0, 1)]
    )
    curve_tree = cKDTree(curve_tiled)
    distance, _ = curve_tree.query(x, k=1)

    # h = median nearest-neighbor spacing among the samples themselves: query k=2
    # (self at distance 0 + the true nearest neighbor) and take the second column.
    sample_tree = cKDTree(x)
    nn, _ = sample_tree.query(x, k=2)
    h = float(np.median(nn[:, 1]))

    np.savez(output_path, distance=distance.astype(np.float64), h=h)
    print(f"samples: {x.shape[0]}  switching-curve points: {curve.points.shape[0]}")
    print(
        f"distance-to-switching-set  min/median/max = "
        f"{distance.min():.4g}/{np.median(distance):.4g}/{distance.max():.4g}"
    )
    n_tube = int(np.sum(distance <= 0.3))
    print(f"switching tube (d <= 0.3): {n_tube} samples")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
