#!/usr/bin/env python3
"""Build the region-eval pool artifact for a pendulum dataset.

The pool is the dense certified two-sided point set used for region metrics:
the basin-restricted body pool plus the envelope-certified switching-band pool
(pad + collar), **minus the emitted training rows** (the training set is thinned
from these very pools, so exact row matches are removed to make the pool
strictly out-of-sample). Each point carries its distance to the ±2π-tiled
switching curve. Saved as ``<dataset stem>_region_eval_pool.npz``
(``x, v, dv, distance``), consumed by the post-fit region eval
(``eval.kind=region_split``) and the post-hoc rescorer.

    python scripts/investigation/build_region_eval_pool.py \
        --data Pendulum_<run>/Pendulum_pmp_value_samples_..._.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.nonsmooth import (  # noqa: E402
    NonsmoothCurve,
    restrict_trajectory_to_curve,
)
from src.OpenLoop.pendulum.solver import (  # noqa: E402
    PendulumPmpSolver,
    PendulumPmpSolverConfig,
)
from src.OpenLoop.value_samples import ValueSamples  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True,
                        help="value-sample .npz under DATA_DIR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = DATA_DIR / args.data
    curve = NonsmoothCurve.load_npz(
        data_path.with_name(data_path.stem + "_nonsmooth_curve.npz")
    )
    raw_pkl = sorted(data_path.parent.glob("*raw_trajectories*.pkl"))[0]
    with open(raw_pkl, "rb") as f:
        raw = pickle.load(f)

    restricted = tuple(restrict_trajectory_to_curve(tr, curve)[0] for tr in raw)
    body = ValueSamples.concatenate([
        ValueSamples(x=tr.state, v=tr.value, dv=tr.costate)
        for tr in restricted if tr.state.size
    ])
    solver = PendulumPmpSolver(config=PendulumPmpSolverConfig(num_trajectories=len(raw)))
    pad, collar = solver.build_collar_samples(tuple(raw), restricted, curve)

    x = np.vstack([body.x, pad.x, collar.x])
    v = np.concatenate([body.v, pad.v, collar.v])
    dv = np.vstack([body.dv, pad.dv, collar.dv])

    # Remove the emitted training rows (exact matches — thinning copies rows).
    with np.load(data_path) as d:
        x_train = np.asarray(d["x"], dtype=np.float64)
    pool_keys = {t.tobytes() for t in np.ascontiguousarray(x_train)}
    keep = np.fromiter(
        (row.tobytes() not in pool_keys for row in np.ascontiguousarray(x)),
        bool, len(x),
    )
    n_removed = int((~keep).sum())
    x, v, dv = x[keep], v[keep], dv[keep]

    from scipy.spatial import cKDTree
    ridge_tiled = np.vstack(
        [curve.points + np.array([2.0 * np.pi * k, 0.0]) for k in (-1, 0, 1)]
    )
    distance, _ = cKDTree(ridge_tiled).query(x, k=1)

    out = data_path.with_name(data_path.stem + "_region_eval_pool.npz")
    np.savez(out, x=x, v=v, dv=dv, distance=distance.astype(np.float64))
    print(f"pool: {len(x)} points (body {body.size} + pad {pad.size} + collar "
          f"{collar.size}, {n_removed} training rows removed)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
