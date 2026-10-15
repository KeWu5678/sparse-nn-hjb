#!/usr/bin/env python3
"""Two-sided oversampling dataset variants for the region-split §3 control.

Rebuilds, from the production 2000-path raw trajectories and their reconstructed
basin, four training-set variants that vary only in the
augmentation share, so the §3 question can be asked on two-sided data:
does spending more samples on the switching band improve the switching fit?

  base6k : 6,000 at the production band share (~23%: 4,615 body + 462 pad + 923 collar)
  band40 : 6,000 reallocated to a 40% band (3,600 body + 800 pad + 1,600 collar)
  band60 : 6,000 reallocated to a 60% band (2,400 body + 1,200 pad + 2,400 collar)
  add2k  : base6k + 2,000 extra band samples (8,000: 4,615 body + 1,129 pad + 2,256 collar)

Shares refer to pad/collar augmentation pools within distance 0.5, not the
fraction of all samples inside the evaluation tube of radius 0.3.
Each variant gets its own ``.npz`` + sibling ``_nonsmooth_curve.npz`` (the
validated curve, copied so the distance-cache script resolves it) under one
run dir; caches are built separately by ``precompute_region_distances.py``.
"""

from __future__ import annotations

import pickle
import sys
import argparse
from datetime import date
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
from src.data import DATA_DIR  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_pendulum_pmp_openloop_example import thin_value_samples  # noqa: E402

# variant -> (body, pad, collar) sample targets
VARIANTS = {
    "base6k": (4615, 462, 923),
    "band40": (3600, 800, 1600),
    "band60": (2400, 1200, 2400),
    "add2k": (4615, 1129, 2256),
}


def additional_samples(pool: ValueSamples, existing: ValueSamples, count: int) -> ValueSamples:
    """Select additional distinct states while preserving every existing sample."""
    seen = {row.tobytes() for row in np.ascontiguousarray(existing.x)}
    available = []
    for i, row in enumerate(np.ascontiguousarray(pool.x)):
        key = row.tobytes()
        if key not in seen:
            available.append(i)
            seen.add(key)
    if len(available) < count:
        raise ValueError(f"need {count} new states, have {len(available)}")
    indices = np.asarray(available)[np.linspace(0, len(available) - 1, count, dtype=int)]
    return ValueSamples(pool.x[indices], pool.v[indices], pool.dv[indices])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="fresh production dataset under rawdata/data")
    parser.add_argument("--tag", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()
    source = DATA_DIR / args.data
    output_dir = DATA_DIR / f"Pendulum_2sided_oversample_{args.tag}"
    curve = NonsmoothCurve.load_npz(source.with_name(f"{source.stem}_nonsmooth_curve.npz"))
    assert curve.basin.shape[0] > 3, "validated basin missing from source curve"
    raw_files = list(source.parent.glob("*raw_trajectories*.pkl"))
    if len(raw_files) != 1:
        raise ValueError("expected exactly one raw-trajectory artifact beside the fresh dataset")
    with raw_files[0].open("rb") as f:
        raw = pickle.load(f)

    restricted = tuple(restrict_trajectory_to_curve(tr, curve)[0] for tr in raw)
    body_pool = ValueSamples.concatenate([
        ValueSamples(x=tr.state, v=tr.value, dv=tr.costate)
        for tr in restricted if tr.state.size
    ])
    solver = PendulumPmpSolver(config=PendulumPmpSolverConfig(num_trajectories=2000))
    body_pool = solver.screen_body_samples(body_pool, tuple(raw))
    pad_pool, collar_pool = solver.build_collar_samples(tuple(raw), restricted, curve)
    print(f"pools: body {body_pool.size}, pad {pad_pool.size}, collar {collar_pool.size}")

    output_dir.mkdir(parents=True, exist_ok=False)
    base = None
    for name, (n_body, n_pad, n_collar) in VARIANTS.items():
        if name == "add2k":
            pad_extra = additional_samples(pad_pool, base, 667)
            existing = ValueSamples.concatenate([base, pad_extra])
            collar_extra = additional_samples(collar_pool, existing, 1333)
            samples = ValueSamples.concatenate([existing, collar_extra])
        else:
            samples = ValueSamples.concatenate([
                thin_value_samples(body_pool, n_body),
                thin_value_samples(pad_pool, n_pad),
                thin_value_samples(collar_pool, n_collar),
            ])
        if samples.size != n_body + n_pad + n_collar:
            raise ValueError(f"insufficient samples for {name}")
        if name == "base6k":
            base = samples
        data_path = samples.save_npz(output_dir / f"{name}.npz")
        curve.save_npz(output_dir / f"{name}_nonsmooth_curve.npz")
        share = 100.0 * (n_pad + n_collar) / samples.size
        print(f"{name}: {samples.size} samples "
              f"({n_body} body + {n_pad} pad + {n_collar} collar, band {share:.0f}%) "
              f"-> {data_path.relative_to(DATA_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
