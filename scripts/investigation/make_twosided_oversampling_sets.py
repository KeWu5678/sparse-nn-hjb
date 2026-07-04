#!/usr/bin/env python3
"""Two-sided oversampling dataset variants for the region-split §3 control.

Rebuilds, from the production 2000-path raw trajectories and the validated
basin (issue #18 workaround), four training-set variants that vary only in the
switching-band share, so the §3 question can be asked on two-sided data:
does spending more samples on the switching band improve the switching fit?

  base6k : 6,000 at the production band share (~23%: 4,615 body + 462 pad + 923 collar)
  band40 : 6,000 reallocated to a 40% band (3,600 body + 800 pad + 1,600 collar)
  band60 : 6,000 reallocated to a 60% band (2,400 body + 1,200 pad + 2,400 collar)
  add2k  : base6k + 2,000 extra band samples (8,000: 4,615 body + 1,129 pad + 2,256 collar)

Each variant gets its own ``.npz`` + sibling ``_nonsmooth_curve.npz`` (the
validated curve, copied so the distance-cache script resolves it) under one
run dir; caches are built separately by ``precompute_region_distances.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_pendulum_pmp_openloop_example import thin_value_samples  # noqa: E402

SOURCE = DATA_DIR / "Pendulum_20260703_ada466c6182948469a197282906c3b6c"
STEM = "Pendulum_pmp_value_samples_2000_20260703"
OUT_DIR = DATA_DIR / "Pendulum_2sided_oversample_20260704"

# variant -> (body, pad, collar) sample targets
VARIANTS = {
    "base6k": (4615, 462, 923),
    "band40": (3600, 800, 1600),
    "band60": (2400, 1200, 2400),
    "add2k": (4615, 1129, 2256),
}


def main() -> int:
    curve = NonsmoothCurve.load_npz(SOURCE / f"{STEM}_nonsmooth_curve.npz")
    assert curve.basin.shape[0] > 3, "validated basin missing from source curve"
    with open(SOURCE / "Pendulum_pmp_raw_trajectories_2000_20260703.pkl", "rb") as f:
        raw = pickle.load(f)

    restricted = tuple(restrict_trajectory_to_curve(tr, curve)[0] for tr in raw)
    body_pool = ValueSamples.concatenate([
        ValueSamples(x=tr.state, v=tr.value, dv=tr.costate)
        for tr in restricted if tr.state.size
    ])
    solver = PendulumPmpSolver(config=PendulumPmpSolverConfig(num_trajectories=2000))
    pad_pool, collar_pool = solver.build_collar_samples(tuple(raw), restricted, curve)
    print(f"pools: body {body_pool.size}, pad {pad_pool.size}, collar {collar_pool.size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (n_body, n_pad, n_collar) in VARIANTS.items():
        samples = ValueSamples.concatenate([
            thin_value_samples(body_pool, n_body),
            thin_value_samples(pad_pool, n_pad),
            thin_value_samples(collar_pool, n_collar),
        ])
        data_path = samples.save_npz(OUT_DIR / f"{name}.npz")
        curve.save_npz(OUT_DIR / f"{name}_nonsmooth_curve.npz")
        share = 100.0 * (n_pad + n_collar) / samples.size
        print(f"{name}: {samples.size} samples "
              f"({n_body} body + {n_pad} pad + {n_collar} collar, band {share:.0f}%) "
              f"-> {data_path.relative_to(DATA_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
