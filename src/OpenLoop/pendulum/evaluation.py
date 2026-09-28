"""Finite-horizon upright diagnostics, not asymptotic stability certificates."""

import numpy as np

from .problem import PendulumSwingUpProblem


def select_stabilizing_fit(candidates, stabilizes):
    """Lowest global validation H1 among fits passing ``stabilizes``.

    Test candidates in ascending H1 order, stopping at the first pass. If none
    pass, return the lowest-H1 fit and False as an explicitly failed comparator.
    Each candidate is a mapping with ``global_h1`` and ``record_path`` keys.
    """
    ordered = sorted(candidates, key=lambda row: (row["global_h1"], row["record_path"]))
    if not ordered or any(not np.isfinite(row["global_h1"]) for row in ordered):
        raise ValueError("expected candidates with finite global validation H1 errors")
    for candidate in ordered:
        if stabilizes(candidate):
            return candidate, True
    return ordered[0], False


def assess_upright(time, states, *, horizon, angle_tolerance=0.1,
                   speed_tolerance=0.5, dwell_time=2.0):
    """Check every recorded sample in the final ``dwell_time`` seconds.

    Angles are measured modulo 2π from upright. The trajectory must cover the
    whole interval [0, horizon], including a sample at the requested horizon.
    Incomplete or non-finite trajectories fail. Longer traces are scored using
    only their prefix. A pass is empirical numerical stabilization; it does
    not establish convergence or finite infinite-horizon cost.
    """
    time = np.asarray(time, dtype=float)
    states = np.asarray(states, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or states.shape != (len(time), 2)
            or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)):
        raise ValueError("expected increasing finite times and aligned (N, 2) states")
    if (not np.all(np.isfinite([horizon, angle_tolerance, speed_tolerance, dwell_time]))
            or not 0 < dwell_time <= horizon or min(angle_tolerance, speed_tolerance) <= 0):
        raise ValueError("require positive tolerances and 0 < dwell_time <= horizon")

    # The reporting horizons lie on the rollout grid; do not silently score a
    # shorter trace when integration stops early or the horizon is off-grid.
    endpoint = np.flatnonzero(np.isclose(time, horizon, rtol=0, atol=1e-10))
    end = int(endpoint[0]) + 1 if len(endpoint) else np.searchsorted(time, horizon, side="right")
    t, x = time[:end], states[:end]
    complete = bool(len(endpoint) and abs(time[0]) <= 1e-10)
    finite = bool(len(x) and np.all(np.isfinite(x)))
    tail = t >= horizon - dwell_time - 1e-10
    result = dict(horizon=float(horizon), angle_tolerance=float(angle_tolerance),
                  speed_tolerance=float(speed_tolerance), dwell_time=float(dwell_time),
                  complete=complete, finite=finite, sustained_upright=False,
                  settling_time=None, terminal_angle_error=None, terminal_speed=None,
                  tail_max_angle_error=None, tail_max_speed=None)
    if not finite:
        return result
    angle = np.abs(PendulumSwingUpProblem.wrap_angle(x[:, 0]))
    speed = np.abs(x[:, 1])
    inside = (angle < angle_tolerance) & (speed < speed_tolerance)
    result.update(terminal_angle_error=float(angle[-1]), terminal_speed=float(speed[-1]))
    if np.any(tail):
        result.update(tail_max_angle_error=float(angle[tail].max()),
                      tail_max_speed=float(speed[tail].max()))
    if complete and np.any(tail) and np.all(inside[tail]):
        outside = np.flatnonzero(~inside)
        result.update(sustained_upright=True,
                      settling_time=float(t[outside[-1] + 1] if len(outside) else t[0]))
    return result
