"""Paper-aligned pendulum PMP value-sample solver."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from src.OpenLoop.pendulum.nonsmooth import (
    NonsmoothCurve,
    compute_nonsmooth_curve,
    restrict_trajectory_to_curve,
)
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.pendulum.trajectories import (
    PmpTrajectory,
    adaptive_boundary_angles,
    integrate_pmp_trajectory,
    uniform_boundary_angles,
)
from src.OpenLoop.value_samples import ValueSamples
from src.paths import DATA_DIR


@dataclass(frozen=True)
class PendulumPmpSolverConfig:
    """Numerical configuration for one pendulum value-sample run."""

    epsilon: float = 2e-4
    value_max: float = 100.0
    t_final: float = 50.0
    max_step: float = 0.005
    rtol: float = 1e-10
    atol: float = 1e-12
    num_trajectories: int = 256
    adaptive_sampling: bool = True
    reference_value: float | None = None
    boundary_distance_power: float = 0.8
    contour_delta: float = 1.0
    periodic_copies: int = 0
    # Switching-set band: harvest envelope-certified samples on BOTH sides of the
    # true arms — near-side pad (central branch beyond the basin's conservative
    # trim) and far-side collar (neighbouring ±2π branch across the arms); see
    # ``build_collar_samples``. Width = max distance from the tiled ridge;
    # 0 disables. Global tiling of the emitted samples (periodic_copies > 0) is
    # NOT the mechanism for two-sided coverage: it multiplies the training
    # domain by identical bowls and collapses the fit.
    collar_width: float = 0.5
    # Value cap on the switching-set arms used to build the upright smooth basin.
    # The reference's hand-tuned trim (boundary_spiral(1:1252)) reaches value ~57
    # (theta-dot ~7.7), matching Han & Yang Fig. 2; 35 (the earlier default) trims
    # to value ~31 / theta-dot ~3.5 — too small. The assembly is unstable at some
    # caps (e.g. 65, 95 collapse); 50 is validated for these parameters (issue #18).
    basin_value_max: float = 50.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.value_max <= self.epsilon:
            raise ValueError("value_max must be larger than epsilon")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("rtol and atol must be positive")
        if self.num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive")
        if self.boundary_distance_power <= 0.0:
            raise ValueError("boundary_distance_power must be positive")
        if self.contour_delta <= 0.0:
            raise ValueError("contour_delta must be positive")
        if self.periodic_copies < 0:
            raise ValueError("periodic_copies must be nonnegative")
        if self.collar_width < 0.0:
            raise ValueError("collar_width must be nonnegative")
        if self.basin_value_max <= self.epsilon:
            raise ValueError("basin_value_max must be larger than epsilon")


@dataclass(frozen=True)
class SolverDiagnostics:
    requested_trajectories: int
    integrated_trajectories: int
    failed_trajectories: tuple[dict[str, object], ...]
    nonsmooth_points: int
    discarded_points: int
    retained_points: int


@dataclass(frozen=True)
class PendulumValueSolution:
    value_samples: ValueSamples
    raw_trajectories: tuple[PmpTrajectory, ...]
    restricted_trajectories: tuple[PmpTrajectory, ...]
    nonsmooth_curve: NonsmoothCurve
    diagnostics: SolverDiagnostics
    # Envelope-certified samples straddling the switching set (see
    # ``build_collar_samples``), kept separate from ``value_samples`` so the
    # generator can control the near-switch share when thinning to the budget:
    # ``pad_samples`` = near-side (central branch, beyond the conservative trim),
    # ``collar_samples`` = far-side (neighbouring branch, across the arms).
    pad_samples: ValueSamples | None = None
    collar_samples: ValueSamples | None = None

    def save_dataset(
        self,
        output_dir: str | Path = DATA_DIR,
        *,
        date_tag: str | None = None,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        date = datetime.now().strftime("%Y%m%d") if date_tag is None else date_tag
        run_dir = output_path / f"Pendulum_{date}_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)

        stem = f"Pendulum_pmp_value_samples_{self.diagnostics.requested_trajectories}_{date}"
        data_path = self.value_samples.save_npz(run_dir / f"{stem}.npz")
        curve_path = self.nonsmooth_curve.save_npz(run_dir / f"{stem}_nonsmooth_curve.npz")
        raw_path = run_dir / f"Pendulum_pmp_raw_trajectories_{self.diagnostics.requested_trajectories}_{date}.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(self.raw_trajectories, f)
        meta_path = run_dir / f"Pendulum_pmp_value_samples_meta_{date}.json"
        failed_path = run_dir / f"Pendulum_pmp_value_samples_failed_{date}.json"

        meta = {
            "requested_trajectories": self.diagnostics.requested_trajectories,
            "integrated_trajectories": self.diagnostics.integrated_trajectories,
            "nonsmooth_points": self.diagnostics.nonsmooth_points,
            "discarded_points": self.diagnostics.discarded_points,
            "retained_points": self.diagnostics.retained_points,
            "nonsmooth_curve": str(curve_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        failed_path.write_text(
            json.dumps(self.diagnostics.failed_trajectories, indent=2),
            encoding="utf-8",
        )
        return {
            "run_dir": run_dir,
            "data": data_path,
            "curve": curve_path,
            "meta": meta_path,
            "failed": failed_path,
            "raw": raw_path,
        }


class PendulumPmpSolver:
    """Run the paper's backward-PMP construction and emit `ValueSamples`."""

    def __init__(
        self,
        problem: PendulumSwingUpProblem | None = None,
        config: PendulumPmpSolverConfig | None = None,
    ) -> None:
        self.problem = problem or PendulumSwingUpProblem()
        self.config = config or PendulumPmpSolverConfig()

    def solve(
        self,
        progress: Callable[[int, int], None] | None = None,
    ) -> PendulumValueSolution:
        raw, failures = self.generate_raw_trajectories(progress=progress)
        curve = self.compute_nonsmooth_curve(raw)
        samples, restricted, discarded = self.build_value_samples(raw, curve)
        pad, collar = self.build_collar_samples(raw, restricted, curve)
        diagnostics = SolverDiagnostics(
            requested_trajectories=self.config.num_trajectories,
            integrated_trajectories=len(raw),
            failed_trajectories=tuple(failures),
            nonsmooth_points=int(curve.points.shape[0]),
            discarded_points=int(discarded),
            retained_points=int(samples.size),
        )
        return PendulumValueSolution(
            value_samples=samples,
            raw_trajectories=raw,
            restricted_trajectories=restricted,
            nonsmooth_curve=curve,
            diagnostics=diagnostics,
            pad_samples=pad,
            collar_samples=collar,
        )

    def generate_raw_trajectories(
        self,
        angles: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[tuple[PmpTrajectory, ...], tuple[dict[str, object], ...]]:
        angle_values = self.sample_boundary_angles(progress=progress) if angles is None else angles
        if len(angle_values) != self.config.num_trajectories:
            raise ValueError("angles must have length num_trajectories")

        trajectories: list[PmpTrajectory] = []
        failures: list[dict[str, object]] = []
        for idx, angle in enumerate(np.asarray(angle_values, dtype=np.float64)):
            if progress is not None:
                progress(idx, len(angle_values))
            try:
                trajectory = self._integrate_angle(float(angle), trajectory_id=idx)
            except Exception as exc:
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": str(exc),
                    }
                )
                continue
            if trajectory.success:
                trajectories.append(trajectory)
            else:
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": trajectory.message,
                    }
                )
        if progress is not None:
            progress(len(angle_values), len(angle_values))
        return tuple(trajectories), tuple(failures)

    def sample_boundary_angles(
        self,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if not self.config.adaptive_sampling:
            return uniform_boundary_angles(self.config.num_trajectories)

        # The adaptive sampler uses a reference value contour to spread raw PMP
        # trajectories in the state plane instead of evenly spacing boundary angles.
        reference_value = (
            min(20.0, self.config.value_max)
            if self.config.reference_value is None
            else float(self.config.reference_value)
        )
        return adaptive_boundary_angles(
            lambda angle: self._integrate_angle(angle, trajectory_id=-1),
            self.problem,
            self.config.num_trajectories,
            self.config.epsilon,
            reference_value,
            boundary_distance_power=self.config.boundary_distance_power,
            progress=progress,
        )

    def compute_nonsmooth_curve(
        self,
        trajectories: tuple[PmpTrajectory, ...],
    ) -> NonsmoothCurve:
        return compute_nonsmooth_curve(
            trajectories,
            value_delta=self.config.contour_delta,
            value_max=self.config.value_max,
            basin_value_max=self.config.basin_value_max,
        )

    def build_value_samples(
        self,
        trajectories: tuple[PmpTrajectory, ...],
        curve: NonsmoothCurve,
    ) -> tuple[ValueSamples, tuple[PmpTrajectory, ...], int]:
        restricted: list[PmpTrajectory] = []
        discarded = 0
        for trajectory in trajectories:
            cut, count = restrict_trajectory_to_curve(trajectory, curve)
            restricted.append(cut)
            discarded += count

        samples = self._trajectories_to_value_samples(tuple(restricted))
        return samples, tuple(restricted), discarded

    def build_collar_samples(
        self,
        raw: tuple[PmpTrajectory, ...],
        restricted: tuple[PmpTrajectory, ...],
        curve: NonsmoothCurve,
        *,
        competitor_radius: float = 0.1,
        competitor_neighbors: int = 32,
        margin: float = 0.3,
    ) -> tuple[ValueSamples, ValueSamples]:
        """Envelope-certified samples straddling the switching set.

        Candidates are the raw trajectories tiled by k ∈ {−1, 0, +1} in θ, gated
        to points within ``collar_width`` of the *tiled* switching set
        (ridge ∪ ridge ± 2π) — anchoring on the ridge, not the basin ring, is
        essential: long ring stretches are value-cap trims ~1 unit short of the
        true arm. A candidate from tile k is certified envelope-optimal iff, with
        branch values estimated by first-order extrapolation from the
        ``competitor_neighbors`` nearest points within ``competitor_radius``:

        * it beats every OTHER tile's branch by ``margin`` (a candidate with no
          other-tile competitor in reach is dropped rather than trusted), and
        * it matches its OWN branch's local lower envelope within ``margin`` —
          raw trajectories re-enter after exiting the basin, so a post-exit point
          can be beaten by a different sheet of its own branch.

        Returns ``(pad, collar)``. k = 0 survivors not already retained by the
        restriction are the *near-side pad*: the central branch is still optimal
        between the basin's conservative trim and the true arm, but those points
        lie beyond each trajectory's first exit (the strip is inside the polygon
        yet unreachable by the first-exit prefix), leaving a hole exactly where
        two-sided coverage is needed. k = ±1 survivors outside the basin are the
        *far-side collar* across the arms. Both sets hug arm stretches where both
        branches carry data — precisely where a kink can be learned.
        """
        import shapely
        from scipy.spatial import cKDTree

        basin = curve.basin_polygon()
        if self.config.collar_width <= 0.0 or basin is None or curve.points.shape[0] == 0:
            empty = ValueSamples.concatenate([])
            return empty, empty

        x0 = np.vstack([tr.state for tr in raw])
        v0 = np.concatenate([tr.value for tr in raw])
        dv0 = np.vstack([tr.costate for tr in raw])
        branch_tree = cKDTree(x0)
        # Raw points already emitted by the restriction (per-trajectory prefixes).
        in_prefix = np.concatenate(
            [
                np.arange(tr.state.shape[0]) < cut.state.shape[0]
                for tr, cut in zip(raw, restricted)
            ]
        )

        period = np.array([2.0 * np.pi, 0.0])
        ridge_tiled = np.vstack([curve.points + k * period for k in (-1, 0, 1)])
        ridge_tree = cKDTree(ridge_tiled)

        def branch_value_at(x: np.ndarray, tile: int) -> np.ndarray:
            """Tile-``tile`` branch value at states ``x``, +inf where no data."""
            dist, nn = branch_tree.query(
                x - tile * period,
                k=competitor_neighbors,
                distance_upper_bound=competitor_radius,
            )
            found = np.isfinite(dist)
            nn_safe = np.where(found, nn, 0)
            value = v0[nn_safe] + np.einsum(
                "cnd,cnd->cn", dv0[nn_safe], (x - tile * period)[:, None, :] - x0[nn_safe]
            )
            return np.where(found, value, np.inf).min(axis=1)

        pad_chunks: list[ValueSamples] = []
        collar_chunks: list[ValueSamples] = []
        for tile in (-1, 0, 1):
            xs = x0 + tile * period
            ridge_dist, _ = ridge_tree.query(xs, k=1)
            candidate = ridge_dist <= self.config.collar_width
            if tile == 0:
                candidate &= ~in_prefix
            else:
                candidate &= ~shapely.covers(basin, shapely.points(xs))
            idx = np.flatnonzero(candidate)
            if idx.size == 0:
                continue
            xc = xs[idx]
            competitor = np.full(idx.size, np.inf)
            for other in (-1, 0, 1):
                if other != tile:
                    competitor = np.minimum(competitor, branch_value_at(xc, other))
            own = branch_value_at(xc, tile)
            keep = np.isfinite(competitor) & (v0[idx] < competitor - margin)
            keep &= v0[idx] < own + margin
            if keep.any():
                chunk = ValueSamples(x=xc[keep], v=v0[idx][keep], dv=dv0[idx][keep])
                (pad_chunks if tile == 0 else collar_chunks).append(chunk)
        return ValueSamples.concatenate(pad_chunks), ValueSamples.concatenate(collar_chunks)

    def _integrate_angle(self, angle: float, trajectory_id: int) -> PmpTrajectory:
        return integrate_pmp_trajectory(
            self.problem,
            angle=angle,
            epsilon=self.config.epsilon,
            value_max=self.config.value_max,
            t_final=self.config.t_final,
            max_step=self.config.max_step,
            rtol=self.config.rtol,
            atol=self.config.atol,
            trajectory_id=trajectory_id,
        )

    def _trajectories_to_value_samples(
        self,
        trajectories: tuple[PmpTrajectory, ...],
    ) -> ValueSamples:
        chunks: list[ValueSamples] = []
        for trajectory in trajectories:
            for copy_index in range(-self.config.periodic_copies, self.config.periodic_copies + 1):
                states = trajectory.state.copy()
                states[:, 0] += 2.0 * np.pi * copy_index
                chunks.append(
                    ValueSamples(
                        x=states,
                        v=trajectory.value.copy(),
                        dv=trajectory.costate.copy(),
                    )
                )
        return ValueSamples.concatenate(chunks)


__all__ = [
    "PendulumPmpSolver",
    "PendulumPmpSolverConfig",
    "PendulumValueSolution",
    "SolverDiagnostics",
]
