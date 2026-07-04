# Open-loop data — pendulum swing-up

Visualisations of the pendulum swing-up open-loop value data and the switching-set
geometry it traces out (backward-PMP, `src/OpenLoop/pendulum`). These follow the
paper's own figure code (Han & Yang, arXiv:2312.17467, `github_main_nosat.m`) and show
the **data only** — no learned model. The dataset path is read from
`conf/data/pendulum.yaml` (currently the 2000-path, 3,900-sample **two-sided** set
under `rawdata/data/Pendulum_20260703_.../`; construction below); the trajectory
figures use the co-located raw-trajectory pickle (2000 ordered paths). Regenerate
with `python generate.py` (or `make openloop`). Figures carry no titles.

| file | what it shows |
| --- | --- |
| `figures/value_scatter.png` | 3D scatter of the raw samples (θ, θ̇, V), coloured by value |
| `figures/value_surface.png` | V(θ, θ̇) over the state plane (the 3000-sample set, periodic-folded into one cell then tiled) |
| `figures/trajectories.png` | every backward-PMP characteristic plotted whole (paper's `plot(θ, θ̇)`), multicolored — the curves spiral into the centers (paper Fig. 2, right) |
| `figures/regions_of_attraction.png` | each state coloured by the upright it belongs to (nearest basin-cut characteristic, tiled by 2πk); the boundaries are the switching-set spirals winding around the hanging points ±π, ±3π (paper Fig. 2, left) |

The regions figure tracks the switching set deeper (`_REGIONS_CAP = 80`) than the wired
training data (`basin_value_max = 50`, which only resolves ~half a spiral turn); the deeper
cut recovers the multi-winding spiral. This is for visualisation only — it does not change
the training samples.

## How the two-sided training set is constructed

The wired dataset (`scripts/run_pendulum_pmp_openloop_example.py`, post-processing
in `src/OpenLoop/pendulum/solver.py`) is built in five steps. The goal of steps
4–5 is to put the gradient jump **in-sample**: the basin restriction alone yields
one-sided data — every trajectory stops *at* the switching curve, so no model
would ever see the jump.

1. **Raw trajectories.** 2000 backward-PMP characteristics are integrated from
   the local LQR boundary `∂L_ε` around the upright until the accumulated cost
   hits `value_max = 100`. Each point carries the exact `(x, V, ∇V)` of *its*
   branch — beyond the switching curve that branch is no longer optimal, but its
   values remain exact branch data.
2. **Switching curve.** Equal-value contours are intersected with their
   2π-shifted copies; the crossings, arm-tracked across value levels, are the
   switching-set spirals, and the reference reflection assembly closes the
   upright smooth basin (the 2000-path auto-assembly is unstable — issue #18 —
   so the validated 256-path basin is injected).
3. **Basin restriction (the one-sided body).** Each trajectory is truncated at
   its **first exit** from the basin polygon (the reference `inpolygon` cut).
   The retained prefixes form the in-basin body pool (~823k points): the smooth
   branch of the upright well, stopping at the curve.
4. **Envelope-certified band (the two sides).** Candidates are the *raw*
   trajectories tiled by k ∈ {−1, 0, +1}·2π in θ, gated to within 0.5 of the
   **±2π-tiled** switching arms. A candidate from tile k is kept only if it is
   certified envelope-optimal there: its branch value must beat every *other*
   tile's branch value (first-order extrapolation from the nearest raw points)
   by a margin, **and** match its *own* branch's local lower envelope (raw
   trajectories re-enter after exiting, so a post-exit point can be beaten by
   another sheet of its own branch). Survivors split into
   - the **near-side pad** (k = 0 beyond the first-exit prefix): the central
     branch is still optimal in the strip between the basin's conservative
     value-cap trim and the true arm, but step 3 discards it;
   - the **far-side collar** (k = ±1 outside the basin): the neighbouring
     upright's branch across the arm — the other side of the jump.
   Anchoring the gate on the tiled *ridge* rather than the basin ring matters:
   long ring stretches are value-cap trims ~1 unit short of the true arm, and a
   collar harvested from the *restricted* trajectories would miss every arm
   stretch whose far-side value exceeds the basin cap. Global tiling of the
   emitted samples (`periodic_copies`) is **not** a substitute: it multiplies
   the training domain by identical wells and collapses the fit.
5. **Thinning to the emitted set.** Body, pad, and collar pools are thinned
   *separately* to the requested shares — the production set is
   3,000 body + 300 pad + 600 collar = 3,900 — so the band share is a design
   parameter rather than the pools' incidental proportion. Each sample carries
   a precomputed distance to the ±2π-tiled switching set
   (`scripts/investigation/precompute_region_distances.py`).

Verified on the production set: 0/900 band samples violate the
first-order-corrected lower envelope; the lowest-decile band (d ≤ 0.25)
straddles the curve (221 near-side / 169 far-side); 44% of samples within 0.3
of the ridge have an opposite-side neighbour within 0.3 (0% in the one-sided
data). The residual one-sided stretches are arms whose far branch lies beyond
the `value_max` integration cap. Downstream use and findings:
`experiments/03_region_split_pendulum/`.
