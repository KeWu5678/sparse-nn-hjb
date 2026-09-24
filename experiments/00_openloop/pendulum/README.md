# Open-loop data — pendulum swing-up

`conf/data/pendulum.yaml` selects the September 24, 2026 production dataset:
`rawdata/data/Pendulum_20260924_ebf42e75ab9748aa855f44aa2387b6df/Pendulum_pmp_value_samples_2000_20260924.npz`.
The same directory holds its 2,000 raw trajectories, native switching curve,
basin polygon, generation metadata, and distance/evaluation artifacts.

## Numerical problem and generation

The infinite-horizon running cost is `2 - 2 cos(theta) + omega² + u²`, with
unit mass and length, damping 0.1, gravity 9.8, and unconstrained control.
Backward PMP integration starts on the local LQR level `epsilon=2e-4`.
Adaptive boundary sampling uses the value-20 contour. Each characteristic is
integrated until value 100 or backward time 50, with maximum step 0.005 and
relative/absolute tolerances `1e-10`/`1e-12`.

From the repository root:

```sh
uv run python scripts/run_pendulum_pmp_openloop_example.py \
  --num-trajectories 2000 --contour-delta 0.05 --basin-value-max 70 \
  --level-set-samples 3900 --tag 20260924
```

The generator creates a new dated directory and prints its paths. For a later
refresh, choose a new tag and update the data and evaluation configurations to
the new outputs before training.

The sample construction is:

1. Integrate the 2,000 characteristics, retaining numerical state, value, and
   costate data for each branch.
2. Intersect equal-value contours with their `2pi` translates, track the four
   switching arms at value spacing 0.05, and assemble the origin's basin using
   the arm extent setting 70. Geometry comes from these same 2,000 fresh paths.
   An empty basin stops generation. The setting 70 is not a sample-value cap.
3. Truncate each path at its first basin exit. Screen all retained body points
   against nearby raw branches before thinning: first-order value estimates
   use the nearest 32 points within radius 0.1. A point must lie within 0.3 of
   its own local lower envelope and beat each available adjacent branch by 0.3.
   Interior body points without an adjacent-branch neighbour remain eligible.
4. Harvest samples within distance 0.5 of the switching arms and their `+/-2pi`
   translates using the same local comparisons. The central branch beyond its
   first-exit prefix supplies the near-side pad; translated branches outside
   the basin supply the far-side collar. Every pad/collar point must have an
   available competing branch.
5. Thin the pools separately to 3,000 body, 300 pad, and 600 collar samples.

The current pools contain 881,672 body, 6,904 pad, and 129,156 collar points.
These local branch checks establish a numerical consistency criterion, not a
certificate of globally optimal values.

## Sampling variants and evaluation inputs

Use `scripts/investigation/make_twosided_oversampling_sets.py --data <production
NPZ relative to rawdata/data> --tag <new tag>` to build a new variant directory.
The current variants are under `rawdata/data/Pendulum_2sided_oversample_20260924/`:

| Variant | Body | Pad | Collar | Total |
| --- | ---: | ---: | ---: | ---: |
| `base6k` | 4,615 | 462 | 923 | 6,000 |
| `band40` | 3,600 | 800 | 1,600 | 6,000 |
| `band60` | 2,400 | 1,200 | 2,400 | 6,000 |
| `add2k` | 4,615 | 1,129 | 2,256 | 8,000 |

`add2k` preserves every `base6k` row and adds 2,000 distinct band samples.
The other variants reallocate a fixed sample budget. Their nominal band shares
refer to pad/collar allocation within the generation width 0.5; they are not
the fraction inside the evaluation tube of radius 0.3. The builder refuses an
existing output directory.

For the production NPZ and each variant, generate a distance cache with
`precompute_region_distances.py --data <NPZ>`. Then run
`build_region_eval_pool.py --data <production NPZ>`, supplying a separate
`--exclude <variant NPZ>` for all four variants and `--out <shared pool NPZ>`.
These scripts live under `scripts/investigation/`; all paths are relative to
`rawdata/data/`. The common pool removes the union of all five sample sets.
Update `conf/eval/region_split.yaml` and each `conf/data/pendulum*.yaml` to the
new shared pool and aligned caches. The region split uses distance at most
0.3 to the switching-set points tiled by `-2pi`, zero, and `+2pi`.

## Plot the configured reference data

```sh
uv run python experiments/00_openloop/pendulum/generate.py
```

`make openloop` also runs this plotter and the VDP plotter. It reads existing
configured artifacts; it does not generate new data or fit a model. Only the
following PNGs are written under `paper/plot/`:

| File | Content |
| --- | --- |
| `pendulum_value_scatter.png` | The configured 3,900 state/value samples |
| `pendulum_value_surface.png` | Periodic interpolation of the configured sample values |
| `pendulum_regions.png` | Nearest-characteristic visualization of periodically translated upright basins |

The surface folds angles into one period, interpolates with seam copies, and
tiles the result on `[-8,8]^2`. Its display construction restricts angular
velocity samples to `[-7.7,7.7]` and clips grid queries to that interval. This
plotting window is not the training support: the fresh samples extend beyond
`[-9,9]^2`. The regions panel uses the saved basin and co-located raw paths;
it does not inject an older switching curve or recompute one at a different cap.

Learned-controller comparisons are generated by
`scripts/paper/pendulum_full_scope.py`. Each controller follows its own
trajectory from each of two starts over `T=10`. The open-loop comparison is a
finite-horizon numerical reference with free terminal state and zero terminal
cost. See the [experiment workflow](../../README.md) for training and validation.

The old branch-restriction diagnosis and superseded data are preserved in the
ignored `outdated/experiment-refresh-20260924/` archive; they are not inputs to
this workflow.
