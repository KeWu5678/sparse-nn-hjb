# VDP Open-Loop Smooth Benchmark

The Van der Pol open-loop problem provides the smooth value-function dataset
for PDAP tests. Pendulum covers the nonsmooth case; VDP uses one fixed
terminal condition when generating the default training dataset.

## Problem

`VdpOptimalControlProblem` owns the mathematics: dynamics
`y1' = y2`, `y2' = -y1 + mu (1 - y1^2) y2 + u`, running cost
`state_weight |y|^2 + beta u^2`, no terminal cost (`p(T) = 0`), the costate
equation, and the stationarity residual `p2 + 2 beta u`. Defaults are
`beta = 0.1`, `mu = 1`, `state_weight = 1`, and `T_final = 3`.

## Solver

`VdpOpenLoopSolver` solves the free-terminal Pontryagin boundary-value problem
with the control eliminated, `u = -p2 / (2 beta)`, and boundary conditions
`y(0) = x`, `p(T) = 0`. It uses SciPy `solve_bvp` from two initial guesses
(state decaying as `exp(-t)`, and constant) and retains the least-cost
converged solution. The value is integrated with three-point Gauss–Legendre
quadrature on the collocation mesh; the gradient label is `dV(x) = p(0)`. The
result is a numerical stationary reference, not a certificate of global
optimality.

Collocation is the only VDP open-loop method; the reduced-gradient profiles
(Barzilai–Borwein and L-BFGS-B) were retired in ADR 0017.

## Dataset

`scripts/generate_vdp_data.py` solves a `30 x 30` grid on `[-3, 3]^2` with 151
initial mesh points and fails if any initial state does not converge. Each run
writes a folder `rawdata/data/VDP_<date>_<uuid>/` containing
`VDP_pmp_grid_30x30_<date>.npz` with arrays `x`, `v = V(x)`, and
`dv = dV(x)`, plus metadata (including data and generator hashes) and a
failed-sample sidecar.

Generation does not update `conf/data/vdp.yaml`; point the training config at
a new dataset manually after inspecting it.

## Retired

- The `DataGenerator` class and the root-level `bvp_bb_optimizer.py`.
- The two-target phase-capture minimum, which could create an artificial
  nonsmooth dataset.
- The structured `.npy` dataset format; `src/data.py` still reads it as legacy.
