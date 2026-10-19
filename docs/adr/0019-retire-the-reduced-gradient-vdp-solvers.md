---
status: accepted
---

# Retire the reduced-gradient VDP solvers

`VdpOpenLoopSolver` used to choose among three open-loop algorithms through a
`profile` flag: `"paper"` (Barzilai–Borwein descent on time-grid controls with
Crank–Nicolson state and costate solves), `"fast"` (L-BFGS-B on Legendre
control coefficients), and `"pmp"` (Pontryagin collocation). Only `"pmp"` was
used to generate data: the current dataset
`VDP_pmp_grid_30x30_20260924.npz` and the manuscript's VDP section both rely
on collocation, while `"paper"` and `"fast"` were exercised only by tests.

We decided to delete the `"paper"` and `"fast"` profiles and their
configuration, so collocation is the only open-loop reference method for the
smooth finite-horizon benchmarks. This removes a class that selected unrelated
algorithms with a flag and a configuration in which each profile read a
different subset of fields. The `"paper"` profile also rejected some accurately
solved Crank–Nicolson steps; that defect was fixed only on the unmerged branch
`fix/vdp-bb-consistency`, which is kept as a record and is not to be merged.

## Considered Options

Keeping Barzilai–Borwein as an optional optimizer backend behind one evaluator,
as the earlier VDP design note proposed, was rejected because no dataset,
experiment, or manuscript result uses it, and the comparison between
reduced-gradient descent and collocation is no longer pursued.
