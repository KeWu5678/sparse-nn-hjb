# 6. Retain the radial search clamp under the moment penalty

Date: 2026-07-25

## Status

Accepted

## Context

The greedy insertion step searches candidate atoms over the parameter domain.
For positively homogeneous activations the atom is determined by its direction,
so the search runs on the compact sphere `S^d`. For non-homogeneous activations
(tanh, softplus, Gaussian, Matérn) the radial scale is a genuine shape
parameter, so the search must also range over `r > 0`.

Without a location-coercive penalty that search need not attain its maximum:
`sup_r |P_t(r·ω̂)|` can stay bounded away from zero as `r → ∞`, which is the
escape mechanism the paper's Example 3.x isolates. The implementation therefore
clamps the log-scale to `s ∈ [-3, 5]`, i.e. `r ∈ [e^-3, e^5] ≈ [0.05, 148]`
(`src/PDAP/insertion.py:160`), which truncates the computational dictionary to
a compact set and makes the subproblem well posed.

The parameter-moment penalty `β·Ψ_p` changes this. The insertion objective
becomes `|P_t(r·ω̂)| − β·w_p(r·ω̂)`, which tends to `−∞` as `r → ∞` whenever
`p > s_1`, so the subproblem is coercive on its own and the clamp is
unnecessary in principle. Removing it would make theory and implementation
agree exactly.

Against that: the confinement lemma bounds the support radius by
`R_* = (2^{s_1}·C_P/β)^{1/(p−s_1)}`, which is finite but scales like
`β^{−1/(p−s_1)}`. For softplus at `p = 2.01` this is of order `10^10` at
`β = 1e-10` — ten orders of magnitude beyond the clamp. So at the small `β`
where the best accuracy sits, the guarantee is numerically vacuous and the
clamp is load-bearing. Removing it there would send atoms to large finite
radii, not to infinity, and would change those cells materially.

## Decision

Keep the clamp at `s ∈ [-3, 5]` for all cells of the sweeps, including `β > 0`.

Report it in the paper for what it is: the compactness substitute that the
`β = 0` model requires, retained under `β > 0` so that every cell — including
the `β = 0` baselines, which have no confinement of their own — is searched
over one common dictionary.

Cells in which at least 95% of the amplitude mass sits at the ceiling
`r = e^5` are recorded in the raw grid but excluded from model selection, on
the stated grounds that there the clamp rather than `β` confines the support.

## Consequences

- The `β = 0` versus `β > 0` comparison is over a single dictionary, so
  differences are attributable to the objective rather than to the search
  space. This is what makes the matched control study
  (`experiments/02_pendulum/moment_penalty/control.py`) interpretable.
- No re-runs. The 272-cell screen, the refinement, and the follow-up stages
  remain valid.
- The paper cannot claim that the moment term is demonstrated to make the
  greedy step well posed — only that it is proved to. The demonstration would
  require dropping the clamp for `β > 0` and re-running both benchmarks.
- `R95` at or near 148 is a diagnostic, not a defect: it marks cells where the
  clamp binds. The censoring rule depends on it, so the ceiling value must not
  be changed without re-deriving the affected tables.
- The `p` axis becomes the reportable confinement evidence, since
  `1/(p − s_1)` controls how fast the guarantee degrades as `β → 0`.

## Alternatives considered

**Drop the upper clamp for `β > 0` and re-run.** Theory and implementation
would agree exactly and the fits would go wherever the moment term puts them.
Rejected for now: a full re-run of screen, refinement, and follow-up on both
benchmarks, and the `β = 0` baselines would still need the clamp, so the
comparison would no longer be over a common dictionary.

**Drop it everywhere, `β = 0` included.** Maximally consistent, and the `β = 0`
rows would exhibit the escape directly instead of having it suppressed.
Rejected: the `β = 0` runs on tanh/softplus may not terminate usefully, which
would cost the accuracy baselines that anchor the tables.
