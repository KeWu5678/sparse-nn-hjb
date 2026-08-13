# 6. Retain the fixed radial search bound

Date: 2026-07-25

## Status

Accepted. Amended 2026-08-12 — see "Amendment: the revised objective" below.
The decision stands; its justification now rests on `alpha` rather than on the
`beta` this record was written against.

## Context

The greedy insertion step searches candidate atoms over the parameter domain.
For positively homogeneous activations the atom is determined by its direction,
so the search runs on the compact sphere `S^d`. For non-homogeneous activations
(tanh, softplus, Gaussian, Matérn) the radial scale is a genuine shape
parameter, so the search must also range over `r > 0`.

Without a location-coercive penalty that search need not attain its maximum:
`sup_r |P^M_μ(r·ω̂)|` can stay bounded away from zero as `r → ∞`, which is the
escape mechanism the paper's Example 3.x isolates. The implementation therefore
clamps the log-scale to `s ∈ [-3, 5]`, i.e. `r ∈ [e^-3, e^5] ≈ [0.05, 148]`
(`src/PDAP/insertion.py:160`), which truncates the computational dictionary to
a compact set and makes the subproblem well posed.

The retired additive-moment comparator changes this. Its insertion objective
becomes `|P^M_μ(r·ω̂)| − β·w_p(r·ω̂)`, which tends to `−∞` as `r → ∞` whenever
`p > s_1`, so the subproblem is coercive on its own and the clamp is
unnecessary in principle. Removing it would make theory and implementation
agree exactly.

Against that: the bounded-support argument gives the radius
`R_* = (2^{s_1}·C_P/β)^{1/(p−s_1)}`, which is finite but scales like
`β^{−1/(p−s_1)}`. For softplus at `p = 2.01` this is of order `10^10` at
`β = 1e-10` — ten orders of magnitude beyond the clamp. So at the small `β`
where the best accuracy sits, the guarantee is numerically vacuous and the
clamp is load-bearing. Removing it there would send atoms to large finite
radii, not to infinity, and would change those runs materially.

## Decision

Keep the bound at `s ∈ [-3, 5]` for all runs of the sweeps, including `β > 0`.

Report it in the paper for what it is: the compactness substitute that the
`β = 0` model requires, retained under `β > 0` so that every run — including
the `β = 0` comparison runs, which have no location-coercive term of their own — is searched
over one common dictionary.

Runs in which at least 95% of the total variation sits at the ceiling
`r = e^5` are recorded in the raw grid but excluded from model selection, on
the stated grounds that there the clamp rather than `β` confines the support.

## Consequences

- The `β = 0` versus `β > 0` comparison is over a single dictionary, so
  differences are attributable to the objective rather than to the search
  space. This is what makes the matched control study
  (`experiments/02_pendulum/moment_penalty/control.py`) interpretable.
- No re-runs. The 272-run screen, the refinement, and the follow-up stages
  remain valid.
- The paper cannot claim that the moment term is demonstrated to make the
  greedy step well posed — only that it is proved to. The demonstration would
  require dropping the clamp for `β > 0` and re-running both benchmarks.
- `R95` at or near 148 is a diagnostic, not a defect: it marks runs where the
  clamp binds. The censoring rule depends on it, so the ceiling value must not
  be changed without re-deriving the affected tables.
- The `p` axis becomes the reportable bounded-support evidence, since
  `1/(p − s_1)` controls how fast the guarantee degrades as `β → 0`.

## Alternatives considered

**Drop the upper clamp for `β > 0` and re-run.** Theory and implementation
would agree exactly and the fits would go wherever the moment term puts them.
Rejected for now: a full re-run of screen, refinement, and follow-up on both
benchmarks, and the `β = 0` comparison runs would still need the bound, so the
comparison would no longer be over a common dictionary.

**Drop it everywhere, `β = 0` included.** Maximally consistent, and the `β = 0`
rows would exhibit the escape directly instead of having it suppressed.
Rejected: the `β = 0` runs on tanh/softplus may not terminate usefully, which
would cost the accuracy comparisons that anchor the tables.

## Amendment: the revised objective

Date: 2026-08-12

The manuscript's Sections 3–5 no longer carry an additive moment term. The
moment norm defines the measure space, and the objective is
`J = L(μ) + α·Φ_φ(μ_p)` with `μ_p = w_p·μ`. So the support-radius bound this
record weighed — `R_* = (2^{s₁}·C_P/β)^{1/(p−s₁)}`, scaling like
`β^{−1/(p−s₁)}` — no longer exists, and the argument above cannot be evaluated
as written.

Its replacement is the quantitative insertion theorem's radius, which is driven
by `α` rather than `β`:

```
R(μ) = max{1, (2^{s₁+1}·C·‖r_μ‖ / (α·L_φ))^{1/(p − s₁)}},
C = C_ρ(A_M^{s₀} + A_M^{s₁−1}).
```

The conclusion is unchanged, for a structurally identical reason. `α` is small
exactly where the best accuracy sits, so the radius is enormous there. Measured
on `VDP_beta_0.1_grid_30x30` after max-abs normalization, with softplus
(`C_ρ = 1`, `s₀ = s₁ = 1`): the sample extent is `A_M = √3 = 1.7321` — the
normalizer puts every coordinate in `[-1,1]`, so `|x| ≤ √2` — giving
`C = 2.7321`, and `‖r_μ‖ = 1.6216` at the zero network, where the radius is
largest.

| `p` | `1/(p − s₁)` | `R(μ)` at `α = 1e-5` | binds against `e⁵ ≈ 148`? |
|---|---|---|---|
| 2.01 | 0.99 | 1.5×10⁶ | no |
| 2.5 | 0.67 | 1.5×10⁴ | no |
| 3 | 0.5 | 1.3×10³ | no |
| 4 | 0.33 | 1.2×10² | **yes** |

So the clamp remains load-bearing across most of the sweep, and
`training.radial_cap=theorem` is applied as `min(R(μ), e⁵)`: the theorem can only
ever tighten the search, never loosen it.

What changes is that the guarantee is now demonstrable somewhere. At `p = 4` the
theorem radius binds strictly inside the clamp for every `s₁ = 1` activation, so
those runs are searched over a region the theory certifies rather than one the
implementation imposes:

| activation | `s₁` | `p = 4`, `α = 1e-4` | `p = 4`, `α = 1e-5` |
|---|---|---|---|
| softplus | 1 | R = 56.5 | R = 121.7 |
| tanh, gaussian | 1 | R = 50.9 | R = 109.7 |
| gelu_squared | 2 | clamp (148.4) | clamp (148.4) |

`gelu_squared` is the exception in both directions: `s₁ = 2` makes the exponent
`1/(p − s₁)` equal to 100 at `p = 2.01` (a valid but vacuous bound) and still only
0.5 at `p = 4`, which is not enough to reach inside the clamp.

The `p`-study measures what this costs. It costs nothing: at `p = 4`, where the
search is restricted to the certified region, the fits are no worse than at
`p = 2.01`, where the clamp governs (`softplus` 0.1158 against 0.1246,
`tanh` 0.1078 against 0.1173, `gaussian` 0.1012 against 0.1104 relative H¹). So
the theorem radius is not merely valid but harmless, and at `p = 4` the
bounded-search claim is demonstrated rather than only proved. Elsewhere the honest
statement remains the one this record already makes.

No threshold guards the vacuous case: the value is a valid upper bound, merely an
uninformative one, and `min(R(μ), e⁵)` reduces it to the clamp with no special
case. `p ≤ s₁` fails the theorem's hypothesis outright, and there
`certificate_radius` returns `None` and the clamp applies.
