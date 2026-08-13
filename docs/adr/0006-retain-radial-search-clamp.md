# 6. Combine the theorem radius with a numerical search bound

Date: 2026-07-25

## Status

Accepted. Rewritten 2026-08-13 for the normalized-measure Algorithm 1.

## Context

For a nonhomogeneous activation, the radial scale of an inner parameter is a
genuine shape parameter. Algorithm 1 therefore searches over the full
parameter `omega=(a,b)`, rather than quotienting the parameter onto the unit
sphere as Algorithm 2 does.

The quantitative insertion theorem confines every point that can violate the
insertion condition to the radius

```
R(mu) = max{1, (2^(s1+1) C ||r_mu|| / (alpha L_phi))^(1/(p-s1))}.
```

The activation registry supplies `C_rho`, `s0`, and `s1`. The samples determine
the remaining factor

```
C = C_rho (A_M^s0 + A_M^(s1-1)).
```

Thus `R(mu)` is computed before candidate generation; it is not fitted from the
candidate locations or from the experiment results.

For small `alpha` and `p` close to `s1`, the theorem radius can be extremely
large. Searching over that entire range is numerically unhelpful. The
implementation therefore also retains the fixed numerical upper bound `exp(5)`.

## Decision

Algorithm 1 uses

```
R_search = min(R(mu), exp(5)).
```

If the theorem does not apply or an activation has no declared growth data,
`R_search=exp(5)`.

The radius is used in two precise places:

1. Draw each random start with a log-uniform radius between `exp(-3)` and
   `R_search` and a uniform direction on the unit sphere.
2. After one unconstrained joint L-BFGS solve over `omega`, discard the final
   point if `|omega| > R_search`.

The optimizer itself is not ball-constrained. A trajectory that leaves the
radius is not clipped or projected back, because a projected boundary point is
not the local maximizer produced by that trajectory. Radius filtering occurs
before Euclidean near-duplicate removal and before the insertion test.

Algorithm 2 is unaffected: positive homogeneity absorbs the radial scale into
the outer coefficient, so its candidate search remains on the unit sphere.

## Consequences

- At large `p`, the theorem radius can tighten the accepted search region below
  `exp(5)`, making the theoretical confinement operational.
- Elsewhere, `exp(5)` remains a documented numerical restriction. The paper
  must not present it as a consequence of the theorem.
- Candidate generation is reproducible across scales because starts are
  log-uniform in radius. Uniform-in-volume sampling would concentrate almost
  every start near the outer boundary, where the normalized profile can be
  nearly flat.
- The final filter may reject an L-BFGS trajectory. It never manufactures an
  accepted candidate by moving that trajectory back onto the boundary.
- Changing either bound changes the candidate distribution and requires the
  affected Algorithm 1 experiments to be rerun.

## Alternatives considered

**Constrained optimization inside the ball.** Rejected. The implementation uses
standard L-BFGS without a reliable ball constraint, and parameterizing the ball
would change the geometry and conditioning of the local search.

**Project the final iterate onto the boundary.** Rejected. Projection changes
the point after optimization and can turn a failed trajectory into a candidate
that was never locally optimized.

**Use only the theorem radius.** Rejected for the current experiments. Near the
admissibility threshold `p>s1`, the certified radius is finite but can be too
large to define a useful numerical search.
