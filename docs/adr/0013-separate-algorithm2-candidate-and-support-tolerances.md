# 13. Separate Algorithm 2 candidate and support tolerances

Date: 2026-08-17

## Status

Accepted

## Context

Algorithm 2 uses random sphere multistart followed by L-BFGS. Different starts
often return the same local maximizer, so candidates from one search are
deduplicated with a cosine-gap tolerance of `1e-2`.

The one-atom increment formula separately requires the inserted location not to
be an atom already in the support. Reusing the `1e-2` candidate tolerance for
this check excluded every point within about 8.1 degrees of an existing atom.
Those nearby points are distinct atoms, and excluding them substantially
degraded the pendulum fits.

## Decision

- Keep exactly `N_trial` random sphere starts; existing support points are not
  added as starts.
- Keep the `1e-2` cosine-gap tolerance for deduplicating candidates returned by
  different starts in the same search.
- Reject a candidate against the existing support only as a numerical repeat,
  using cosine gap `1e-8`.

Both tolerances live at their use sites in
`src/PDAP/insertion.py::_generate_candidates`: `merge_tol` (configurable, `1e-2`
by default) for candidate deduplication, and a literal `1e-8` for the existing-
support filter.

## Consequences

Exact numerical returns to the current support are not inserted, while nearby
distinct ReLU knots remain admissible. Algorithm 2 experiments produced with
the wider support exclusion must be rerun.

## Alternative considered

**Use one tolerance for both comparisons.** Rejected because candidate
deduplication and equality with the existing support serve different purposes.
