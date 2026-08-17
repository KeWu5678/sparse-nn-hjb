# 9. Use only verified closed-form global proximal maps

Date: 2026-08-17

## Status

Accepted

## Context

For activation power `k`, the homogeneous coefficient penalty has exponent
`q = 2/(k+1)`. Its coefficient correction needs the global scalar proximal map
of `mu |c|^q`.

The former general-`q` Newton fallback did not compute that map. It found a
positive stationary root and activated it at the turning-point threshold. For
`q<1`, that root can be only a local minimizer whose scalar objective is larger
than the value at zero. A finite-step warm start does not repair an incorrect
proximal map during the subsequent coefficient correction.

Verified closed forms are implemented for exactly these cases:

| activation power `k` | penalty exponent `q` | scalar solve |
|---|---|---|
| 1 | 1 | soft thresholding |
| 2 | 2/3 | quartic closed form |
| 3 | 1/2 | cubic closed form |

The fractional maps compare the positive stationary branch with zero through
the global objective-switching threshold and select zero at an exact tie.

## Decision

Support coefficient correction only for activation powers `k in {1, 2, 3}`.
Validate this contract when `PDAP` is constructed, independently of the
insertion strategy, so an unsupported profile configuration cannot fail midway
through its first correction.

Algorithm 2 studies use `k in {2, 3}`. The `k=1` ReLU--L1 study remains a
separate soft-thresholding baseline.

## Consequences

- Powers such as `2.01`, `4`, and `5` are deliberately outside the current
  solver contract; this is an implementation scope decision, not a claim that
  their mathematical proximal maps do not exist.
- Restoring the former Newton fallback is not an acceptable way to broaden the
  scope because it reinstates the wrong switching rule.
- Supporting another power requires a verified global scalar proximal map, its
  active-branch derivative, deterministic tie handling, and insertion and
  end-to-end correction tests. This ADR must then be revisited.

## Alternative considered

**Keep accepting arbitrary powers and fail only when the proximal map is
called.** Rejected because configuration would appear valid, data loading and
candidate insertion could run, and training would then fail inside SSN.
