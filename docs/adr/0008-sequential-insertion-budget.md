# 8. Give sequential insertion a matched neuron budget, not a matched iteration count

Date: 2026-08-12

## Status

Accepted

## Context

The revised manuscript is explicit that inserting `N_ins > 1` candidates against
one frozen residual forfeits the insertion theorem's guarantees: the joint
fidelity change carries cross terms
`sum_{i<j} c_i c_j <K^M(omega_i), K^M(omega_j)>` that no individual acceptance
test controls, so "neither a decrease per outer iteration nor monotonicity of
`J^M` along the iterates is claimed." The reported numerics in Section 6 use
`N_ins = 15` and therefore run the unguaranteed variant.

`training.insert_mode=sequential` restores the guarantee by admitting exactly one
atom per outer iteration — the maximizer of `|P_p|`, which is what the rate bound
`eq:certificate-violation-rate` requires at every insertion.

That changes what an iteration buys. Under batch insertion the support grows by
up to `N_ins` per iteration; the recorded runs reach a maximum of 150 neurons,
which is exactly `T_out * N_ins = 10 * 15`, so the cap binds for the widest fits.
Under sequential insertion the support grows by at most one, so `T_out = 10`
caps a run at 10 neurons — below the observed medians of 17 (VDP) and 27
(pendulum), and far below the p90 of 70 and 115.

Comparing the two at equal `T_out` therefore compares a converged batch fit
against a truncated sequential one, and any accuracy difference is dominated by
the width difference rather than by the insertion rule.

## Decision

Run sequential insertion at `T_out = 150`, matching the batch neuron budget
`T_out * N_ins`, and compare the two on the neurons-versus-error frontier rather
than per iteration.

## Consequences

- Sequential can reach every width batch can, so the frontier is comparable over
  its whole range instead of only at the sparse end.
- **The budget is not what binds, and the cost is far below the naive estimate.**
  Reasoning from the iteration count alone predicts a 15x slowdown (150 solves
  instead of 10). Measured on the `frac_exp` sweeps, it is closer to 3x, because
  `insertion_first` also adopts the algorithms' termination rule and stops as
  soon as no candidate clears the threshold — which fires long before 150:

  | sweep / mode | median elapsed | median iterations | `T_out` | terminated early |
  |---|---|---|---|---|
  | vdp batch | 4.7 s | 4 | 10 | 37/40 |
  | vdp sequential | 14.1 s | 15 | 150 | 40/40 |
  | pendulum batch | 6.0 s | 6 | 10 | 30/40 |
  | pendulum sequential | 16.1 s | 25.5 | 150 | 36/40 |

  So `T_out = 150` is a ceiling that is never reached, not a budget that is
  spent. The decision stands — a smaller `T_out` *would* truncate, since
  sequential runs to 25+ iterations and the cap must not be the thing that stops
  them — but it is nearly free, and the earlier framing of this as a multi-hour
  trade was wrong.
- The same measurement says something about the preserved loop order, which never
  terminates: most of its configured iterations were inserting nothing. The
  paper's stopping rule is not only cheaper, it is the only one that distinguishes
  "converged" from "ran out of budget".
- The frontier gains resolution as a side effect. Batch runs move the neuron axis
  in jumps of up to 15, while sequential advances one neuron at a time, so
  `fig:neuron_h1_frontier` becomes a genuine curve rather than a coarse
  staircase.
- The comparison is budget-matched, not compute-matched. A sequential run still
  costs about 3x the batch run it is compared against, so nothing here supports a
  claim about time-to-accuracy — only about accuracy at a given width.
- "Terminated" versus "hit the cap" is reported per sweep, since only the former
  means the algorithm converged in its own terms. On the evidence above it is
  almost always the former.

## Alternatives considered

**Keep `T_out = 10`.** Same compute as today. Rejected: it caps sequential runs
at 10 neurons, so the comparison would only cover the very sparse regime and
would not touch the width range where the existing results live.

**Insert one atom at a time within an iteration, refreshing the residual after
each.** Keeps the neuron budget and the ten coefficient solves, and is the other
remedy the manuscript names. Rejected: it still corrects only once per iteration,
so it does not satisfy the rate bound's hypothesis of a maximizing insertion
followed by a correction — it would be a third mode whose guarantee is weaker
than sequential and stronger than batch, without a statement in the paper to
anchor it.

**Match compute instead of neurons.** Rejected: it makes the width the free
variable, which is the axis the results are reported on.
