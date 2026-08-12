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
- Cost rises roughly 15x per run: the candidate search and the coefficient solve
  both run 150 times instead of 10. At the recorded 14 s median this is a few
  minutes per run, so a 224-cell sweep moves from about 7 minutes to under two
  hours at `JOBS=8`. Affordable, but no longer free — a full paper-conforming
  sweep of both problems in both modes is a multi-hour job, not an interactive
  one.
- The frontier gains resolution as a side effect. Batch runs move the neuron axis
  in jumps of up to 15, while sequential advances one neuron at a time, so
  `fig:neuron_h1_frontier` becomes a genuine curve rather than a coarse
  staircase.
- The comparison is budget-matched, not compute-matched. A sequential run costs
  far more than the batch run it is compared against, so nothing here supports a
  claim about time-to-accuracy — only about accuracy at a given width.
- `insertion_first` terminates when no candidate clears the threshold, so a
  sequential run that exhausts its candidates stops well before 150 iterations.
  "Terminated" versus "hit the cap" is worth reporting, since only the former
  means the algorithm converged in its own terms.

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
