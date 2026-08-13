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
- **The cost is below the naive estimate, but quote the right number.** Reasoning
  from the iteration count alone predicts a 15x slowdown (150 solves instead of
  10). It is less, because `insertion_first` also adopts the algorithms'
  termination rule and stops as soon as no candidate clears the threshold, which
  usually fires well before 150. But the *median* run and the *whole sweep* give
  different ratios, because the distribution is right-skewed — a minority of
  long sequential runs dominates the total:

  | sweep | median-run ratio | total-sweep ratio |
  |---|---|---|
  | vdp log | 2.2x | 5.5x |
  | pendulum log | 2.0x | 4.4x |
  | vdp frac_exp | 3.0x | 3.6x |
  | pendulum frac_exp | 2.7x | 5.6x |

  Plan sweeps with the total (4–6x); describe a single run with the median
  (2–3x). An earlier revision of this record quoted only the median and so
  understated the cost of a sweep.

  | sweep / mode | `T_out` | median iters | max iters | hit the cap | median elapsed |
  |---|---|---|---|---|---|
  | vdp frac_exp batch | 10 | 4 | 10 | 3/40 | 4.7 s |
  | vdp frac_exp sequential | 150 | 15 | 108 | 0/40 | 14.1 s |
  | pendulum frac_exp batch | 10 | 6 | 10 | 10/40 | 6.0 s |
  | pendulum frac_exp sequential | 150 | 25.5 | 150 | 4/40 | 16.1 s |
  | vdp log batch | 10 | 10 | 10 | 150/224 | 7.2 s |
  | vdp log sequential | 150 | 27.5 | 150 | 30/224 | 16.0 s |
  | pendulum log batch | 10 | 10 | 10 | 144/224 | 8.0 s |
  | pendulum log sequential | 150 | 25 | 150 | 35/224 | 16.0 s |

  The full paper-conforming re-run of both problems in both modes took about
  75 minutes at `JOBS=8`, not the several hours first estimated.
- **The cap still binds, which is why it is 150 and not smaller.** A median
  sequential run stops itself at 25–28 iterations, but 30–35 of 224 log-penalty
  cells reach 150 and are truncated by the budget. Those are the widest, least
  regularized fits — exactly the top of the frontier the comparison is for. A
  smaller `T_out` would have cut them, so the decision holds; what was wrong in
  the first version of this record was the cost, not the choice.
- The same measurement indicts the preserved loop order, which never terminates.
  Its batch runs hit the configured 10 iterations in roughly two thirds of the
  log-penalty cells and stopped early in the rest — but having no stopping rule,
  it could not report which was which. The paper's rule is the only one that
  distinguishes "converged" from "ran out of budget".
- The frontier gains resolution as a side effect. Batch runs move the neuron axis
  in jumps of up to 15, while sequential advances one neuron at a time, so
  `fig:neuron_h1_frontier` becomes a genuine curve rather than a coarse
  staircase.
- The comparison is budget-matched, not compute-matched. A sequential run still
  costs 2–3x the batch run it is compared against, so nothing here supports a
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
