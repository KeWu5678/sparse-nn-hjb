# 8. Give sequential insertion a matched neuron budget, not a matched iteration count

Date: 2026-08-12

## Status

Accepted

## Context

Inserting `N_ins > 1` candidates against one frozen residual forfeits the
one-candidate decrease statement: the joint fidelity change carries cross terms
`sum_{i<j} c_i c_j <K^M(omega_i), K^M(omega_j)>` that no individual acceptance
test controls.

`training.insert_mode=sequential` restores the single-candidate condition by
admitting exactly one atom per outer iteration. The implementation selects the
highest-ranked candidate returned by multistart L-BFGS; the rate bound
`eq:certificate-violation-rate` additionally requires that candidate to be an
exact global maximizer, which the practical search does not certify.

Under batch insertion the support grows by up to `N_ins` per iteration. With
`T_out=10` and `N_ins=15`, the available width is 150 atoms. Under sequential
insertion the support grows by at most one, so the same iteration count would cap
every run at ten atoms. Comparing the modes at equal `T_out` would therefore make
available width, rather than the insertion rule, the dominant difference.

## Decision

Run sequential insertion at `T_out = 150`, matching the batch neuron budget
`T_out * N_ins`, and compare the two on the neurons-versus-error frontier rather
than per iteration.

## Consequences

- Sequential can reach every width batch can, so the frontier is comparable over
  its whole range instead of only at the sparse end.
- Reasoning from the iteration count alone predicts a 15x slowdown. The measured
  cost is smaller because the finite multistart search often returns no accepted
  candidate before 150 iterations. The median-run and whole-sweep ratios differ
  because a minority of long sequential runs dominates the total:

  | sweep | median-run ratio | total-sweep ratio |
  |---|---|---|
  | VDP Algorithm 1 | 1.9x | 5.4x |
  | pendulum Algorithm 1 | 1.2x | 3.5x |

  | sweep / mode | `T_out` | median iters | max iters | hit the cap | median elapsed |
  |---|---|---|---|---|---|
  | VDP batch | 10 | 10 | 10 | 160/224 | 5.0 s |
  | VDP sequential | 150 | 36.5 | 150 | 48/224 | 9.4 s |
  | pendulum batch | 10 | 10 | 10 | 152/224 | 6.9 s |
  | pendulum sequential | 150 | 39 | 150 | 60/224 | 8.1 s |

- The 150-iteration cap remains necessary: 48 VDP cells and 60 pendulum cells
  reach it. A smaller value would truncate the widest part of the frontier.
- The frontier gains resolution as a side effect. Batch runs move the neuron axis
  in jumps of up to 15, while sequential advances one neuron at a time, so
  `fig:neuron_h1_frontier` becomes a genuine curve rather than a coarse
  staircase.
- The comparison is budget-matched, not compute-matched. A sequential run still
  costs more in aggregate, so nothing here supports a claim about
  time-to-accuracy — only about accuracy at a given width.
- "No candidate returned" versus "hit the cap" is reported per sweep. The former
  is a stopping event of the finite multistart search, not a global optimality
  certificate.

## Alternatives considered

**Keep `T_out = 10`.** Rejected: it caps sequential runs at ten neurons, so the
comparison would cover only the very sparse regime.

**Insert one atom at a time within an iteration, refreshing the residual after
each.** Keeps the neuron budget and the ten coefficient solves, and is the other
remedy the manuscript names. Rejected: it still corrects only once per iteration,
so it does not satisfy the rate bound's hypothesis of a maximizing insertion
followed by a correction — it would be a third mode whose guarantee is weaker
than sequential and stronger than batch, without a statement in the paper to
anchor it.

**Match compute instead of neurons.** Rejected: it makes the width the free
variable, which is the axis the results are reported on.
