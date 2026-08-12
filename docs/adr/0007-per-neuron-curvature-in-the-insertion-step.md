# 7. Use the per-neuron curvature in the insertion step, not the uniform bound

Date: 2026-08-12

## Status

Accepted

## Context

The quantitative insertion theorem (`paper/paper_0805.tex`, Section 5) gives an
accepted candidate the initial coefficient

```
kappa(omega) = -Delta(mu, omega) / B_p^2 * sign(P_p(omega)),   c = kappa / w_p,
```

where `B_p = sup_omega ||K_p(omega)||` is a **uniform** bound over the whole
parameter domain, and concludes a decrease of at least `Delta^2 / (2 B_p^2)`.

Implementing that literally requires a computable `B_hat >= B_p`. `B_p` is
residual-independent, so it could be found once per run by multistart
maximization of `||K^M(omega)|| / w_p(omega)`. But such a search returns a local
maximum, so it is not certified; and the error hurts both ways. Under-estimating
`B_hat` makes the step too large and voids the very guarantee it exists to
provide. Over-inflating it drives `kappa` toward zero, so the atom enters at
essentially nothing and depends on the SSN correction to grow it — which is the
behaviour of the coordinate-descent warm start this step was meant to replace.

The proof, however, does not need a uniform constant at this point. Its chain is

```
J(mu + c delta_omega) - J(mu)
    <= kappa P_p(omega) + (kappa^2/2) ||K_p(omega)||^2 + alpha L_phi |kappa|
    <= kappa P_p(omega) + (kappa^2/2) B_p^2            + alpha L_phi |kappa|.
```

`B_p` enters only at the second step, which uniformizes the bound so that the
telescoped rate statement has one constant to sum against. The first inequality
is exact and per-neuron.

## Decision

Initialize an accepted atom by minimizing the *first* bound exactly, with the
per-neuron curvature `A = ||K_p(omega)||^2`:

```
c(omega) = -Delta(mu, omega) * sign(P_p(omega)) / (w_p(omega) * A),
decrease >= Delta(mu, omega)^2 / (2 A).
```

Compute no global `B_hat` anywhere.

This is not a weakening of the paper. Since `A <= B_p^2`, the per-neuron
decrease `Delta^2/(2A)` is at least the printed `Delta^2/(2 B_p^2)`, so every
statement proved with `B_p` still holds — including the telescoped rate, whose
summation only needs each step to decrease by at least its `B_p` amount.

The reused quantity is `S_sq` in `src/PDAP/insertion.py`, already computed per
candidate for the finite-step test.

## Consequences

- No calibration stage, no safety factor, and no configuration for either.
  Nothing in the algorithm depends on an uncertified numerical maximization.
- The step is strictly larger than the paper's, so an atom enters closer to
  where the coefficient solve will put it, and the correction has less to do.
- Algorithm 1's insertion becomes structurally identical to Algorithm 2's
  finite-step solve, which already minimizes its increment exactly per
  candidate. The two insertion strategies differ only in the penalty they
  minimize against, not in how the initial coefficient is found.
- A reader comparing the code against `eq:normalized-inserted-mass` will find
  `||K_p(omega)||^2` where the manuscript writes `B_p^2`. That is the reason
  this record exists; `tests/test_paper_conformance.py` asserts the resulting
  decrease against the theorem's own bound.
- If the manuscript is ever revised to state the sharper per-neuron form, this
  decision becomes a no-op rather than something to undo.

## Alternatives considered

**Multistart L-BFGS for `B_hat`, once per run.** Cheap enough, since `B_p` does
not depend on the residual. Rejected: it is an uncertified local maximum
standing in for a supremum, so the "guarantee" it buys is conditional on a
search succeeding — and a search failure silently produces too large a step.

**Candidate-set maximum, per iteration.** Free, since the norms are already
computed. Rejected for the same reason, more sharply: a maximum over the current
candidates is a lower bound on the supremum, so it systematically overshoots.

**Per-activation analytic `B_p`.** Certified and free at runtime. Rejected as
disproportionate: it would have to be derived per activation and re-derived
whenever one is added, to obtain a constant the per-neuron form does not need.
