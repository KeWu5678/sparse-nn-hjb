# single_property_activation

**status:** refuted

**statement (the claim that was made).** There is a single qualitative property $P(\sigma)$ of
the activation such that
$$P(\sigma)\iff\text{the cone has a (divergent) sparsity advantage}.$$
Two candidates were proposed: $P=$ "$\sigma$ smooth", and $P=$ "$t\mapsto\sigma(t)^p$ convex" (C1).

**refutation.** Both fail on the faithful VDP data (repo PDAP):
- **smoothness:** `gaussian` is smooth yet shows essentially **no** advantage (cone 13 vs
  signed 12).
- **convexity (C1):** `tanh` (non-convex atom) shows a **strong** advantage (cone 2 vs signed
  19) while `gaussian` (also non-convex) shows none. Convexity neither gates nor sizes it.

So no single qualitative $\sigma$-property is equivalent to the advantage.

**what survived.** The correct invariant is **quantitative + metric-dependent**:
`activation_quadratic_cost` ($\sigma''$, best-$n$-term vs penalized). Not one property of $\sigma$.

**attempts.** 1. smoothness (refuted by gaussian). 2. atom convexity C1 (refuted by tanh).
3. dropped single-property search; moved to $\sigma''$ + metric. Detail:
`../../_archive/direction1-smooth.md` (contains a now-FALSE "Lemma 2"),
`../../scripts/activation_condition_derive.py`.
