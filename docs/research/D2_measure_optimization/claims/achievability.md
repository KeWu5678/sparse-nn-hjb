# achievability

**status:** open ($d\ge2$); $d=1$ via `certificate_dichotomy`

**statement.** With the penalized objective (log/nonconvex penalty, strength $\alpha$) solved by
PDAP, the **penalized** cone solution attains the best-$n$-term cone count of the proved legs
(it adds no spurious atoms beyond $N^+(g,\varepsilon)$), and the penalized signed solution does not
beat it. I.e. the representational separation (`separation_flat`) is **realized by the actual
solver**.

**why needed.** Legs `head_reduction`/`signed_lower_bound`/`correction_cost` bound
*representability* (best-$n$-term, solver-independent). This claim bounds what **PDAP** finds —
the only half where optimality conditions (D2 machinery) enter. The lower bound on the signed
model transfers to the solver automatically; the cone *upper* bound (achievability) does not,
hence this claim.

**status of pieces.** $d=1$: `certificate_dichotomy` (cone one-sided, ratio→2). $d\ge2$: open —
needs the $d\ge2$ certificate count.

**uses (when closed).** `certificate_dichotomy` ($d\ge2$), `separation_flat`.

**caveat.** Nonnegativity can *raise* the count (no cancellation); $\mu^\dagger$ may be non-atomic
(curved switching). So this is achievability *given* the representational legs hold (flat
switching), not a standalone sparsity source.

**attempts.** 1. source-condition route → refuted ($d=1$, `source_condition_separation_free`).
2. certificate dichotomy → $d=1$ done, $d\ge2$ open.
