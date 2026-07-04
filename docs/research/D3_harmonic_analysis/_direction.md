# D3 — Harmonic analysis (Radon / ridgelet / R-norm)

**Machinery.** Radon and ridgelet transforms; the $\mathcal R$-norm (variation norm of a ReLU
net = total variation of $\partial_b^{d+1}\mathcal R\{f\}$ in the Radon domain). Linearizes ridge
norms: a ridge $\sigma(w\!\cdot\!x-b)$ becomes a point mass in $(w,b)$, so "how expensive is $f$ for
ridge nets" becomes a Radon-domain integrability question. Refs: Ongie–Willett–Soudry–
Srebro (`Relu(15)`, `Relu(33)`), Parhi–Nowak (Radon-domain TV), Helgason (`Radon transform`),
Candès (ridgelets), E–Wojtowytsch (Barron spaces).

**Role.** Supplies the **norm facts** the other directions use — not currently an active
engine on its own, but the source of:
- **`budget_1d`** — 1-D budget $=\|g''\|=\mathcal R$-norm (proved). `refs/t1-1d-budget-lemma.md`.
- **`curved_switching_rnorm`** — curved switching set ⟹ ridge $\mathcal R$-norm $=\infty$ (proved $d=2$).
  `refs/t2-separation-draft.md`. This is what makes D1's `correction_cost` (curved case) diverge.
- the certificate identity $\partial_b^2 q = \mathcal R[\eta]$ used in D2.

## Status
Used to obtain `budget_1d`, `curved_switching_rnorm`; could become an active engine if we pursue
the variation-norm currency directly. Currently feeds D1 (`correction_cost`) and D2.

## Claims → `claims/`  (registry in `../CLAIMS.md`)
- `budget_1d.md` — **proved**.
- `curved_switching_rnorm.md` — **proved** ($d=2$).

## Refs → `refs/`
`t1-1d-budget-lemma.md`, `t2-separation-draft.md`.
