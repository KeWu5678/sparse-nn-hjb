# source_condition_separation_free

**status:** refuted (as a $d\ge2$ separation); valid only $d=1$ / recovery-only

**statement (the claim that was made).** Via Bredies' source condition (Rmk 4.4), the cone
($\mu\ge0$) has empty negative-support set $\widetilde\Omega_-=\varnothing$, removing the opposite-sign
separation requirement, so the cone is in the "separation-free sparse recovery" regime
(de Castro–Gamboa) with $O(\delta)$ rate — in **general $d$** — while the signed model needs a
minimal separation (Candès–Fernández-Granda).

**refutation (why not a $d\ge2$ separation).**
1. **Mairhuber–Curtis:** separation-free nonneg recovery rests on Chebyshev/T-systems, which
   essentially **do not exist in $d\ge2$**. The clean statement is $d=1$ only.
2. **Recovery $\ne$ count:** the source condition stabilizes *recovery of $\mu^\dagger$*; it does not
   make $\mu^\dagger$ sparse. Nonnegativity can *increase* atom count (no cancellation), and $\mu^\dagger$
   may be non-atomic (curved switching, `correction_cost`) — no sparse target to recover.
3. Even with $\widetilde\Omega_-=\varnothing$, separation from $\partial\Theta$ is still required.

**what survived.** The one-sided certificate idea, folded into `certificate_dichotomy` ($d=1$).

**attempts.** 1. claimed separation-free in general $d$ (overclaim). 2. Mairhuber–Curtis +
recovery$\ne$count scoped it to $d=1$. Detail: `../refs/semiconcave-source-condition.md`.
