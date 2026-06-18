# Sparsity statements and the structural difference between the two models

Literature-grounded synthesis, 2026-06-17. Answers two questions:
(1) what statement does one *propose* to establish a sparsity effect, and
(2) what is the structural difference in the two models' inductive bias.
Vocabulary per `CONTEXT.md`. Papers in `~/Documents/NotePaper/MasterThesis/`.

## 0. The closest paper is NOT a sparsity result

Kunisch–Vásquez-Varas, *Structure Preserving Approximation of Semiconcave
Functions* (`semiconcave_approximaton.pdf`) — nearest to our setting — proves
(Thm 2.1) every $C$-semiconcave $v$ = $\inf_{i\in\mathcal I}\phi_i$ over a $C^2$ family with
$\sup\|\nabla^2\phi_i\|\le C$, and that a smooth parametrization **preserves
semiconcavity** and **converges** in $C(\overline\Omega)$ and $W^{1,p}$. It is a
*structure-preserving approximation / convergence* result — **not** a "fewer
neurons than a baseline" statement. The sparsity statement must be imported from
the variation-space / representer literature below.

## 1. The three templates for a sparsity statement

**Template A — Representer / sparse-minimizer.** *The minimizer of [data fit +
variation norm] is a finite atomic network with $K\le N$ atoms.*
- Parhi–Nowak (`Banach space representer theorem.pdf`): $\min_f G(\mathcal Vf)+\|f\|_{(m)}$
  ($\|\cdot\|_{(m)}$ = TV of $\partial_b^m\Lambda^{d-1}\mathcal R f$, Radon domain) has solutions
  $f=\sum_{k\le K} v_k\,\rho_m(w_k\!\cdot\!x-b_k)+\text{poly}$, $K\le N$; $m=2$ ⇒ ReLU.
- Bredies–Pikkarainen (`Inverse problem in the space of measures.pdf`):
  $\min_\mu \tfrac12\|K^*\mu-f\|^2+\alpha\|\mu\|_{\mathcal M}$ over Radon measures; the TV
  norm forces **atomic (delta-peak) minimizers**. Selection via the dual
  certificate $\eta$: $\|\eta\|_*\le\alpha$ everywhere, **equality on $\mathrm{supp}(\mu)$**,
  sign fixed by $\mathrm{Sgn}$.
- Boyd–Schiebinger–Recht (`CG/…Sparse Inverse Problems.pdf`, = **PDAP's ancestor**):
  $\min_\mu \ell(\Phi\mu-y)$ s.t. $\mu\ge0$, $|\mathrm{supp}\,\mu|\le N$; convex surrogate uses
  a mass bound; solutions supported on $\le d+1$ atoms (§5).
- *Buys:* explains **why the solution is sparse at all** (L¹/TV over measures has
  atomic extreme points). Does not, by itself, compare two architectures.

**Template B — Variation-space (Barron) rate.** *If the target has finite
variation norm $\gamma(f)$, then $n$ atoms approximate it to error
$\le \gamma(f)\,n^{-1/2-(2k+1)/2d}$.*
- Barron_93; Yang–Zhou (`optimal_rate_Relu^k.pdf`); Bach
  (`convex-neural-networks-Paper.pdf`); Ongie/Parhi $\mathcal R$-norm (`Relu(15)`,`Relu(33)`).
- *Buys:* sparsity $\iff$ **small variation norm**; the neuron count is *derived*
  from $\gamma$. Unifies "count" and "variation norm" — $\gamma$ is the currency.

**Template C — Separation / lower bound.** *Target $T$ needs $\ge N(\varepsilon)$ neurons
in model A but $\le M(\varepsilon)$ in model B, $N/M\to\infty$.*
- Yang–Zhou pseudo-dimension / $n$-width lower bounds (§2); our **T2** (curved
  switching set ⇒ ridge $\mathcal R$-norm $=\infty$ vs $2d$ exact atoms for min-net).
- *Buys:* the head-to-head we ultimately want.

**For our problem:** the standard route is **C in B's currency** — show the value
function $V$ has **small/finite variation norm in the semiconcave dictionary but
large/infinite in the signed dictionary**. This is what T2 does; the count gap is
then a corollary via B. The literature does not prove sparsity by counting neurons
directly; it proves a **variation-norm gap**.

## 2. The structural difference in inductive bias — TWO features × TWO layers

Template A's certificate lens, applied to our exact models (`CONTEXT.md` §1),
shows the bias differs in **two independent structural inputs**:

**Feature 1 — null space (unpenalized subspace).** Penalty acts only on atoms $c_i$.
Free part: semiconcave = span$\{\tfrac12 C\|x\|^2,\,x,\,1\}$ (**includes the
quadratic**); signed = at most $\{x,1\}$ (**no quadratic**). Null-space representer
theorems (Unser/Boyer) ⇒ solution = (sparse atoms) + (free null-space part). So the
semiconcave model fits $V$'s quadratic component at **zero regularizer cost**; its
atoms carry only the remainder. Structural gift the signed model cannot access.

**Feature 2 — constraint cone.** semiconcave $\mu\ge0$ ($c_i\ge0$, a **cone**) ⇒
**one-sided** certificate $q\le\alpha$, atoms at $q=+\alpha$; signed = signed measure ⇒
**two-sided** $|q|\le\alpha$, atoms at $q=\pm\alpha$. ADCG (Boyd et al.) is *natively*
nonnegative — so the **cone model is the natural sparse-recovery object; the signed
model is the *extended* (signed-measure) case.**

These two features are independent (different null space AND different cone), so the
inductive bias genuinely **differs in two ways** — and they are the *correct*,
non-confounded form of the earlier (botched) "capacity vs selection": capacity =
Feature 1 (head/null-space), selection = Feature 2 (cone). The representer theorem
**separates them by construction** (null-space component vs one-/two-sided bound) —
the right tool, instead of the confounded subtraction that failed before.

**Two LAYERS.** Independently of the two features:
- **Variational layer** = what the regularized *minimum* is (Template A certificate).
- **Algorithmic layer** = what **greedy CG/ADCG = PDAP actually finds** (insert one
  atom at the certificate max, then local search). These can **differ** — the ReLU²
  case showed greedy missing a symmetric pair the variational optimum contains. So
  "what controls the bias" must be answered at *both* layers.

## 3. What this says about "what controls the inductive bias" (open, but now structured)

The bias is a $2\times2$: {Feature 1 head, Feature 2 cone} × {variational, algorithmic}.
- We do **not** yet know which cell drives the observed value-function sparsity.
- The literature gives the precise tools per cell: Template A certificate
  (variational, both features), ADCG analysis (algorithmic, both features),
  Template B variation norm (the currency for the eventual separation).
- The clean next step (non-confounded): work the **dual certificate** for each
  model — Feature 1 enters as null-space orthogonality conditions
  $\langle\eta,\tfrac12\|x\|^2\rangle=\langle\eta,x\rangle=\langle\eta,1\rangle=0$; Feature 2 as one-sided
  vs two-sided $q$ — and identify which makes $\gamma_{\text{signed}}(V)\gg\gamma_{\text{semiconcave}}(V)$.

## Recommended statement to prove

> Under a condition on $\sigma$ (TBD), the general $C$-semiconcave value function $V$
> satisfies $\gamma_{\text{semiconcave}}(V) < \infty$ (or $\le$ poly) while
> $\gamma_{\text{signed}}(V) = \infty$ (or $\gg$), where $\gamma$ is the variation/$\mathcal R$-norm in
> each model's dictionary; the neuron-count separation follows by Template B.

This is Template C in Template B's currency, with the structural mechanism
(Features 1–2) supplied by Template A. It subsumes T2 (its $\infty$ case) and
Theorem A-d (its ReLU count corollary).
