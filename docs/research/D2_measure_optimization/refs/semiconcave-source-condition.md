# Structural sparsity of the semiconcave model via Bredies' optimality + source condition

2026-06-17. **Uses** the optimality/source-condition framework of Bredies–Pikkarainen
(`Inverse problem in the space of measures.pdf`, Prop 3.6, source cond. 4.2–4.4, rate
Rmk 4.3) to derive a *structural* property of the semiconcave model that makes its
solution sparse, **with a convergence rate**. New relative to T5-d1 (which was the
$d=1$ certificate dichotomy, no source condition / no rate). Vocabulary per NOTATION.md.

## Setup in the Bredies framework

Atom parameter space $\Theta = S^{d-1}\times\mathbb{R}$; representing measure $\mu$ on $\Theta$;
predual map $(K\mu)(x) = \int_\Theta \sigma(w\!\cdot\!x-b)\,d\mu(w,b)$ (the network), data $f =\nabla V$
fitted in $H = L^2(\Omega;\mathbb{R}^d)$ (gradient training). Tikhonov problem
$\min_\mu \tfrac12\|K\mu - f\|_H^2 + \alpha\|\mu\|_{\mathcal M}$. The **certificate** is
$q := K^\ast(f - K\mu^\ast)$ on $\Theta$, i.e. $q(w,b) = \langle \nabla\sigma_{w,b},\, \eta\rangle$, $\eta$ the residual.
- **signed model:** $\mu$ signed.
- **semiconcave (cone) model:** $\mu \ge 0$ (and the free head/affine = unpenalized null
  space; see Note 2).

## 1. Sparsity mechanism (Bredies Prop 3.6)

$\mu^\ast$ optimal $\iff \|q\|_\infty \le \alpha$ and $\mathrm{supp}\,|\mu^\ast| \subset \{(w,b): |q(w,b)| = \alpha\}$
(the **contact set**), with the polar sign of $\mu^\ast$ matching $\mathrm{sign}\,q$ there. So the
support lives on the contact set — generically a **finite** set of points (the atoms).
This is *why the solution is atomic/sparse at all*. Sign-specialized (Bredies Rmk 3.8):
- **signed:** atoms at $q = +\alpha$ **or** $q = -\alpha$ (both bounds).
- **cone ($\mu\ge0$):** atoms only at $q = +\alpha$ (one bound). [matches T5-d1 Lemma 3]

## 2. The rate (Bredies source condition 4.2 + rate Rmk 4.3)

Define the source condition on the min-norm solution $\mu^\dagger$ (Bredies 4.2):
$$
\exists\, h\in H:\quad \langle \mu^\dagger, Kh\rangle = \|\mu^\dagger\|_{\mathcal M},\quad \|Kh\|_\infty = 1 .
$$
**If it holds**, then with $\alpha \sim \delta$ the regularized solution satisfies the
**Bregman-distance rate $D_{\|\cdot\|_{\mathcal M}}(\mu_\alpha, \mu^\dagger) = O(\delta)$** (Bredies Rmk 4.3,
Thm 2 of [9]) — i.e. the model recovers the *sparse* truth at rate $O(\delta)$. The
source condition is exactly the existence of a **nondegenerate dual certificate**:
$Kh$ saturates ($=\!\pm1$) on the support and stays $|Kh|\le1$ off it.

## 3. The structural property (Bredies Rmk 4.4, specialized by sign)

Bredies Rmk 4.4: the source condition holds **iff** there exist sets
$\widetilde\Omega_+ \supset \mathrm{supp}\,\mu^\dagger_+$, $\widetilde\Omega_- \supset \mathrm{supp}\,\mu^\dagger_-$ and $h\in H$ with
$Kh = +1$ on $\widetilde\Omega_+$, $Kh = -1$ on $\widetilde\Omega_-$, $|Kh|\le1$ else — and crucially
**$\widetilde\Omega_+$, $\widetilde\Omega_-$ must be separated from each other and from $\partial\Theta$ by a
positive distance.** Now specialize:

- **Cone ($\mu^\dagger \ge 0$):** the polar $\sigma^\dagger \equiv +1$, so $\widetilde\Omega_- = \varnothing$. The
  source condition reduces to: $Kh = +1$ near $\mathrm{supp}\,\mu^\dagger$, $|Kh|\le1$ else, support
  separated **only from $\partial\Theta$**. **No opposite-sign separation is required.**
- **Signed:** both $\widetilde\Omega_+, \widetilde\Omega_-$ present ⟹ the certificate must separate
  **opposite-sign clusters from each other** by a positive distance.

**This is the structural property.** The sign constraint $c_i \ge 0$ removes the
opposite-sign separation requirement of the source condition. Concretely:

> **The cone model is in the *separation-free* sparse-recovery regime; the signed
> model is in the *minimal-separation-required* regime.**

This is precisely the established dichotomy: nonnegative sparse recovery succeeds with
**no minimal-separation condition** (de Castro–Gamboa; Schiebinger–Recht; Slawski–Hein),
whereas signed/complex recovery requires a minimal separation $\gtrsim 1/f_c$
(Candès–Fernandez-Granda). Bredies Rmk 4.4 is the bridge that makes this a statement
about *our* Tikhonov problem and ties it to the $O(\delta)$ rate.

## 4. Consequence for sparsity (the payoff)

- **Cone:** for *any* nonnegative target configuration whose support is separated from
  $\partial\Theta$ — **including arbitrarily clustered atoms** — the source condition holds,
  so the regularized cone solution recovers the sparse $\mu^\dagger$ at rate $O(\delta)$: it
  **stays as sparse as the truth**.
- **Signed:** when the optimal representation uses both signs and the opposite-sign
  clusters are closer than the minimal separation, the source condition **fails**; the
  regularized signed solution then **spreads** (the mass smears over a region rather
  than concentrating on the true atoms) — *more* effective atoms, no $O(\delta)$ support
  recovery.

So the cone's nonnegativity is not just "one bound instead of two" (T5-d1, the
$2\times$ count) — via the source condition it is **the difference between
separation-free sparse recovery and separation-limited recovery**, and it carries the
$O(\delta)$ rate. This is the optimality-framework answer to "what structural property
makes the semiconcave solution sparse."

## Notes / honest status

1. **What is cited vs new.** Prop 3.6, source condition, $O(\delta)$ rate, Rmk 4.4 are
   Bredies' theorems (cited, not re-proved — "rely on existing proof"). The *new*
   content is the sign-specialization ($\widetilde\Omega_-=\varnothing$ for the cone) and its reading
   as separation-free vs separation-required recovery, with the de Castro–Gamboa /
   Candès–FG anchors. This generalizes T5-d1's one-sided certificate from $d=1$/$\ell_1$
   to general $d$ and **adds the rate**.
2. **The free head is a *separate* structural feature** (the penalty null space
   $\{\tfrac12\|x\|^2, x, 1\}$), not covered by basic Bredies (which has no null space). It
   enters as $\eta \perp \{\tfrac12\|x\|^2, x, 1\}$ (KKT for the unpenalized block) — the
   $d$-dim "endpoint zeroing" of T5-d1. Folding the null space into the source
   condition (a *partial* source / Bregman setup) is the next concrete step.
3. **To verify (not yet done):** that the cone's min-norm target $\mu^\dagger$ for an actual
   HJB $g$ indeed satisfies the separated-from-$\partial\Theta$ condition (so the source
   condition genuinely holds), and to measure the signed model's spreading vs the
   cone's concentration on a clustered-atom target (the falsifiable prediction of §4).
