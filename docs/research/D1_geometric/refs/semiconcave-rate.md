# Semiconcave (cone) model — structural reduction, curvature budget, and rate

Intermediate result, 2026-06-17. The *semiconcave side* of the eventual
separation (Template C in Template B's currency, structure from Template A; see
`sparsity-statements-and-structure.md`). Vocabulary per `NOTATION.md`.

**Setting.** $V = \tfrac{C}{2}\|x\|^2 - g$ on a bounded convex $\Omega \subset \mathbb{R}^d$, $g$
convex; $V$ Lipschitz up to $\partial\Omega$. Cone model
$f = \tfrac{\tilde C}{2}\|x\|^2 + a\!\cdot\!x + b_0 - \sum_{i\le n} c_i\,\sigma(w_i\!\cdot\!x - b_i)^p$,
$c_i \ge 0$, $\tilde C \ge 0$, $\sigma = \mathrm{ReLU}$, atom power $k := p$ (so atoms are
$\mathrm{ReLU}^k$). Cost $N(M, V, \varepsilon)$ = best-$n$-term neuron count to relative
gradient-$L^2(\Omega)$ error $\varepsilon$ (NOTATION.md).

---

## (a) Structural reduction [exact, all $d$]

**Prop A.** The head represents $\tfrac{C}{2}\|x\|^2$ exactly ($\tilde C = C$, $0$ atoms),
so approximating $V$ by the cone model is *exactly* approximating the convex
correction $g$ by a nonnegative atomic measure:
$$
N(\text{cone}, V, \varepsilon) = N^+(g, \varepsilon), \quad\text{independent of } C,
$$
where $N^+(g,\varepsilon)$ is the fewest **nonnegative** $\mathrm{ReLU}^k$ atoms approximating $g$ to
gradient error $\varepsilon$.

*Proof.* Set $\tilde C = C$, $a=b_0=0$; then $f - V = -(\sum c_i\sigma^p - g)$ and
$\nabla(f-V) = -\nabla(\sum c_i\sigma^p - g)$. The head contributes $0$ to the error and
costs $0$ atoms; the cone constraint $c_i\ge0$ is exactly nonnegativity of the atomic
measure representing $g$. $\square$

This is "Feature 1 (null space)" quantified: the curvature $C$ is **free**.

---

## (b) Curvature budget [rigorous, all $d$]

**Definition (budget).** $\mathrm{bud}(g) := \|D^2 g\|_{\mathcal M} = \int_\Omega \mathrm{tr}\,d(D^2 g)$
= total mass of the Hessian (Alexandrov) measure of the convex correction $g$ (a
property of $g$ alone, no activation). Distinct from the rate constant $\gamma^+(g)$
(the nonneg *ridge* variation norm); see NOTATION.md and (d).

**Prop B.** $g$ convex $\Rightarrow D^2 g$ is a nonnegative-matrix-valued Radon
measure with
$$
\mathrm{bud}(g) = \int_\Omega d(\Delta g) = \int_{\partial\Omega} \partial_n g \, dS \;\le\; \mathrm{Lip}(g)\,|\partial\Omega|,
$$
**finite and independent of the number/curvature of switching surfaces.**

*Proof.* $g$ convex $\Rightarrow D^2g \succeq 0$ (Alexandrov), so $\mathrm{tr}\,D^2g =
\Delta g \ge 0$ as a measure. Gauss–Green for the measure Laplacian on convex $\Omega$:
$\int_\Omega d(\Delta g) = \int_{\partial\Omega}\partial_n g\,dS$. Finally
$\partial_n g \le |\nabla g| \le \mathrm{Lip}(g)$. $\square$

This is the $d$-dimensional generalization of **T1** ($\mathrm{TV}(V'') \le 2C|\Omega| + 2L$,
proved in 1-D, `t1-1d-budget-lemma.md`): the *curvature mass* of the correction is
bounded regardless of how complex the switching set is.

---

## (c) Rate [Maurey/Barron; Yang–Zhou improvement]

**Prop C.** Let $\gamma^+_\nabla(g) = $ the nonnegative variation norm of the gradient
field, i.e. $\inf\{\|\mu\| : \nabla g = \int \nabla[\sigma(w\!\cdot\!x-b)^k]\,d\mu,\ \mu\ge0\}$. If
$\gamma^+_\nabla(g) < \infty$, the cone model achieves, **dimension-free in the exponent**,
$$
\|\nabla f - \nabla V\|_{L^2(\Omega)} \le \frac{B\,\gamma^+_\nabla(g)}{\sqrt n},
\qquad\Longrightarrow\qquad
N(\text{cone},V,\varepsilon) \le \Big(\tfrac{B\,\gamma^+_\nabla(g)}{\varepsilon\,\|\nabla V\|}\Big)^2,
$$
$B = \sup_\theta \|\nabla\sigma_\theta\|_{L^2(\Omega)}$.

*Proof.* Maurey: $\nabla g/\gamma^+_\nabla$ is in the closed convex hull of
$\{\pm\nabla\sigma_\theta\}$ scaled; $n$-term convex combinations approximate any hull
element at rate $B/\sqrt n$ in the Hilbert space $L^2(\Omega;\mathbb{R}^d)$. Nonnegativity is
no obstruction — Maurey *is* an argument about convex combinations. $\square$

Under activation smoothness the exponent improves to $n^{-1/2-(2k+1)/2d}$
(Yang–Zhou, `optimal_rate_Relu^k.pdf`), giving
$N \le (\gamma^+/\varepsilon)^{2d/(d+2k+1)}$.

---

## (d) Finiteness of $\gamma^+(g)$ — and the architecture insight

- **$d=1$ [unconditional]:** $g$ convex on $[a,b]$ $\Rightarrow$
  $g(x) = (\text{affine}) + \int (x-t)_+ \, dg''(t)$ with $g'' \ge 0$, so the nonneg ReLU
  measure is exactly $g''$ and $\gamma^+(g) = \|g''\| = \mathrm{bud}(g) \le 2\,\mathrm{Lip}(g)$
  (**budget $=$ rate constant in 1-D**). Thus **(b) $\Rightarrow$ (c) with no extra
  hypothesis**:
  $$
  N(\text{cone}, V, \varepsilon) \le \big(2\,\mathrm{Lip}(g)/(\varepsilon\|\nabla V\|)\big)^2 \quad(d=1),
  $$
  independent of the number of switching points — a **proved** rate, extending T1.
- **$d \ge 2$, $g$ smooth or flat (piecewise-linear) switching set:** $\gamma^+(g)<\infty$
  (Barron for smooth $g$; finitely many ridges for PL $g$). Rate (c) holds.
- **$d \ge 2$, curved switching set:** $\gamma^+(g)$ (the **ridge** norm) **exceeds**
  $\|D^2g\|$ and is **infinite** — **T2** (`t2-separation-draft.md`), CONFIRMED below.

**Confirmed negative result (T2 + empirical, 2026-06-17).** Concrete counterexample
$g=(\|x\|-1)_+$ on $[-2,2]^2$ (convex; *circular* switching set): finite budget
(circle crease mass $2\pi$ + smooth exterior) but $\gamma^+(g)=\infty$. Since
$\|V\|_{\mathcal R}=\infty$ for a curved kink and $\|\tfrac C2\|x\|^2\|_{\mathcal R}<\infty$, we get
$\|g\|_{\mathcal R}=\infty$, and $\gamma^+(g)\ge\|g\|_{\mathcal R}=\infty$ (nonneg is more constrained
than signed). **Faithful repo cone-ReLU-ridge PDAP** (`../../scripts/curved_vs_flat_switching.py`):

| switching set | atoms @ relGrad $0.05$ | best relGrad | slope $d\log n/d\log\tfrac1\varepsilon$ |
|---|---|---|---|
| flat $g=(x_1)_+{+}(x_2)_+$ | **5** | 0.006 | **0.00** (bounded) |
| curved $g=(\|x\|-1)_+$ | — (unreachable) | **0.138 floor** (92→124 atoms) | **+0.65** (growing) |

The curved case **plateaus at relGrad $\approx0.14$, unbreakable even at 124 atoms**,
while flat reaches $0.006$ with $5$ — the accuracy floor + atom explosion is the
signature of $\gamma^+=\infty$.

**Architecture insight (with the necessary/sufficient distinction made honest).**
The curvature budget (b) is *always* finite, but the **ridge representation blows up
on curved switching sets** — solid negative result (above), for any ridge power $k$
(ReLU$^k$ ridges have *flat* curvature jumps, so cannot tile a curved switching
surface). So the repo cone-ridge model inherits budget$\to$rate **only for smooth/flat
$g$**; for curved switching sets (the generic HJB case) **something beyond ridges is
*necessary*.**

*What the min architecture does and does not fix.* Kunisch–Vásquez-Varas's
**min-of-paraboloids** is the natural candidate. It **succeeds on $v_d$** (min of
$2d$ paraboloids: finite atoms, $\gamma^+_{\min}<\infty$, curved switching) where the
ridge net's $\gamma^+$ diverges — the clean "min beats ridge" separation. But **min is
not a universal fix**: rotationally symmetric corrections like $g=(\|x\|-1)_+$ (our
ridge counterexample) appear to need a *continuum* of paraboloids in the min
architecture too (the inf-convolution sources fill the circle). So:
- **Necessary** (proved): beyond-ridge is required for curved switching.
- **Sufficient** (partial): min suffices for *finitely-min-representable* curved sets
  ($v_d$-type); **open** for general curved sets. Which HJB value functions are
  finitely-min-representable is the right question (cf. max-plus rank, McEneaney).

---

## Status & relation to the program

- **Rigorous:** (a) all $d$; (b) all $d$; (c) given $\gamma^+<\infty$; (d) $d=1$ ⟹ full
  proved $d=1$ rate; (d) $d\ge2$ flat/smooth ⟹ rate holds.
- **Resolved (was "the gap"):** $d\ge2$ **curved switching sets** give $\gamma^+(g)=\infty$
  for the ridge model — confirmed (T2 + empirical, counterexample $(\|x\|-1)_+$). Ridge
  **cannot** achieve budget→rate there; **beyond-ridge is necessary** (min suffices for
  $v_d$-type, open in general).
- **Role:** this is the semiconcave half of the separation. The other half (signed
  model needs $\gg$ neurons / $\gamma_{\text{signed}}(V)=\infty$) is Theorem A-d (ReLU, count)
  and T2 (curved switching, $\mathcal R$-norm).
- **Next (positive curved result):** show the **min-of-paraboloids** architecture
  achieves budget→rate on **finitely-min-representable** $V$ (clean case $v_d$, 2d
  atoms), and characterize that class for HJB (max-plus rank; `curse-free-max-plus`,
  `Gaubert–McEneaney–Qu`). $v_d$ is the concrete first target.
