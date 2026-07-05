# Why the semiconcave model is much sparser for HJB value functions — the real setting

## THE ANSWER (quantifiable, real-data confirmed)

**Statement.** Write the value function $V = \tfrac{C}{2}\|x\|^2 - g$, $g$ convex
($C$ = semiconcavity constant). A ReLU sum is piecewise-linear, so its Hessian is
**zero a.e.**; to reproduce $V$'s curvature ($D^2 V \approx -CI$ on the bulk) the
**signed** model must lay down a growing number of flat pieces, whereas the
**semiconcave** model supplies that curvature with the single head parameter $C$
and spends its atoms only on the $O(1)$ non-quadratic (convex-correction /
switching) structure. Quantitatively, with gradient-$L^2$ accuracy $\varepsilon$:

$$
n_{\mathrm{cone}}(\varepsilon) = O(1), \qquad
n_{\mathrm{signed}}(\varepsilon) \ge \frac{\kappa_d\, C\, \mathrm{vol}(\Omega)^{\frac12+\frac1d}}{\varepsilon}
\quad\text{(Theorem A-d)},
\qquad
\boxed{\;\frac{n_{\mathrm{signed}}}{n_{\mathrm{cone}}} = \Theta(\varepsilon^{-1})\;\text{ — it DIVERGES.}}
$$

The gap is **not a constant factor**; it grows without bound as accuracy tightens.
That is the *why*: curvature is free for the head, $\Theta(1/\varepsilon)$-expensive for ReLU.

**Real-data confirmation (repo PDAP, VDP value function, `../../scripts/why_curve_vdp.py`):**
the slope $d(\log n)/d(\log\tfrac1\varepsilon)$ is **cone $\approx -0.00$, signed $\approx +1.35$** —
$n_{\mathrm{cone}}$ flat, $n_{\mathrm{signed}} \sim \varepsilon^{-1.35}$ (the predicted $\gtrsim 1$). Matched-accuracy
ratio: 5× at $\varepsilon=0.35$ → 8× at $\varepsilon=0.25$, climbing as predicted. This is the
clean, *un-confounded* evidence (two $n(\varepsilon)$ curves; no bulk-subtraction).

---


Status: analysis grounded in the `experiments/activationsearch/vdp + experiments/activationsearch/pendulum` data, 2026-06-16.
Setting: **d = 2** (VDP state space), **gradient-augmented** loss (`h1` =
loss_weights [1,1]), **non-convex log penalty** (γ swept), repo PDAP+SSN,
**power p = 1 (ReLU)**. This supersedes the d=1, convex-ℓ1 selection toy (now
the appendix `t5-d1-certificate-dichotomy.md`).

## The empirical gap (activationsearch, VDP, h1 loss, best-γ per config)

At matched accuracy, semiconcave (cone) vs signed neuron counts:

| activation, insertion | semiconcave | signed | gap |
|---|---|---|---|
| softplus, finite_step | **1** @ H1 0.312 | 11 @ 0.305 | **11×**, matched acc |
| softplus, profile | 2 @ 0.275 | 10 @ 0.378 | 5×, cone also better acc |
| tanh, finite_step | 2 @ 0.235 | 19 @ 0.484 | ~10×, cone also better |
| gausscent_1, finite_step | 4 @ 0.268 | 15 @ 0.266 | ~4×, matched acc |
| matern52, finite_step | 5 @ 0.255 | 12 @ 0.238 | ~2.5× |

Consistent 3–11× sparsity at matched accuracy. Two tells about the *mechanism*:
the cone frequently attains its best score at **γ = 0 (ℓ1)** while the signed
model needs **γ > 0 (non-convex)** to reduce neurons; and the gap is largest for
activations that cannot cheaply produce curvature (tanh, softplus). Both indicate
the cone is sparse **structurally**, before the non-convex penalty contributes.

## Decomposition: capacity (head) vs selection — and which dominates at p = 1

Write the semiconcave target as $V = \tfrac{C}{2}\|x\|^2 - g$, $g$ convex (this is
what $C$-semiconcavity *means*). For an HJB value function $g$'s ridges are the
switching/synthesis-cell surfaces — a thin (codim-1) set — while $\tfrac{C}{2}\|x\|^2$
is the smooth strongly-convex **bulk**.

- **Cone model** (`SemiconcaveModel`): the $\tfrac{C}{2}\|x\|^2$ **head** represents
  the bulk *exactly* (its gradient $Cx$ is an exact model output, 0 atoms), so the
  $c_i \ge 0$ atoms carry only $g$ — the thin switching part. $n_{\mathrm{cone}} \approx
  \#\text{switching ridges}$.
- **Signed model** (`SignedModel`, ReLU + affine head only): no quadratic head, so
  it must *approximate the entire curved bulk with atoms*. At **p = 1 there is no
  head emulation** ($(x)_+^2 + (-x)_+^2 = x^2$ needs p = 2), so the bulk's linear
  gradient field $Cx$ must be built from piecewise-constant ReLU-gradient atoms —
  a **growing** number as accuracy tightens.

**Consequence (Theorem A, lifted to d ≥ 2 — see below): at p = 1 the head is a
genuine capacity advantage, not merely selection.** This is penalty-independent
(a best-$n$-term representation fact) and it is the *dominant* driver of the
table above: the cone needs $O(\#\text{ridges})$ atoms; the signed needs that
*plus* the bulk-approximation atoms, and the latter grows with $1/\varepsilon$ and with $d$.

## ⚠ ACTIVATION SCOPE (2026-06-16): Theorem A-d's Θ(1/ε) is ReLU-specific — the general statement is ε^{−κ_σ}

ReLU is the wrong base case: we do *gradient* fitting (ReLU's gradient is
discontinuous) and the repo uses smooth/squared activations. What survives for
**any** activation σ is only the head: the quadratic head represents the
**second-order part of V exactly** with O(d) parameters, for free. So
$n_{\mathrm{cone}}(\varepsilon) = N_\sigma(g,\varepsilon)$, $n_{\mathrm{signed}}(\varepsilon) = N_\sigma(V,\varepsilon)$, and the gap is
governed by **$N_\sigma(Q,\varepsilon)$ = the σ-dictionary cost of a nondegenerate
quadratic** — which is **activation-dependent**.

**Measured (pure quadratic $V=\tfrac{C}{2}\|x\|^2$, $g=0$, so $n_{\mathrm{cone}}=0$ and
$n_{\mathrm{signed}}=N_\sigma(Q,\varepsilon)$ exactly; repo PDAP;
`../../scripts/quadratic_cost_by_activation.py`):** define the **quadratic-cost
exponent** $\kappa_\sigma := d(\log n)/d(\log\tfrac1\varepsilon)$.

| activation | $N_\sigma(Q,\varepsilon)$ | $\kappa_\sigma$ | head advantage |
|---|---|---|---|
| ReLU ($p=1$) | $\Theta(1/\varepsilon)$ (proved) | $1$ | capacity, diverging |
| matérn52 | 12→30 | $\approx 0.86$ | capacity, $\sim\varepsilon^{-0.86}$ |
| gaussian | 9→18 | $\approx 0.56$ | capacity, $\sim\varepsilon^{-0.56}$ |
| softplus, tanh | plateau (can't reach ε<0.24) | — | quadratic *hard* (ceiling-confounded) |
| gelu², silu² (squared) | $\approx 22$, **flat** | $\approx 0$ | **closes** — gap is *selection* |

**General statement (replaces the ReLU-only Θ(1/ε)):**
$$
\frac{n_{\mathrm{signed}}(\varepsilon)}{n_{\mathrm{cone}}(\varepsilon)} \;\gtrsim\; \frac{N_\sigma(Q,\varepsilon)}{N_\sigma(g,\varepsilon)} \;=\; \Theta\!\big(\varepsilon^{-\kappa_\sigma}\big)\big/N_\sigma(g,\varepsilon),
\qquad
\kappa_\sigma = \text{activation's quadratic-cost exponent}.
$$
The head's advantage **diverges iff $\kappa_\sigma > 0$** — true for kernel /
saturating activations (matérn, gaussian, and the proved ReLU), **false for
polynomial (squared) activations** where $\kappa_\sigma = 0$ and the observed gap is
selection, not capacity. *This is the useful, activation-faithful form;* the ReLU
theorem below is the sharp $\kappa_\sigma = 1$ endpoint, not the general case.

## Theorem A-d (the $\kappa_\sigma = 1$ endpoint: ReLU, Tier-1, two-sided, all d — PROVED)

**Setting (Tier 1, as general as the cone represents exactly).** Bounded $\Omega
\subset \mathbb{R}^d$ with $\mathrm{vol}(\Omega) > 0$; fixed $C > 0$; target
$$
V(x) = \tfrac{C}{2}\|x\|^2 - g(x), \qquad
g(x) = \sum_{i=1}^{K} c_i\,(w_i\!\cdot\!x - b_i)_+,\ c_i \ge 0,\ \|w_i\|=1,
$$
i.e. $g$ is any nonnegative combination of $K$ ReLU ridges (a convex,
piecewise-linear "switching" correction; this is exactly the convex part the cone
model can carry). **Metric:** $N(\text{model}, \varepsilon) = $ fewest atoms reaching
relative gradient error $\|\nabla f - \nabla V\|_{L^2(\Omega)} \le \varepsilon\,\|\nabla V\|_{L^2(\Omega)}$.
Activation ReLU ($p=1$). $N(\cdot)$ is **best-$n$-term** (minimum over all
representations), hence independent of the regularizer (ℓ1 or log non-convex).

**Theorem.**
$$
N(\text{cone}, \varepsilon) = K \ \ (\forall \varepsilon>0), \qquad
N(\text{signed}, \varepsilon) = \Theta(1/\varepsilon),
\qquad\Longrightarrow\qquad
\boxed{\ \frac{N(\text{signed},\varepsilon)}{N(\text{cone},\varepsilon)} = \Theta\!\Big(\frac{1}{K\,\varepsilon}\Big) \xrightarrow[\varepsilon\to0]{} \infty\ }.
$$
The advantage is a **diverging** factor (not constant), in **every dimension** $d$.

**Proof.**

*Cone, upper ($=K$, exact).* $V$ is literally a cone element: head $\tfrac C2\|x\|^2$,
$C$ exact, and $K$ nonnegative atoms reproducing $g$. Zero error at $K$ atoms, so
$N(\text{cone},\varepsilon)=K$ for all $\varepsilon$ — independent of accuracy.

*Signed, lower ($\Omega(1/\varepsilon)$).* The $K$ ridge hyperplanes have measure zero, so
(int $\Omega \ne \emptyset$) there is a ball $B \subseteq \Omega$ on which every
$(w_i\!\cdot\!x-b_i)_+$ is in a fixed activation state; there $g$ is affine and
$\nabla V = Cx - v_0$ ($v_0$ const), with $D^2V = C\,I$. Let $f = \sum_{i\le n}
c_i(w_i\!\cdot\!x-b_i)_+ + (\text{affine})$ be any signed model. $\nabla f$ is
**constant on each cell** of the arrangement of its $n$ hyperplanes; the number of
cells meeting $B$ is $N_{\mathrm{cell}} \le \sum_{j=0}^d \binom{n}{j} \le \kappa_d\, n^d$
(cells $K_\ell$, and the global affine gradient is absorbed into each cell
constant). On $K_\ell \cap B$, minimizing over an *arbitrary* constant vector
(a relaxation — the true $\nabla f$ is one specific such field, so this only
lowers the bound):
$$
\int_{K_\ell\cap B} |\nabla f - \nabla V|^2
\ge C^2 \min_{v}\!\int_{K_\ell\cap B}\!|x-v|^2
= C^2 \!\int_{K_\ell\cap B}\!|x-\bar x|^2
\ge C^2 c_d\,\mathrm{vol}(K_\ell\cap B)^{1+\frac2d},
$$
with $c_d = \tfrac{d}{d+2}\,\omega_d^{-2/d}$ (the **moment-of-inertia inequality**:
the ball minimizes $\int_S|x-\bar x_S|^2$ at fixed $\mathrm{vol}(S)$; $\omega_d=$ unit-ball
volume). Sum over the $N_{\mathrm{cell}}$ cells covering $B$ (so $\sum_\ell
\mathrm{vol}(K_\ell\cap B)=\mathrm{vol}(B)$) and apply the **power-mean inequality** (exponent
$1+\tfrac2d>1$, Jensen): $\sum_\ell s_\ell^{1+2/d} \ge N_{\mathrm{cell}}^{-2/d}\,
\mathrm{vol}(B)^{1+2/d}$. Hence
$$
\|\nabla f-\nabla V\|_{L^2(B)}^2 \ge C^2 c_d\,\kappa_d^{-2/d}\,\mathrm{vol}(B)^{1+\frac2d}\,n^{-2},
\quad\Rightarrow\quad
\|\nabla f-\nabla V\|_{L^2(\Omega)} \ge \frac{C\sqrt{c_d}\,\kappa_d^{-1/d}\,\mathrm{vol}(B)^{\frac12+\frac1d}}{n}.
$$
Dividing by $\|\nabla V\|_{L^2(\Omega)}$ and demanding relative error $\le\varepsilon$ gives
$$
n \ge \frac{\sqrt{c_d}\,\kappa_d^{-1/d}\,\mathrm{vol}(B)^{\frac12+\frac1d}}
{\|\nabla V\|_{L^2(\Omega)}/C}\cdot\frac1\varepsilon \;=\; \frac{\kappa(d,\Omega)}{\varepsilon}
= \Omega(1/\varepsilon).
$$
(The factor $C$ cancels against $\|\nabla V\|\sim C$, so the constant is purely
geometric — the *relative* bulk cost is $\Omega(1/\varepsilon)$ regardless of $C$.)

*Signed, upper ($O(1/\varepsilon)$, matching).* Coordinatewise grids make the rate tight:
$\partial_k V = C x_k - (\partial_k g)$; on each cell $\partial_k g$ is constant, and
the linear ramp $Cx_k$ on a length-$L$ segment is approximated to gradient error
$\varepsilon$ by $\lceil CL/\varepsilon\rceil$ equally-spaced ReLU kinks with $w=e_k$ (a staircase
gradient). Doing this in each of $d$ coordinates plus the $K$ ridges:
$N(\text{signed},\varepsilon)\le d\lceil CL/\varepsilon\rceil + K = O(1/\varepsilon)$. With the lower
bound, $N(\text{signed},\varepsilon)=\Theta(1/\varepsilon)$. $\blacksquare$

**Remarks.** (i) **Penalty-independent**: both bounds are best-$n$-term; the log
non-convex penalty (vs ℓ1) governs only *achievability* — whether PDAP attains
$N(\cdot)$ — which is direction **D2**, not this theorem. (ii) **All $d$**: the
proof is the $d\ge2$ argument; $d=1$ recovers the proved Theorem A. The mechanism
is exactly "*ReLU sums have Hessian $0$ a.e., so synthesizing the bulk curvature
$C\,I$ costs $\Theta(1/\varepsilon)$ atoms; the quadratic head supplies it with one
parameter.*" (iii) The affine head of the signed model is already included (it is
the global constant gradient, absorbed per cell) — it kills the constant $v_0$ but
**not** the linear-in-$x$ field $Cx$, which is the curvature it cannot afford.
(iv) **Real-data confirmation** (un-confounded): the $n(\varepsilon)$ slopes on VDP are
cone $\approx 0$, signed $\approx +1.35$ (`../../scripts/why_curve_vdp.py`) — exactly
$N(\text{cone})$ flat, $N(\text{signed})\sim\varepsilon^{-1}$ (observed $\varepsilon^{-1.35}$, the
predicted $\gtrsim 1$, the excess being PDAP above best-$n$-term). (v) **Tier-2
(smooth convex $g$) is open**: then $N(\text{cone},\varepsilon)=N_{\mathrm{ridge}}(g,\varepsilon)$ and the
ratio diverges iff $N_{\mathrm{ridge}}(g,\varepsilon)=o(1/\varepsilon)$ — "$g$ ridge-cheaper than a
quadratic" — the genuinely $d$-sensitive frontier (direction D3).

**Self-referee (inline, 2026-06-16).** Checked: the a-fortiori direction
(relaxing to arbitrary per-cell constants lowers the bound — correct sign); cell
count $\le\kappa_d n^d$ (arrangement of $n$ hyperplanes in $\mathbb{R}^d$); moment-of-
inertia constant $c_d=\tfrac{d}{d+2}\omega_d^{-2/d}$ (ball, direct integration);
power-mean exponent $1{+}2/d>1$ ⟹ $N_{\mathrm{cell}}^{-2/d}$; affine-head subsumption;
relative-error normalization (the $C$-cancellation). No gap found; an external
referee pass is *offered* but not auto-run (token discipline).

## ⚠ RETRACTION (2026-06-16, real-data PDAP): the "capacity is everything" verdict below was an artifact of a rigged synthetic target

The synthetic isolation (next section) concluded **selection ≈ 0, capacity (head)
= whole effect**. The repo-PDAP test on the **real VDP value function**
(`../../scripts/outdated/selection_isolation_repo_pdap.py`) **refutes that**:

| target relGrad | cone | signed | signed **given the bulk for free** |
|---|---|---|---|
| 0.30 (γ=0) | **2** | 12 | **44** |
| 0.30 (γ=1) | **1** | 10 | **28** |

Subtracting the cone's learned head $\tfrac12\tilde C\|x\|^2$ from the data made the
signed model **worse** (44 vs 12), not cone-sparse. So on real data the head is
**not** the separable driver.

**Why the synthetic experiment misled:** it built $V = \tfrac12\|x\|^2 + 3$
*dictionary* ridges, so subtracting the head trivially left 3 atoms — the
conclusion was baked into the target. The real VDP value function is **not**
"quadratic + a few ReLU ridges," so the head and the $c\ge0$ constraint cannot be
separated by bulk-subtraction.

**Corrected mechanism (honest):** the advantage is **structural and joint** — the
*whole* semiconcave parametrization ($V = \tfrac{C}{2}\|x\|^2 - g$, $g$ convex via
$c\ge0$) matches the value function, which genuinely is a smooth strongly-convex
bulk minus a convex correction. It does **not** decompose additively into
"head capacity" + "cone selection"; removing either component alone leaves a
target the signed model cannot fit sparsely. Theorem A-d (the head's
penalty-independent capacity advantage) remains **true and proved as a lower
bound on the signed model**, but it is a *contributing* lower bound, not "the whole
effect." The "capacity vs selection decomposition" framing itself was the error.

What stands: (a) the empirical 3–11× gap (activationsearch); (b) Theorem A-d as a
genuine $\Theta(1/\varepsilon)$ signed-model lower bound; (c) that the gap is a property of
the joint semiconcave structure, not isolable into head-only or c≥0-only. The
synthetic-experiment section below is **kept for the record, marked superseded.**

## [SUPERSEDED] Isolation experiment — synthetic result (rigged target; see retraction above)

Controlled test (`../../scripts/outdated/selection_isolation_d2.py`): synthetic d=2 semiconcave
target $V = \tfrac12\|x\|^2 - g$, $g$ = 3 switching ridges drawn *from the
dictionary* (so exact recovery is possible), gradient-augmented, p=1 ReLU. Three
models — **cone** ($c_i\ge0$ + quad head), **signed+head** (free $c$ + *matched*
quad head), **signed** (free $c$, affine head only) — at matched gradient accuracy
(gradRMSE ≤ 0.03). Recall the **matched-head fact**: signed+head $\supseteq$ cone,
so $n_{\mathrm{signed+head}} - n_{\mathrm{cone}}$ is **100% selection/optimization**, 0% capacity.

| γ | cone | signed+head | signed | selection gap (sh−cone) | capacity gap (s−sh) |
|---|---|---|---|---|---|
| 0 (ℓ1) | 5 | 4 | 138 | **−1** | **+134** |
| 5 (non-convex) | 3 | 3 | 59 | **0** | **+56** |

**Verdict — the requested "selection primary" hypothesis is REFUTED by the
controlled experiment:**
- **Selection gap ≈ 0.** With the head matched, the signed model recovers the
  *same* 3–4 atoms as the cone; the $c_i \ge 0$ constraint contributes nothing to
  atom count (at γ=0 it is even slightly *worse*, 5 vs 4 — the cone occasionally
  must add a positive atom where signed uses one negative).
- **Capacity gap is the whole effect.** The signed model *without* the head spends
  **56–134 atoms** approximating the smooth $\tfrac12\|x\|^2$ bulk that the head
  gives for free — this is the entire 3–11× gap in the activationsearch table.
- **The non-convex penalty** sharpens *both* models toward exact recovery
  (3 = true ridges) equally; it does not favor the cone.

**Conclusion: the semiconcave model's much-better sparsity for HJB value functions
is the quadratic head (capacity / Theorem A-d), essentially 100% of it at p=1, not
selection.** The cone constraint is what makes $V$ semiconcave (the structural
prior / well-posedness), but the *sparsity* comes from the head absorbing the
bulk. The d=1 ℓ1 "selection dichotomy" (appendix) describes a real but, here,
*negligible* secondary effect.

*Caveats:* synthetic target whose bulk is exactly the head's quadratic (real HJB
bulks are only approximately quadratic — residual smooth structure costs both
models equally, so the head gap persists). Clean-recovery regime; heavier
noise/ill-conditioning could surface a small selection effect, but the
first-order answer is unambiguous.

## Plan (revised by the result)

1. **Formalize Theorem A-d** — the d≥2, penalty-independent capacity advantage of
   the quadratic head: signed ReLU needs $\Omega(\varepsilon^{-d/2})$ atoms for the smooth
   strongly-convex bulk; cone needs 0. This is now *the* answer to "why sparser"
   and the priority. (1-D case = proved Theorem A; lift the ReLU $n$-term lower
   bound for a quadratic in H1 to d≥2.)
2. **Retire "selection" as the headline** — keep the d=1 ℓ1 dichotomy as a
   labeled appendix documenting a secondary effect that the controlled experiment
   shows is ≈0 here.
3. **Optional real-data confirmation:** add a quadratic head to `SignedModel` in
   the repo, rerun activationsearch on VDP; predict $n_{\mathrm{signed+head}} \approx n_{\mathrm{cone}}$,
   confirming the head (not c≥0) is the driver on real data too.

## Relation to the program

- Capacity core (penalty-independent, d≥2): T2 boundary, T3′ head-completeness,
  T1-d′ budget, **Theorem A-d** (this doc) — explains the bulk of the gap.
- Selection layer (the residual, non-convex + d≥2): the open target; the d=1 ℓ1
  certificate dichotomy is its simplest shadow (appendix).
- Honest headline: **for HJB value functions the semiconcave model's quadratic
  head represents the smooth bulk for free while its nonnegative atoms carry only
  the thin switching set; at p = 1 this is a genuine, growing capacity advantage
  (Theorem A-d), with a secondary selection advantage from the cone constraint
  under the non-convex penalty.**
