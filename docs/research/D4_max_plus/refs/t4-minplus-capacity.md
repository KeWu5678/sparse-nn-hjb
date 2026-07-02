# T4 — Min-plus capacity for HJB value functions (framed, two-sided)

Status: framed from citations with proof obligations listed, 2026-06-10. Ladder
position: see `t2-separation-draft.md`. Contains the companion result T2′
(dictionary incomparability), proved modulo one flagged extension.

The headline change forced by Gaubert–McEneaney–Qu (GMQ): **there is no uniform
no-curse theorem for min-plus approximation of semiconcave functions.** T4 must be
conditioned on min-plus *rank* (branch count), not on a Barron-type norm. The
two-sided picture then mirrors the ridge side exactly:

| dictionary | no-curse iff | matching lower bound |
|---|---|---|
| signed ridge $(+)$ | variation/$\mathcal{R}$-norm finite (Ongie; Yang–Zhou) | pseudo-dim: $N^{-\alpha/d}$ (YZ) |
| min-plus quadratic | min-plus rank small (DDM; McE07) | Bregman volume: $n^{-2/d}$ (GMQ 3.3) |

## Sign dictionary

GMQ/McEneaney work in max-plus/semiconvex convention: $\psi$ is $(c-\varepsilon)$-semiconvex
($\psi + \tfrac{c}{2}|x|^2$ convex), approximated from below by maxima of shared-Hessian
quadratics $-\tfrac{c}{2}|x|^2 + p^\top x + a$. Our convention is min-plus/semiconcave: $V = -\psi$,
$V$ $C$-semiconcave, approximated from above by minima of $\tfrac{C}{2}|x|^2 - p^\top x - a$ — which
is precisely the max-affine limit of this repo's `SemiconcaveModel`
($V = \tfrac{C}{2}\|x\|^2 - g$, $g$ max-affine). All cited results transfer under $V = -\psi$,
$\sup \leftrightarrow \inf$, minorant $\leftrightarrow$ majorant. (Writing this transfer out is obligation 1.)

## Inputs

**DDM** (Darbon–Dower–Meng 2023; file `CDC-max-plus-complexity bounds.pdf` — the
filename is misleading, see References). Setting (A1)–(A2): LQ dynamics
$f = A(t)x + B(t)u$, LQ Lagrangian, terminal cost $\Psi(x) = \min_{i \le m} \{\tfrac12 x^\top G_i x + a_i^\top x + b_i\}$.
Their Prop 1: the unique viscosity solution of the HJ PDE is

$$
V(t,x) = \min_{i \le m} \Bigl\{ \tfrac12 x^\top P_i(t)\, x + q_i(t)^\top x + r_i(t) \Bigr\},
$$

with $P_i, q_i, r_i$ solving decoupled Riccati / linear / scalar final-value problems
(their (14)–(16)). Min-plus linearity of the solution operator $\implies$ **the atom count
$m$ is conserved by the flow**. Exact representation only — no approximation theory,
no rates, no lower bounds, no learning.

**McE07** (McEneaney 2007, SIAM JCO 46(4)). $H$ = pointwise max of $M$ LQ
Hamiltonians: dual-space semigroup propagates quadratics with per-step cost
polynomial in $d$ (no spatial grid — "curse-of-dimensionality-free"), but the basis
count multiplies by $M$ per propagation step — the "curse of complexity" (GMQ §I–II,
their eq. (13)), controlled in practice by pruning.

**GMQ** (Gaubert–McEneaney–Qu 2011). For $\psi$ $(c-\varepsilon)$-semiconvex of class $C^2$ on a full-
dimensional compact convex $X$, approximated from below by $n$ quadratics
$-\tfrac{c}{2}|x|^2 + p^\top x + a$ (shared Hessian), the optimal errors satisfy (their Thms
3.1–3.2; via Gruber's asymptotics for circumscribed polytopes)

$$
\delta^1_{X,n}(\psi,c) \sim \frac{\alpha_1}{n^{2/d}} \left( \int_X \det\bigl(\psi''_x + cI\bigr)^{\frac{1}{d+2}} dx \right)^{\frac{d+2}{d}},
\qquad
\delta^\infty_{X,n}(\psi,c) \sim \frac{\alpha_2}{n^{2/d}} \left( \int_X \det\bigl(\psi''_x + cI\bigr)^{\frac12} dx \right)^{\frac{2}{d}},
$$

with $\alpha_1 \sim \frac{d+1}{\pi e}$, $\alpha_2 \sim \frac{1}{2\pi}\bigl(\Gamma(\tfrac{d}{2}+1)\,\vartheta_d\bigr)^{2/d}$ — both growing only $\sim$
linearly in $d$; the curse sits in the exponent $n^{-2/d}$. Thm 3.3: **every** max-plus
basis method of this class inherits $\Omega(n^{-2/d})$ on $C^2$ $c$-semiconvex value
functions. Crucially, their own remark: *"the integral term can be small if the
Hessian of $\psi$ is nearly constant and close to $-cI_d$ (attenuation of the curse of
dimensionality)"* — the lower-bound constant is a **Bregman volume** that
degenerates exactly where the Hessian saturates the semiconvexity bound.

## Lemma N (sup-norm nonexpansiveness of the semigroup)

Let $S_t$ be the value-function (Lax–Oleinik / dynamic-programming) semigroup of a
fixed optimal control problem, acting on terminal data. $S_t$ is order-preserving
and commutes with additive constants, hence

$$
\|S_t \Phi_1 - S_t \Phi_2\|_\infty \le \|\Phi_1 - \Phi_2\|_\infty.
$$

Proof: $\Phi_1 \le \Phi_2 + \|\Phi_1-\Phi_2\|_\infty \implies S_t\Phi_1 \le S_t\Phi_2 + \|\Phi_1-\Phi_2\|_\infty$ (monotonicity +
constants); symmetrize. $\blacksquare$ (Immediate from the inf-representation of the value
function; standard.)

## Theorem T4 (conditional no-curse, assembled from citations)

Setting: LQ data (DDM (A1)–(A2)), or $H$ a pointwise min of $M$ LQ Hamiltonians.

(a) **Exact rank conservation** [DDM Prop 1]. $\Psi$ = min of $m$ quadratics $\implies$
$V(t,\cdot)$ = min of the same $m$ Riccati-propagated quadratics for all $t$. Cost:
$m$ decoupled $d \times d$ Riccati ODEs — $\mathrm{poly}(d)\cdot m$, no grid, **no curse in $d$**.

(b) **Switching Hamiltonians** [McE07]. $H$ = min of $M$ LQ Hamiltonians $\implies$ after $N$
semigroup steps the representation rank is $\le m \cdot M^N$: curse of *complexity* in $N$,
not of *dimensionality* in $d$. Pruning (GMQ §V–VI) controls the growth; the
pruned-rank notion is obligation 3.

(c) **Reduction to the data**. For general $C$-semiconcave $L$-Lipschitz $\Psi$ and any
$n$-atom surrogate $\Psi_n$: by Lemma N, $\|S_t\Psi - S_t\Psi_n\|_\infty \le \|\Psi - \Psi_n\|_\infty$, and $S_t\Psi_n$ is
exactly a min of (a)/(b)-many quadratics. **The capacity question for the flow
reduces entirely to approximating the terminal data.**

(d) **Sharp data rates, shared Hessian** [GMQ 3.1–3.2, transferred]. For $C^2$
$C$-semiconcave data, $n(\varepsilon) \asymp \varepsilon^{-d/2} \cdot \bigl(\int \det(C I - D^2\Psi)^{1/2}\bigr) \cdot (\alpha_2\text{-type constant})$ —
cursed in general, but **attenuated to near-zero exactly on the Hessian-saturated
class $D^2\Psi \approx C I$ a.e.** This is the precise sense in which HJB value functions are
min-plus-favorable: along optimal flows the Riccati dynamics push Hessians toward
saturation, and the kink set (where the lower bound says nothing — $\Psi \notin C^2$) is
handled at 2-atoms-per-shock (T2 part (A) pattern).

(e) **No uniform no-curse** [GMQ 3.3]. On the full $C^2$ $c$-semiconvex class, every
shared-Hessian min-plus method is $\Omega(n^{-2/d})$. Any honest T4 must therefore be
stated relative to rank / Bregman volume, never uniformly.

## Proposition T2′ (incomparability of the two dictionaries)

Let $s \in C^2(\mathbb{R})$, not affine, with $s''$ bounded, and $\psi(x) = s(w \cdot x)$ on a full-
dimensional compact convex $X \subset \mathbb{R}^d$, $|w| = 1$.

(i) *Ridge side, cheap*: $\psi$ is intrinsically one-dimensional; its ridge variation
cost equals the 1-D cost of $s$ (Savarese/PDW characterization: $\sim \int |s''|$, finite,
dimension-independent), and $N$ ridge atoms with direction $w$ achieve 1-D rates
($N^{-2}$-type for ReLU; cf. T1 Cor 4 logic).

(ii) *Shared-Hessian min-quadratic side, cursed*: for $c > \sup(-s'')$, $\psi$ is
$c$-semiconvex and $C^2$, and $\det(\psi''_x + cI) = c^{d-1}\,(c + s''(w \cdot x)) \ge
c^{d-1}\,(c - \sup(-s'')) > 0$ on $X$. GMQ Thm 3.2 $\implies$ the best $n$-atom minorant
approximation satisfies $\delta^\infty \gtrsim n^{-2/d}$: **the curse, on a target the ridge
dictionary represents with one neuron.**

Together with T2 ($v_d$: $2d$ min-atoms vs $\|v_d\|_{\mathcal{R}} = \infty$) this proves the signed-ridge
and shared-Hessian-quadratic dictionaries are **incomparable** — neither dominates
— and HJB value functions with curved switching structure sit on the min-plus-
favorable side, plain ridge-like profiles on the ridge-favorable side.

*Caveat discharged (obligation 2 closed, 2026-06-10):* GMQ's lower bound is
stated for **minorant** (max-plus-projection) approximation. Lemma P below
removes the one-sidedness — at the price of a uniform-eigenvalue constant in
place of GMQ's sharp Bregman volume — so the $\Omega(n^{-2/d})$ bound binds *trained*
shared-Hessian models, not just max-plus projections. T2′(ii) is thereby proved
for the unconstrained class.

*Note on free Hessians:* the per-branch-curvature dictionary (free Hessians,
KVV/DDM) strictly contains the shared-Hessian one **and** emulates 1-D ridge
structure via rank-one-Hessian atoms $q(x)$ = quadratic in $(w \cdot x)$ — so it is cheap on
*both* test families ($2d$ atoms on $v_d$; 1-D rates on ridges). The incomparability
is specifically signed-ridge vs shared-Hessian; the per-branch head dominates
both on these tests. On generic targets it is *conjectured* to face an
$n^{-3/d}$-type curse (matching the piecewise-quadratic upper bound; the covering
proof does not extend naively — see the third-order remark after Lemma P): its
true power is exactness at finite min-plus rank, not generic smoothing.

## Lemma P (unconstrained covering lower bound) — obligation 2 closed

**Lemma.** Let $X \subset \mathbb{R}^d$ be compact convex with $\mathrm{vol}(X) > 0$, $V \in C^2(\bar X)$, and $C \in \mathbb{R}$
such that $D^2\bigl(\tfrac{C}{2}|x|^2 - V\bigr) \succeq \varepsilon_0 I$ on $X$ for some $\varepsilon_0 > 0$ (the Hessian of $V$ stays
$\varepsilon_0$ below its semiconcavity bound on $X$). Then every
$\tilde V = \min_{i \le n} \{\tfrac{C}{2}|x|^2 - \ell_i(x)\}$ with $\ell_i$ affine — **no minorant constraint** —
satisfies

$$
\|V - \tilde V\|_{L^\infty(X)} \;\ge\; \frac{\varepsilon_0}{4} \left( \frac{\mathrm{vol}\, X}{n\,\omega_d} \right)^{2/d},
\qquad \omega_d = \text{vol of unit ball}.
$$

**Proof.** Set $g := \tfrac{C}{2}|x|^2 - V$ and $\hat g := \max_{i \le n} \ell_i$. The shared quadratic
cancels in the difference, so $\delta := \|V - \tilde V\|_{L^\infty(X)} = \|g - \hat g\|_{L^\infty(X)}$, and $g$ is
$\varepsilon_0$-strongly convex on the convex set $X$.
(1) *One-sided global bound from the max structure*: for every $i$,
$\ell_i \le \hat g \le g + \delta$ on all of $X$, hence $h_i := g - \ell_i \ge -\delta$ on $X$.
(2) *Two-sided bound on the active region*: on $R_i := \{x \in X : \hat g(x) = \ell_i(x)\}$,
$h_i = g - \hat g \in [-\delta, \delta]$.
(3) *Strong convexity localizes the active region*: $h_i$ is $\varepsilon_0$-strongly convex on
$X$; let $\bar x_i$ minimize $h_i$ over $\bar X$. First-order optimality on a convex set gives
$\langle \nabla h_i(\bar x_i),\, x - \bar x_i \rangle \ge 0$ for $x \in X$, hence
$h_i(x) \ge h_i(\bar x_i) + \tfrac{\varepsilon_0}{2}|x - \bar x_i|^2 \ge -\delta + \tfrac{\varepsilon_0}{2}|x - \bar x_i|^2$, using (1).
Combined with (2): $R_i \subseteq B\bigl(\bar x_i,\, 2\sqrt{\delta/\varepsilon_0}\bigr)$.
(4) *Covering count*: $X = \bigcup_i R_i$, so $\mathrm{vol}\, X \le n\,\omega_d\,(4\delta/\varepsilon_0)^{d/2}$; solving for $\delta$
gives the claim. $\blacksquare$

**Sharpness (referee pass 2026-06-11).** The constant $\varepsilon_0/4$ is attained: in $d = 1$
with $V$ quadratic, $D^2V = (C-\varepsilon_0) I$ on $[-R, R]$, the optimal $n$-piece max-affine
approximant (equioscillating secants on an equal partition) achieves error
exactly $\varepsilon_0 R^2/(4n^2)$ — Lemma P is sharp, not merely order-correct.

**Corollary 1 (T2′(ii) unconstrained).** $\psi(x) = s(w \cdot x)$ with $s \in C^2$, $C > \sup s''$
and $C > 0$: take $\varepsilon_0 = \min(C - \sup s'',\, C) > 0$ — the Hessian $C I - s'' w w^\top$ has
eigenvalue $C - s''$ along $w$ and $C$ on $w^\perp$, so $\succeq \min(C - \sup s'',\, C)\, I$ (the two can
order either way depending on the sign of $\sup s''$). The curse $\Omega(n^{-2/d})$ holds
for unconstrained shared-Hessian approximation of a target the ridge dictionary
represents with one neuron.

**Corollary 2 ($v_d$ smooth branches; proves the table's row-2 prediction).** Fix
the model curvature $C \ge C^*$. Inside the active cone of the branch centered at
$-e_1$, take $B = B(1.4\,e_1,\, 0.1)$ (valid in every $d$: points have $x_1 \ge 1.3 \ge |x_j|$).
There $r^2 = |x + e_1|^2 \in [5.29,\, 6.25]$ and the active branch's largest Hessian
eigenvalue $\lambda(r^2) = (r^2-1)e^{-r^2/2} \le 0.305$, so $\varepsilon_0 = C - 0.305 \ge 0.14$. Lemma P on
$B$: $n \gtrsim \mathrm{vol}(B)\,(\varepsilon_0/(4\varepsilon))^{d/2}$ atoms for accuracy $\varepsilon$.
*Caution:* a branch's Hessian at its **own** center is $-I$ ($g'' = (C^*+1) I$ there),
but centers lie in *other* branches' active cones, so that value is irrelevant;
on active regions $\varepsilon_0$ ranges in $\approx (0,\, 0.45]$, vanishing only on the saturation
sphere $r^2 = 3$.

**Remark (learnable $C$ — the dichotomy).** Corollaries 1–2 fix $C$; a trained $C$ does
not escape. For $v_d$: on $B_{\mathrm{hi}} := B(0.74\,e_1,\, 0.05)$ (inside the active cone,
$r^2 \in [2.86,\, 3.21]$) the target's radial curvature is $\ge m_1 \approx 0.444$; on the
far-field $B_{\mathrm{lo}}$ (e.g. $r^2 \approx 16$) the branch curvature is $\le M_2 \approx 0.01$. If the model's
$C \le 0.43$, then on $B_{\mathrm{hi}}$ the target curves faster than any $C$-semiconcave function
along a radial chord of length $L$, giving the **$n$-independent** floor
$\|V - \tilde V\|_\infty \ge (m_1 - C)\, L^2/16$ (midpoint inequality for $f$ with distributional
$f'' \ge m_1 - C$). If $C > 0.43$, Lemma P applies on $B_{\mathrm{lo}}$ with $\varepsilon_0 \ge 0.42$. Either way
the error is $\ge \min(\mathrm{const},\, \mathrm{const} \cdot n^{-2/d})$. The same dichotomy covers Cor 1 for
non-quadratic $s$; for exactly-quadratic $s$ ($s''$ constant) restrict to a slice
$\{w \cdot x = 0\}$ — the model class restricts to the model class in $d-1$ variables — and
apply Lemma P there: $n^{-2/(d-1)}$ (requires $0 \in \mathrm{int}\, X$ so the slice has positive
$(d-1)$-volume).

**Remark (free Hessians, third-order — CONJECTURE only).** Free-Hessian atoms can
match value, gradient, and Hessian at a point, leaving cubic contact; an
$n^{-3/d}$ lower bound would match the piecewise-quadratic upper bound
($h^3$ with $h \sim n^{-1/d}$) and is conjectured. The covering proof does **not** extend
naively: an atom's accuracy region can hug the zero variety of the third-order
form (a codim-1 cone whose $\delta$-tubes have volume *linear* in $\delta$, far exceeding
$\delta^{d/3}$), so the per-atom volume bound collapses. Not needed for the program's
claims.

**Remark (orientation duality).** GMQ's class is max-of-*concave* quadratics
(semiconvex side; condition $c > \sup(-s'')$ in T2′(ii) above); Lemma P's class is
the repo's min-of-*convex* paraboloids (semiconcave side; condition $C > \sup s''$
in Cor 1). Mirror images under $V = -\psi$ — both statements true, with different
constants; the repo's model is the Lemma P side.

**Relation to GMQ.** GMQ Thms 3.1–3.2 = sharp anisotropic constant (Bregman
volume $\int \det(g'')^{1/2}$), minorant class; Lemma P = uniform-eigenvalue constant,
unconstrained class. Use GMQ for sharpness, Lemma P for scope; both degenerate
on the Hessian-saturated class ($\varepsilon_0 \to 0 \iff$ Bregman volume $\to 0$ locally), which is
the favorable-HJB mechanism of T4(d).

## Repo predictions (three-way; $v_d$ and ridge test functions)

| model (ideal dictionary) | on $v_d$ | on $s(w \cdot x)$ (T2′) | status |
|---|---|---|---|
| `SignedModel` — signed $\sigma_k$ ridges | weight cost $\to \infty$ (T2 B) | $\sim$1 atom, 1-D rates | proved ($d=2$) / cited |
| `SemiconcaveModel` max-affine limit — shared-Hessian paraboloids | kinks cheap, $\Omega(n^{-2/d})$ on smooth branches (Lemma P Cor 2) | $\Omega(n^{-2/d})$ (Lemma P Cor 1) | **proved** |
| per-branch quadratic head (KVV/DDM) — free-Hessian min-quadratics | **$2d$ exact** | 1-D rates via rank-1 Hessians | assembled |

PDAP experiment: train all three on $v_d$ samples ($d = 2, \dots, 6$), plot atom count vs
target accuracy $\varepsilon$; predicted slopes: divergent-with-mass (row 1), $\sim \varepsilon^{-d/2}$
(row 2), flat at $2d$ (row 3).

## Proof obligations

1. **Sign transfer**: write the GMQ semiconvex/max-plus $\to$ semiconcave/min-plus
   dictionary once, carefully (mechanical; Lemma P is already stated natively in
   the semiconcave orientation, so this matters only when citing GMQ's sharp
   constants).
~~2. **Minorant $\to$ unconstrained** — closed 2026-06-10 by Lemma P above.~~
3. **Rank growth and pruning**: precise $m \cdot M^N$ bound for $H$ = min of $M$ LQ
   Hamiltonians and a clean definition of pruned/effective rank (from McE07 +
   GMQ §V).
4. **Ridge-convex intermediate model**: `SemiconcaveModel` with $\sigma^p$ ridges
   ($p > 1$) is neither max-affine (GMQ doesn't bind) nor fully covered by T2(C)
   on smooth targets; characterize its position.
5. **Branch-count capacity for general HJB data** (the genuinely open piece):
   combine Cannarsa–Sinestrari Ch. 4 (rectifiability of singular sets) with
   Subbotina's characteristic representative formula to produce a finite atlas
   of $C^2$ branches with controlled overlap, giving $n(\varepsilon) \lesssim \#\text{branches} \times \text{per-branch}$
   cost for the per-branch-Hessian dictionary — the rigorous form of
   "min-plus rank of a value function".

## References (local paths under /Users/chaoruiz/Documents/NotePaper/MasterThesis/representation theorem/)

- Gaubert, McEneaney, Qu (2011). *Curse of dimensionality reduction in max-plus
  based approximation methods: theoretical estimates and improved pruning
  algorithms*. CDC-ECC 2011, arXiv:1109.5241. — `Gaubert–McEneaney–Qu-2011.pdf`
- Darbon, Dower, Meng (2023). *Neural network architectures using min-plus
  algebra for solving certain high-dimensional optimal control problems and
  Hamilton–Jacobi PDEs*. MCSS 35:1–44, arXiv:2105.03336.
  ⚠ file misleadingly named — `CDC-max-plus-complexity bounds.pdf`
- McEneaney (2007). *A curse-of-dimensionality-free numerical method for solution
  of certain HJB PDEs*. SIAM J. Control Optim. 46(4):1239–1276.
  — `curse-of-dimensionality-free_McEeany.pdf`
- Dower, McEneaney. *A max-plus fundamental solution semigroup for a class of
  lossless wave equations* (adjacent only). ⚠ file misleadingly named —
  `curse-free-max-plus_McEneay.pdf`
- Cannarsa, Sinestrari (2004). *Semiconcave Functions, Hamilton–Jacobi Equations,
  and Optimal Control*. Birkhäuser. — `semiconcave-functions_Cannarsa.pdf`
- Kunisch, Vásquez-Varas (2026), Ongie et al. (2019), Yang–Zhou (2025),
  Petrosyan et al. (2020), Subbotina (2006): see `t2-separation-draft.md`.
