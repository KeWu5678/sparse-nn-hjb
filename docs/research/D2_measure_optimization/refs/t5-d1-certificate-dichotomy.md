# T5 (d = 1) — the certificate dichotomy: why the cone is $\approx 2\times$ sparser

Status: theorem with proof, 2026-06-15. The rigorous core (Lemmas 1–3 + the
structural restriction) is proved; the $\approx 2\times$ selection factor follows
from that core plus one genericity hypothesis (equioscillation), which is
numerically confirmed. Builds on Proposition C (Theorem B) and the
gradient-training amendments. Detail/overview: `theory-ladder.md`. Numerics:
`../../scripts/t5_certificate_dichotomy.py`.

## Setting

Gradient training, $d = 1$, domain $\Omega = [-1,1]$, target gradient $y \in L^2(\Omega)$.
Power-2 ridge dictionary; the **gradient features** are one-sided ramps

$$
\varphi_{\varepsilon,b}(x) = 2\varepsilon(\varepsilon x - b)_+, \qquad \varepsilon \in \{\pm 1\}, \; b \in \mathbb{R}.
$$

- **Signed model** (`SignedModel`): fit $y$ by $\sum_i c_i \varphi_{\varepsilon_i, b_i} + \mathrm{affine}$, coefficients
  $c_i$ **free**, penalty $\alpha \sum_i |c_i|$.
- **Cone model** (`SemiconcaveModel`): fit $y$ by $\tilde C x + a - \sum_i c_i \varphi_{\varepsilon_i, b_i}$ with
  $c_i \ge 0$, $\tilde C \ge 0$ (quadratic head), $a$ free, penalty $\alpha \sum_i c_i$.

Both are continuum ($1$-D total-variation) lasso problems over the knee
parameter $b$. Write $\eta := y - (\text{fit})$ for the optimal residual.

## Lemma 1 (certificate $=$ double antiderivative of the residual)

Define the dual certificate $q_\varepsilon(b) := \langle \varphi_{\varepsilon,b}, \eta \rangle$. For the $+$ branch and $b$ in
the hull,

$$
q_+(b) = 2\int_b^1 (x - b)\, \eta(x)\, dx, \qquad
q_+'(b) = -2\int_b^1 \eta(x)\, dx, \qquad
\boxed{\,q_+''(b) = 2\eta(b)\,}
$$

(Proposition C, already proved and machine-verified). The $-$ branch satisfies
the mirror identity $q_-''(b) = -2\eta(-b)$ (sign verified numerically
2026-06-15; the doc previously stated $+2\eta(-b)$ — wrong sign, but it does not
affect any count below: a global sign flip of $q_-''$ swaps which critical points
are maxima vs minima while preserving their number, alternation, and the
sign-change count of the driving residual). **Consequence:** each certificate is
$C^2$ wherever $\eta$ is continuous, and *its second derivative is the residual* —
so the certificate's convexity/concavity tracks the sign of $\eta$ pointwise.

**Endpoint zeroing (free affine head).** With the unpenalized affine term, the
optimal residual is orthogonal to $\{1, x\}$, so $\int_\Omega \eta = \int_\Omega x\eta = 0$. Then
$q_+'(\pm 1) = 0$ and $q_+(\pm 1) = 0$ (verified numerically): the head drives both
certificate endpoints to zero, so boundary knees are never active and "active $\Rightarrow$
interior critical point" in Lemma 3 holds automatically. (Also $\int\eta = 0$ forces
$K \ge 1$: the residual must change sign.)

## Lemma 2 (critical points alternate — counts CONTACT REGIONS, not atoms)

Let $K$ be the number of sign changes of $\eta$ on the hull. By Lemma 1,
$q_+'' = 2\eta$ changes sign $K$ times, so $q_+'$ is monotone on each of the $K+1$
resulting intervals and has at most $K+1$ zeros. Hence $q_+$ has at most $K+1$
**critical points**, and since $q_+''$ flips sign at each, they **alternate**
between local maxima and local minima. Therefore the certificate touches each
bound in alternating, equinumerous-up-to-one sets of critical points.

**Caveat (referee 2026-06-15): this bounds CERTIFICATE CRITICAL POINTS, not the
number of active atoms.** At finite penalty $\alpha$ the solver places a *cluster*
of $m(\alpha) \propto 1/\alpha$ grid atoms around each bound-contact region (the
adjacent near-null knees of Theorem B), so the raw atom count far exceeds $K+1$
(observed $\sim 3\times$). The right invariant is the number of **contact regions**
(maximal arcs where $|q| = \alpha$), one *continuum* atom each; Lemma 2 counts those.
Write $R_\pm$ for the number of $\pm\alpha$ contact regions; alternation gives
$R_+ = R_- \pm 1$, i.e. $R_+ \approx R_-$.

## Lemma 3.5 (dual-intersection structure — sharpens hypothesis M to LP geometry)

The three models' dual-feasible sets are constraints on the certificate
$q_\varepsilon(b) = \langle \varphi_{\varepsilon,b}, \eta \rangle$ (atoms enter the cone with a sign; the KKT reduced-cost
condition is one-sided):

$$
\mathcal{F}_{\mathrm{semi}} = \{\eta : q_\varepsilon(b) \ge -\alpha \ \forall \varepsilon,b\} =: \mathcal{A}
\quad\text{(atoms } -c\varphi,\, c \ge 0\text{)},
$$
$$
\mathcal{F}_{\mathrm{conv}} = \{\eta : q_\varepsilon(b) \le +\alpha \ \forall \varepsilon,b\} =: \mathcal{B}
\quad\text{(atoms } +c\varphi,\, c \ge 0\text{)},
$$
$$
\mathcal{F}_{\mathrm{signed}} = \{\eta : |q_\varepsilon(b)| \le \alpha\} = \mathcal{A} \cap \mathcal{B}.
$$

As *sets* this is a triviality (it is the definition of the three norms:
$|q| \le \alpha \iff q \ge -\alpha$ and $q \le +\alpha$). Its value is the **projection**
reading it unlocks (external referee, 2026-06-16), which closes hypothesis (M)
down to a single inequality:

**The projection mechanism (replaces the earlier disjoint-support story, which
was wrong).** Each model's optimal residual is the projection of the (head-
deflated) data $y$ onto its dual-feasible polytope:

$$
\eta_{\mathrm{cone}} = \mathrm{Proj}_{\mathcal{A}}(y),
\qquad
\eta_{\mathrm{signed}} = \mathrm{Proj}_{\mathcal{A} \cap \mathcal{B}}(y),
\qquad \mathcal{A} \cap \mathcal{B} \subseteq \mathcal{A}.
$$

The one load-bearing fact: **the cone certificate never exceeds the upper bound,
$q_{\mathrm{cone}}(b) \le +\alpha$ for all $b$** (equivalently $\eta_{\mathrm{cone}} \in \mathcal{B}$; the cone
certificate *equioscillates*, reaching $+\alpha$ at its maxima — verified to machine
precision, $\max q_{\mathrm{cone}} - \alpha \le 0$ on every target tried). Given it,
$\eta_{\mathrm{cone}}$ is already feasible for the *tighter* signed problem, so by **uniqueness
of projection onto a convex set**

$$
\boxed{\;\eta_{\mathrm{signed}} = \eta_{\mathrm{cone}}\;}
$$

— *one shared residual, hence one shared certificate.* Everything follows:

1. **Hypothesis (M) is now a corollary, not an assumption.** Same certificate $\Rightarrow$
   same $-\alpha$ contacts $\Rightarrow$ $R_-^{(\mathrm{signed})} = R_-^{(\mathrm{cone})}$.
2. **The factor $2$.** Apply Lemma 2 to that single certificate: $q'' = 2\eta$ forces
   bound-contacts to alternate $+\alpha, -\alpha, +\alpha, \dots$, so $R_+ = R_-$ **exactly**
   (endpoint-zeroing $q(\pm1)=0$ + alternation removes the $\pm1$ defect). Signed uses
   both bound families, cone only $-\alpha$ $\Rightarrow$ ratio $= 2$ *exactly* (not "$\to 2$").
3. **Target-independence.** This uses no semiconcavity. *Verified on convex
   ($\tfrac12 x^2$), oscillatory ($\sin 15x$), and convex-kink ($|x|$) targets:*
   $\|\eta_{\mathrm{signed}} - \eta_{\mathrm{cone}}\| \le 10^{-6}$ and $\max q_{\mathrm{cone}} - \alpha = 0$ in every case
   (`../../scripts/t5_projection_mechanism.py`). The earlier claim that (M) holds
   "because the target is semiconcave / disjoint support" is **retracted** — (M)
   holds for convex targets too, where there are no downward kinks at all.

**The remaining open analytic step is now a single inequality:**
$q_{\mathrm{cone}} \le +\alpha$ everywhere (the cone optimum's certificate respects the upper
bound — i.e. equioscillates). By KKT the cone already gives $q_{\mathrm{cone}} \ge -\alpha$; the
claim is that its optimum also never overshoots $+\alpha$. This is far more tractable
than the discarded "disjoint support" and is numerically airtight; it is the lone
gap between the $\approx 2\times$ observation and a theorem.

This reframes T5-d1 as **projection onto nested dual polytopes** — the home of the
cited nonnegative-recovery theory (Slawski–Hein; de Castro–Gamboa).

## Lemma 3 (KKT: which bound each model may touch)

At the optimum, $|q_\varepsilon(b)| \le \alpha$ for all $(\varepsilon, b)$, and a knee is **active** (carries an
atom) only at an *interior* point where its certificate attains a bound; interior
equality of $|q_\varepsilon| \le \alpha$ forces $q_\varepsilon'(b) = 0$, i.e. **active $\Rightarrow$ critical point at a
bound.** The sign constraint then selects the bound:

- **Signed** ($c_i$ free): active where $q_\varepsilon(b) = +\alpha$ (a maximum) **or** $q_\varepsilon(b) = -\alpha$ (a
  minimum) — *both* bounds.
- **Cone** ($c_i \ge 0$, atoms enter as $-c_i \varphi$): the reduced-cost condition is
  $\langle -\varphi_{\varepsilon,b}, \eta \rangle \le \alpha$ with equality when active, i.e. $q_\varepsilon(b) = -\alpha$ — the
  **single** $-\alpha$ bound only. (On the $+$ branch this is a minimum of $q_+$ since
  $q_+'' = 2\eta > 0$ there; on the $-$ branch, after the sign correction $q_-'' = -2\eta(-b)$,
  the same $-\alpha$ *value* bound may be attained at a maximum of $q_-$. The
  bound-selection is by *value* $q = -\alpha$, not by extremum type, so the count is
  unaffected — replace any "$-\alpha$ minima" reading by "$-\alpha$ bound".)

(Verified: in `../../scripts/t5_certificate_dichotomy.py` the cone's active atoms sit
at the $-\alpha$ bound in **every** run — $0$ atoms at $+\alpha$ — while the signed model
splits $\approx$ evenly between $+\alpha$ and $-\alpha$.)

## Theorem (certificate dichotomy — corrected to a RATIO statement, 2026-06-15)

The original draft asserted absolute counts $n_{\mathrm{signed}} = K+1$,
$n_{\mathrm{cone}} = \lceil (K+1)/2 \rceil$. **That is false** — atoms cluster, so counts scale as
$1/\alpha$ at fixed $K$ (referee + numerics below). The correct, robust statement is
about *continuum* atom counts (= contact-region counts), where the clustering
multiplicity cancels:

**Theorem.** Let the signed and cone problems be solved at the same penalty $\alpha$
on the same data, with optimal residuals $\eta_s, \eta_c$. Let $R_\pm^{(s)}$, $R_-^{(c)}$ be
the numbers of $\pm\alpha$ contact regions of the respective certificates, and let
$N^{\mathrm{cont}}$ denote continuum atom counts (grid clusters merged). Then:

$$
N^{\mathrm{cont}}_{\mathrm{signed}} = R_+^{(s)} + R_-^{(s)},
\qquad
N^{\mathrm{cont}}_{\mathrm{cone}} = R_-^{(c)},
$$

and **if** (E) every certificate critical point attains its bound (so contact
regions $=$ critical points) **and** (M) the two problems form the same number of
$-\alpha$ contact regions, $R_-^{(s)} = R_-^{(c)}$, then by Lemma 2's alternation
$R_+^{(s)} = R_-^{(s)} \pm 1$ and

$$
\frac{N^{\mathrm{cont}}_{\mathrm{signed}}}{N^{\mathrm{cont}}_{\mathrm{cone}}}
= \frac{R_+^{(s)} + R_-^{(s)}}{R_-^{(c)}} \longrightarrow 2.
$$

**Proof.** Lemma 3: a signed continuum atom sits at a $+\alpha$ or $-\alpha$ contact
region; a cone continuum atom only at a $-\alpha$ region. So the displayed equalities
hold by definition of contact region. Hypothesis (E) makes the contact-region
count equal the critical-point count, and Lemma 2 makes the two bounds'
critical points alternate, hence $R_+^{(s)} = R_-^{(s)} \pm 1$. Hypothesis (M)
identifies the cone's $-\alpha$ region count with the signed's. The ratio follows. $\blacksquare$

**Why the clustering multiplicity cancels (now a corollary of $\eta_s = \eta_c$).**
Lemma 3.5 gives $\eta_{\mathrm{signed}} = \eta_{\mathrm{cone}}$ (modulo the one open inequality
$q_{\mathrm{cone}} \le +\alpha$). The two problems therefore see the *same* residual, so the
$\propto 1/\alpha$ cluster of grid atoms around each contact region is *literally identical*
between them — not merely "the same local problem." Thus raw counts are
$m(\alpha) \cdot (R_+ + R_-)$ vs $m(\alpha) \cdot R_-$ with the *same* $m(\alpha)$ and (by the shared
certificate) the *same* $R_-$, so $m(\alpha)$ cancels and the ratio is $(R_+ + R_-)/R_- = 2$
exactly. This is why the **raw** ratio ($1.67$–$1.81$) is a finite-grid artifact and
the **merged** ratio is exactly $2.00$ (table below). The lone remaining gap is the
inequality $q_{\mathrm{cone}} \le +\alpha$ (see Honest status), not a separate "multiplicity
matching" assumption.

*Two-branch bookkeeping.* The $+$ and $-$ ridge branches contribute atoms at
distinct knee locations: $q_+$ critical points are zeros of the right-tail
$\int_b^1 \eta$, $q_-$ critical points zeros of the left-tail $\int_{-1}^{-b}\eta$ — distinct
functionals of $\eta$. (The often-stated fact $\varphi_{+,b} = 2(x-b) + 2(b-x)_+$ with the
leftover outside the $-$ dictionary is *true but irrelevant* to the count — referee.)
The counts $R_\pm$ above are summed over *both* branches; per-branch ratios are
erratic and are **not** each $2$ — the factor $2$ is a both-bounds-vs-one-bound
effect on the totals, not a per-branch effect.

**The eviction reading.** The cone activates only at $-\alpha$ contact regions and
**cannot place atoms at the $+\alpha$ regions** the signed model uses. Those $+\alpha$
regions are the two-sided-optimality atoms of Theorem B — overhead the cone is
structurally free of. (Note: the certificate of the *cone* still *reaches* $+\alpha$ at
its maxima; it simply cannot activate there — "one-sided active set," not
"one-sided certificate.")

## Numerical confirmation (`../../scripts/t5_certificate_dichotomy.py`)

Mollified saturated target (3 shocks, $\delta = 0.04$), coordinate descent (robust;
LARS provably breaks here, Prop C). RAW = grid atom count; MERGED = continuum
atoms (grid-adjacent knees of the same branch merged, gap $0.08$); $K$ = residual
sign changes; $+/-$ = signed actives bucketed by certificate sign:

| $\sigma$ | $\alpha$ | $K$ | sgn raw | $+$ | $-$ | sgn **mrg** | cone raw | cone **mrg** | acc (s/c) | raw ratio | **mrg ratio** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | $3\!\cdot\!10^{-3}$ | 7 | 5 | 2 | 3 | 4 | 3 | 2 | 12.3/12.3% | 1.67 | **2.00** |
| 0 | $1\!\cdot\!10^{-3}$ | 7 | 12 | 6 | 6 | 10 | 7 | 5 | 8.1/8.1% | 1.71 | **2.00** |
| 0 | $3\!\cdot\!10^{-4}$ | 7 | 29 | 14 | 15 | 12 | 16 | 6 | 3.4/3.4% | 1.81 | **2.00** |
| 0.05 | $3\!\cdot\!10^{-3}$ | 33 | 5 | 2 | 3 | 4 | 3 | 2 | 12.3/12.3% | 1.67 | **2.00** |
| 0.05 | $1\!\cdot\!10^{-3}$ | 59 | 12 | 7 | 5 | 10 | 7 | 5 | 7.9/7.9% | 1.71 | **2.00** |
| 0.05 | $3\!\cdot\!10^{-4}$ | 102 | 22 | 12 | 10 | 12 | 13 | 6 | 3.4/3.4% | 1.69 | **2.00** |

What the data say (referee-corrected reading): (a) **signed splits across both
bounds** (genuine, not forced); (b) cone sits entirely on $-\alpha$ — *definitional*
from KKT (Lemma 3), so not independent evidence; (c) **matched accuracy** — in fact
$\eta_s = \eta_c$ exactly (Lemma 3.5), so (b), (c) and the equal $R_-$ are *one fact
measured three ways*, not three confirmations; (d) the **merged ratio is exactly
$2.00$ in every row**, including under noise where $K$ inflates to $102$ — so the
factor $2$ is *not* a $K$-effect; (e) the **raw** ratio ($1.67$–$1.81$) is the
clustering artifact ($m(\alpha)$ shrinking), *not* convergence to $2$. Cross-check: the
independent debiased control ratios (1.8/2.6/1.8/2.1 in `theory-ladder.md`) are
raw-count ratios and so likewise understate the continuum $2$.

## Referee record (three passes, 2026-06-15/16)

**Self-pass** (the first external Opus subagent died on a session limit): found &
fixed the mirror-sign error $q_-'' = -2\eta(-b)$ (machine-checked: my sign err
$4\!\cdot\!10^{-4}$, the old $+$ version err $5.15$); confirmed Lemmas 1–3.

**External Opus pass #1** (independent re-derivation): confirmed Lemmas 1–3 and
found a **BLOCKER** — the absolute-count law $n_{\mathrm{signed}} = K+1$,
$n_{\mathrm{cone}} = \lceil(K+1)/2\rceil$ is numerically false. $K$ is $\approx$ constant across $\alpha$ while
atom counts grow $\propto 1/\alpha$ (the solver clusters $\sim m(\alpha)$ atoms per contact
region, $\sim 3\times$ the critical points). Fix: re-state as a **ratio** over continuum
(contact-region) counts; merged numerics give exactly $2.00$. Other fixes applied:
endpoint-zeroing is unconditional only for the **signed** model ($\tilde C > 0$ needed for
the cone); "cone $+\alpha$-count $= 0$" is *definitional*; per-branch ratios erratic.

**External Opus pass #2** (the dual-intersection material): confirmed Lemmas 1–3 +
the corrected ratio theorem, and **solved hypothesis (M)** by finding the right
mechanism — $\eta_{\mathrm{signed}} = \eta_{\mathrm{cone}}$ via projection onto nested polytopes (Lemma
3.5). Demolished two of my explanations with evidence: (a) "(M) holds because the
target is semiconcave / disjoint support" is **false** — (M) holds for *convex* and
oscillatory targets too (verified); (b) "factor 2 from reflection symmetry / no
bias" is **wrong** — the targets are one-sided yet $R_+ = R_-$ exactly; the real
reason is Lemma 2 alternation on the *shared* certificate. Both retracted above.
Also: $\mathcal{F}_{\mathrm{signed}} = \mathcal{A} \cap \mathcal{B}$ as a set identity is *trivial* (it's the norm
definition) — the content is the projection argument, not the set algebra; the
"R tracks $m$" claim is an **overclaim** (R tracks *resolvable* shocks and saturates
when knots are closer than the $0.08$ merge gap — at $m \ge 5$ the test knots collide,
so R$\approx$4–7 there, not $m$); $Q1/Q2/$onbnd are *one fact* ($\eta_s = \eta_c$) measured
thrice, and onbnd is definitional. All applied.

## Honest status — what is proved vs assumed (post pass #2)

- **Proved (rigorous, three referees):** Lemma 1 ($q_+'' = 2\eta$, $q_-'' = -2\eta(-b)$);
  Lemma 2 (bound-contact critical points alternate); Lemma 3 (KKT: signed at both
  bounds, cone only $-\alpha$); Lemma 3.5's projection structure **given** the one
  inequality below. Conditional on it: $\eta_s = \eta_c$, hence (M) is a corollary,
  $R_+ = R_-$ exactly, and **ratio $= 2$ exactly** (not "$\to 2$").
- **The single open analytic step (collapsed from two):** the cone optimal
  certificate satisfies $q_{\mathrm{cone}} \le +\mathrm{bound}$ everywhere (equivalently
  $\eta_{\mathrm{cone}} \in \mathcal{B}$; the cone certificate equioscillates, reaching $+\mathrm{bound}$
  at its maxima), where $\mathrm{bound} = -\min_b q_{\mathrm{cone}}$ is the level the cone's
  active atoms sit at. KKT gives the lower bound $q_{\mathrm{cone}} \ge -\mathrm{bound}$; this is
  the matching upper bound. **Strongly supported numerically (2026-06-16):** the
  convention-free real-overshoot $R = (\max q + \min q)/(-\min q)$ is $\le 10^{-10}$
  across 387 random + signed-jump + adversarial interior targets (and $\sim 10^{-13}$
  on the convex-heavy / $|x|$ cases that most stress it), i.e. no genuine interior
  overshoot; not yet proved analytically. This lone inequality *replaces* the
  earlier (E)+(M) pair and the mislabeled $|K_s - K_c|$ gap.
  `../../scripts/t5_overshoot_search.py`.
- **Measurement caveat (2026-06-16, three traps recorded so they don't recur).**
  The "overshoot" must be measured convention-free as $R$ above. Wrong ways that
  each produced a *spurious* verdict: (i) comparing $\max q$ to the penalty $\alpha$ —
  sklearn's $(1/2n)$ objective puts the KKT level at $n\alpha$, so this fabricated a
  $119\times$ "overshoot"; (ii) "$\max q - \max|q|$" is $\le 0$ by definition and can
  never detect overshoot (vacuous); (iii) thresholding $\|\eta_s - \eta_c\|$ at
  $10^{-4}$ flags coordinate-descent slack as a refutation. The trustworthy
  quantities are $R$ and $\|\eta_s - \eta_c\|$ *cross-checked against* $R$.
- **Independent confirmation (inline referee, 2026-06-16):** fresh seed, fine
  2001-point certificate (evaluated off the 301-point fitting grid to catch
  between-knot overshoot), tol $10^{-9}$, 20 interior stress targets ($\tilde C$ up to
  $+124$): worst interior $R = 1.35\times10^{-12}$ — no overshoot. The reduction
  was also re-derived from LASSO–projection duality and is rigorous in the
  explicit **two-hypothesis** form: $\eta_{\mathrm{signed}} = \eta_{\mathrm{cone}}$ needs both
  $q_{\mathrm{cone}} \le \mathrm{bound}$ **and** $\tilde C > 0$ (the latter pins $\langle x,\eta\rangle = 0$;
  the cone's $\tilde C \ge 0$ is one-sided, the signed head is free, so $D_{\mathrm{signed}}
  \subseteq \mathcal{A}$ requires the equality).
- **Caveat (referee):** on concave-bulk targets ($\tilde C$ pinned at $0$) the cone head
  cannot supply the bulk slope and $q_{\mathrm{cone}} \le +\alpha$ may fail — the mechanism, and
  the whole result, is for the $\tilde C > 0$ (semiconcave/head-interior) regime, which
  is the relevant one. Off it the dichotomy genuinely changes.
- **Not claimed:** any absolute atom-count law. Atom counts scale $\propto 1/\alpha$; only
  the *ratio* is invariant. The earlier $n = \lceil(K+1)/2\rceil$ claim is retracted.

## Relation to the rest of the ladder

This is the **selection** mechanism (Part 0, item 3) made precise in $d = 1$ at
the repo's power $p = 2$. It composes with:
- **Theorem A** ($p = 1$, saturated): there the head gives an *unbounded*
  capacity separation; here at $p = 2$ capacity is equal (emulation) and the
  surviving advantage is exactly this $\approx 2\times$ selection factor.
- **Theorem B** (B1–B4, Prop C): supplies the certificate calculus this theorem
  runs on, and identifies the discarded $+\alpha$ atoms as the degenerate
  cancellation family the cone evicts.
- **T5 (general)**: this is the $d = 1$, $p = 2$ instance of the open
  conjecture; the continuum cluster-dichotomy ($\ge 2$-D, and hypothesis (M) —
  the $-\alpha$-region/multiplicity match) remains the target, now with a proved
  $1$-D mechanistic backbone and the
  literature anchors (nonnegative spline recovery without separation:
  Slawski–Hein, de Castro–Gamboa, Schiebinger et al.; two-sided minimal
  separation: Candès–Fernandez-Granda).

## References

- Proposition C, Theorem B, gradient-training amendments: `theory-ladder.md`.
- Tibshirani (2013), *The lasso problem and uniqueness* — general-position /
  uniqueness frame.
- Numerics: `../../scripts/t5_certificate_dichotomy.py` (this result),
  `../../scripts/t5_penalized_path.py` (control debiased ratios).
