# The theory ladder — statements and intuition

Status: recentered 2026-06-11 (user decision, Track 1). **The thesis core is the
semiconcave structure *within* the ridgelet framework** — the repo's actual
`SemiconcaveModel` — not the min-architecture. Min-network results are retained
in Part II (moved, not deleted) as motivation & landscape. Detail docs:
`t1-1d-budget-lemma.md` (T1), `t2-separation-draft.md` (boundary theorem, T2),
`t4-minplus-capacity.md` (T2′, T4, Lemma P).

Throughout: $V$ $C$-semiconcave means $x \mapsto V(x) - \tfrac{C}{2}|x|^2$ concave. Ridge dictionary
$= \{\pm\sigma_k(w \cdot x - b)\}$. The repo's models: `SignedModel` $g(x) = \sum c_i\, \sigma(w_i \cdot x + b_i)^p$ with
free $c_i$; `SemiconcaveModel` $V = \tfrac{C}{2}\|x\|^2 - g$ with $c_i \ge 0$ (so $g$ convex, $V$
$C$-semiconcave). Cone-variation $\gamma^+(g)$ = minimal total mass of a *nonnegative*
ridge representation of $g$.

---

# Part 0 — the problem and the shape of the answer

**Problem.** Explain, with theorems, why the semiconcave-structured ridge network
fits HJB value functions with fewer atoms than the signed ridge network — the
empirical observation that started the program.

The framework splits "fewer neurons" into **three mechanisms plus one boundary**:

1. **Capacity (T3 → T3′).** *Proved in $d = 1$ (T1c). Resolved numerically in
   $d = 2$ (2026-06-11): the pure cone is NOT free — a smoothed triple junction
   sits 54%-of-its-Laplacian-mass outside it — but the **quadratic head buys
   completeness back** at a finite curvature price $\lambda$. See T3/T3′ below; the
   head is exactly the feature that completes the model's capacity.*
2. **Budget (T1a / T1-d′).** Semiconcavity is a one-sided second-order law:
   gradient jumps only point down, so their total mass is $\le C|\Omega| + 2L$ no matter
   how many shocks there are. *Proved in $d = 1$; in $d \ge 2$ the uniform version is
   provably FALSE and the correct form is scale-resolved:
   $\mathrm{mass}(\varepsilon) \lesssim (C+L)\,\varepsilon^{-(d-1)}$, with the blowup localized at the switching set
   (T1-d′ below).*
3. **Selection (T5).** The cone problem has a one-sided dual certificate
   (PDAP insertion test $p > \alpha$ instead of $|p| > \alpha$, cf. `src/PDAP/insertion.py`):
   no sign-paired atoms, no cancellation, generically smaller contact sets $\implies$
   sparser minimizers at matched accuracy. *Open; this is the mechanism that
   matches the measured sparsity.*

**Boundary (T2-B/C).** At curved or modulated switching sets, *no* ridge model —
signed or semiconcave-structured — survives with bounded weight mass.
*Proved for $d = 2$, referee-verified.*

**One sentence:** the semiconcave structure does not change the ridge
dictionary's rates — it buys a free half-space constraint (capacity), a
one-sided shock budget (mass), and a one-sided certificate (solver-level
sparsity), which together explain "fewer neurons" away from the switching set;
at the switching set the theory proves the whole ridgelet framework fails,
which is the honest boundary of the claim.

---

# Part I — core: semiconcave structure in the ridgelet framework

## T1 — the 1-D semiconcavity budget (PROVED)

**Theorem.** Let $V$ be $C$-semiconcave and $L$-Lipschitz on a bounded interval $\Omega$. Then
$V'' = C\,dx - \mu$ with $\mu \ge 0$ and $\mu(\Omega) \le C|\Omega| + 2L$, so $\mathrm{TV}(V'') \le 2C|\Omega| + 2L$ —
independent of the number of shocks. Moreover $V$ is *exactly* a cone-constrained
ReLU network plus a paraboloid (= `SemiconcaveModel`, $p = 1$), the representing
measure is unique (so the $c_i \ge 0$ constraint costs nothing: T3 holds in $d = 1$),
and $N$ atoms achieve sup-error $\lesssim (C|\Omega| + 2L)\,|\Omega|\, N^{-2}$ uniformly over the class.

**Intuition.** Semiconcavity is a one-sided second-order law: upward curvature is
capped at $C$ and gradient jumps can only go *down*. Since $V'$ starts and ends
within $[-L, L]$, total descent is paid for by total ascent plus the boundary
allowance — like a hiker whose total drop is fixed by total climb plus net
elevation change. So a value function may have arbitrarily many shocks, but their
*total strength* is budgeted by $C|\Omega| + 2L$, and approximation cost depends on the
budget, never the shock count. Because every kink bends the same way, the optimal
ReLU weights are automatically nonnegative — the repo's cone constraint is free.

## T2-B/C — the negative boundary theorem (PROVED d = 2; referee-verified + numerics)

**Theorem.** Let $v_2(x) = \min_{i \le 4} e^{-\frac12 |x \mp e_i|^2}$ — an HJB viscosity solution,
$C^*$-semiconcave, $C^* = 2e^{-3/2}$. Then $\|v_2\|_{\mathcal{R}} = \infty$, and every sequence of shallow
ridge networks converging to $v_2$ uniformly on compacta has total weight cost $\to \infty$.
This covers **both repo models**: `SignedModel` for every $\sigma_k$, and
`SemiconcaveModel` — the quadratic head changes the Laplacian by a constant
(no singular part) and the cone is a subclass of the signed measures.

**Intuition.** $v_2$'s creases lie on straight lines, but the gradient-jump
*density* along each is Gaussian-modulated, not constant. A ridge atom is a
constant-density kink along an *entire* hyperplane — perfect for straight uniform
creases, wrong for modulated ones. In Fourier space the modulated crease spreads
its energy over a whole fan of directions around the crease normal with slow
$r^{-2}$ decay, and each dyadic angular band of that fan demands a fixed quantum of
ridge mass — infinitely many bands, log-divergent total.

**Role in the program.** This fences the efficiency claim. The semiconcave
structure does *not* rescue the ridge dictionary at switching sets — nobody in
the ridgelet framework survives there. The thesis claim therefore lives away
from the switching set, exactly where mechanisms 1–3 operate — and where the
experiments to date actually measure (the pendulum data currently misses the
switching set, issue #18). It also certifies (via Part II) that the cost is the
dictionary's fault, not the function's: the same $v_d$ is trivial for min-atoms.

## T3 — cone-optimality of the signed problem (OPEN; d = 1 settled by T1)

**Conjecture.** If $g$ is convex and representable in the signed $\mathrm{ReLU}^k$ ridge
dictionary with finite total variation, the minimal-TV signed representing
measure is nonnegative; equivalently, min over signed measures = min over the
cone: $\gamma(g) = \gamma^+(g)$.

**Intuition.** Convexity should be visible in the optimal Radon-domain weights:
along every line the second derivative of $g$ is $\ge 0$, and each ridge atom
contributes a positive curvature bump along its own normal — so negative
coefficients mean deliberately carving concave dents that other atoms must then
cancel, pure waste under a TV objective. In 1-D this is immediate because the
representation is unique (the measure *is* $g''$). In $d \ge 2$ uniqueness fails (the
Radon transform has a nontrivial null space on measures — Ongie's even/odd
splitting), so the question is whether TV-minimization always *selects* the
nonnegative representative. If yes, the cone constraint costs nothing in
expressive power while halving the search space and strengthening the PDAP
dual certificate — the cleanest open pure-math question in the program.

**Sharpened to an exact criterion (2026-06-11).** By Ongie's uniqueness of the
even representing measure (their Lemma 1) and the evenness bound (Prop 7), the
minimal signed mass of a ridge-representable $g$ equals $\|c_g\|$, where $c_g$ is the
even ridgelet density of $g$, and **$g$ admits a nonnegative representation iff
$c_g \ge 0$**. Hence T3 $\iff$ "$g$ convex (ridge-representable) $\implies c_g \ge 0$" — concrete and
testable. And genuinely in doubt:

**Counterexample program (T3 may be FALSE in $d \ge 2$).** $c_g$ involves $d+1$
$b$-derivatives of slice profiles; for smooth bumps these oscillate (Hermite-type
lobes with negative regions of amplitude comparable to $\|D^2 \mathrm{bump}\|$), while a
quadratic's density is a finite positive constant per unit bias. Candidate:
$g = \tfrac{a}{2}|x|^2 + \varphi$ with $\varphi$ a smooth bump scaled so $g$ stays convex but the bump's
negative ridgelet lobe beats the quadratic's density. The scaling
$\varphi_\lambda = \lambda^2 \varphi(\cdot/\lambda)$ leaves both $\|D^2\varphi\|$ and the dip amplitude invariant, so the
comparison is a pure **shape constant computable by one FFT in $d = 2$**.
($d = 3$ heuristic: the radial even density is $\propto g'''(b)\, b + 3 g''(b)$; convexity
bounds $g'' \ge 0$ but not $g'''$, and rapidly decaying $g''$ flips the sign.)

**Stronger candidate — the smoothed triple junction (likely the truth).**
$g = \mathrm{LogSumExp}_\eta(\ell_1, \ell_2, \ell_3)$, three affine functions: convex, Lipschitz, smooth,
with three crease-rays *ending* at a junction. A ridge atom's crease runs along
its entire line; to terminate a crease at the junction, the representing density
must go **negative** on the line's continuation — nonnegative weights cannot do
it. $d = 1$ has no junctions, which is exactly why the miracle holds there.
**Confirmed numerically (below): T3-pure is false by exactly this mechanism.**

**Numerical verdict (2026-06-11; `scripts/t3_cone_certificate.py`).**
Dual-certificate LP on $\Omega = [-3,3]^2$, $61^2$ grid, 6 914 sampled lines, HiGHS, with an
in-cone sanity target (sum of softplus ridges) calibrating the noise floor:

| certificate class | sanity (in-cone) | smoothed junction ($\eta = 0.3$) |
|---|---|---|
| pure cone | 1.33 % of $\|\Delta g\|_1$ (noise) | **53.99 %** (40.6× noise; feasibility $5 \cdot 10^{-13}$) |
| mean-zero (= refutes model with free quadratic head) | $-0.00$ % | **$-0.00$ %** |

Reading (wording referee-corrected 2026-06-11). Row 1: $\exists \psi$ with all *sampled*
line-integrals $\le 0$ but $\langle \Delta g, \psi \rangle$ = 54 % of the total mass $\implies$ **T3-pure is false on
the sampled line family** (40× the calibrated noise floor). The passage to a
genuine "no cone sequence at any mass" theorem uses: the $\Delta g_n$ are *positive*
measures, uniformly locally bounded (convexity + uniform bound), converging
*vaguely* to $\Delta g$; the smoothed-junction target has $\Delta g$ **absolutely continuous**
(a load-bearing hypothesis — it fails for sharp targets!), so it charges no
cell boundary and **Portmanteau** passes the bang-bang pairing to the limit.
(The naive "$\mathcal{D}'$-convergence $\implies$ pairing passes" is FALSE in general — referee
counterexample: $\tfrac{n}{2}\mathbf{1}_{[0,1/n]} \rightharpoonup \delta_0$ against $\mathbf{1}_{[0,\infty)}$.) Two links remain to
harden for the all-lines theorem: mollify $\psi$ in $x$ and verify the constraint over
all $(\theta, b)$ with explicit moduli (controlled by $b$-sampling density), and rule
out mass escaping to $\partial\Omega$.
Row 2: refuting the model with a *free* head requires a mean-zero certificate —
the head adds a uniform background $\lambda \cdot \mathbf{1}$ to $\Delta g$, and every feasible $\psi$
automatically has $\int \psi \le 0$ by the **Cauchy–Crofton identity**
($\int_{\text{lines}} \int_{\ell \cap \Omega} \psi \, ds \, d\theta \, db = \pi \int_\Omega \psi \, dx$ — referee-verified incl. the $\ell \cap \Omega$
truncation, so boundary effects do not break the sign). The optimum is exactly
$0 \implies$ by Farkas duality — **exact in the discretized LP** — $\Delta g_{\mathrm{disc}} + \lambda \cdot \mathbf{1}$ is a
discrete cone element: **the quadratic head rescues capacity** (numerically).
Continuum version needs: line-family refinement, $\lambda$ **bounded uniformly in
mesh** (not yet certified), and quadrature consistency.

**T3′ (head-completeness — the reformulated capacity claim).** For every convex
Lipschitz $g$ on $\Omega$ (junctions allowed) there is $\lambda \ge 0$ with $\Delta g + \lambda \cdot \mathbf{1}_\Omega$ in the
closed cone of nonnegative line superpositions; equivalently `SemiconcaveModel`
represents the semiconcave target at the price of an enlarged curvature
constant $C' = C + \lambda/d$. The minimal such $\lambda =: \pi(g)$, the **curvature price** of
the crease-ending structure. The quadratic head is exactly the feature that
completes the cone's capacity — the LP explains the architecture. Open items:
(i) prove T3′ constructively, including $\lambda$ uniform in mesh; (ii) bound $\pi$ by
junction geometry (jump sizes and ray angles); (iii) decide whether straight
kinks with *non-constant* density ($v_2$'s diagonals) have $\pi = \infty$ even in closure —
conjectured yes, which would make the boundary theorem strict even for the
head-completed model. **Note (referee):** the LP verdicts above concern the
*smoothed* ($\eta = 0.3$, a.c.-Laplacian) junction only; for sharp targets the
Portmanteau step fails as-is ($\Delta g$ charges the kink set), so (iii) and the sharp
limit $\eta \to 0$ are genuinely open and are NOT covered by the row-1 headline.

**$v_2$-kink run (2026-06-11; `scripts/t3_v2_kink_certificate.py`).**
Targets = convex part of $v_2$: sharp (singular $J$-deposits), const-$J$ control (same
kink mass, density averaged), mollified $\delta \in \{0.2, 0.1\}$. Pure cone: sharp 7.48 %,
control 6.81 % (floor 1.33 %) — even the constant-$J$ + varying-a.c. combination
is outside the pure cone; the $J$-variation-specific part is only ~0.7 %.
Mean-zero: **0 for all four targets**, including sharp. Two lessons, both
recorded as method-facts:
1. **Fixed-mesh mean-zero LPs are structurally blind to the singular
   obstruction.** On a grid the kink is numbers in a one-cell tube, and tilted
   lines emulate variable density inside it; the structure theorem's
   constant-density constraint emerges only as $h \to 0$. So "sharp mean-zero = 0"
   is NOT evidence against $\pi = \infty$; item (iii) requires **mesh-scaling of the
   minimal $\lambda$** (or pure analysis), not a fixed-mesh certificate.
2. **Degenerate-dual caveat:** at optimum 0 (attained at $\psi = 0$) the dual set is
   large; the reported mean-zero-row marginals ($\approx 4\,400$–$4\,700$) are arbitrary
   dual vertices, NOT curvature prices. The measurement instrument for $\pi$ is the
   *primal* two-phase min-$\lambda$ program (minimize fit residual, then minimize $\lambda$ at
   near-optimal residual), run at several mesh sizes: flat $\lambda(h) \implies$ finite $\pi$;
   growing $\lambda(h) \implies \pi = \infty$ revealing itself as mesh-divergence.

**Curvature-price study (2026-06-11;
`scripts/t3_curvature_price.py`).** Two-phase min-$\lambda$ LPs (all
residual floors 0.00% — exact fits):

| target | $\lambda_{\min} = \pi$ (discrete) |
|---|---|
| junction ($\eta = 0.3$), $n=61$ | **2.373** — finite, modest: $C' \approx C + 1.19$ |
| $v_2$ mollified $\delta = 0.4$ / $0.3$ | **0 / 0** (exactly — in the *pure* cone, no head needed) |
| $v_2$ mollified $\delta = 0.2$ / $0.1$ | 0.291 / 2.113 (local exponent $\alpha \approx 2.9$) |
| $v_2$ sharp, $n=45 \to n=61$ | 19.47 → **25.12** (grows with resolution) |

Readings. (1) **The junction price is real but affordable** — T3′ with its first
measured constant. (2) **Threshold behavior**: wide-mollified kinks cost
*nothing* ($\lambda = 0$ exactly, $\delta \ge 0.3$); the head becomes load-bearing only below a
critical sharpness $\delta^* \approx 0.25$, after which the price explodes ($\lambda \sim \delta^{-3}$ on the
resolvable range). (3) **Sharp kinks**: $\lambda$ grows monotonically with mesh
resolution, consistent with $\pi = \infty$ — exactly what the proved structure theorem
predicts (non-constant singular density unreachable by any finite cone measure).
Caveats: $\delta = 0.1 \approx h$ partially under-resolved; the mesh pair ($n=45$, $61$) is a
direction signal, not a measured divergence exponent; all values are
discrete-cone measurements at the stated line family. Compute lesson (recorded
in the script): per-iteration LP cost is forecastable (dense normal equations,
cubic in cell count), iteration count on near-degenerate instances is not —
design cheap (coarsen/threshold/time-box), don't forecast; the $n=45$ coarsening
check delivered in 87 s what $n=76$/$n=91$ refinement could not in hours.

**Either outcome is a thesis result.** T3 true $\implies$ capacity is free, as hoped.
T3 false $\implies$ the cone class is *characterized* ($c \ge 0$ — a clean description of
what `SemiconcaveModel` can express), the cone-projection price becomes a new
measurable quantity, and the efficiency story rests on T1-d′ (budget) + T5
(selection) — where the empirical evidence pointed anyway.

## Gradient-training amendments (2026-06-12 — the data contain $\nabla V$)

The training is gradient-augmented ($V$ *and* $\nabla V$ in the loss; Gradient-Augmented
Regression reference; `src/data.py`, `src/eval.py`). Consequences:

1. **Emulation verdict unchanged**: $(w \cdot x)_+^2 + (-w \cdot x)_+^2 = (w \cdot x)^2$ is a function
   identity — capacity separation stays impossible at $p = 2$ in any Sobolev norm.
2. **Lemma G (gradient-data mass floor; provable in 5 lines).** For $C^{1,1}$
   activations, $\mathrm{Lip}(\nabla \mathrm{net}) \le \|\sigma''\|_\infty M + C$; a gradient jump $J$ along a kink of
   length $\ell$ forces a transition layer of width $\ge J/(\|\sigma''\| M + C)$, hence
   gradient-$L^1$ accuracy $\varepsilon$ requires **$M \gtrsim J^2 \ell / (\|\sigma''\| \varepsilon)$** — both models, any
   signs. Value training has no such floor: **gradient data is what makes the
   switching set expensive at finite accuracy**, and it upgrades the T2-A vs
   ridge separation to a finite-$\varepsilon$ mass law (min-net: exact gradients off $\Omega_\delta$
   with $2d$ atoms; ridge: $M \sim 1/\varepsilon$).
3. **Experimental regime explained**: kinks are $O(1)$-visible in gradient data
   vs $O(\delta^2)$ in values; the pendulum data currently missing the switching set
   (issue #18) is exactly why both models look comfortable. After the fix,
   expect Lemma-G mass inflation in both models near the switching set.
4. **The semiconcave model under gradient training = monotone vector-field
   regression**: $\nabla V = Cx - \nabla g$, $\nabla g$ monotone (gradient of convex). The
   one-sidedness moves into the data-fidelity term — the right statistical
   frame for T5.
5. **T5 in $d = 1$ becomes classical**: cone-constrained gradient-LS = isotonic
   regression (unique, PAV structure, atoms = active blocks) vs two-sided
   fused-lasso-type signed fit (non-unique, cancellation-prone). The most
   realistic path to a rigorous T5 instance. The PDAP certificate must be
   re-derived in the gradient-augmented pairing (atoms enter via $w\, \sigma'(w \cdot x - b)$:
   angular sensitivity).

## Theorem A — d = 1 head separation on saturated targets (PROVED modulo referee; 2026-06-12)

**Theorem (referee-verified 2026-06-12; all constants recomputed exactly).**
Let $V'(x) = V'(a) + C(x-a) - \sum_{j=1}^m s_j\, \mathbf{1}_{x > t_j}$ on $\Omega = [a,b]$
($C$-semiconcave with **saturated Hessian** $V'' = C$ off $m$ shocks — the LQ-like
value-function class), gradient training. Then:
(i) the cone + quadratic-head model ($p = 1$) represents $V'$ **exactly with $m$
atoms** (plus the parameter $C$ and a free affine term, slope $p_1 = V'(a) - Ca$;
atoms: $\varepsilon_i = +1$, $b_i = t_j$, $c_i = s_j \ge 0$ — feasible exactly because semiconcavity
makes all gradient jumps downward);
(ii) any headless piecewise-constant-gradient model (signed ReLU, $p = 1$) with $n$
atoms has gradient-$L^2(\Omega)$ error $\ge C\, \ell_*^{3/2} / (\sqrt{12}\,(n+1))$, $\ell_* = |\Omega|/(m+1)$.
*Proof:* the $m$ shocks leave a shock-free interval $I^*$ with $|I^*| \ge \ell_*$
(pigeonhole). The model's $n$ jumps cut $I^*$ into $\le n+1$ cells whose lengths sum to
$|I^*|$; on each cell (inside $I^*$, so the target is pure slope $C$) the best constant
leaves squared error $C^2 \ell^3 / 12$; the power mean $\sum \ell_i^3 \ge |I^*|^3/(n+1)^2$ finishes. $\blacksquare$
The bound is **tight**: an equispaced staircase achieves error
$C\, \ell_*^{3/2}/(\sqrt{12}\, n)$, so the signed model's rate is exactly $\Theta(1/\varepsilon)$. Hence
**$n_{\mathrm{signed}}(\varepsilon) \asymp C\, \ell_*^{3/2}/\varepsilon$ while $n_{\mathrm{cone}} = m$**: unbounded, two-sided
separation at $p = 1$, on exactly the Hessian-saturated class — the class value
functions inhabit (Riccati saturation; the GMQ-attenuation class of T4(d)).
(iii) At $p = 2$ the separation collapses to **2 atoms in $d = 1$ ($2d$ in dimension
$d$)** via head emulation $(x)_+^2 + (-x)_+^2 = x^2$: at the repo's actual power,
capacity is equal and any observed sparsity gap must be **selection** (below).

**Numerical evidence (2026-06-12;
`scripts/t5_d1_gradient_sparsity.py`; exact solvers — fused lasso
via dual box-LS, cone via PAVA + endpoint shift $\alpha$ + scalar-$C$ search; an
$\alpha/2$-vs-$\alpha$ scaling inconsistency found by the referee is fixed, headline numbers
unchanged).** Staircase target (saturated, 5 shocks): signed needs 123 / 56
atoms at 2% / 5% accuracy vs cone 5 / 5 (noiseless); under noise $\sigma = 0.15$ at
5%: 28 vs 12. Smooth *non-saturated* target: no meaningful separation
(2%: 109 vs 132; 5%: 87 vs 87; noisy slices 54 vs 54, 32 vs 33) — the head
advantage is a saturation phenomenon, as the theorem says (the small cone edge
at 2% matches the smooth target's partial edge-saturation). The residual cone
advantage under noise at matched accuracy — **$\approx 1.2$–$5\times$ depending on the accuracy
slice (2.3–3× at the 5% slices)** — is the **selection factor**: T5's empirical
signature, isolated from the head effect.

## The p = 2 algorithmic experiments (2026-06-12) — where the sparsity gap actually lives

Two $d = 1$ experiments on the mollified saturated target (3 shocks, $\delta = 0.04$,
$p = 2$ dictionary, gradient training), probing how the capacity parity of
Theorem A(iii) interacts with the *algorithm*:

**1. Greedy discovery (`scripts/t5_greedy_emulation.py`, OMP-style: insertion by max
certificate + exact refit — PDAP insertion logic without the path).** My
"emulation pair is algorithmically hidden" hypothesis is **REFUTED**: on a
bounded domain the dictionary contains single-atom head emulators — a distant
atom $(x - b)_+^2$ with $b$ outside $\Omega$ is a full parabola on $\Omega$ (its *gradient*
feature, the object actually selected, is the slope-2 ramp carrying the bulk
trend) — and signed greedy selects one **first**. With
parameter-honest counting (cone's free head + affine = +2), greedy shows
**no meaningful gap**: cone 3/8/10/15 atoms at 10/5/2/1% vs signed 6/11/13/15
($\sigma = 0$). Capacity parity is algorithmically *reachable* — by greedy-with-
exact-refit. (Artifact note: the cone run's late path (>15 atoms, $\sigma = 0$) went
numerically unstable from collinear refits; thresholds were reached before.)

**2. Penalized path (`scripts/t5_penalized_path.py`, exact LARS lasso paths — the
PDAP-faithful algorithm class; "debiased" view = LS refit on support $\approx$ SSN
polish). ⚠ Superseded in part: see Proposition C + control in the Theorem B
section — the stall is a LARS artifact; the robust gap is $\approx 2\times$.** As
originally observed: the **signed path STALLS**
— LARS hits degenerate active sets (Cholesky pivots at machine eps; the
sign-cancellation directions of the two-sided problem) and early-stops, never
reaching 2% accuracy at any sparsity; the **cone path sails** to 2%/1% with
10–29 atoms (debiased, $\sigma = 0$: signed 5/8/--/-- vs cone 4/7/10/29; $\sigma = 0.05$:
signed 5/9/--/-- vs cone 4/7/15/--; cone counts exclude its 2 free head
params — cf. experiment 1's param-honest adjustment — and the
Frisch–Waugh formulation leaves $\tilde C \ge 0$ unenforced, violated only at 14/297
off-threshold path points, $\tilde C \approx +10$ at the reported thresholds).

**Combined verdict for the thesis claim at $p = 2$.** Three candidate
explanations for the observed semiconcave-model sparsity, now adjudicated:
capacity — dead (Theorem A(iii), emulation); greedy non-discovery — dead
(experiment 1); **regularized-path degeneracy — alive and demonstrated**
(experiment 2). The mechanism is consistent with T5's: the two-sided problem
has cancellation directions that render active sets degenerate and stall
path-following/Newton methods, while the cone+head structure removes them
(Theorem B; the stalling active sets are not yet shown to be exactly the
B1 family — see Theorem B's remaining items).
Caveat for honesty: experiment 2 shows the failure of LARS-type path tracking,
not of the abstract signed lasso minimizer; but PDAP/SSN solves the same
degenerate systems, so the algorithmic reading is the relevant one for the
repo. T5's theorem target is now sharp and literature-anchored (lasso
uniqueness/general-position theory, Tibshirani 2013): **on saturated
semiconcave targets, the signed $\ell_1$ path generically encounters non-unique /
degenerate active sets, the nonneg path does not.**

## Theorem B — the path-degeneracy dichotomy (d = 1, p = 2; mechanism lemmas PROVED, 2026-06-12)

The analytic explanation of experiment 2's stall. Both lemmas verified to
machine precision (inline check, 2026-06-12).

**Lemma B1 (signed degeneracy — explicit null vectors; referee-corrected
2026-06-12).** On bounded $\Omega$, every atom with knee outside the data hull — of
*either* sign $\varepsilon$ — is a full parabola, so out-of-hull **value** atoms span only
$\mathrm{span}\{x^2, x, 1\}|_\Omega$: degeneracy at $\ge 4$ atoms, null conditions
$\sum v = \sum vb = \sum vb^2 = 0$ (Vandermonde, alternating signs). In the
**gradient-augmented fit actually used** the atom features are
$\varphi = 2\varepsilon(\varepsilon x - b)$, which span only **$\mathrm{span}\{x, 1\}$** (all out-of-hull gradient
features have slope $+2$ regardless of $\varepsilon$): degeneracy onsets at **three**
atoms, null conditions $\sum v = \sum vb = 0$. In both pictures $\sum v_i = 0$, so any signed
$\ell_1$ solution whose active set contains enough out-of-hull atoms with *equal*
coefficient signs lies on a **fit-preserving, penalty-preserving continuum**
($\ell_1$ change $= t \cdot s \cdot \sum v = 0$ while signs persist) — non-unique minimizers, and the
singular active-set systems (machine-eps Cholesky pivots) that stall LARS and
PDAP/SSN. Verified: $\|\sum v_i \varphi_i\|_\infty \approx 7 \cdot 10^{-16}$, $\ell_1$ drift 0; gradient-feature rank of
out-of-hull family = 2 (mixed $\varepsilon$ included).

**Lemma B2 (cone exclusion — conditional; referee-corrected 2026-06-12).** In
the cone + head model, every out-of-hull atom's gradient feature
$2\varepsilon(\varepsilon x - b) = 2x - 2\varepsilon b$ lies **exactly** in the free head+affine span $\{x, 1\}$
(projection residual $\approx 2 \cdot 10^{-15}$). Removing an atom of mass $c > 0$ and
compensating with the head requires $\tilde C \to \tilde C - 2c \ge 0$. **Hence: whenever the
target's bulk curvature is representable with $\tilde C \ge 0$ — in particular on the
$C$-semiconcave saturated class ($C_{\mathrm{true}} \ge 0$), where the optimum has
$\tilde C = C_{\mathrm{true}} + 2\sum c_i$ so the transfer is always feasible — the transfer preserves
the fit and strictly reduces $\sum c$, and no optimal cone solution activates an
out-of-hull atom.** The condition is necessary: for a strictly concave bulk
($V' = -Kx$), $\tilde C$ pins at 0 and the optimal cone solution is *forced* to carry
out-of-hull mass $K/2$ (referee counterexample) — outside the semiconcave class.
On the relevant class the quadratic head does not merely emulate capacity
(Theorem A(iii)) — it **structurally evicts the degenerate atom family from
the penalized problem**. The signed model, having no free head, must **spend
atoms** on its bulk-affine content — one globally-affine out-of-hull ramp or
several near-boundary in-hull ramps — content the cone gets free from the
head (experiment 2: its first selection is the globally-affine out-of-hull
atom, $b = -3.00$).

**Lemma B3 (certificate affinity; PROVED + diagnostic-confirmed 2026-06-12).**
In gradient space the lasso certificate on the out-of-hull family
$\{\varphi_t = 2x - 2t\}$ is affine in $t$: $\langle \varphi_t, \hat\eta \rangle = A - t\tilde B$, $A = 2\langle x, \hat\eta \rangle$, $\tilde B = 2\langle 1, \hat\eta \rangle$.
Dichotomy: (a) if a signed solution has **two** same-sign active out-of-hull
atoms, then $\tilde B = 0$, $|A| = \alpha$, and the **entire out-of-hull continuum** ties in
the equicorrelation set — solution polytope of dimension (#family $-$ 2);
(b) if $\tilde B \ne 0$, an affine function hits $\pm\alpha$ at most once each $\implies$ **at most two**
out-of-hull actives (opposite signs). Diagnostic at experiment 2's stall:
$\tilde B = 4.8 \cdot 10^{-2} \ne 0$ and exactly **1** out-of-hull active — case (b), as predicted.

**Lemma B4 (KKT eviction — the clean form of B2; PROVED).** The cone+head
KKT conditions include $\hat\eta \perp \mathrm{span}\{x, 1\}$ whenever the head is interior
($\tilde C > 0$; the free affine always gives $\langle 1, \hat\eta \rangle = 0$). Since every out-of-hull
feature lies in $\mathrm{span}\{x, 1\}$, its certificate is $\langle -\varphi_t, \hat\eta \rangle = 0 < \alpha$: **strictly
inactive, unconditionally, whenever $\tilde C > 0$ at the optimum** — which holds on
the $C$-semiconcave saturated class ($\tilde C = C_{\mathrm{true}} + 2\sum c > 0$). At the boundary
$\tilde C = 0$ (concave bulk, the referee's counterexample), $\langle x, \hat\eta \rangle = -\alpha/2 < 0$ and the
whole family ties at $\alpha$ — the cone inherits the degeneracy exactly there, and
only there. B2's condition is thus *equivalent* to head-interiority.

**Stall diagnosis (2026-06-12) — the out-of-hull forcing hypothesis is
REFUTED; the true mechanism identified.** At experiment 2's stall ($\alpha \approx 7.5 \cdot 10^{-4}$,
22 actives): one out-of-hull atom only (B3(b)); the degeneracy lives in
**sign-alternating and same-sign clusters of adjacent in-hull knees at the
mollified shocks** — e.g. $(+1, b = -0.34/-0.33/-0.32)$ with signs $+/-/+$ (a
discrete second-difference spike polishing the shock profile) and an 8-run of
same-sign adjacent knees at $b \approx 0.49$–$0.57$. Adjacent-knee ramps differ by
tiny-norm tent functions, so these clusters are near-null directions with
exploding coefficients — the machine-eps pivots. The endpoint even violates
equicorrelation (one atom at 0.230 vs bound 0.179): LARS returned a
non-KKT point — the solver genuinely broke. **The cone cannot form either
cluster type**: sign alternation is infeasible, and same-sign mass-splitting
is penalty-flat only along exact ties, which the one-sided certificate
generically prevents.

**Proposition C (certificate calculus & active separation; PROVED + verified
2026-06-12).** For the $p = 2$ gradient dictionary, the signed certificate
$q(b) = \langle \varphi_{+,b}, \eta \rangle$ satisfies the identity **$q''(b) = 2\eta(b)$** — the
certificate's curvature in knee-position *is* the residual. Consequence
(separation bound): consecutive opposite-bound touches at distance $\Delta$ require
$\sup |q''| \gtrsim 8 \cdot (2\bar\alpha)/\Delta^2$, i.e. $\|\eta\|_\infty \gtrsim 8\bar\alpha/\Delta^2$. At experiment 2's stall the observed
$+/-/+$ cluster ($\Delta = 0.02$, $\bar\alpha = 0.23$, $\|\eta\|_\infty = 5.2$) would need $\|\eta\|_\infty \approx 4600$ —
**infeasible by $\sim 9 \times 10^2$: the stall endpoint is provably not a lasso solution.**
Exact signed minimizers have separated actives; the stall is a *solver*
breakdown (consistent with the endpoint's equicorrelation violation).

**Control experiment (2026-06-12) — the gap is real but $\approx 2\times$, not $\infty$.**
Re-solving the signed problem with coordinate descent (robust, no path
tracking; sklearn Lasso, tol $10^{-12}$) shows **no stall**: all accuracies
reachable. Fair debiased comparison (LS refit on support, both models):

| accuracy | cone (debiased) | signed CD (debiased) | ratio |
|---|---|---|---|
| 10 % | 4 | 7 | 1.8× |
| 5 % | 7 | 18 | 2.6× |
| 2 % | 10 | 18 | 1.8× |
| 1 % | 29 | 60 | 2.1× |

**Revised verdict (supersedes the stall narrative above):** at $p = 2$, $d = 1$,
the semiconcave model's advantage decomposes into (i) a **robust $\approx 2\times$
selection factor** at matched accuracy, solver-independent, consistent with
the $p = 1$ exact-solver factor (1.2–5×); and (ii) **solver fragility of the
signed problem for path-following methods** — LARS provably returns non-KKT
points (Proposition C), and PDAP/SSN belongs to the same algorithm class —
which in practice inflates the gap further. The "signed path never reaches
2%" phrasing in the experiments section above describes the LARS artifact,
not the signed problem itself.

**Remaining for the full dichotomy theorem (re-aimed by the diagnosis):**
prove the cluster dichotomy on the continuum dictionary — the one-sided
(cone) certificate $p(b) \le \alpha$ touches tangentially at isolated knees
(generically separated actives), while the two-sided $|p(b)| \le \alpha$ admits
sign-alternating touch clusters near shocks. Anchor literature: nonnegative
sparse recovery without separation (Slawski–Hein; de Castro–Gamboa;
Schiebinger et al.) vs two-sided minimal-separation theory (Candès–
Fernandez-Granda). Plus the spline-independence argument for in-hull
uniqueness. Tibshirani (2013) remains the uniqueness frame.

## T5 — PDAP sparsity under the cone (d = 1 PROVED; general OPEN)

**d = 1, p = 2 — MECHANISM PROVED, ratio modulo one hypothesis:** the certificate
dichotomy, `t5-d1-certificate-dichotomy.md` (two referee passes, 2026-06-15).
The certificate is a double antiderivative of the residual ($q_+'' = 2\eta$,
$q_-'' = -2\eta(-b)$, Prop C), so its bound-contact critical points alternate; the
**signed** model activates at *both* bounds, the **cone** only at $-\alpha$ (KKT,
Lemma 3). Hence at the **continuum (contact-region) level** $N_{\mathrm{signed}} = R_+ + R_-$
vs $N_{\mathrm{cone}} = R_-$, and with $R_+ \approx R_-$ (alternation) the ratio $\to 2$. ⚠ The
original absolute-count law $n_{\mathrm{cone}} = \lceil(K+1)/2\rceil$ was found **false** by the
external referee — atoms *cluster* ($\propto 1/\alpha$ per shock), outnumbering critical
points $\sim 3\times$; the clustering multiplicity $m(\alpha)$ cancels only in the *ratio*.
**Merged numerics: ratio exactly $2.00$** across two orders of $\alpha$ and into noise
(raw $1.67$–$1.81$ was the clustering artifact). **Hypothesis (M) SOLVED — the
projection mechanism (Lemma 3.5, external referee #2, 2026-06-16):** in residual
variables $\mathcal{F}_{\mathrm{signed}} = \mathcal{A} \cap \mathcal{B}$ ($\mathcal{A} = \{q \ge -\alpha\}$ cone,
$\mathcal{B} = \{q \le +\alpha\}$), so $\eta_{\mathrm{cone}} = \mathrm{Proj}_{\mathcal{A}}(y)$,
$\eta_{\mathrm{signed}} = \mathrm{Proj}_{\mathcal{A} \cap \mathcal{B}}(y)$ with $\mathcal{A} \cap \mathcal{B} \subseteq \mathcal{A}$. **If** the cone
certificate satisfies $q_{\mathrm{cone}} \le +\alpha$ everywhere (the single open inequality;
verified to machine precision), then $\eta_{\mathrm{cone}} \in \mathcal{B}$, so by projection
uniqueness **$\eta_{\mathrm{signed}} = \eta_{\mathrm{cone}}$** — one shared certificate, whence (M) is a
corollary and $R_+ = R_-$ exactly (Lemma 2 on that certificate), giving ratio $= 2$
*exactly*. **Target-independent** (verified on convex $\tfrac12 x^2$, oscillatory,
$|x|$) — so the earlier "semiconcave / disjoint-support / reflection-symmetry"
explanations are **retracted** (M holds for convex targets too). Open analytic gap
collapses to the lone inequality $q_{\mathrm{cone}} \le +\alpha$ (cone certificate equioscillates),
for the head-interior $\tilde C > 0$ regime. `scripts/t5_projection_mechanism.py`,
`scripts/t5_dual_intersection.py`.

**Conjecture (general $d$).** For the cone-constrained PDAP problem (minimize
data-misfit + TV-regularizer over *nonnegative* measures on the dictionary),
minimizers are atomic with atom count governed by the contact set of the
one-sided dual certificate $p \le \alpha$; generically this contact set is smaller than
the two-sided contact set $\{|p| = \alpha\}$ of the signed problem at matched accuracy —
explaining the observed sparsity gap between `SemiconcaveModel` and `SignedModel`.

**Intuition.** In PDAP, atoms live where the dual certificate touches its bound.
A signed problem must guard the certificate from above *and* below, and its
minimizers can contain pairs of opposite-sign atoms partially cancelling each
other — capacity spent on bookkeeping rather than approximation. The cone problem
guards one side only, and cancellation is infeasible by constraint. Combined with
T1/T3 (the cone loses no expressive power on semiconcave targets), the constraint
is a free lunch: same approximation class, structurally sparser solutions. This
is the layer that converts variation norms into the literal atom counts the
repo's experiments measure.

## T1-d — the $d \ge 2$ budget: uniform version FALSE; scale-resolved T1-d′ (REFORMULATED 2026-06-11)

**The naive conjecture is false.** There is no $K(d, \Omega)$ with
$\gamma^+(\tfrac{C}{2}|x|^2 - V) \le K (C + L)$ over all $C$-semiconcave, $L$-Lipschitz $V \in C^2(\bar\Omega)$,
$d \ge 2$. Proof mechanism: mollification preserves both constants
($V * \rho_\delta - \tfrac{C}{2}|x|^2 = \text{concave} * \rho_\delta + \text{const}$), so $V_\delta = v_2 * \rho_\delta \to v_2$ is a sequence of
$C^\infty$, $C^*$-semiconcave, $L$-Lipschitz functions; a uniform budget would give
bounded-mass representations along the sequence, and bounded mass passes to the
uniform limit — contradicting the boundary theorem T2-B/C on $v_2$. Status split:
with $\Omega$-uniform $K$ (growing balls) this contradicts the *proved* T2-B via Ongie's
$\bar{\mathcal{R}}$ definition (rigorous); for fixed $\Omega$ it rests on the weak-* lemma — **now
proved** (see the lemma section; referee pass 2026-06-11, all mollification
claims confirmed). Either way: **the $d = 1$ miracle — budget independent of
smoothness — dies in $d \ge 2$, and its failure mode IS the boundary theorem**
(resolving a mollified kink at scale $\delta$ costs mass diverging as $\delta \downarrow 0$).

**T1-d′ (scale-resolved budget — the correct positive statement; proof
program).** For every $C$-semiconcave, $L$-Lipschitz $V$ on $\Omega$ — kinks allowed — and
every $\varepsilon > 0$, there is a quadratic-head + nonneg-ridge representation with
sup-error $\le \varepsilon$ and mass $\lesssim K(d, \Omega)(C + L)\,\varepsilon^{-(d-1)}$. Mechanism: mollify at
$\delta \sim \varepsilon/L$; every Radon slice of $g = \tfrac{C}{2}|x|^2 - V$ obeys the 1-D budget at second
order ($\partial_b^2 \mathcal{R}\{g\}(w, \cdot) = \mathcal{R}\{\langle w, D^2 g\, w \rangle\} \ge 0$, slice-budgeted — one-sidedness
*does* survive slicing); each of the ridgelet filter's remaining $d-1$
$b$-derivatives costs $\delta^{-1}$ on a $\delta$-mollified profile. The blowup exponent is the
filter order. Gap to close: positivity of the *filtered* density is not
automatic — that is exactly T3 — so the bound as sketched holds for the signed
mass, and the cone version needs T3 or an explicit nonneg construction.

**Localization (companion statement).** On any $\Omega' \Subset \Omega$ with
$\mathrm{dist}(\Omega', \Sigma(V)) > 0$ — away from the switching set — the mass for accuracy $\varepsilon$ on $\Omega'$
is bounded by $K (C + L + \|D^2 V\|_{C^{0,1}(\Omega')})$ with **no $\varepsilon$-blowup**: the
divergence localizes at $\Sigma$. This is the quantitative form of "the thesis claim
lives away from the switching set."

## The weak-* lemma — PROVED (referee pass 2026-06-11; decomposition paragraph supplied per referee recipe)

**Lemma.** Let $\Omega \subset \mathbb{R}^d$ be bounded and full-dimensional,
$g_n = \int \sigma(w \cdot x - b)\, d\mu_n + \mathrm{affine}_n$ with ReLU $\sigma$ and $\|\mu_n\| \le M$ (signed or $\mu_n \ge 0$),
and $g_n \to v$ uniformly on $\bar\Omega$. Then $v = \int \sigma\, d\mu + \mathrm{affine}$ with $\|\mu\| \le M$ and
$\Delta v|_\Omega = \mathcal{R}^*\mu|_\Omega$, where $\mathcal{R}^*\mu(E) = \int H^{d-1}(\ell_{w,b} \cap E)\, d\mu$.

**Proof.** (1) *Decomposition* (the previously missing step): split
$\mu_n = \mu_n^K$ ($|b| \le R_\Omega$) + dead atoms ($b > R_\Omega$: $\equiv 0$ on $\Omega$, drop) + far atoms
($b < -R_\Omega$: affine on $\Omega$, fold into $\mathrm{affine}_n$). The kept part
$K_n = \int \sigma\, d\mu_n^K$ satisfies $|K_n| \le 2 R_\Omega M$ and $\mathrm{Lip}(K_n) \le M$ on $\bar\Omega$ $\implies$
Arzelà–Ascoli: $K_n \to K^*$ uniformly (subsequence). The remainder
$A_n = g_n - K_n$ is affine and converges uniformly (both terms do); uniform
convergence of affine functions **on a full-dimensional bounded set** pins the
coefficients (finite-dimensional), so $A_n \to A^*$ affine. (2) *Compactness*: $\mu_n^K$
lives on the compact set $S^{d-1} \times [-R_\Omega, R_\Omega]$ with mass $\le M$ $\implies$ $\mu_n^K \rightharpoonup^* \mu$,
$\|\mu\| \le M$ by lower semicontinuity. (3) *Identification*: for $\varphi \in C_c^\infty(\Omega)$,
$\langle \Delta g_n, \varphi \rangle = \langle g_n, \Delta\varphi \rangle \to \langle v, \Delta\varphi \rangle$, while $\langle \Delta K_n, \varphi \rangle = \langle \mu_n^K, \mathcal{R}\varphi \rangle \to \langle \mu, \mathcal{R}\varphi \rangle$ since
the X-ray transform $\mathcal{R}\varphi$ is continuous on the compact parameter set (dominated
convergence; tangency harmless as $\varphi$ vanishes smoothly). Hence $\Delta v = \mathcal{R}^*\mu$ on $\Omega$. $\blacksquare$

**Structure theorem (referee-confirmed solid).** For *fixed finite* $\mu$: the
$H^{d-1}$-singular part of $\mathcal{R}^*\mu$ on a hyperplane $H$ is exactly
$\mu(\{\mathrm{param}(H)\}) \cdot H^{d-1}|_H$ — constant density, whole-hyperplane support — because
every other line/hyperplane meets $H$ in an $H^{d-1}$-null set. Consequently $\mathcal{R}^*\mu$ can
match neither a non-constant jump density ($v_2$'s diagonals, $d = 2$: **boundary
theorem re-proved, qualitatively, without Fourier analysis**) nor a singular
part supported on a proper cone face (the $d \ge 3$ mechanism — *conditional* on
computing the per-face densities $J_F$, t2 gaps 1–2; "all $d$" not yet earned).

**$C^{1,1}$ activations (the repo's $\sigma^2$, gelu²) — the most bulletproof claim in
the program (referee).** Each atom's Laplacian is a.c. with density $\le \|\sigma''\|_\infty$,
so $|\Delta(\mathrm{net})| \le \|\sigma''\|_\infty \|\mu\|$ pointwise: uniform limits of bounded-mass nets are
$C^{1,\alpha}$ — **no function with a gradient jump is approximable at all** with
bounded weight mass, head or no head.

**Downstream (referee verdicts).** Fixed-$\Omega$ T1-d falsification: valid, exactly as
strong as this lemma (all mollification claims confirmed). The growing-ball
version is independently rigorous via the Fourier-proved T2-B.

---

# Part II — motivation & landscape (min-architecture results)

These are proved results about the **KVV/DDM min-architecture**, *not* about the
current `SemiconcaveModel`. Moved here 2026-06-11 (Track-1 decision); retained
because they motivate future model-zoo extensions (per-branch quadratic head)
and delimit the landscape: the boundary theorem's cost is dictionary-relative,
and no dictionary dominates.

## T2-A — min-net exact sparsity on $v_d$ (assembled from KVV; all d)

**Theorem.** The KVV soft-min network with the $2d$ Gaussian atoms and Moreau
regularizer achieves sup-error $\le (2d-1)\varepsilon$, *exact* gradients off any
$\delta$-neighborhood of the kink set, and uniform Hamiltonian consistency there —
with **$2d$ neurons, linear in $d$** — while (T2-B) every ridge net needs unbounded
weight mass on the same target.

**Intuition.** The min architecture never pays the crease cost: a crease is just
two smooth branches crossing, so *two atoms* produce exactly the right modulated
jump. The dictionary matches the geometry of value functions — branches of
cost-to-go crossing along the switching set (Subbotina's representative
formula). This certifies that $v_d$ is genuinely simple in the right coordinates;
the expense in T2-B is the ridge dictionary's fault.

## T2′ — incomparability of the dictionaries (PROVED, unconstrained class, via Lemma P)

**Proposition.** Let $s \in C^2(\mathbb{R})$ be non-affine with bounded $s''$ and $\psi(x) = s(w \cdot x)$
on a compact convex full-dimensional $X$.
(i) Ridge nets: one neuron represents $\psi$; its ridge variation cost is the 1-D cost
of $s$ — dimension-free, with 1-D rates.
(ii) Shared-Hessian min-quadratic nets: for $C > \max(\sup s'', 0)$ (semiconcave
orientation), any $n$-atom approximant has sup-error $\ge \tfrac{\varepsilon_0}{4} \bigl(\mathrm{vol}\, X/(n\, \omega_d)\bigr)^{2/d}$
with $\varepsilon_0 = \min(C - \sup s'',\, C)$ — the curse, on a target the ridge dictionary gets
for free. Hence neither dictionary dominates; with T2-A/B, curved-switching value
functions sit on the min-plus-favorable side, ridge-like profiles on the
ridge-favorable side.

**Intuition.** A ridge function is flat along $d-1$ directions; a paraboloid with
full Hessian $C I$ curves in all $d$. Each min-quadratic atom can therefore hug the
ridge target only on a small ball (its strong-convexity contact region, radius
$\sim \sqrt{\delta}$), and tiling $X$ needs $\sim \delta^{-d/2}$ balls. Symmetrically, $v_d$ curves like a
Gaussian in every direction and creases along modulated faces — cheap for
quadratic branches, ruinous for ridges. Approximation power is not a total
order: each dictionary is efficient exactly on targets sharing its invariance
structure.

## T4 — rank-conditioned min-plus capacity (FRAMED; lower bound unconstrained)

**Theorem (assembled).** (a) LQ data, terminal $\Psi$ = min of $m$ quadratics $\implies$ the
value function is the min of the *same $m$* Riccati-propagated quadratics for all
$t$ (DDM Prop 1): min-plus rank is conserved; cost $\mathrm{poly}(d)\cdot m$; no curse in $d$.
(b) $H$ = min of $M$ LQ Hamiltonians $\implies$ rank $\le m \cdot M^N$ after $N$ steps: a curse of
*complexity* in time, not of *dimensionality* in $d$ (McE07; pruning).
(c) The semigroup is sup-norm nonexpansive (Lemma N), so the capacity question
reduces entirely to approximating the terminal data.
(d) For $C^2$ semiconcave data and shared-Hessian atoms, $n(\varepsilon) \asymp \varepsilon^{-d/2}$ with a
Bregman-volume constant (GMQ 3.1–3.2; unconstrained version: Lemma P) that
**degenerates exactly on the Hessian-saturated class $D^2\Psi \approx C I$**.
(e) No uniform no-curse theorem exists (GMQ 3.3).

**Intuition.** The right complexity measure for min-plus approximation is not a
norm but a *rank* — how many smooth branches the function is a min of — because
the HJB flow is min-plus linear: it propagates branches independently through
Riccati equations and never increases their number. Low-rank data $\implies$ the curse
vanishes (McEneaney's curse-of-dimensionality-free phenomenon). GMQ shows rank
is genuinely necessary: a *generic* smooth semiconcave function keeps its
Hessian strictly inside the semiconcavity bound on positive volume, and there
min-quadratic approximation is as cursed as gridding. The saving grace for
control: value functions are not generic — Riccati dynamics push Hessians toward
saturation (where the lower-bound constant vanishes), and non-smoothness
concentrates on the switching set, where branch crossings are 2-atom events.
"Value functions are min-plus sparse" — rank $\approx$ number of synthesis cells — is the
precise conjecture (obligation 5 in `t4-minplus-capacity.md`).

## Lemma P — unconstrained covering lower bound (PROVED & SHARP)

**Lemma.** Let $X \subset \mathbb{R}^d$ be compact convex with $\mathrm{vol}(X) > 0$, $V \in C^2(\bar X)$, $C \in \mathbb{R}$ with
$D^2\bigl(\tfrac{C}{2}|x|^2 - V\bigr) \succeq \varepsilon_0 I$ on $X$ for some $\varepsilon_0 > 0$. Then every
$\tilde V = \min_{i \le n} \{\tfrac{C}{2}|x|^2 - \ell_i(x)\}$, $\ell_i$ affine — **no minorant constraint** —
satisfies

$$
\|V - \tilde V\|_{L^\infty(X)} \;\ge\; \frac{\varepsilon_0}{4} \left( \frac{\mathrm{vol}\, X}{n\, \omega_d} \right)^{2/d}.
$$

**Proof.** Set $g := \tfrac{C}{2}|x|^2 - V$ and $\hat g := \max_i \ell_i$; the shared quadratic cancels,
so $\delta := \|V - \tilde V\|_\infty = \|g - \hat g\|_\infty$, with $g$ $\varepsilon_0$-strongly convex on the convex $X$.
(1) One-sided global bound: $\ell_i \le \hat g \le g + \delta$ on all of $X$, so $h_i := g - \ell_i \ge -\delta$
everywhere. (2) On the active region $R_i = \{\hat g = \ell_i\}$, $h_i = g - \hat g \le \delta$. (3) $h_i$ is
$\varepsilon_0$-strongly convex; with $\bar x_i$ its minimizer over $\bar X$ (first-order optimality on a
convex set), $h_i(x) \ge h_i(\bar x_i) + \tfrac{\varepsilon_0}{2}|x - \bar x_i|^2 \ge -\delta + \tfrac{\varepsilon_0}{2}|x - \bar x_i|^2$. Combining
with (2): $R_i \subseteq B\bigl(\bar x_i, 2\sqrt{\delta/\varepsilon_0}\bigr)$. (4) The active regions cover $X$, so
$\mathrm{vol}\, X \le n\, \omega_d\, (4\delta/\varepsilon_0)^{d/2}$; solve for $\delta$. $\blacksquare$
The bound is **sharp**: for quadratic $V$ with $D^2 V = (C - \varepsilon_0) I$ in $d = 1$,
equioscillating secants attain $\varepsilon_0 R^2/(4n^2)$ with equality (referee pass
2026-06-11, which also verified every step and corollary adversarially).

**Corollary 1 (T2′(ii), unconstrained).** $\psi = s(w \cdot x)$, $C > \max(\sup s'', 0)$ $\implies$
$\varepsilon_0 = \min(C - \sup s'',\, C)$; the curse for shared-Hessian nets on a one-neuron ridge
target.

**Corollary 2 ($v_d$ smooth branches).** For model curvature $C \ge C^*$, the ball
$B = B(1.4\, e_1,\, 0.1)$ lies in the active cone of the branch at $-e_1$ (every $d$), with
$r^2 \in [5.29,\, 6.25]$ and active-branch Hessian eigenvalues $\le 0.305$, so $\varepsilon_0 \ge 0.14$:
the max-affine shared-Hessian class needs $n \gtrsim \mathrm{vol}(B)\,(\varepsilon_0/4\varepsilon)^{d/2}$ atoms for
accuracy $\varepsilon$ on $v_d$. (Note: a branch's own center, where $g'' = (C^*+1) I$, lies in
*other* branches' active cones — on active regions $\varepsilon_0 \le \sim 0.45$, vanishing only on
the saturation sphere $r^2 = 3$.) For *learnable* $C$ a two-ball dichotomy closes the
escape route: $C \le 0.43$ hits an $n$-independent error floor on a high-curvature
ball near $r^2 = 3$ (target curves faster than any $C$-semiconcave function along a
radial chord); $C > 0.43$ hands Lemma P $\varepsilon_0 \ge 0.42$ on a far-field ball. Details in
`t4-minplus-capacity.md`.

**Remark (free Hessians — conjecture only).** Atoms with free Hessians can cancel
the second-order gap, leaving cubic contact; an $n^{-3/d}$ lower bound is
*conjectured* (it would match the piecewise-quadratic upper bound), but the
covering argument does not extend naively — accuracy regions can hug the zero
variety of the cubic form, whose $\delta$-tubes have volume $\sim \delta$, not $\sim \delta^{d/3}$. Either
way, the per-branch dictionary's true power is exactness at finite min-plus rank
(T4a), not generic smoothing.

**Caveat for Part I readers.** Lemma P does *not* curse the repo's
`SemiconcaveModel` in terms of its own atom count: $n$ ridge atoms induce up to
$\sim n^d$ max-affine pieces, so the bound translates only to $\delta \gtrsim n^{-2}$ — dimension-
free and consistent with the ridge variation-space rates. Lemma P's force is
against the *max-affine piece count* (the min-architecture's currency), which is
why it lives in Part II.
