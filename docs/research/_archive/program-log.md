---
name: semiconcave-theory-program
description: Theory program proving why semiconcave/min networks beat ridge nets on HJB value functions; T2 draft in docs/research/t2-separation-draft.md
metadata: 
  node_type: memory
  type: project
  originSessionId: ca4ff4ed-bb58-477b-89ad-ca74f6f71544
---

Research program (started 2026-06-10) to prove the semiconcave architecture's
sparsity advantage for HJB value functions. Full draft: `docs/research/t2-separation-draft.md`.

PROBLEM LOCKED 2026-06-17 (user-agreed, in docs/research/CONTEXT.md §1): TARGET =
general C-semiconcave V = ½C‖x‖² − g, g convex (arbitrary; smooth OR kinks/switching
set). ACTIVATION CONDITION: OPEN. (C1) atom convexity (t↦σ(t)^p convex) was proposed then
REFUTED 2026-06-17 by faithful activationsearch data: tanh (NON-convex atom) cone 2 vs
signed 19 = STRONG advantage; gaussian (non-convex) cone 13 vs signed 12 = NO advantage.
So convexity neither gates nor sizes the effect. The convexity-based "class matching /
monotone gradient" WHY is also retracted. DATA PATTERN (finite_step, h1, matched ~acc):
count advantage STRONG for softplus(1v11),tanh(2v19),lisht(1v19),snake(1v18); ~NONE for
gaussian(13v12); modest matern(5v12); squared gelu²/silu² cone picks head-only(0 atoms).
DERIVED FROM FAITHFUL DATA 2026-06-17 (scripts/activation_condition_derive.py): the
condition is a RATIO not a single property. ROBUST: if signed σ-atoms can cheaply build
the quadratic ½C‖x‖², advantage VANISHES (gaussian fits Q@14 atoms→adv 1.0; necessary
not sufficient). MODEL (= Theorem A-d split per activation): advantage ≈ 1 + N_σ(Q)/N_σ(g)
where Q=½C‖x‖², g=convex correction. Large iff atoms MUCH WORSE at smooth quadratic than
at sharp correction g. Explains all 5: softplus(bad@Q,good@kinks→8), tanh(bad@Q AND
bad@kinks→1.4), gaussian/matern/gelu²(good@Q→≈1). So softplus≈smoothed-ReLU is the strong
case (good at kinks via ReLU-likeness, bad at smooth curvature). RATIO MODEL = HYPOTHESIS,
counts config/accuracy-sensitive (1 seed). CONFIRM by measuring N_σ(g) directly across
seeds. KEY LESSON: stop seeking a single qualitative property of σ; the answer is the
RELATIVE cost (quadratic vs correction), which is exactly the proved Theorem A-d
decomposition applied per-activation — not a new mechanism. COST = neuron count
N(M,V,ε)=min{n: inf_{f∈M_n}‖∇f−∇V‖_{L²(Ω)}≤ε‖∇V‖_{L²}}, locked choices: relative
gradient-L², BEST-n-TERM (solver/penalty-independent; PDAP count = empirical upper bound).
Neuron = σ-atom; head+affine not counted. Variation norm ≠ count (kept as proof tool).
GOAL THEOREM (not yet proved): under (C1), for general semiconcave V (esp. with curved
switching set), N(signed,ε)/N(semiconcave,ε)→∞. Proved special cases: Theorem A-d (ReLU,
piecewise-linear g, Θ(1/ε)) and T2 (min-arch, curved switching, ∞ R-norm, d=2).

BREDIES OPTIMALITY FRAMEWORK USED 2026-06-17 (docs/research/semiconcave-source-condition.md;
user pushed: I'd never actually USED the optimality framework, only described it). Read
Bredies-Pikkarainen optimality (Prop 3.6: supp|μ*|⊆ contact set {|q|=α}), source condition
(4.2-4.4), O(δ) Bregman rate (Rmk 4.3). KEY NEW STRUCTURAL RESULT: Bredies Rmk 4.4 = source
condition (⟹ O(δ) rate + exact sparse support recovery) requires support sets Ω̃₊,Ω̃₋ separated
from EACH OTHER and from ∂Θ by positive distance. SPECIALIZE BY SIGN: cone (μ≥0) ⟹ σ†≡+1 ⟹
Ω̃₋=∅ ⟹ NO opposite-sign separation needed (only sep from boundary); signed ⟹ both Ω̃₊,Ω̃₋ ⟹
opposite-sign clusters must be minimally separated. ⟹ CONE = SEPARATION-FREE recovery regime
(de Castro-Gamboa, Schiebinger-Recht, Slawski-Hein); SIGNED = MINIMAL-SEPARATION-REQUIRED
(Candès-Fernandez-Granda). PAYOFF: cone recovers sparse μ† at O(δ) even for ARBITRARILY
CLUSTERED atoms; signed SPREADS (more atoms, no support recovery) when opposite-sign clusters
too close. This is the optimality-framework answer to "what structural property makes
semiconcave solution sparse" — generalizes T5-d1 one-sided certificate from d=1/ℓ1 to general
d AND adds the rate. CITED not re-proved (Bredies thms); NEW = sign-specialization + sep-free
reading. Free head = SEPARATE null-space feature (η⊥{½‖x‖²,x,1}), not in basic Bredies, =
next step (partial-source/Bregman). TO VERIFY: cone μ† for real HJB g meets sep-from-∂Θ;
measure signed-spreading vs cone-concentration on clustered target (falsifiable §4 prediction).

BUDGET DEFINED 2026-06-17 (CONTEXT.md, user asked): bud(g) := ‖D²g‖_M = total mass of
Hessian (Alexandrov) measure of convex correction g = ∫_Ω tr(dD²g) = ∫_∂Ω ∂_n g (g convex)
≤ Lip(g)|∂Ω| ≈ C|Ω|+2L. Property of g ALONE (no activation). ALWAYS FINITE for semiconcave V,
indep of switching complexity. d-dim form of T1. DISTINCT from γ⁺(g) = nonneg RIDGE variation
norm (Maurey/Barron constant, what the RATE needs). RELATION (crux): bud(g) ≤ c(Ω)γ⁺(g),
EQUAL in d=1; but γ⁺ can be ∞ while bud finite (d≥2 curved). So bud<∞ NECESSARY not sufficient
for rate; "budget→rate" is a d=1 statement. Don't conflate the two.

INTERMEDIATE RESULT 2026-06-17 (docs/research/semiconcave-rate.md): semiconcave/cone
model structure+budget+rate. (a) head represents ½C‖x‖² EXACTLY ⟹ N(cone,V,ε)=N⁺(g,ε)
indep of C (approximating V reduces to approximating convex correction g by NONNEG atoms).
(b) CURVATURE BUDGET all d: g convex ⟹ ‖D²g‖=∫tr D²g=∫_∂Ω ∂_n g ≤ Lip(g)|∂Ω| FINITE,
indep of switching complexity (divergence thm; d-dim generalization of T1). (c) RATE: if
nonneg ridge varnorm γ⁺(g)<∞, gradient-L² error ≤ B γ⁺(g)/√n (Maurey, dim-free), N≤(γ⁺/ε)²;
Yang-Zhou improves exponent. (d) FINITENESS of γ⁺: d=1 γ⁺=‖g''‖=‖D²g‖ (g=affine+∫(x-t)_+
dg'', g''≥0) ⟹ (b)⟹(c) UNCONDITIONAL, full proved d=1 rate extending T1; d≥2 smooth/flat
g finite; d≥2 CURVED switching set γ⁺=∞. NEGATIVE RESULT CONFIRMED (T2+empirical): counterex
g=(‖x‖-1)_+ on [-2,2]² (circular switching) has FINITE budget but γ⁺=∞ (γ⁺≥‖g‖_R=∞ by T2,
since ‖V‖_R=∞ for curved kink & quadratic has finite R-norm). FAITHFUL repo cone-ReLU-ridge
PDAP (scripts/curved_vs_flat_switching.py): FLAT g=(x1)_++(x2)_+ reaches relGrad 0.006 w/ 5
atoms (bounded, slope 0); CURVED g=(‖x‖-1)_+ PLATEAUS at relGrad 0.138 unbreakable even w/
124 atoms (slope +0.65) = signature of γ⁺=∞. ARCHITECTURE BOUNDARY: repo cone-RIDGE model
inherits budget→rate ONLY for smooth/flat g; for CURVED switching (generic HJB) BEYOND-RIDGE
is NECESSARY (ridge^k has flat curvature jumps, can't tile curved switching surface, any
power k). NECESSARY/SUFFICIENT (corrected 2026-06-17, was overclaimed): min-of-paraboloids
SUFFICES for v_d-type (finitely-min-representable, 2d atoms) but NOT universally — rotationally-
symmetric (‖x‖-1)_+ likely needs a CONTINUUM of paraboloids in min too (inf-conv sources fill
the circle). So (‖x‖-1)_+ shows RIDGE FAILS (solid negative); the clean MIN-BEATS-RIDGE example
is v_d, NOT (‖x‖-1)_+. Don't conflate. NEXT: positive curved result = min architecture on
FINITELY-MIN-REPRESENTABLE V (target v_d, 2d atoms) + characterize that class for HJB (max-plus
rank; curse-free-max-plus, Gaubert-McEneaney-Qu).

LITERATURE GROUNDING 2026-06-17 (docs/research/sparsity-statements-and-structure.md;
user wanted to know what statement establishes a sparsity effect + uncover structural/
inductive-bias difference). FINDINGS: (0) closest paper Kunisch-Vásquez-Varas
(semiconcave_approximaton.pdf) is structure-preserving APPROXIMATION/convergence, NOT a
sparsity separation — must import statement form. (THREE TEMPLATES): A=representer/sparse-
minimizer (Parhi-Nowak Banach rep thm; Bredies-Pikkarainen measures; Boyd-Schiebinger-Recht
ADCG = PDAP'S ANCESTOR) — minimizer of [datafit+TV/variation norm] is atomic, ≤N atoms,
selected by dual certificate ‖η‖_*≤α w/ equality on supp; B=variation-space/Barron rate
(Barron93,Yang-Zhou,Bach,Ongie R-norm) — n atoms give error γ(f)·n^{-1/2-(2k+1)/2d}, γ=
variation norm=the currency, count derived from γ; C=separation/lower bound (Yang-Zhou
pseudo-dim; our T2). FOR US: route = C in B's currency = show γ_semiconcave(V)<∞ but
γ_signed(V)=∞/≫; count gap follows by B. Literature proves sparsity via VARIATION-NORM gap,
NOT direct neuron counting (vindicates user's T2/variation-norm steer; my neuron-count
detours were off-template). STRUCTURAL DIFF (inductive bias) = 2 FEATURES × 2 LAYERS:
Feature1 NULL SPACE (semiconcave free part = span{½C‖x‖²,x,1} incl QUADRATIC; signed = only
affine) → head fits V's quadratic at zero penalty cost; Feature2 CONE (μ≥0 one-sided cert
q≤α vs signed two-sided |q|≤α; ADCG natively nonneg ⟹ cone=natural object, signed=extended).
These are the CORRECT non-confounded form of botched "capacity(=F1)/selection(=F2)";
representer thm SEPARATES them by construction (null-space orthogonality ⟨η,½‖x‖²⟩=⟨η,x⟩=
⟨η,1⟩=0 vs one/two-sided bound) instead of confounded subtraction. 2 LAYERS: variational
(regularized min, Template A cert) vs ALGORITHMIC (greedy CG/ADCG=PDAP: insert at cert max
+ local search) — CAN DIFFER (ReLU² greedy missed symmetric pair). "What controls bias" =
open 2×2 {F1,F2}×{variational,algorithmic}. NEXT: work dual certificate per model to find
which makes γ_signed≫γ_semiconcave. RECOMMENDED STATEMENT: under σ-condition(TBD), general
C-semiconcave V has γ_semiconcave(V)<∞ while γ_signed(V)=∞/≫; subsumes T2 + Theorem A-d.

VOCABULARY AUTHORITY: docs/research/CONTEXT.md (created 2026-06-17, user-requested).
§1 = agreed terms only (may use freely); §2 = Claude coinages awaiting ratification
(bulk[drop], Q, κ_σ, Tier1/2) — do NOT use until approved; §3 = retired (capacity/
selection, the variation-norm/broad-atom churn). Consult/maintain it before introducing
any term. Distinct from repo-root CONTEXT.md (open-loop domain language).

⚠ REWIND 2026-06-17 01:20 (user, repeated reliability complaint about churn/flip-flops):
TRUSTED BASELINE = docs/research/sparsity-gap-real-setting.md (mtime 06-16 17:22) and
everything earlier (theory-ladder, t5-d1, T1/T2/T4 docs). DISCARDED AS UNTRUSTED:
direction1-smooth.md (06-16 22:51) + direction2-relu2.md (06-16 22:43) + tonight's
verbal variation-norm / bulk-vs-switching 2×2 / T2-reconciliation reasoning (no files).
SPECIFIC ERROR found in direction1-smooth.md: "Lemma 2" (non-poly σ ⟹ N_σ(t²,ε)→∞,
"head advantage diverges for all non-poly σ") is FALSE — refuted by the broad-atom
construction (σ''≠0 ⟹ c[σ(at)+σ(−at)]=const+t²+O(a²) ⟹ 2 atoms fit t² to any ε with
weights ~1/a²). Files NOT deleted (user said keep). When resuming, do NOT build on the
D1/D2 conclusions or tonight's variation-norm claims; re-derive carefully if needed.
NOTE: trusted baseline's κ_σ table = MEASURED/greedy exponents (honest as such);
fundamental-vs-greedy per activation is OPEN (the broad-atom fact, though true, is set
aside per the rewind). LESSON (recurring, now explicit): stop emitting confident
headlines then retracting; verify on real data/solver before concluding; one careful
claim beats five churned ones.

Key facts established:
- Repo's `SemiconcaveModel` (src/models/semiconcave.py) is V = ½C‖x‖² − g, g = Σ cᵢσ(w·x+b)^p
  with cᵢ ≥ 0 — ridge-convex Legendre-dual form, NOT the Kunisch–Vásquez-Varas
  min-of-C²-functions architecture (per-branch Hessians). Equivalent to inf of
  paraboloids with shared Hessian C·I.
- Yardstick: dictionary variation norms; for ReLU ridges this is exactly Ongie et al.'s
  R-norm = γ_d‖∂_b^(d+1) R{f}‖₁ (Radon domain).
- T2 (flagship, drafted): for v_d = min_{2d} exp(−½|x∓eᵢ|²) (KVV §5 test function,
  HJB viscosity solution): min architecture is exact with 2d atoms; ‖v_d‖_R = ∞ for
  d ≥ 2 ⟹ any ReLU-net sequence converging to v_d has weight cost → ∞. Proof via
  kink density J(t) = √2·exp(−(t²+|t|+½)) on bisector faces ⟹ v̂_d ~ r^(−2) on a fan
  of rays ⟹ dyadic angular bands each need fixed mass quantum. 4 rigor gaps listed in doc.
- Smooth radial parts are poly(d)-cheap (Ongie Ex. 3: ~d²); the separation is carried
  entirely by the semiconcave shocks.
- Theorem ladder T1–T5 in the doc; T3 (cone-optimality: minimal signed ridge measure of
  ridge-convex g is automatically nonneg) is the cleanest new open question justifying
  the cᵢ ≥ 0 constraint.
- T1 proved (docs/research/t1-1d-budget-lemma.md): TV(V'') ≤ 2C|Ω|+2L, exact cone representation,
  1-D uniqueness ⟹ cone constraint free in d=1; N^(−2) shock-independent rate.
- T2 part (B) made rigorous for d=2 via matched-filter witness inside Ongie Def. 1
  (no disintegration needed); remaining gaps: d≥3 cone-face FT decay, general-d kink
  density, k≥2 slice criterion, constant-chasing.
- GMQ 2011 (Gaubert–McEneaney–Qu, arXiv:1109.5241) gives the min-plus-side LOWER bound:
  shared-Hessian quadratic approximation of C² semiconvex targets is Ω(n^(−2/d))
  (Bregman volume det(ψ″+cI) constant; attenuates when Hessian saturates). ⟹ no
  uniform no-curse theorem; T4 must be rank-conditioned. Framed in
  docs/research/t4-minplus-capacity.md, with T2′ (incomparability: smooth ridge cheap for
  ridge nets, cursed for shared-Hessian min-quadratics) and Lemma N (semigroup
  sup-norm nonexpansiveness ⟹ capacity reduces to terminal data).
- DDM (Darbon–Dower–Meng 2023) Prop 1: LQ flow conserves min-plus rank (m quadratics
  in ⟹ m Riccati-propagated quadratics out). Closest competitor; T4 positions vs it.
- Lemma P (proved, docs/t4 + docs/research/theory-ladder.md): unconstrained covering lower
  bound ‖V−Ṽ‖∞ ≥ (ε₀/4)(vol X/(n·ω_d))^(2/d) for shared-Hessian min-quadratic nets
  whenever D²((C/2)|x|²−V) ⪰ ε₀I. Closes T4 obligation 2 (binds trained models, not
  just minorants); proves T2′ and the ε^(−d/2) row-2 PDAP prediction.
- Audit 2026-06-11 (self-check after Lemma P, agent review predated it): core proof
  sound; 3 fixes: (a) Cor 2 illustration — branch centers are NEVER in their own
  active cones (min picks farthest center), valid ball B(1.4e₁, 0.1) gives ε₀ ≥ 0.14,
  not 1.45; (b) learnable C needs the two-ball dichotomy (C ≤ 0.43 ⟹ n-free curvature
  floor near saturation sphere r²=3; C > 0.43 ⟹ Lemma P far-field with ε₀ ≥ 0.42);
  (c) free-Hessian n^(−3/d) DOWNGRADED to conjecture — covering fails on the cubic's
  zero variety (δ-tubes have volume ~δ, not δ^(d/3)).
- Opus adversarial referee 2026-06-11 (Lemma P + T2(B) d=2): BOTH PROOFS HOLD, no
  blockers. Lemma P is SHARP (d=1 quadratics: equioscillating secants attain
  ε₀R²/(4n²) exactly). Fixes applied: Cor 1 ε₀ = min(C−sup s″, C) (wrong when
  sup s″ < 0); √2 arclength Jacobian in T2 sheet FT (Ĵ now = FT of arclength
  density, κ ≈ 1.32; conclusion scale-invariant in ρ₀); log-coefficient wording
  (errors are fractions of main term, P ≥ ½c₁κρ₀log − O(C_Jθ₀²)); referee numerics
  confirm P → ∞ with exact log-tracking (≈0.0125 per θ_min halving). Remaining for
  full rigor: double-cover factor c ∈ {1,2}, FT-convention pinning vs Ongie (18),
  honest cross-sheet constant (t2 gap 4); d ≥ 3 (gaps 1–2). Lesson: Sonnet fine for
  citation checks, Opus needed for adversarial proof verification.
- docs/research/theory-ladder.md = overview with clean statements + intuition for T1, T2, T2′,
  T3, T4, T5, Lemma P.
- ⚠ File identities in 'representation theorem/': 'CDC-max-plus-complexity bounds.pdf'
  is actually DDM 2023; 'curse-free-max-plus_McEneay.pdf' is actually Dower–McEneaney
  wave-equation semigroup (adjacent only); the real McEneaney 2007 is
  'curse-of-dimensionality-free_McEeany.pdf'; GMQ is 'Gaubert–McEneaney–Qu-2011.pdf'.
- Architecture recommendation, three-way (sharpened by GMQ): signed ridge blocked by
  T2(B); shared-Hessian max-affine SemiconcaveModel limit pays GMQ ε^(−d/2) on smooth
  branches; per-branch-curvature min-quadratic head (KVV/DDM) gets 2d atoms on v_d AND
  1-D rates on ridges (rank-1 Hessians). PDAP three-way experiment predicted slopes:
  divergent / ε^(−d/2) / flat-at-2d.

- DECISION 2026-06-11 (user): thesis core = Track 1 — semiconcave structure WITHIN
  the ridgelet framework (the actual SemiconcaveModel), NOT the min-architecture.
  docs/research/theory-ladder.md restructured: Part 0 = problem & three mechanisms
  (capacity T3 / budget T1·T1-d / selection T5) + boundary theorem T2-B/C;
  Part I = core (T1 proved, T2-B/C proved d=2, T3/T5/T1-d open, parked weak-*
  all-d proof UNVERIFIED); Part II = min-net results moved to bottom (T2-A, T2′,
  T4, Lemma P) as motivation/landscape — user wants them retained, not deleted.
  T1-d (new open lemma): d≥2 cone-variation budget γ⁺(½C|x|²−V) ≲ C+L for C²
  semiconcave V; mechanism = slice one-sidedness ∂_b²R{g} ≥ 0; obstacle = ramp
  filter (d+1 b-derivatives). Caveat recorded: Lemma P does NOT curse the repo
  model in ridge-atom count (n ridges ⟹ ~n^d max-affine pieces ⟹ only δ ≳ n^(−2)).

- T1-d resolved-as-reformulated 2026-06-11 (UNVERIFIED batch): uniform d≥2 budget is
  FALSE — mollify v₂ (constants preserved), bounded mass passes to uniform limits,
  contradicts boundary theorem (rigorous for Ω-uniform K via proved T2-B + Ongie R̄;
  fixed-Ω version rests on parked weak-* lemma). Correct form T1-d′: mass(ε) ≲
  K(d,Ω)(C+L)ε^(−(d−1)) via mollification + slice budget (∂_b²R{g} ≥ 0 survives
  slicing) + (d−1) filter derivatives à δ^(−1); blowup localizes at Σ(V). Cone
  version of T1-d′ gated on T3.
- T3 RESOLVED NUMERICALLY 2026-06-11 (docs/research/scripts/t3_cone_certificate.py,
  dual-certificate LP, 61² grid, 6914 lines, in-cone sanity = 1.33% noise floor):
  (a) T3-pure FALSE — smoothed triple junction (LSE, η=0.3) has certificate worth
  53.99% of ‖Δg‖₁ (40.6× noise): pure cone can't represent OR uniformly approximate
  junctions ("creases can't end", quantified); (b) mean-zero certificate (the class
  that refutes the model with FREE quadratic head, since head adds uniform background
  λ·1 and feasible ψ automatically has ∫ψ ≤ 0) has optimum EXACTLY 0 ⟹ THE QUADRATIC
  HEAD RESCUES CAPACITY: Δg + λ·1 ∈ cone closure, C′ = C + λ/d. New formulation T3′
  (head-completeness) + curvature price π(g) := minimal λ. The LP explains WHY the
  architecture has the head. Next experiments: compute minimal λ for the junction
  (bisection over cone-feasibility LPs); run the v₂-diagonal mean-zero LP (conjecture:
  certificate exists via ψ ~ J − mean(J) ⟹ π = ∞ for non-constant-density straight
  kinks ⟹ boundary theorem strict even head-completed). Solver note: dual simplex
  churned >76 min on the mean-zero LPs (dense coupling row + bang-bang degeneracy);
  highs-ipm solved them in 5–20 min; pure-cone numbers identical under both.

- v₂-kink LP 2026-06-11 (docs/research/scripts/t3_v2_kink_certificate.py): pure cone
  sharp 7.48% / const-J control 6.81% / floor 1.33% (J-variation-specific ≈ 0.7%);
  mean-zero = 0 for ALL targets incl. sharp. Lessons: (1) fixed-mesh mean-zero LPs are
  STRUCTURALLY BLIND to singular obstructions (tilted lines emulate variable density
  inside the one-cell kink tube; structure theorem bites only as h→0) ⟹ sharp result
  inconclusive by design, item (iii) needs mesh-scaling of minimal λ; (2) duals at the
  degenerate optimum are NOT curvature prices (reported ~4400-4700 = arbitrary dual
  vertices); the instrument is the primal two-phase min-λ LP at several meshes
  (flat λ(h) ⟹ finite π; growing ⟹ π = ∞).
- T5-d1 CERTIFICATE DICHOTOMY 2026-06-15 (docs/research/t5-d1-certificate-dichotomy.md;
  scripts/t5_certificate_dichotomy.py): the ≈2× selection factor turned into a theorem
  with mechanism. Core (rigorous): Lemma 1 = Prop C (q''=2η ⟹ certificate is double
  antiderivative of residual); Lemma 2 (η has K sign changes ⟹ q has ≤K+1 critical
  points, alternating max/min, ≤⌈(K+1)/2⌉ each); Lemma 3 (KKT: SIGNED active at both
  bounds ±α, CONE active only at −α bound — one-sided). Theorem: under equioscillation
  n_signed=K+1, n_cone=⌈(K+1)/2⌉ ⟹ ratio→2. Eviction reading: cone keeps the −α atoms,
  discards the +α (Theorem-B cancellation) half. Numerics CONFIRM: signed splits both
  bounds, cone +α-count=0 in EVERY run (all at −α), matched acc%, signed −α count ≈
  cone total (15v16, 6v7), ratio 1.67→1.81→2. Honest gaps: (a) equioscillation
  genericity for exact 2; (b) cross-problem reduction |K_s−K_c| (different residuals) —
  the analytic step. TWO referee passes 2026-06-15: (1) self-pass (first subagent died
  on limit) fixed mirror-sign q_-''=-2η(-b); (2) EXTERNAL Opus pass found a BLOCKER —
  the absolute-count law n_signed=K+1, n_cone=⌈(K+1)/2⌉ is NUMERICALLY FALSE: K≈const
  in α (7 noiseless) but atom counts grow ∝1/α because the solver CLUSTERS ~m(α) atoms
  per shock/contact region (~3× more atoms than critical points). FIX: theorem
  restated as a RATIO over CONTINUUM (contact-region) counts where m(α) cancels —
  N_signed=R_+ + R_-, N_cone=R_-, R_+≈R_- by alternation ⟹ ratio→2. MERGED numerics
  (cluster-adjacent knees) = EXACTLY 2.00 across two orders of α and into noise (raw
  1.67-1.81 was the clustering artifact). Other fixes: endpoint-zeroing unconditional
  only for SIGNED (cone needs head interior C̃>0); "cone +α=0" is DEFINITIONAL (KKT),
  not independent evidence; per-branch ratios erratic (factor 2 is totals effect, not
  per-branch). Real open gap relabeled: NOT |K_s−K_c| (measured =0) but hypothesis (M):
  the two problems form the same -α-region count + multiplicity. DUAL-INTERSECTION
  THEOREM 2026-06-16 (Lemma 3.5, scripts/t5_dual_intersection.py): in residual vars
  F_signed = A ∩ B, A={q≥−α}=semiconcave cone, B={q≤+α}=convex cone=−A (exact LP
  duality). Factor 2 is FORCED BY LEMMA 2 ALTERNATION (R+=R-±1 for ANY residual — q''=2η
  forces +α,−α,+α,... alternation; verified on ASYMMETRIC targets: lopsided/uneven/
  sub-gap-pair all give R+=R-, ratio 2.00 — so my initial "reflection symmetry B=−A /
  no-bias" explanation was WRONG REASON for a correct fact, caught by my own asymmetry
  probe before referee returned; B=−A is the semiconcave/convex model duality, not the
  R+=R- reason). Signed uses both bound families, cone one. (M) = facet-stability
  R_-(signed)=R_-(cone) — VERIFIED
  EXACT across m∈{1,2,3,5,8} × 2 α × 4 seeds (Q2=1.00 every row; onbnd=1.00 confirms
  F_signed=A∩B; R tracks m=structure, not K=2..9 nor 1/α). Proof skeleton: reflection
  symmetry + disjoint-support (the +α facets bind on the convex region, inert on
  semiconcave targets). Reframes T5-d1 as facet-counting on intersection of two
  reflected polyhedra.
- EXTERNAL OPUS REFEREE #2 2026-06-16 (dual-intersection material) SOLVED (M) and
  killed two of my explanations: the real mechanism is η_signed = η_cone via
  PROJECTION onto nested polytopes — η_cone=Proj_A(y), η_signed=Proj_{A∩B}(y), A∩B⊆A;
  IF q_cone ≤ +α everywhere (cone cert never overshoots upper bound, = equioscillates)
  THEN η_cone∈B ⟹ by projection uniqueness η_signed=η_cone ⟹ ONE shared certificate ⟹
  (M) is a COROLLARY + R+=R- exactly + ratio=2 exactly. VERIFIED to machine precision
  (scripts/t5_projection_mechanism.py): η_s=η_c (1e-6..1e-16) and max(q_cone)-α=0 on
  CONVEX (½x²), oscillatory (sin15x), |x| targets — TARGET-INDEPENDENT, so my
  "semiconcave/disjoint-support" AND "reflection-symmetry/no-bias" stories are BOTH
  RETRACTED (M holds for convex targets with no downward kinks at all; R+=R- by Lemma 2
  alternation on the shared cert, not by symmetry). Two open hypotheses (E)+(M) +
  mislabeled |K_s−K_c| ALL collapse to ONE inequality: q_cone ≤ +α everywhere (for the
  C̃>0 head-interior regime; fails on concave-bulk where C̃ pins at 0). "F_signed=A∩B"
  is a trivial set identity (norm definition); content is the projection argument.
  "R tracks m" was an overclaim (tracks RESOLVABLE shocks; m≥5 test knots collide
  below the 0.08 merge gap). onbnd/Q1/Q2 are ONE fact (η_s=η_c) measured thrice.
  Mechanism (Lemmas
  1-3, signed-both-bounds/cone-one-bound) is PROVED; the exact 2 needs (E)
  equioscillation + (M).
- OVERSHOOT SEARCH 2026-06-16 (scripts/t5_overshoot_search.py): the lone open
  inequality q_cone ≤ +bound (interior C̃>0 regime) is SUPPORTED — convention-free
  real overshoot R=(max q+min q)/(−min q) ≤ 1e-10 across 387 random+signed+adversarial
  interior targets, ~1e-13 on convex-heavy/|x| cases that stress it most; boundary
  (C̃=0 concave) is the excluded regime. Reduction (Lemma 3.5: η_s=η_c iff q_cone≤bound)
  is PROVED; the inequality is the sole remaining analytic gap. THREE MEASUREMENT TRAPS
  hit & recorded (each gave a SPURIOUS "REFUTED"; nothing wrong reached the docs):
  (i) sklearn (1/2n) ⟹ KKT bound is n·α NOT α (fabricated 119× overshoot); (ii)
  "max q − max|q|" is ≤0 by definition = vacuous; (iii) thresholding ‖η_s−η_c‖ at 1e-4
  flags CD solver slack as refutation. Trustworthy metric = R, cross-checked vs
  ‖η_s−η_c‖. PROCESS: ProcessPoolExecutor leaves orphan spawn-children that pkill -f
  MISSES (they show as 'python -c from multiprocessing.spawn') — kill by PID; prefer
  single-process loops in these scripts. Also: my runtime estimates keep being wrong —
  per-solve cost forecastable, loop totals are not; design lean + time-box.
- RUNTIME/PROCESS DISCIPLINE (2026-06-16, after stranding long jobs & spinning wrappers
  that frustrated user): (1) TIME ONE SOLVE FIRST, quote measured×count — tol=1e-13 was
  16s/solve=2.65M CD iters → 500-trial run silently 4 HOURS; tol=1e-9=7s, tol=1e-7=3s;
  TOL is the cost lever not count; tol=1e-9 suffices (R detect at 1e-6). (2) NO
  'until grep VERDICT; do sleep; done' wrappers — spin forever if watched job killed,
  pkill -f misses them; rely on harness completion notification. (3) TaskStop every
  abandoned bg job BY ID (registered tasks linger = 'shell running 3h'). (4) NO
  ProcessPoolExecutor. (5) live per-iter prints+flush.
- T5-d1 INLINE REFEREE 2026-06-16 (Opus subagent ran 1h opaque, stopped; done in-session
  max reasoning): reduction RE-DERIVED rigorous, TWO-HYPOTHESIS form — η_signed=η_cone
  needs q_cone≤bound AND C̃>0 (cone C̃≥0 one-sided ⟹ ⟨x,η⟩≤0; signed head free ⟹
  ⟨x,η⟩=0; D_signed⊆A needs the equality). Independent hunt seed 777, fine 2001-pt
  cert, 20 stress targets C̃ up to +124: worst interior R=1.35e-12 → SUPPORTED. NET:
  reduction PROVED; inequality q_cone≤bound (interior) strongly supported, sole open step.
  Robustness probe (scratch, non-saturated targets): cone-on-
  one-bound universal, merged ratio ~2 even on smooth tanh target — dichotomy is a
  p=2 SELECTION effect, target-general (vs Theorem A's p=1 saturation-specific
  capacity effect; no contradiction, different powers).
- PROPOSITION C + CONTROL 2026-06-12 (major honest revision): certificate-curvature
  identity q″(b) = 2η(b) ⟹ separation bound ⟹ the stall's +/−/+ cluster is
  infeasible for exact lasso solutions by ~9e2× ⟹ LARS endpoint provably non-KKT
  (stall = SOLVER breakdown, not solution structure). Coordinate-descent control:
  signed problem reaches ALL accuracies (no stall); fair debiased table cone vs
  signed-CD = 4/7 (10%), 7/18 (5%), 10/18 (2%), 29/60 (1%) ⟹ ROBUST ≈2× selection
  factor (solver-independent, consistent with p=1's 1.2–5×) PLUS path-method
  fragility on the signed problem (LARS provably breaks; PDAP/SSN same class).
  Experiment-2 "stall = never reaches 2%" narrative SUPERSEDED (marked in ladder).
  Final p=2 d=1 decomposition: capacity dead, greedy parity, robust ≈2× selection
  gap, plus solver-fragility amplification for path methods.
- LEMMAS B3/B4 + STALL DIAGNOSIS 2026-06-12: B3 (proved+confirmed) — gradient
  certificate on out-of-hull family is AFFINE in t ⟹ either 2 same-sign actives
  force the whole continuum to tie (massive non-uniqueness) or ≤2 actives total;
  diagnostic at the stall: B̃ ≠ 0, exactly 1 oh active — case (b) as predicted.
  B4 (proved) — cone+head KKT: head interior (C̃>0) ⟹ η ⊥ span{x,1} ⟹ out-of-hull
  certificates ≡ 0 — unconditional eviction; C̃=0 boundary reproduces the referee
  counterexample exactly (B2's condition ⟺ head interiority). FORCING HYPOTHESIS
  REFUTED: the stall is NOT the out-of-hull family — it is sign-alternating and
  same-sign CLUSTERS OF ADJACENT IN-HULL KNEES at the mollified shocks (discrete
  2nd-difference spikes; near-null, exploding coefficients; LARS endpoint even
  violates equicorrelation = genuine solver breakdown). Cone cannot form either
  cluster (alternation infeasible; one-sided certificate prevents ties). Re-aimed
  T5 target: cluster dichotomy on continuum dictionaries — one-sided certificate ⟹
  isolated actives (anchors: Slawski–Hein NNLS, de Castro–Gamboa, Schiebinger) vs
  two-sided ⟹ alternating clusters (Candès–Fernandez-Granda separation theory).
- Theorem B referee pass #5 2026-06-12: all numerics reproduced exactly; both
  scripts sound. TWO MAJOR fixes applied: (1) B2 eviction is CONDITIONAL — needs
  C̃−2c ≥ 0, i.e. bulk curvature representable with C̃ ≥ 0 (holds on the
  C-semiconcave saturated class where C̃ = C_true + 2Σc; referee counterexample
  V′ = −Kx forces out-of-hull mass K/2 — concave bulk, outside the class);
  (2) B1 was value-space — in the GRADIENT fit out-of-hull features span {x,1}
  (2-dim, all slopes +2 regardless of ε): degeneracy onsets at 3 atoms, conditions
  Σv = Σvb = 0 only; "same-ε" dropped (mixed-ε same family). Minor: EXP2 counts
  exclude cone's 2 head params; FW leaves C̃ ≥ 0 unenforced (14/297 off-threshold
  path points, C̃ ≈ +10 at thresholds — headline stands, code comment added);
  "must route through that family" weakened (near-boundary in-hull ramps also
  carry bulk); "precisely T5's mechanism" → "consistent with" (stalling active
  sets not yet shown to be the B1 family).
- THEOREM B 2026-06-12 (path-degeneracy dichotomy, d=1 p=2; mechanism lemmas
  PROVED, machine-precision verified inline): B1 — out-of-hull atoms span only
  {x²,x,1}; any 4 same-ε carry the alternating Vandermonde null vector with Σv=0
  ⟹ signed solutions with ≥4 same-sign out-of-hull actives lie on fit- AND
  ℓ1-preserving continua (the singular systems that stall LARS/PDAP);
  B2 — out-of-hull atom gradients lie EXACTLY in the cone model's free head span
  ⟹ transferring mass to head strictly reduces penalty ⟹ cone optima never
  activate the degenerate family: the head structurally EVICTS the degeneracy
  (not just capacity emulation). Open for full dichotomy: saturated targets force
  the signed 4-same-sign config; in-hull independence (spline argument) for cone
  uniqueness. This is the analytic core of T5 at p=2.
- p=2 ALGORITHMIC experiments 2026-06-12 (t5_greedy_emulation.py,
  t5_penalized_path.py; d=1 saturated target, p=2 dictionary): (1) greedy-discovery
  hypothesis REFUTED — on bounded Ω a single distant atom (b outside Ω) IS a
  parabola = head emulator, and signed greedy picks it FIRST; param-adjusted OMP
  parity (cone 3/8/10/15 vs signed 6/11/13/15 at 10/5/2/1%). (2) penalized path
  (LARS, PDAP-faithful): SIGNED PATH STALLS at ~5% — degenerate active sets
  (machine-eps Cholesky pivots = sign-cancellation directions), never reaches 2%;
  CONE PATH SAILS to 1-2% (debiased σ=0: signed 5/8/--/-- vs cone 4/7/10/29).
  VERDICT: at p=2 the sparsity gap is a REGULARIZED-PATH DEGENERACY phenomenon —
  capacity dead (emulation), greedy-discovery dead, path degeneracy demonstrated.
  T5 theorem target now sharp: signed ℓ1 path generically hits non-unique/degenerate
  active sets on saturated semiconcave targets (anchor: Tibshirani 2013 lasso
  uniqueness/general position), nonneg path does not. Caveat recorded: experiment
  shows LARS-type path-tracking failure, not abstract-minimizer failure; PDAP/SSN
  solves the same degenerate systems.
- Theorem A referee pass #4 2026-06-12: math CONFIRMED (all constants exact:
  C²ℓ³/12, power mean, C·ℓ*^(3/2)/(√12(n+1)); numbers reproduced verbatim); fixes
  applied: (a) p=2 collapse is 2 atoms in d=1 (2d in dim d) — my "≤4" was the d=2
  figure; (b) cone_fit α/2-vs-α shift inconsistency = silent half-strength
  regularization — FIXED (shift α for ½-scaled LS, matched to fused lasso); after
  fix the smooth-target marginal signed wins disappeared (they WERE the bias);
  (c) tightness added (equispaced staircase achieves Θ(1/ε) — separation two-sided);
  (d) affine-term parenthetical; (e) selection factor honest range ≈1.2–5×
  (2.3–3× at 5% slices). Referee also confirmed both solver dualities (fused-lasso
  box-dual KKT; PAVA endpoint-shift complete-the-square).
- THEOREM A 2026-06-12 (d=1 head separation; docs/research/theory-ladder.md; experiment
  docs/research/scripts/t5_d1_gradient_sparsity.py with exact solvers): on
  Hessian-SATURATED targets (V″ = C off m shocks — the LQ/value-function class),
  cone+head = m atoms exact vs signed PWL n(ε) ≳ C·ℓ*^(3/2)/ε — unbounded separation
  at p=1 (10-line proof: per-cell slope-vs-constant + power mean). At p=2 collapses
  to ≤ 4 atoms (head emulation) ⟹ at repo's power the observed sparsity gap must be
  SELECTION. Numerics: staircase 123/56 signed vs 5/5 cone (2%/5%, noiseless);
  noise σ=0.15: 28 vs 12 (the ~2× selection factor, isolated); smooth non-saturated
  target: NO separation — head advantage is a saturation phenomenon (matches GMQ
  attenuation / T4(d) story). Two harness bugs found & fixed before trusting numbers:
  cone atom-count must count jumps of m = CΔx − Δv (not Δv — slope counts as fake
  jumps); matched-accuracy thresholds must be absolute (relative-to-best forces
  interpolation at σ=0). Open T5 core sharpened: can PDAP-greedy DISCOVER the 4-atom
  paired head emulation at p=2? (plausibly not ⟹ algorithmic separation.)
- GRADIENT TRAINING amendments 2026-06-12 (user-prompted; data contain ∇V): emulation
  verdict survives (function identity ⟹ capacity separation impossible at p=2 in any
  Sobolev norm; signed emulates the head with 4 relu² atoms ⟹ n_signed ≤ n_semic + 4 —
  sparsity claim MUST be a PDAP-selection theorem, T5); NEW Lemma G: C^(1,1)
  activations ⟹ Lip(∇net) ≤ ‖σ″‖M + C ⟹ gradient-L¹ accuracy ε at a jump J, length ℓ
  costs M ≳ J²ℓ/(‖σ″‖ε) for BOTH models — gradient data makes switching sets expensive
  at finite ε (explains issue #18 regime; predicts mass inflation in both models once
  pendulum data reaches the switching set); semiconcave model under gradient training
  = MONOTONE VECTOR-FIELD regression (∇V = Cx − ∇g, ∇g monotone); d=1 T5 via ISOTONIC
  REGRESSION (unique, PAV, atoms = active blocks) vs two-sided fused-lasso signed fit
  — most realistic path to rigorous T5; certificate must be re-derived in gradient
  pairing (w·σ′ atoms, angular sensitivity).
- Curvature-price study COMPLETE 2026-06-11 (scripts/investigation/
  t3_curvature_price.py, two-phase min-λ LPs, residuals 0.00%): π(junction) = 2.37
  (finite — head-completeness with measured price, C′ ≈ C + 1.19); v₂ mollified:
  THRESHOLD λ = 0 exactly for δ ≥ 0.3, then 0.29 (δ=0.2) → 2.11 (δ=0.1), α ≈ 2.9;
  v₂ sharp: λ grows with mesh 19.47 (n=45) → 25.12 (n=61) — consistent with π = ∞
  per the structure theorem. Critical sharpness δ* ≈ 0.25 below which the quadratic
  head becomes load-bearing. COMPUTE LESSON: LP normal equations are dense (every
  cell pair coupled by a line) ⟹ per-iteration cost cubic in cells (n=91 = 11× n=61);
  iteration count on near-degenerate sharp targets unpredictable (n=76 phase 1 alone
  >77 min, killed); coarsening check n=45 delivered the mesh signal in 87 s. Policy:
  design cheap + time-box, never trust point forecasts; beware orphaned spawn workers
  (pkill -f misses them — they don't carry the script name; kill by PID).
- Opus referee pass #2, 2026-06-11 (weak-* lemma + LP logic): lemma UPGRADED TO
  PROVED after supplying the missing decomposition paragraph (Arzelà–Ascoli on kept
  atoms FIRST, then finite-dim affine pinning; needs Ω full-dimensional) — structure
  theorem solid (constant density on whole hyperplanes; fixed finite μ); boundary
  theorem re-proved qualitatively WITHOUT Fourier for d=2; d≥3 still conditional on
  per-face densities J_F (t2 gaps 1–2); C^(1,1)-activation no-kink claim = "most
  bulletproof in the program". BLOCKER caught in my LP reading: "D′ convergence ⟹
  pairing passes" is FALSE (counterex (n/2)1_{[0,1/n]} vs 1_{[0,∞)}); rescue = positive
  measures + vague convergence + Portmanteau, REQUIRES Δg a.c. (true for smoothed
  junction, FAILS for sharp targets — so sharp-junction/v₂-in-closure is genuinely
  open). "T3-pure FALSE" weakened to "on the sampled line family" (all-lines needs
  ψ-mollification + ∂Ω tightness). Crofton identity verified (π normalization, ℓ∩Ω
  truncation OK); Farkas duality exact in discretization; head algebra C′ = C + λ/d
  exact; mesh-uniformity of λ uncertified. Weakest point: LP→theorem passage for the
  pure-cone separation (3 stacked links). All fixes applied to theory-ladder.md.

- PROBLEM FORMULATION LOCKED 2026-06-16 (user-approved). CLASS: V = (C/2)|x|² − g,
  g convex, on bounded Ω⊂ℝ^d, gradient-L² metric N(model,ε) = min neurons at relative
  grad error ε. Models = repo signed (free c, affine head) vs cone (c≥0, QUADRATIC
  head). Penalty = nonconvex log (γ>0) / ℓ1 (γ=0). TWO TIERS: Tier-1 = g a nonneg sum
  of K ReLU ridges (saturated/switching, sharp); Tier-2 = smooth convex g (open,
  d-sensitive). STANDARD = ratio N(signed)/N(cone). Penalty role: the divergence is
  REPRESENTATIONAL (best-n-term, penalty-independent); penalty enters only ACHIEVABILITY
  (D2). Dimension: lower bound ALL d (proved); cone upper bound clean all d for Tier-1,
  open for Tier-2. Directions: D1 = close Tier-1 two-sided theorem (DONE below);
  D2 = achievability under log penalty (PDAP attains the count); D3 = Tier-2 smooth g
  in d≥2 (bound N_ridge(g,ε); the d-frontier). Recommendation order D1→D2→D3.
- THREE DIRECTIONS (user, 2026-06-16, go one-by-one): D1 smooth/kernel (matern,gaussian);
  D2 ReLU² (well-studied in MasterThesis/representation theorem, simple Jacobian/Hessian);
  D3 sigmoidal/softplus (Barron space). Started with D2.
- D1 (smooth/kernel) DONE 2026-06-16 (docs/research/direction1-smooth.md). The head
  advantage for smooth activations is SMOOTHNESS-GRADED. (1) RESTRICTION LEMMA (rigorous):
  d-D σ-ridge sum restricted to a line = 1-D σ-ridge sum with ≤n atoms, Q→1-D quadratic
  ⟹ N_σ^d(Q,ε) ≥ N_σ^1(t²,cε); d-D head advantage ≥ 1-D, and 1-D tractable. (2) DICHOTOMY
  (rigorous): non-poly σ ⟹ t² not a finite ridge combo ⟹ N→∞ ⟹ head advantage DIVERGES
  for ALL non-poly σ (gaussian,matern,softplus,tanh) — UNLIKE ReLU² (poly,κ=0). So smooth
  advantage is REAL (representational), not just algorithmic. (3) RATE graded by σ's OWN
  smoothness: ReLU C⁰ κ=1 (proved) > matern-ν C^s κ~ε^{-1/s} polynomial > C^∞ gaussian/
  softplus/tanh κ→0 (spectral/polylog, Mhaskar) > poly κ=0. Measured GREEDY κ respect the
  ordering: ReLU 1.0 > matern 0.86 > gaussian 0.56. (4) CAVEAT (ReLU² lesson): measured κ
  are GREEDY upper bounds; for C^∞ σ the FUNDAMENTAL (best-n-term) advantage may be only
  log, big empirical gap = greedy/solver (like ReLU²). MATERN = sweet spot (smooth enough
  for gradient fitting, rough enough for genuine κ>0). OPEN: (i) prove 1-D matern lower
  bound N_σ^1(t²,ε)≳ε^{-1/s} (lifts via restriction lemma to a d-D theorem); (ii) measure
  best-n-term (non-greedy) for gaussian/softplus to separate greedy from fundamental
  (predict polylog). NEXT: matern lower bound, or D3 (sigmoidal/Barron).
- D2 (ReLU²) DONE 2026-06-16 (docs/research/direction2-relu2.md). KEY FINDING: for ReLU²
  the semiconcave advantage is ALGORITHMIC not representational. (1) Hessian of atom =
  2H(w·x−b)wwᵀ ⟹ cᵢ≥0 ⟺ D²V⪯CI ⟺ C-semiconcave EXACTLY (cone class = exactly C-semiconcave
  ReLU² fns). (2) NO diverging capacity gap: quadratic exactly = 2d atoms via (w·x)₊²+
  (−w·x)₊²=(w·x)², so N(signed)≤N(cone)+2d, κ_σ^bestnterm=0 (vs ReLU Θ(1/ε)). Ref Yang-Zhou
  optimal_rate_Relu^k.pdf: H^α⊆F_σ2 for α>(d+5)/2, rate N^(−1/2−5/(2d)). (3) BUT observed
  gap is GREEDY EMULATION FAILURE: confirmed (scripts/quadratic_cost_by_activation, true
  ReLU² relu power=2, pure quadratic) κ_σ^greedy≈+0.50 (n grows 6→16; at n=4 only ε=0.53,
  exact 2d emulation never found). Mechanism: PDAP inserts single max-correlation atom but
  emulation is symmetric PAIR whose one-sided halves each poorly match symmetric quadratic
  ⟹ greedy lays many one-sided bumps; head hard-codes the pair. CONCLUSION: squared
  activations = WRONG place for fundamental semiconcave advantage (cone only fixes a solver
  pathology). Fundamental (capacity, diverging) advantage lives in sub-quadratic-growth
  activations: ReLU (κ=1 proved), smooth/sigmoidal (D1/D3, κ>0). OPEN: prove κ_greedy>0.
  NEXT: D1 (smooth/kernel, diverging capacity, generalizes Thm A-d) or D3 (sigmoidal/Barron).
- ACTIVATION GENERALIZATION 2026-06-16 (user: ReLU not ideal — we do gradient fitting,
  use smooth/squared activations; want useful/general result). KEY: head represents
  V's quadratic part EXACTLY with O(d) params for ANY σ ⟹ gap governed by N_σ(Q,ε) =
  σ-cost of a nondegenerate quadratic. Define κ_σ = d(log n)/d(log 1/ε) = activation's
  QUADRATIC-COST EXPONENT. MEASURED (scripts/quadratic_cost_by_activation.py, pure
  quadratic so n_cone=0, n_signed=N_σ(Q,ε), repo PDAP): ReLU κ=1 (proved Θ(1/ε));
  matern52 κ≈0.86; gaussian κ≈0.56; gelu²/silu² κ≈0 (FLAT 22 atoms, error 0.018 — quadratic
  CHEAP/bounded for squared activations); softplus/tanh plateau ~0.24 (can't fit quadratic
  well — expensive but accuracy-ceiling-confounded, need more iters). GENERAL STATEMENT
  replacing ReLU Θ(1/ε): n_signed/n_cone ≳ Θ(ε^−κ_σ)/N_σ(g,ε); head advantage DIVERGES
  iff κ_σ>0 — TRUE for kernel/saturating (matern,gaussian,ReLU), FALSE for polynomial
  (squared) where κ_σ=0 and the gap is SELECTION not capacity. Theorem A-d (ReLU Θ(1/ε))
  is just the κ_σ=1 endpoint. This is the useful activation-faithful form. OPEN: prove
  N_σ(Q,ε) lower bound for kernel activations (matern/gaussian — softened moment-of-
  inertia via localized σ'' slabs, exponent κ_σ); resolve softplus/tanh ceiling.
- D1 DONE 2026-06-16 (docs/research/sparsity-gap-real-setting.md, "Theorem A-d ReLU
  κ_σ=1 endpoint, Tier-1 two-sided"): for V = (C/2)|x|² − Σ_{i≤K} cᵢ(wᵢ·x−bᵢ)₊ (cᵢ≥0), N(cone,ε)=K (exact,
  ε-independent), N(signed,ε)=Θ(1/ε), ⟹ ratio = Θ(1/(Kε)) DIVERGES, all d. PROOF:
  cone trivial (in class); signed LOWER = arrangement cells (≤κ_d n^d) + moment-of-
  inertia (ball min, c_d=(d/(d+2))ω_d^(−2/d)) + power-mean (exp 1+2/d) on a g-affine
  ball ⟹ ‖∇f−∇V‖_{L²(B)} ≥ C√c_d κ_d^(−1/d) vol(B)^(1/2+1/d)/n; C cancels in relative
  metric ⟹ n ≥ κ(d,Ω)/ε; signed UPPER = coordinatewise ReLU staircase grids O(1/ε)
  matching ⟹ Θ(1/ε) tight. Affine head subsumed (kills const v₀ not linear Cx).
  Penalty-independent (best-n-term). Self-refereed inline (a-fortiori sign, cell count,
  c_d, power-mean, normalization — no gap); external referee OFFERED not auto-run.
- THE WHY (quantifiable, real-data confirmed) 2026-06-16, scripts/why_curve_vdp.py,
  docs/research/sparsity-gap-real-setting.md headline: V = (C/2)|x|² − g, g convex.
  ReLU sum is piecewise-linear ⟹ Hessian 0 a.e. ⟹ signed must build V's curvature
  (D²V≈−CI) from Θ(1/ε) flat pieces (Theorem A-d, proved Θ(1/ε) all d); semiconcave
  supplies curvature with ONE param C (head), atoms carry only O(1) convex correction.
  ⟹ n_cone=O(1), n_signed=Ω(C·vol^(1/2+1/d)/ε), RATIO n_signed/n_cone = Θ(1/ε) DIVERGES
  (not a constant factor). CONFIRMED on real VDP via repo PDAP: slope d(log n)/d(log 1/ε)
  = cone −0.00, signed +1.35 (n_cone flat, n_signed~ε^−1.35); matched-acc ratio 5×@0.35
  → 8×@0.25, climbing. This is the un-confounded evidence (two n(ε) curves, NO
  subtraction) and VINDICATES Theorem A-d as the mechanism — the bulk-subtraction
  "retraction" below was about a confounded PROBE, not about A-d (which stands and is
  now the confirmed why). The curvature argument (free for head, Θ(1/ε) for ReLU) is
  the clear quantifiable WHY the user asked for.
- ⚠ Bulk-subtraction PROBE confounded (retraction re decomposition framing only):
  the "capacity is everything / selection≈0"
  verdict was an ARTIFACT of a RIGGED synthetic target (V = head + 3 DICTIONARY ridges,
  so subtracting head trivially left 3 atoms — circular). REAL VDP via REPO PDAP
  (scripts/selection_isolation_repo_pdap.py, matched-accuracy α-sweep): at relGrad 0.30,
  cone=2, signed=12, signed-GIVEN-THE-BULK=44 (γ=0); cone=1/signed=10/sgn-nobulk=28
  (γ=1). Subtracting the cone's learned C̃ head made signed WORSE not sparser ⟹ head is
  NOT the separable driver on real data. CORRECTED MECHANISM: the advantage is STRUCTURAL
  & JOINT — the whole semiconcave parametrization (½C‖x‖²−g, g convex via c≥0) matches
  the VDP value fn (smooth strongly-convex bulk minus convex correction); does NOT
  decompose additively into head-capacity + cone-selection. The "capacity vs selection
  decomposition" FRAMING ITSELF was the error. STANDS: (a) empirical 3–11× gap; (b)
  Theorem A-d (docs/research/sparsity-gap-real-setting.md) = proved Θ(1/ε) signed lower
  bound for the smooth bulk (moment-of-inertia/arrangement-cells argument, all d) — a
  CONTRIBUTING bound, not the whole effect; (c) gap is a joint-structure property.
  LESSON: never validate a mechanism on a target constructed to exhibit it; test on real
  data with the real solver. Original (now superseded) synthetic finding:
- [SUPERSEDED] SELECTION-vs-CAPACITY 2026-06-16 (docs/research/sparsity-gap-real-setting.md,
  scripts/selection_isolation_d2.py). Real setting = experiments/activationsearch:
  d=2 VDP, gradient-aug (h1), nonconvex log penalty (γ swept), p=1 ReLU; cone 3–11×
  fewer neurons than signed (softplus 1 vs 11, tanh 2 vs 19, matched accuracy).
  CONTROLLED ISOLATION (synthetic d=2 semiconcave, 3 models, matched-head fact:
  signed+head ⊇ cone ⟹ any gap is 100% selection): cone=5/signed+head=4/signed=138
  at γ=0; cone=3/sh=3/signed=59 at γ=5. ⟹ SELECTION GAP ≈ 0 (even slightly negative);
  CAPACITY GAP = +56..+134 = THE WHOLE EFFECT. So the "selection primary" hypothesis
  (user's and mine) is REFUTED by controlled experiment. The sparsity advantage is
  the QUADRATIC HEAD (capacity, Theorem A-d): at p=1 (no emulation) signed ReLU needs
  Ω(ε^(-d/2)) atoms for the smooth ½C‖x‖² bulk; cone gets it free from the head; atoms
  then carry only the thin switching set. Nonconvex penalty sharpens BOTH to exact
  recovery equally, doesn't favor cone. c≥0 = the semiconcavity/well-posedness prior,
  NOT the sparsity driver. NEXT: formalize Theorem A-d (ReLU n-term lower bound for a
  quadratic in H1, d≥2); d=1 ℓ1 selection dichotomy demoted to appendix (secondary,
  ≈0 here). Optional: add quadratic head to repo SignedModel, rerun activationsearch
  to confirm n_signed+head ≈ n_cone on real data.

**Why:** user wants a provable framework for the empirically observed sparsity of the
semiconcave model; this is the thesis-theory backbone.
**How to apply:** when discussing approximation theory, activations, or model-zoo
extensions, anchor to the T1–T5 ladder and the R-norm yardstick; key references live in
/Users/chaoruiz/Documents/NotePaper/MasterThesis/ (paths in the doc). Related:
[[project_pdap_consolidation]].
