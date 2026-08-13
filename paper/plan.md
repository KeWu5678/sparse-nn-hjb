# Current manuscript plan and authority

Last updated: 2026-08-13

## 1. Status and authority

The normalized nonhomogeneous objective is implemented in
`paper/paper_0805.tex` and is the current formulation.

- `paper/paper_0805.tex` is the working manuscript.
- This file records current decisions and current state. It replaces the
  obsolete root `prompt.md` and the historical rewrite proposal previously
  stored here.
- `paper/notation.md` is the notation contract.
- `paper/term.md` is the controlled terminology list.
- Historical reviews and the comparison draft are evidence, not authority.

Authority is scoped as follows:

1. `CONTEXT.md` defines project-wide domain meanings and vocabulary.
2. This file defines the manuscript formulation, scope, and current decisions.
3. `notation.md` and `term.md` refine the manuscript's symbols and controlled
   vocabulary without contradicting the first two sources.
4. `docs/adr/` records implementation and experiment decisions.

Conflicts between these sources must be resolved explicitly. No source silently
overrides another outside its stated scope.

## 2. Non-negotiable decisions

### 2.1 Adopted nonhomogeneous objective

Use

`J(μ) = L(μ) + α Φφ(μₚ)`

with

- `wₚ(ω) = 1 + |ω|ᵖ`,
- `μₚ = wₚ μ`,
- `Kₚ(ω) = K(ω)/wₚ(ω)`, and
- `𝒩μ = ∫ K dμ = ∫ Kₚ dμₚ`.

For distinct finite nodes, the penalty is

`α Σₙ φ(wₚ(ωₙ)|cₙ|)`.

There is no additive `β‖μ‖Mp` term in the revised nonhomogeneous theory or
Algorithm 1. The comparison functional `J₀ = L + αΦφ` is retained only for
the escape examples.

### 2.2 Activation assumption

All activation hypotheses are centralized in Assumption 3.1.

- `ρ ∈ C¹(ℝ)`.
- Point (1) bounds `ρ` with exponent `s₀`.
- Point (2) bounds `ρ′` with exponent `s₁−1`.
- `s₁ ≥ max{s₀,1}`.

Theorem 3.16 assumes all of Assumption 3.1 but its proof uses only point (1),
which is why existence requires `p>s₀`. Lemma 3.17 derives continuity of
`K:Ω→H¹(D)` and full-atom growth of order `s₁`; later gradient-based results
use `p>s₁`.

Do not replace these activation-level hypotheses with an abstract assumption
about `K`. Plain ReLU is outside this smooth gradient-fitting framework;
`ReLUᵏ` with `k≥2` remains covered.

### 2.3 Scalar and measure penalties

- `Lφ = φ′(0+)` is finite and strictly positive.
- `φ` is nondecreasing, concave, coercive, and strictly increasing under the
  stated assumptions.
- The bounded-interval curvature constant is
  `γ_A = −sup{φ″(t): 0<t≤A}>0`.
- The relaxed measure penalty is
  `Φφ(ξ) = Lφ‖ξ_cont‖TV + Σω∈atom(ξ) φ(|ξ({ω})|)`.
- Use “continuous part,” never “diffuse part.”

The homogeneous fractional penalty has infinite right slope only as a
structural endpoint. Never substitute `Lφ=∞` into the finite-slope results.

### 2.4 Scope discipline

Keep changes surgical. Mathematical proofs, Algorithm 2, data generation, and
reported numerical values remain unchanged unless a specific inconsistency is
identified and approved. Generated reports and the manuscript may be refreshed
from existing records when their source or wording changes. Do not alter the
homogeneous proofs without a separate mathematical review.

### 2.5 Writing rules

- The deliverable is standalone: do not refer to an earlier version, source
  file, revision process, or comparison draft in the manuscript.
- Preserve correct mathematical material and thesis-level intermediate proof
  steps. Delete or compress only when it improves clarity without removing a
  needed argument or dependency.
- Define every variable before use.
- Use established mathematical terminology, direct prose, and explicit
  parameters instead of invented labels or vague descriptions.
- In the numerical section, present each investigation as question, result,
  and illustration; do not narrate experimental bookkeeping.
- Keep stable symbols in `notation.md`.
- Add a term to `term.md` only when it is an established mathematical
  concept, appears in the body of a definition, theorem, lemma, corollary, or
  remark, and occurs in at least three distinct places in the paper.
- Do not invent result names or import names from the comparison draft.
- Do not add a new result title without approval.
- Number an equation only when it is referenced later.
- Refer to every theorem, lemma, proposition, corollary, assumption, and
  remark by number in the prose.
- Use the paper's notation: `μₚ`, `Kₚ`, `P̄ₚ`, `wₚ`, and `N_cov`.
- Reserve `ν` for the population measure and `η` for the control-energy
  coefficient.
- When reporting mathematics in the terminal, prefer readable Unicode over
  raw TeX source.

## 3. Current mathematical backbone

The implemented chain is:

1. Theorem 3.9 gives the disjoint-test representation of `Φφ`.
2. Corollary 3.10 gives weak-* lower semicontinuity and bounded TV sublevels.
3. Theorem 3.13 identifies `Mₚ(Ω)` isometrically with `M(Ω)` through `μₚ`.
4. Lemma 3.15 proves strong `L²` convergence by adapting the original cutoff
   proof to `μₚ` and `Kₚ`.
5. Theorem 3.16 proves existence using normalized weak-* compactness and weak
   identification of the gradient.
6. Lemma 3.17 derives the full `H¹` atom bound from Assumption 3.1.
7. Theorem 3.18 gives first-order optimality conditions in normalized
   coordinates.
8. Lemma 3.20 localizes atomic and continuous support in a compact superlevel
   set.
9. Theorem 3.21 removes the continuous part, separates normalized atoms, and
   gives a covering-number atom bound.
10. Corollary 3.22 gives attained finite-network equality and an a priori
    neuron bound for global minimizers.
11. Theorem 3.23 gives quantitative insertion decrease, square-summability of
    the threshold excess, the best-iterate `n⁻¹ᐟ²` rate, and a bounded search
    radius.
12. Theorem 3.26 gives a sufficient condition for measure-local optimality of
    a finite network.

The insertion result proves decay of the threshold excess. It does not prove
convergence of the iterates to a global minimizer or an objective-gap rate.

## 4. Actual current result map

These numbers come from the current `paper_0805.aux`. Do not use the old
numbering map from the historical plan.

| Number | Current role |
| --- | --- |
| Assumption 3.1 | `C¹` activation; value and derivative growth |
| Assumptions 3.2–3.4 | scalar slope, concavity/coercivity, and curvature |
| Lemma 3.6 | subadditivity and tangent bound |
| Proposition 3.7 | second-order scalar bounds |
| Lemma 3.8 | disjoint small-mass test construction |
| Theorem 3.9 | disjoint-test representation of `Φφ` |
| Corollary 3.10 | weak-* lower semicontinuity and TV-sublevel bound |
| Examples 3.11–3.12 | tanh escape and softplus nonattainment |
| Theorem 3.13 | weighted-measure normalization |
| Definition 3.14 | local minimizer in the `Mₚ` norm |
| Lemma 3.15 | strong `L²` continuity under normalized weak-* convergence |
| Theorem 3.16 | existence on the unbounded parameter domain |
| Lemma 3.17 | derived `H¹` continuity and growth of `K` |
| Theorem 3.18 | normalized first-order optimality conditions |
| Remark 3.19 | compactness of positive superlevel sets |
| Lemma 3.20 | support and normalized-atom bounds |
| Theorem 3.21 | finite support, separation, and atom count |
| Corollary 3.22 | finite-network equality and global neuron bounds |
| Theorem 3.23 | quantitative insertion and search radius |
| Remark 3.24 | boundary nondecay example |
| Corollary 3.25 | escaping directions in the `H²` regime |
| Theorem 3.26 | sufficient condition for local optimality |

Inserting or deleting any numbered result changes this map. Check the
generated auxiliary file after every structural edit.

## 4a. Section 4 result map

| Number | Current role |
| --- | --- |
| Assumption 4.1 | `k`-homogeneity of `K` |
| Remark 4.2 | `k ≥ 1` for rescaling, `k ≥ 2` for gradient fitting |
| Lemma 4.3 | general `ξ,ζ` radial rescaling; induced exponent and `C_{k,ξ,ζ}` |
| Remark 4.4 | quadratic case, explicit `Cₖ`, `α = Cₖ α_orig` |
| Lemma 4.5 | `ReLUᵏ` dictionary: homogeneity, gradient, continuity, `0 < C_S < ∞` |
| Definition 4.6 | extended measure penalty `Φ_{ψₖ}` on `M(Sᵈ)` |
| Lemma 4.7 | well-definedness, positivity, `Φ ≥ ‖μ‖^q`, merging inequality |
| Definition 4.8 | total-variation local minimizer; reduced width `N = #atom(μ)` |
| Remark 4.9 | countably atomic measure of finite penalty |
| Proposition 4.10 | finite atomic truncations; `i_disc = i_meas` |
| Theorem 4.11 | atomwise stationarity; `c_min(μ̄)`; the `B = 0` case |
| Remark 4.12 | three consequences of `ψₖ'(0⁺) = ∞`; zero is a TV-local minimizer |
| Theorem 4.13 | existence, attainment, and the uniform `N_max` for every global minimizer |
| Corollary 4.14 | finite support of TV-local minimizers, a posteriori bound |
| Remark 4.15 | empirical analogue |

Retired labels: `prop:homogeneous_reformulation`, `cor:singular_penalty`,
`cor:discrete_existence`. Their live references were repointed, and
`prob:homogeneous` now exists.

## 5. Algorithm 1 contract

The empirical objective is

`Jᴹ(ω⃗,c) = lᴹ(μω⃗,c) + αΣₙφγ(wₚ(ωₙ)|cₙ|)`.

The empirical derivative and insertion quantities are:

- `Pᴹₚ,μ(ω) = Pᴹ_μ(ω)/wₚ(ω)`,
- insertion condition `|Pᴹₚ,μ(ω)|>αLφ`,
- certificate violation
  `Δ(μₜ,ω)=max{|Pᴹₚ,μₜ(ω)|−αLφ,0}`,
- per-candidate normalized curvature `‖Kᴹₚ(ω)‖²`, and
- outer coefficient
  `c(ω)=−Δ(μₜ,ω)sign(Pᴹₚ,μₜ(ω)) /
  (wₚ(ω)‖Kᴹₚ(ω)‖²)`.

No numerical global bound substitutes for the theorem's uniform `Bₚ`. The
per-candidate curvature minimizes the same increment bound exactly and implies
the printed uniform decrease because `‖Kᴹₚ(ω)‖²≤Bₚ²`.

Algorithm 1 must:

- sample and locally refine nonzero directions;
- search each retained ray over
  `exp(-3) ≤ λ ≤ min{R(μ_t), exp(5)}` when the theorem radius is available,
  and up to `exp(5)` otherwise;
- select one candidate per outer iteration;
- merge an exact repeated location;
- insert using the candidate-local coefficient above; and
- accept the finite-dimensional correction only when it does not increase
  the post-insertion objective.

The one-candidate decrease estimate applies to every accepted candidate. The
theorem's square-summability and best-iterate rate additionally require an
exact global maximizer. The practical multistart L-BFGS search is not covered
by those rate conclusions without a quantitative search-accuracy condition;
do not call its returned point an argmax or maximizing candidate.

Do not modify Algorithm 2.

## 6. Completed implementation details worth preserving

- Lemma 3.8 explicitly handles atomic and nonatomic mass in its small-mass
  partition and constructs its cutoffs.
- Theorem 3.9 explicitly handles finite-to-countable atomic sums, large-atom
  localization, and the continuous part.
- Corollary 3.10 states mutual singularity before adding variation norms.
- Lemma 3.15 uses the original cutoff proof, with only the substitutions
  `μ→μₚ` and `K→Kₚ` plus the normalized tail estimate.
- Theorem 3.18 treats both signs of an atomic coefficient perturbation and
  handles the continuous part separately.
- Lemma 3.20 states the normalized atom bound used downstream.
- Theorem 3.21 selects the continuous-part concentration point from the
  intersection of all required full-measure sets.
- Corollary 3.22 displays the repeated-location merging inequality.
- Theorem 3.23 distinguishes exact global insertion from practical local
  search and states precisely what converges.
- Algorithm 1 uses a candidate-local guaranteed-decrease coefficient and
  rolls back an increasing correction.
- The homogeneous mathematics, data generation, and Algorithm 2 remain outside
  the current numerical-conformance repair.

## 7. Editing and review protocol

### 7.1 Change-justification gate

Before changing an existing argument, record:

```text
Obligation:
Baseline argument:
Exact step that fails under the new assumptions:
Minimal repair:
Downstream results requiring the repair:
```

If “Exact step that fails” is empty, do not replace the argument. Adapt it
only where notation or hypotheses changed.

Use the least powerful sufficient tool, in this order:

1. notation substitution;
2. local modification;
3. an existing lemma;
4. a standard citation;
5. new machinery.

Do not generalize unless a numbered downstream result needs the additional
generality.

### 7.2 Agent allocation

For a tightly coupled proof, use one writer and multiple independent
reviewers. Do not split a proof merely because it is long.

Every reviewer checks:

- mathematical correctness;
- exact hypotheses and conclusion;
- defined variables and stable notation;
- necessity and minimality.

Give reviewers different primary failure modes, for example:

- independent proof reconstruction;
- comparison with the prior proof;
- cross-result and notation consistency; or
- limiting cases and counterexamples.

Reviewers return evidence and proposed repairs. One writer owns all edits.
Delegation never replaces the orchestrator's comparison with the prior proof and final
proof reconstruction.

### 7.3 Four independent acceptance gates

A change is accepted only after four separate judgments:

1. Correct: every inference is justified.
2. Consistent: notation, hypotheses, numbering, and downstream uses agree.
3. Necessary: the previous argument genuinely fails or the statement changed.
4. Minimal: no weaker edit achieves the same obligation.

Compilation does not establish any of these four properties.

## 8. Numerical-conformance decisions

The following decisions govern the paper-conforming studies and their report:

- The reported moment-order study uses the existing sequential records under
  `rawdata/logs/multirun/{vdp,pendulum}/paper_log_penalty/p_study_sequential`,
  with one insertion per outer iteration and `T_out = 150`.
- The earlier batch records under `p_study/` remain evidence and are not
  overwritten, but they do not supply the reported moment-order figure.
- The practical candidate search is multistart L-BFGS. It selects the best
  candidate found; it does not certify the exact global maximum required by
  the rate statement.
- The nonzero radial search retains the numerical lower bound `exp(-3)` and the
  fixed upper comparison bound `exp(5)`, tightened by the theorem radius when
  available. The algorithm does not evaluate the origin separately.
- Stable configuration values such as `radial_cap="fixed"` remain unchanged;
  mathematical prose uses the controlled terms in `CONTEXT.md` and `term.md`.

## 9. Verification checklist

### 9.1 Mathematical

- Reconstruct every changed proof independently.
- Check atomic and continuous components separately whenever `Φφ` is used.
- Check every factor of `½` and `α` in separation and insertion estimates.
- Check that each compact superlevel threshold is strictly positive.
- Check the normalized/original coefficient conversion in every insertion.
- Check that Theorem 3.23 claims only threshold-excess convergence.
- Check `V=0` separately in the global-minimizer corollary.

### 9.2 Notation and language

- No undefined variable.
- No obsolete `ass:H1-growth` reference.
- No “diffuse part.”
- No tilded normalized measure or generic normalized `ν`.
- No invented result name.
- Every stable symbol is in `notation.md`.
- Every `term.md` entry occurs in at least three paper locations.
- Every result is cited by its actual number.
- Every numbered equation is referenced later, except inherited equations in
  explicitly protected text.

### 9.3 Build and structure

- Compile `paper_0805.tex` with `latexmk`.
- Inspect the log for undefined references and citations, duplicate labels,
  and layout warnings.
- Compare the auxiliary-file result numbers with Section 4.
- Confirm every citation has a tracked bibliography entry.
- Run the repository pre-commit hooks and full pytest suite.
- Regenerate the tracked PDF from the tracked source and bibliography.
- Confirm `make paper-p-study` targets the same sequential record directories
  used by `p_study_figure.py`.

### 9.4 Scope

- Review the diff before every handoff.
- Trace every changed line to an approved decision in this plan.
- Keep local run records intact unless a separate destructive operation is
  explicitly approved.

## 10. Next safe action

After each manuscript or generator change, rebuild the affected derived
artifact and run the verification checklist before handoff.
