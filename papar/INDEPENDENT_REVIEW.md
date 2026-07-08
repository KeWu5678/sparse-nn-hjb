# Independent Referee Report — Sections 3–4

Reviewer: independent (re-derived from source; `REVIEW.md` deliberately not read).
Document: `papar/Mthesis.tex` (line numbers below refer to this file).
Scope: proof correctness of the mathematical core (§3 existence/optimality/finite
support, §4 k-homogeneous reformulation and singular-penalty corollaries), plus an
assessment of whether the machinery can be replaced by standard tools.

---

## 1. Verdict

Sections 3–4 are, with one exception, **mathematically sound and unusually careful**:
the existence proof on the unbounded domain, the finite-support merging argument, the
boundary-representation lemma, and the k-homogeneous algebra all check out line by
line, and every constant I recomputed (Prop. `homogeneous_penalty`, Cor.
`discrete_existence`) is correct. The cited generalizations of Pieper–Petrosyan and the
transcriptions of Bouchitté–Buttazzo, Li et al., and Heeringa are faithful.

**The one real defect** is a gap in Theorem `opt_cdt` (first-order conditions): the
optimality statement on the *continuous* part of the measure is derived by
differentiating the loss along `u = μ̄_cont`, but under the standing assumptions this
directional derivative need not be finite — the gradient kernel `∇ₓσ` is not
`|μ̄|`-integrable (it grows like `|a|^{1/2}` in `L²`), so `𝒩(μ̄_cont)` need not lie in
`H¹`. This gap propagates into Theorem `finite_support` (Step 1), whose merging
competitor relies on `p̄ = -α` holding `μ̄⁺_cont`-a.e. It is **repairable** by adding a
gradient-integrability (half-moment) hypothesis on candidate minimizers, which is weaker
than a moment condition the author already invokes elsewhere.

**Biggest structural concern (not an error):** almost the entire unbounded-domain
apparatus — localization `ρ,ρ'∈L²`, the value-decay lemma, the no-escape condition, and
the `C₀` half of the boundary lemma — is avoided outright by taking `Ω` compact, which
the author's own Remark `rem:sharpness` + §2 Barron embeddings say costs nothing in
approximation power. See §3.

---

## 2. Proof correctness (ordered by severity)

### GAP-1 — Theorem `opt_cdt` (line 785), continuous-part optimality; propagates to Theorem `finite_support`
**Issue.** Claim (2), `p̄ = -α·sign(μ̄_cont)` for `|μ̄_cont|`-a.e. `ω`, is proved (lines
820–828) by taking `u = μ̄_cont`, dividing `J(μ̄+τu)-J(μ̄)` by `τ`, and asserting the
fidelity difference `→ ⟨p̄, μ̄_cont⟩`. That step silently requires
`𝒩(μ̄+τμ̄_cont) ∈ H¹`, i.e. `𝒩(μ̄_cont) ∈ H¹`.
**Why it is not justified.** Under Assumption `ass:ridge`,
`‖∇ₓσ(·,ω)‖_{L²(D)} ~ |a|^{1/2} → ∞`, so `∫_Ω ‖∇ₓσ‖_{L²} d|μ̄|` can be `+∞`. Then
`𝒩(μ̄_cont)` and `𝒩(μ̄_atom)` may each fail to be in `H¹` while their sum `𝒩μ̄` (known
to be `H¹` because `L(μ̄)<∞`) is fine — cancellation is possible. In that case
`L(μ̄+τμ̄_cont) = +∞`, the difference quotient is `+∞`, and the inequality
`-⟨p̄,μ̄_cont⟩ ≤ α‖μ̄_cont‖` cannot be extracted. (Claim (1), and claim (2) *at atoms*,
use only single-atom perturbations `u=δ_ω` with `σ(·,ω)∈H¹` and are fully rigorous.)
**Where it bites.** Theorem `finite_support` Step 1 (line 863) uses exactly
"`p̄ = -α` holds `μ̄⁺_cont`-a.e." to make the first-order term of the merging competitor
vanish (line 910–914). Only `|p̄|≤α` (claim 1) is not enough there — one needs
`p̄(ω̂) = -α`. So the gap is load-bearing, not cosmetic.
**Fix.** Add to the optimality/finite-support theorems the standing hypothesis
`∫_Ω (1+|a|)^{1/2} d|μ̄|(ω) < ∞` (equivalently `∫‖∇ₓσ‖_{L²}d|μ̄|<∞`, the very condition
the loss definition on lines 362–370 flags). Then `𝒩(μ̄_cont)∈H¹`, the direction is
admissible, and both proofs go through verbatim. This is strictly weaker than the
`3/2`-moment already assumed in Remark `rem:no_escape` (line 1166). On a compact `Ω_R`
the issue disappears (`∇ₓσ` bounded), so the compact-domain statements are unaffected.
**Severity: GAP** (true as stated once the moment hypothesis is added; the current proof
is incomplete).

### GAP-2 — Existence (Thm `existence`) and finite support (Thm `finite_support`) are not connected on unbounded `Ω`
Theorem `existence` produces a global minimizer `μ̄` but says nothing about its support.
Theorem `finite_support` needs the no-escape condition `(H)` (line 850) *for that
minimizer*, and `(H)` is only verified a posteriori (Remark `rem:no_escape`) when the
residual `𝒩μ̄-V ∈ H²`, which in turn needs `∫(1+|a|)^{3/2}d|μ̄|<∞` — never established
for the minimizer from Thm `existence`. So on `ℝ^{d+1}` the chain "a global minimizer
exists ⟹ it is finitely supported" is **not closed**. It *is* closed on compact `Ω_R`
(where `(H)` is void). Worth stating explicitly; right now the narrative reads as if
existence + finite support compose, and they do not without extra regularity on the
minimizer. **Severity: GAP (exposition/logic).**

### IMPRECISION-1 — "`p̄ ∈ C(Ω)`" (Remark, line 767) is stated unconditionally
As written this is fine only as *continuity* (which does hold: for fixed `L²` residual,
`ω↦(σ,∇ₓσ)(·,ω)` is continuous into `L²×(L²)^d`). But the reader is likely to read it as
*bounded* continuity; `p̄` is **unbounded** as `|ω|→∞` in general (the gradient kernel
grows like `|a|^{1/2}`), and boundedness/`C₀` genuinely requires the residual to be `H²`
(Lemma `dual_pairing`(1),(3)). The thesis does localize boundedness correctly later, so
this is only a wording risk at line 767 — add "continuous (boundedness requires the
residual in `H²`, Lemma `dual_pairing`)". **Severity: IMPRECISION.**

### Checked and correct (with the reason I am confident)

- **Lemma `firstorderestimate` (subadditivity, `φ(z)≤z`, line 484).** FTC + `φ'`
  nonincreasing; `φ` differentiable and concave ⟹ `φ'` continuous ⟹ FTC valid. Correct.
- **Prop. `secondorderestimate` (line 526).** Integrate `φ'(s)-φ'(0)` against the two-sided
  bounds of `pen3`. Both envelopes correct.
- **Lemma `coercivity` (line 560).** Countable subadditivity + `φ(z)≤z`; coercivity from
  `φ` coercive. Correct; `|μ|(Ω∖atom μ)=‖μ_cont‖` used correctly.
- **Lemma `lsc` (line 590).** The heavy/light split is exactly the Bouchitté–Buttazzo
  Thm 3.3 argument specialized. Heavy part cites their Lemma 3.6 (≤k-atom set weak-*
  closed + `Σφ` l.s.c.) — I verified the source states precisely this. Light part: chord
  bound `φ(z) ≥ (φ(ε)/ε)z` on `[0,ε]` from concavity, `δ_ε→0` from `pen1`; continuous
  part enters with weight `1 ≥ (1-δ_ε)`. Recombination uses `liminf(a+b) ≥ liminf a +
  liminf b` and subadditivity of `Φ₁`. The opposite-sign atom-collision case is covered
  by `φ(|a-b|) ≤ φ(a+b) ≤ φ(a)+φ(b)` (monotone + subadditive). Correct, and the "only
  `pen1,pen2` needed" claim is accurate.
- **Lemma `value_decay` (line 663).** Rotation bound `∫_D|ρ(a·x+b)|² ≤
  (2R_D)^{d-1}|a|^{-1}‖ρ‖²_{L²}` recomputed and correct; bounded-`a`/large-`b` regime
  correct; hence `κ∈C₀`, and `𝒩μ_n ⇀ 𝒩μ̄` in `L²`. Correct.
- **Theorem `existence` (line 696).** Coercivity ⟹ bounded minimizing sequence ⟹ weak-*
  compact (`C₀(Ω)` separable). Value part converges weakly by `value_decay`; gradient
  part `∇𝒩μ_n ⇀ g`, and `g = ∇𝒩μ̄` distributionally via `∫g·ψ = -lim∫𝒩μ_n divψ =
  -∫𝒩μ̄ divψ`. This is where the **distributional-gradient/case-split convention does
  real work**: it yields `𝒩μ̄∈H¹` without any moment assumption. Norms weakly l.s.c.,
  `Φ₁` weak-* l.s.c. Chain of `liminf`s correct. **The existence proof is the strongest
  part of the thesis** and I found no defect.
- **Remark `rem:sharpness` (line 731).** Both counterexamples check out: `tanh` with
  `b_n→∞` gives `μ_n⇀*0` but `𝒩μ_n→1` in `H¹` (fidelity not weak-* l.s.c.); softplus
  `μ_n = ρ(b_n)^{-1}δ_{(0,b_n)}` fits `V≡1` exactly at penalty `→0`, and
  `n^{-1}ρ(n(e·x-t)) → (e·x-t)_+` in `H¹`. Correct and genuinely instructive.
- **Lemma `dual_pairing` (line 1014).** (1) Green `⟨∇r,∇σ⟩ = -⟨Δr,σ⟩+⟨∂_n r,σ⟩_{∂D}`,
  bound `≤ C_{D,ρ}‖r‖_{H²}` correct. (2) The key move — Green transfers the derivative
  off `σ` so both kernels are bounded by `‖ρ‖_∞`, making Fubini against `μ'` legitimate
  *without* gradient integrability; the trace-of-continuous-function identity is used
  correctly (`𝒩μ'∈C(D̄)` by dominated convergence). (3) The `C₀` argument (band `B_j(τ)`
  splitting; `H^{d-1}(∂D∩hyperplane)=0` via continuity-from-above) is correct, including
  the `|a_j|` bounded vs. `→∞` cases and the `|c|=∞` degenerate case. This is a clean,
  correct lemma.
- **Remark `rem:no_escape` (line 1125).** Flat-face limit `p̄(ω_j) → ρ(s_0)∫_F ∂_n r`
  correct (argument constant `≡ s_0` on `F`, `→±∞` off `F`); reduction of `(H)` to
  `‖ρ‖_∞|∫_F ∂_n r| < α` per face is right; the `H²`/`3/2`-moment sufficient condition
  scales correctly (`‖∂²σ‖_{L²} ~ |a|^{3/2}`).
- **Theorem `sc_opt` (line 1192).** Steps 1–6 all verified: strict `φ'(|c̄_n|)<1` from
  `pen3(2)`; uniform gap `sup|p̄| ≤ (1-δ)α` from compact-ball max + `(H)`; competitor
  decomposition norm-additivity `‖c-c̄‖_{ℓ¹}+‖μ̃‖=‖μ-μ̄‖`; fidelity expansion using
  `dual_pairing`(2) for the `p̄`-term and `dual_pairing`(1) for the `C_{ω̄}ε‖μ̃‖`
  remainder; penalty lower bound `Φ₁(μ̃) ≥ (1-γ₁ε/2)‖μ̃‖` from Prop.
  `secondorderestimate`(1); final assembly with the stated `ε` makes the bracket `≥0`.
  Correct. Notably this proof **avoids GAP-1** by using `dual_pairing`(2) directly rather
  than the `μ̄_cont` directional derivative — the same fix that would repair `opt_cdt`.
- **Prop. `homogeneous_penalty` (line 1330).** I recomputed `τ* = (k|c̃|^p)^{1/(kp+q)}`,
  both substituted terms, and the coefficient collapse to `(2/s)k^{-kp/(kp+q)}` with
  `s=2pq/(kp+q)` — all correct. `k=1` recovers Pieper's harmonic-mean `s=2pq/(p+q)` and
  constant `1`; `p=q=2` gives exponent `2/(k+1)` and constant `(k+1)/2·k^{-k/(k+1)}`
  (e.g. `k=2`: `(3/2)2^{-2/3}`). Correct.
- **Prop. `homogeneous_reformulation` (line 1449).** Projection to the sphere: network
  invariance under `(ã,c̃)=(λω̂, ĉ/λ^k)`, and `|c̃|≤|ĉ|` (since `λ>1,k≥1`) with `φ_k`
  nondecreasing gives non-increase of penalty. Two-sided feasibility ⟹ equal optima.
  Correct.
- **Cor. `singular_penalty` (line 1516).** Stationarity at atoms via `±δ_ω` (finite `φ'`
  at `c̄_n≠0`); lower bound `|c̄_n| ≥ (φ')^{-1}(‖p̄‖/α)` and `N ≤ ‖μ̄‖/c_min` on compact
  `Ω`. Correct. The accompanying remark (global bound lost; automatic minimum atom size;
  `μ=0` always a local min) is correct and honest.
- **Cor. `discrete_existence` (line 1596).** Weierstrass per `N` (continuity by
  dominated convergence; coercivity of `φ_k`); uniform `N`-bound via
  `αφ_k'(|c_n^*|) ≤ ‖V‖_{H¹}·sup_ω‖σ‖_{H¹}` and inversion, then `N·φ_k(c_min) ≤
  ‖V‖²/(2α)`. `N_max` and `c_min` recomputed and correct.

---

## 3. Simplification / standard tools

### 3.1 The single biggest simplification: take `Ω` compact
On a compact parameter set `Ω_R`, `∇ₓσ(·,ω)` is bounded in `L²` (continuous on
`D̄×Ω_R`), so
`𝒯μ = (𝒩μ, ∇𝒩μ)` is a **bounded linear operator** `𝓜(Ω_R) → 𝓨 = L²×(L²)^d` that is
weak-*-to-weak continuous. Then:
- Existence is the one-line direct method (bounded minimizing sequence, weak-* compact,
  `𝒯` weak-*-to-weak continuous, `L` weakly l.s.c., `Φ₁` weak-* l.s.c. by Bouchitté–
  Buttazzo). Lemma `value_decay`, the `L²`-localization `ρ,ρ'∈L²`, and the whole
  escape-to-infinity discussion become unnecessary.
- **GAP-1 evaporates** (`∇ₓσ` bounded ⟹ `𝒩(μ̄_cont)∈H¹` automatically), so Theorem
  `opt_cdt` and Theorem `finite_support` need no moment hypothesis.
- The no-escape condition `(H)` is void (the author already says so), so
  `finite_support` and `sc_opt` hold unconditionally.
- Lemma `dual_pairing`(3) and the flat-face analysis of Remark `rem:no_escape` are not
  needed for the theorems (they only serve to *verify* `(H)`, which is vacuous on `Ω_R`).
  Part (2) of the lemma is still a nice way to organize `sc_opt`, but on `Ω_R` even the
  direct interchange `⟨∇r,∇𝒩μ'⟩ = ∫⟨∇r,∇ₓσ⟩dμ'` is legitimate (bounded kernel), so
  Green's identity is optional.

**What is genuinely lost by compactifying `Ω`:** only atoms with unbounded inner weight
`|a|` (arbitrarily sharp/wide Gaussian ridges) and unbounded bias. By the author's own
Remark `rem:sharpness` and the §2 Barron embeddings (`B_σ ↪ B_{ReLU^k}` on bounded `D`),
this costs nothing in approximation power for the activation classes considered. My
recommendation: **make `Ω_R` the primary setting**; present the unbounded-domain results
(Thm `existence`, Lemma `dual_pairing`(3), Remark `rem:no_escape`) as an optional
"localized-ridge" appendix. This removes ~40% of §3.3–3.4 machinery with no loss of
usable conclusions.

### 3.2 Cite, don't re-derive, the l.s.c. lemma
Lemma `lsc` is a special case of **Bouchitté–Buttazzo (1990), Theorem 3.3** — precisely
the theorem Pieper–Petrosyan themselves cite ([5] in their paper) for the same functional.
Hypotheses (H4)–(H6) reduce here to: `φ` l.s.c. (it is continuous), subadditive (Lemma
`firstorderestimate`), `φ(0)=0`. The self-contained proof is correct but reproduces their
argument verbatim; a one-paragraph citation would do, with the current proof demoted to a
remark. (If kept, it is fine.)

### 3.3 Sparsity / finite support vs. standard representer theorems
The *convex* representer theorems — Bredies–Pikkarainen (2013), Fisher–Jerome, **Bredies–
Carioni (2020)** "Sparsity of solutions for variational inverse problems," Boyd–
Schiebinger–Recht (ADCG), Duval–Peyré — give existence + finite support (`≤` number of
measurements) for TV/`ℓ¹`-regularized measure problems and would subsume the `k=1`,
convex-penalty case immediately. They do **not** subsume Theorem `finite_support`: the
nonconvex penalty (strict subadditivity from `pen3(2)`) is exactly what is beyond those
results, and the merging argument is the right tool. So the finite-support theorem is a
legitimate non-standard contribution (following Pieper–Petrosyan), not reinvention. Only
the *existence* and *l.s.c.* scaffolding is reinvention.

### 3.4 Assumption-by-assumption necessity

| Assumption | Where used | Necessary as stated? |
|---|---|---|
| `pen1` `φ'(0)=1` | `φ(z)≤z`; `δ_ε→0` in `lsc`; threshold `|p̄|≤α` | Normalization only (rescale `α`); harmless. |
| `pen2` diff./concave/nondecr./coercive/`φ(0)=0` | subadditivity, coercivity, existence | All used; concavity is the workhorse. Only *differentiability* is needed, **not `C²`**. |
| `pen3(1)` (`γ₁`-convex at 0) | `sc_opt` Step 5 (Prop `secondorderestimate`(1)) | Needed only there. |
| `pen3(2)` (strong concavity at 0) | `finite_support`, `sc_opt` Step 1 | Needed only there; drives strict subadditivity/merging. |
| `ass:ridge`(1) `ρ,ρ'` Lipschitz | local Lipschitz of atoms (`assumption on actfun`) | Used throughout §3.4. |
| `ass:ridge`(2) `ρ,ρ'∈L²` (localization) | `value_decay`, existence on `ℝ^{d+1}` | **Only for the unbounded domain.** Void if `Ω` compact (§3.1). |
| `ρ∈C²`, `ρ''∈L²` | `sc_opt`, Remark `rem:no_escape` (`H²` residual) | Needed only there; natural. |
| `3/2`-moment `∫(1+|a|)^{3/2}d|μ̄|` | Remark `rem:no_escape` (residual `∈H²`) | Natural (`‖∂²σ‖~|a|^{3/2}`); only in a remark, not load-bearing for a theorem. |

**Under-assumed:** Theorem `opt_cdt` needs the half-moment `∫(1+|a|)^{1/2}d|μ̄|<∞`
(GAP-1) that is not stated. Everything else is either exactly used or explicitly local.

### 3.5 Barron material (§2.2)
The Barron–Sobolev embedding (Li et al. Thm 7) and the activation embeddings (Heeringa
Thm 1) are **motivational, not load-bearing** for any §3–4 theorem; they justify using
`ReLU^k` for gradient training and back the "compactness loses no approximation power"
claim. Fine to keep as context; not overkill, but could be trimmed to the two facts
actually used (`B_1^k ⊂ H^k`, and smooth activations embed into the `ReLU^k` scale).

---

## 4. Cited-result fidelity

- **Pieper–Petrosyan, Prop. 3** → thesis Prop. `homogeneous_penalty`. Faithful
  generalization from `k=1` to `k`-homogeneous. Source: `s = 2pq/(p+q)` (harmonic mean),
  penalty `(2α/s)Σ|c|^{s/2}`. Thesis: `s = 2pq/(kp+q)`, penalty
  `(2α/s)k^{-kp/(kp+q)}Σ|c|^{s/2}`; reduces to the source at `k=1`. **Constants verified
  correct.**
- **Pieper–Petrosyan, Thm 4 (sufficient local optimality)** → thesis Thm `sc_opt`.
  Same hypotheses (i) `c̄` local min of `J_ω̄`, (ii) `|p̄(ω)|<α` off the support; thesis
  adds the gradient term and hypothesis (iii) no-escape to drop compactness of `Ω` (the
  source assumes `Ω` compact, their line "Let `Ω` be a compact subset of `ℝ^{d+1}`").
  Faithful.
- **Pieper–Petrosyan, Thm 3 (finite support)** → thesis Thm `finite_support`. Same
  conclusion; thesis removes compactness at the cost of `(H)`. Faithful (modulo GAP-1,
  which is the thesis's own, not a mis-citation).
- **Bouchitté–Buttazzo, Lemma 3.6 & Thm 3.3** → used in Lemma `lsc`. I read the source:
  Lemma 3.6 states exactly "`{λ : #(spt λ) ≤ k}` is sequentially weak-* closed, and
  `λ↦Σg` is weak-* l.s.c. on it for l.s.c. subadditive `g`." Thm 3.3's subadditivity
  hypothesis (H5) matches. **Faithful**, and the thesis's `Φ₁` (continuous part weight
  `φ'(0)=1`) is the correct l.s.c. representation (diffuse mass penalized at the slope at
  0), consistent with the source's singular-integrand treatment.
- **Li–Lu–Mathé–Pereverzev, Thm 6 & 7, Def 1** → thesis §2.2. Source Thm 7:
  `B_1^k ⊂ H^k(Ω)`, `‖f‖_{H^m} ≤ C(Ω,d,k)‖f‖_{B_1^k}`, `0≤m≤k`. Thesis Proposition
  reproduces this verbatim. Thm 6 (`B_1^k` normed) cited correctly. **Faithful.**
- **Heeringa et al., Thm 1** → thesis §2.2 embeddings proposition. Source: two points,
  the `∂²σ∈L¹ ⟹ B_σ↪B_{ReLU}` case and the `C^k` + one-sided Caputo `(k+1)`-derivative
  `∈L¹ ⟹ B_σ↪B_{ReLU^k}` case, with the "distributional derivative a finite measure
  suffices" remark. Thesis transcribes both points and the remark accurately.
  **Faithful.**

---

## 5. Things the author seems to have missed (errors and easy wins)

1. **GAP-1 (top priority).** State the half-moment `∫(1+|a|)^{1/2}d|μ̄|<∞` for
   candidate minimizers in Thm `opt_cdt`/`finite_support`, or (cleaner) reroute the
   continuous-part optimality through Lemma `dual_pairing`(2) as is already done in
   `sc_opt`. Easy win, closes the only real hole.
2. **GAP-2.** Add one sentence acknowledging that on `ℝ^{d+1}` the minimizer from Thm
   `existence` is finitely supported *only if* it also satisfies `(H)` (equivalently the
   `3/2`-moment for `H²`-regularity), which is not proved — so existence and finite
   support compose unconditionally only on compact `Ω_R`.
3. **Structural easy win (§3.1).** Lead with compact `Ω_R`. It removes the moment
   subtleties, voids `(H)`, and turns Lemma `dual_pairing` into an optional convenience —
   with zero cost by the author's own approximation argument. The unbounded/localized-
   ridge theory is a nice optional section, not the backbone.
4. **Wording (line 767).** "`p̄∈C(Ω)`" should say *continuous* (bounded/`C₀` needs the
   `H²` residual). As currently phrased it invites the reader to assume a boundedness that
   fails at infinity.
5. **`sc_opt` hypothesis (2)** (`|p̄(ω)|<α` for all `ω` off the support) is a
   non-degeneracy / strict-complementarity condition that is essentially "assume the hard
   part." That is standard for such sufficient conditions (cf. Duval–Peyré non-degenerate
   source condition), but it is worth naming it as such so the reader knows it is not
   checkable a priori — it is verified in practice by the algorithm's dual variable.
6. **Cite Bouchitté–Buttazzo Thm 3.3 for Lemma `lsc`** and demote the reproduction to a
   remark; and trim §2.2 to the two facts actually used. Cosmetic, shortens the paper.

---

### Bottom line
The proofs are correct except for the continuous-part optimality gap (GAP-1) and the
existence↔finite-support composition gap (GAP-2), both repairable with a moment
hypothesis already in the author's toolkit. The heavier message is architectural: a
large share of §3.3–3.4 is machinery for an unbounded parameter domain that the author's
own results show can be replaced by "work on compact `Ω_R` and cite Bouchitté–Buttazzo /
the convex representer-theorem literature," at no cost to any usable conclusion. The
genuinely novel, non-reducible contributions are the **nonconvex** finite-support/merging
theorem and the `k`-homogeneous reformulation with its correct constants — both sound.
