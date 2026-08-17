# Notation contract for `paper_0805.tex`

This file records stable symbols, not temporary variables local to a proof.
It excludes symbols used only in Section 6. A symbol marked **reserved**
already has another meaning and must not be repurposed.

## 0. Optimal-control data

**`t₀ < T`, `m_u`, and `η > 0`**
Initial and terminal times, control dimension, and control-energy weight in
the open-loop problem.

**`(\mathcal C,y,u,f,g,h,p^*)`**
Cost functional, state, control, drift, control matrix, running state cost,
and optimal PMP costate. The corresponding unstarred `p` is used for a
costate along a nonoptimal trial control in Section 5.1.

## 1. Domains and basic spaces

**`d`**
State-space dimension.

**`D ⊂ ℝᵈ`**
Bounded Lipschitz state domain.

**`ν` — reserved**
Population measure `Lebesgue measure restricted to D`. Never use it for the
normalized parameter measure.

**`L²(D)` and `H¹(D)`**
Population value space and value-gradient Hilbert space, both formed with the
population measure `ν`.

**`Ω = ℝ^(d+1)`**
Unbounded inner-parameter domain in the nonhomogeneous formulation.

**`d_Ω`**
Generic parameter-space dimension in the finite-moment preliminaries; it is
fixed to `d+1` for the network parameter domain used thereafter.

**`ω = (a,b)`**
One inner parameter, with slope `a ∈ ℝᵈ` and bias `b ∈ ℝ`.

**`Sᵈ`**
Unit sphere used only in the homogeneous formulation and its linking
paragraph.

**`C₀(Ω)`, `C₀(Ω;L²(D))`, and `C₀(Ω;H¹(D))`**
The scalar- and function-valued continuous functions on `Ω` that vanish in
absolute value or norm at infinity.

**`C_c(U)`**
Continuous functions with compact support in an open set `U ⊂ Ω`; in
particular, `C_c(Ω)` is used in the representation theorem for `Φφ`.

**`C_{wₚ⁻¹}(Ω)`**
Continuous scalar functions `f` such that `f/wₚ` belongs to `C₀(Ω)`.

## 2. Measures

**`M(Ω)`**
Finite signed Radon measures on `Ω`, with total-variation norm.

**`|μ|` and `‖μ‖TV`**
Total-variation measure and total-variation norm.

**`δ_ω`**
Dirac measure at `ω`.

**`atom(μ)`**
At most countable set of points at which `μ` has nonzero point mass.

**`μ_atom`**
Purely atomic part of `μ`.

**`μ_cont`**
Continuous part of `μ`, meaning the part with no atoms.

**`sign(μ)`**
Polar sign in `μ = sign(μ)|μ|`.

**`⇀*`**
Weak-* convergence is the measure convergence used in the
nonhomogeneous theory. For weighted measures it is applied to `μₚ`.

## 3. Weight and normalized coordinate

**`p > 0`**
Polynomial weight order. The value-level normalization, continuity, and
existence results use `p > s₀`; the gradient-level optimality, support,
local-sufficiency, and insertion results use `p > s₁`.

**`wₚ(ω) = 1 + |ω|ᵖ`**
Positive polynomial parameter weight.

**`Mₚ(Ω)`**
Measures with finite weighted total variation.

**`‖μ‖Mp = ∫Ω wₚ d|μ|`**
Weighted variation norm and local-minimality topology. The objective applies
the scalar penalty to the normalized measure `μₚ`.

**`μₚ = wₚ μ`**
Normalized measure. The normalization theorem identifies it isometrically
with an element of `M(Ω)`.

## 4. Network map and fidelities

**`ρ`**
Activation function, required to belong to `C¹(ℝ)` by Assumption 3.1.

**`K(ω)`**
Unnormalized ridge atom, with `K(a,b)(x) = ρ(a·x+b)`.

**`Kₚ(ω) = K(ω)/wₚ(ω)`**
Normalized ridge atom. It belongs to `C₀(Ω;L²(D))` for `p > s₀` and to
`C₀(Ω;H¹(D))` for `p > s₁` under the stated growth assumptions.

**`𝒩`**
Network or value operator. The normalization identity is
`𝒩μ = ∫Ω K dμ = ∫Ω Kₚ dμₚ`.

**`N`**
Finite network width. For theoretical width bounds it denotes the number of
distinct nonzero atoms in a reduced representation, after zero coefficients
are discarded and exact repeated locations are merged. Do not use it for a
covering number.

**`ω⃗ = (ωₙ)`, `c = (cₙ)`**
Finite vectors of inner parameters and outer coefficients.

**`μ_{ω⃗,c} = Σₙ cₙδ_{ωₙ}`**
Finite atomic measure representing a shallow network. Its reduced
representation has distinct locations and nonzero coefficients; its width is
`#atom(μ_{ω⃗,c})`.

**`V`**
Target value function.

**`r_μ = 𝒩μ − V`**
Population residual.

**`L(μ) = ½‖r_μ‖²H¹`**
Population gradient-augmented fidelity, with value `+∞` when the network is
not in `H¹(D)`.

**`M`**
Number of empirical samples, always with `M ≥ 1`.

**`xᵐ`**
Empirical sample points.

**`gᵐ = p^{*,m}(0)`**
Costate label in the empirical gradient channel. It equals `∇V(xᵐ)` only
at samples where the value function is differentiable.

**`lᴹ(μ)`**
Empirical gradient-augmented fidelity.

**value-only substitution — no separate symbol**
When the gradient summand of the fidelity is dropped, the sampled norm is read
with its gradient entries deleted. No separate symbol is introduced.

**`s₀`**
Polynomial growth exponent of `ρ`, and hence of `K` in `L²(D)`.

**`s₁`**
One plus the polynomial growth exponent assigned to `ρ′`. Assumption 3.1
requires `s₁ ≥ max{s₀,1}`, and the bound `eq:H1-growth` records growth of
`K` in `H¹(D)` of order `s₁`.

**`Cρ`, `C_D`, `C_K`**
Common activation-growth constant in Assumption 3.1 and the resulting `L²`
and `H¹` atom-growth constants, respectively.

## 5. Scalar and measure penalties

**`φ`**
Generic increasing, concave, coercive scalar penalty in the finite-slope
nonhomogeneous theory.

**`Lφ = φ′(0+)`**
Finite positive right slope at zero. The subscript prevents collision with
the fidelity `L`.

**`φ′`, `φ″`**
First and second derivatives on `(0,∞)`; `φ′(0+)` is used at zero.

**`φ⁻¹`**
Inverse of the strictly increasing coercive function `φ`, whose existence is
established immediately after Assumption 3.3.

**`γ₁` and `ẑ₁`**
Local lower-curvature-control constants used in the sufficient condition for
local optimality.

**`γ_A = −sup{φ″(t): 0 < t ≤ A}`**
Positive strict-curvature constant on a bounded coefficient interval, defined
for each fixed `A > 0`.
For the paper's `φγ` with `γ > 0`, it equals
`γ/(1+2γA)²`; it vanishes in the linear case `γ = 0`.

**`Φφ(ξ)`**
Measure penalty
`Lφ ‖ξ_cont‖TV + Σω∈atom(ξ) φ(|ξ({ω})|)`.
Here `ξ` denotes a generic finite signed Radon measure; the argument in the
objective is normally the normalized measure `μₚ`.

**`α > 0`**
Single global coefficient multiplying `Φφ(μₚ)`.

**`J₀(μ) = L(μ) + αΦφ(μ)`**
Unnormalized auxiliary functional used only in Examples 3.11 and 3.12 and
their immediate discussion.

**`J(μ) = L(μ) + αΦφ(μₚ)`**
Adopted nonhomogeneous population objective.

**`J_{ω⃗}(c)`**
Reduced coefficient objective obtained by fixing distinct inner parameters
and varying only their outer coefficients.

**`Jᴹ(ω⃗,c)`**
Adopted empirical finite-network objective
`lᴹ(μ_{ω⃗,c}) + αΣₙφγ(wₚ(ωₙ)|cₙ|)` in Algorithm 1.

**`φγ`**
Scalar penalty used in Algorithm 1. Its argument becomes
`wₚ(ωₙ)|cₙ|`.

## 6. Derivative functions and quantitative bounds

**`P̄(ω) = ⟨r_{μ̄},K(ω)⟩H¹`**
Function representing the derivative of the population fidelity at a fixed
candidate `μ̄` in an unnormalized Dirac direction.

**`P̄ₚ(ω) = P̄(ω)/wₚ(ω)`**
Weighted representative of the population fidelity derivative. Under
`p > s₁`, it belongs to `C₀(Ω)`.

**`Pᴹ_μ(ω)`**
Derivative of empirical fidelity at `μ` in the direction `δ_ω`.

**`Pᴹₚ,μ(ω) = Pᴹ_μ(ω)/wₚ(ω)`**
Weighted representative of the empirical fidelity derivative used by
Theorem 5.1 and Algorithm 1.

**`Pₚ,μ(ω) = P_μ(ω)/wₚ(ω)`**
Normalized population derivative profile defined in Section 3. At a fixed
candidate `μ̄`, it is denoted by `P̄ₚ = Pₚ,μ̄`.

**`T(μ) = φ⁻¹(J(μ)/α)`**
Upper bound on every normalized atom magnitude of a positive-objective local
minimizer.

**`𝒦(μ)`**
Compact superlevel set
`{ω : |Pₚ,μ(ω)| ≥ αφ′(T(μ))}`.

**`T₀ = φ⁻¹(‖V‖²H¹/(2α))`**
A priori coefficient bound for global minimizers when `V ≠ 0`.

**`𝒦₀`**
Known compact set
`{ω : ‖V‖H¹ ‖Kₚ(ω)‖H¹ ≥ αφ′(T₀)}` containing every atom of every global
minimizer.

**`N_cov(A,r)`**
For a subset `A` of a metric space and `r > 0`, the least number of balls of
radius `r` needed to cover `A`.

**`#atom(μ)`**
Cardinality of the atom set, bounded in the finite-support theorem and its
global-minimizer corollary.

**`Bₚ = supω ‖Kₚ(ω)‖H¹(D)`**
Uniform normalized feature bound defined in Section 3 and used in
Theorem 5.1.

**`Δ(μ,ω) = max{|Pₚ,μ(ω)| − αLφ,0}`**
Nonnegative excess above the insertion threshold at a single candidate. It is
the quantity the one-step decrease of Theorem 5.1 is stated in, so the same
estimate covers a candidate returned by a local search.

**`Δ(μ) = sup_ω Δ(μ,ω) = max{‖Pₚ,μ‖∞ − αLφ,0}`**
Its supremum, which drives the certificate-rate corollary. The two-argument
and one-argument forms are distinguished by arity.

**`ω*`**
Parameter attaining the maximum of `|Pₚ,μ|` in Theorem 5.1, so that
`Δ(μ,ω*) = Δ(μ)`.

**`κ`**
Free normalized inserted mass, `κ = wₚ(ω)c`, used in the one-step estimate of
Theorem 5.1. The symbol `q` is reserved for the homogeneous exponent.

**`κ(ω) = −Δ(μ,ω)/B_p² · sign Pₚ,μ(ω)`**
Value of `κ` inserted at the candidate `ω` by Theorem 5.1.

**`c(ω) = κ(ω)/wₚ(ω)`**
Corresponding outer coefficient in the original network coordinate. The symbol
`c*` is left to Section 5.2, where it is the initial outer weight of the
finite-step criterion.

**`R(μ)`**
Computable Euclidean search radius from Theorem 5.1.

**`R_search(μ) = min{R(μ),e⁵}`**
Algorithm 1 sampling and final-candidate filter radius. When `R(μ)` is
unavailable, `R_search=e⁵`.

**`Aᴹₚ(ω) = ‖K(ω)/wₚ(ω)‖²_M`**
Candidate-specific empirical curvature used by Algorithm 1's warm start.

**`κ_A(ω)`, `c_A(ω)`**
Normalized inserted mass and physical outer coefficient obtained from the
candidate-specific one-atom quadratic estimate.

**`μₖ`**
Finite atomic iterate sequence in the certificate-rate corollary.

**`‖K(ω)‖_M`**
Empirical value-gradient seminorm,
`‖K(ω)‖²_M = M⁻¹ Σₘ (|K(ω)(xᵐ)|² + |∇K(ω)(xᵐ)|²)`, used by both insertion
criteria.

**`ΔJᴹ_{ψₖ}(c;ω)`**
Actual change caused by adding one atom with outer weight `c` at a new location
`ω∉atom μ`:
`cPᴹ_μ(ω) + (1/2)c²‖K(ω)‖²_M + α|c|ᑫ`.

**`c*(ω)`**
Selected global minimizer of the one-atom increment,
`prox_{(α/A)|·|ᑫ}(−P/A)`, where `P=Pᴹ_μ(ω)` and `A=‖K(ω)‖²_M`. A candidate is
accepted only when `c*(ω)≠0` and its actual increment is negative.

## 7. Homogeneous link

**`k`**
Positive homogeneity degree. The radial rescaling result allows `k ≥ 1`; the
gradient-augmented homogeneous learning problem and its measure theory assume
`k ≥ 2`.

**`M(Sᵈ)`**
Finite signed Radon measures on the unit sphere, with the total-variation
norm `‖·‖_{M(Sᵈ)}`. Identified with the measures in `M(Ω)` vanishing outside
`Sᵈ`. This is the whole domain of the homogeneous theory; “local minimizer”
in Section 4 always means local with respect to `‖·‖_{M(Sᵈ)}`.

**`(P_{ψₖ})`**
Tag of the all-measures homogeneous minimization problem
`min_{μ ∈ M(Sᵈ)} J_{ψₖ}(μ)`.

**`ξ, ζ > 0`**
Exponents of the separable inner/outer penalty `|c|^ξ/ξ + |ω|^ζ/ζ`, local to
the general rescaling lemma and its remark.

**`C_{k,ξ,ζ} = (1/q) k^{−kξ/(kξ+ζ)}`**
Constant induced by minimizing that separable penalty over radial
representations.

**`q`**
In the homogeneous rescaling lemma, `q = ξζ/(kξ+ζ)`. Its quadratic
specialization at `ξ = ζ = 2` is
`q = 2/(k+1)`, which under the gradient-training assumption `k ≥ 2` lies
strictly between zero and one. Do not use `q` outside these homogeneous
exponents.

**`Cₖ = ((k+1)/2) k^{-k/(k+1)}`**
The value `C_{k,2,2}`. Satisfies `C₁ = 1` and `C₂ = (3/2)·2^(−2/3)`. If the
original separable penalty has coefficient `α_orig`, the sphere-normalized
objective has effective coefficient `α = Cₖ α_orig`. Equal nominal values of
`α` across different `k` do not correspond to equal `α_orig`.

**`ψₖ(t) = tᑫ`**
Homogeneous scalar penalty, with infinite right derivative at zero when
`q < 1`.

**`Φ_{ψₖ}`**
Extended homogeneous measure penalty on `M(Sᵈ)`. For a purely atomic measure
it is the sum of `ψₖ` applied to the absolute atom weights; it is `+∞` when
the continuous part is nonzero. Repeated locations are merged before its
atomic sum is evaluated.

**`J_{ψₖ}` and `Jᴹ_{ψₖ}`**
Population and empirical homogeneous objectives on `M(Sᵈ)`, obtained by
adding `αΦ_{ψₖ}` to the corresponding fidelity. Their finite-network
restrictions are evaluated through `μ_{ω⃗,c}`.

**`C_S = sup_{ω∈Sᵈ} ‖K(ω)‖_{H¹(D)}`**
Uniform dictionary bound on the sphere, finite whenever `K : Sᵈ → H¹(D)` is
continuous. Defined once at the beginning of the homogeneous formulation and
used by every later homogeneous result. It plays the role that `Bₚ` plays in the
nonhomogeneous theory; do not conflate the two.

**`P̄(ω) = ⟨r_μ̄, K(ω)⟩_{H¹(D)}`**
In Section 4 the function representing the Gâteaux derivative is evaluated on
the sphere and is unweighted; there is no `P̄ₚ` in the homogeneous theory.

**`B(μ̄) = ‖P̄‖_{C(Sᵈ)}`**
Sup-norm of that derivative function at the candidate `μ̄`. Finite by
continuity and compactness. `B(μ̄) = 0` forces `μ̄ = 0`.

**`c_min(μ̄) = (αq/B(μ̄))^{1/(1−q)}`**
Minimizer-dependent lower bound on every nonzero atom weight of a
total-variation local minimizer. A posteriori: it depends on `μ̄`.

**`c_min,glob = (αq/(‖V‖_{H¹(D)} C_S))^{1/(1−q)}`**
A priori counterpart, valid at every global minimizer because
`B(μ̄) ≤ ‖V‖_{H¹(D)} C_S` there.

**`N_max = ‖V‖²_{H¹(D)} / (2α c_min,glob^q)`**
Uniform a priori bound on the reduced width of every global minimizer.
Referenced outside its theorem, hence recorded here.

**`N` in the homogeneous existence proof — local convention**
Section 4 keeps `N = #atom(μ)` as the reduced width in all statements. Inside
the existence proof only, `N` is the nominal representation width and `N′`
the reduced one.

**`Lφ = ∞` — structural endpoint, not a substitution rule**
It motivates the separate definition of `Φ_{ψₖ}` in Section 4. It is not an
allowed value in Assumption 3.2 and must not be substituted into any of the
finite-slope results of Section 3.

## 8. Other reserved symbols

**`η` — reserved**
Control-energy weight. Do not use it for a fidelity-derivative function or
measure remainder.

**`p` — reserved within the theory**
Polynomial weight order and, in the optimal-control context, PMP costate only.
Do not reuse it as a generic exponent or derivative function.

**`𝒩` — reserved**
Network operator. Covering numbers use `N_cov`, never `𝒩`.

## 9. Algorithm parameters, solver symbols, and reported errors

**`U = L²(t₀,T;ℝ^{m_u})`**
Control space of the open-loop problem. The reduced objective is
`u ↦ 𝒞(u;t₀,x)` on `U`; no separate symbol is used for it.

**`uₙ = wₚ(ωₙ)cₙ`**
Normalized outer coefficient used by Algorithm 1's semismooth Newton
correction. The physical coefficient is recovered as `cₙ=uₙ/wₚ(ωₙ)`.

**`prox_{λ|·|ᑫ}`**
Selected global proximal map used by Algorithm 2. At the switching input, where
zero and a nonzero point are both global minimizers, the implementation selects
zero. Closed forms are used for `q=1/2` and `q=2/3`; `q=1` is soft thresholding.

**`mₜ`, `λₜ`, `ρ_prox`**
Smallest nonzero warm-start coefficient magnitude and the fixed proximal scale
used during one Algorithm 2 correction:
`λₜ=min{α,ρ_prox mₜ^(2−q)/(2(1−q))}`, with `ρ_prox=0.1` for `q<1`; for
`q=1`, `λₜ=α`.

**`Gₜ`**
Algorithm 2 normal map in the proximal preimage. With
`c(z)=prox_{λₜ|·|ᑫ}(z)`, it is
`Gₜ(z)=(α/λₜ)(z−c(z))+∇₍c₎lᴹ(μ_{ω⃗,c(z)})`.

**`T_out`**
Number of outer insertion iterations.

**`N_trial`**
Number of random candidate starts sampled per outer iteration.

**`N_ins`**
Maximum number of candidates inserted per outer iteration. The one-candidate
decrease guarantees hold only for `N_ins = 1`.

**`ε_prune`**
Pruning tolerance: atoms with `|cₙ| ≤ ε_prune` are removed after the
correction step.

**`θ_merge`**
Near-duplicate tolerance. In Algorithm 1, two refined candidates are duplicates
when their Euclidean parameter distance is at most `θ_merge = 10⁻²`. In
Algorithm 2, whose parameters lie on the unit sphere, the cosine rule
`ω̂ᵢ·ω̂ⱼ > 1 − θ_merge` is used. Only the first candidate in the enumeration is
kept.

**`Err_{L²}`, `Err_{H¹}`**
Relative validation errors of a fit `V̂` against `V`, defined at the opening of
Section 6 and computed on the evaluation set of each experiment.
