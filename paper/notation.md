# Notation contract for the objective rewrite

Status: notation contract for `paper_0805.tex`. This file records stable
symbols, not temporary variables local to a proof. It excludes symbols used
only in Section 6.

The statuses are:

- **retained** — same object and meaning as in the current paper;
- **changed** — existing symbol whose definition or role changes;
- **new** — genuinely new object required by the revised results;
- **retired** — must not be used in the revised nonhomogeneous theory; and
- **reserved** — already has another meaning and must not be repurposed.

## 0. Optimal-control data

**`t₀ < T`, `m_u`, and `η > 0` — retained**
Initial and terminal times, control dimension, and control-energy weight in
the open-loop problem.

**`(\mathcal C,y,u,f,g,h,p^*)` — retained**
Cost functional, state, control, drift, control matrix, running state cost,
and optimal PMP costate. The corresponding unstarred `p` is used for a
costate along a nonoptimal trial control in Section 5.1.

## 1. Domains and basic spaces

**`d` — retained**
State-space dimension.

**`D ⊂ ℝᵈ` — retained**
Bounded Lipschitz state domain.

**`ν` — reserved**
Population measure `Lebesgue measure restricted to D`. Never use it for the
normalized parameter measure.

**`L²(D)` and `H¹(D)` — retained**
Population value space and value-gradient Hilbert space, both formed with the
population measure `ν`.

**`𝓗` — new**
Abstract Hilbert space used only in Theorem 5.1 before its population and
empirical instantiations.

**`Ω = ℝ^(d+1)` — retained**
Unbounded inner-parameter domain in the nonhomogeneous formulation.

**`d_Ω` — retained**
Generic parameter-space dimension in the finite-moment preliminaries; it is
fixed to `d+1` for the network parameter domain used thereafter.

**`ω = (a,b)` — retained**
One inner parameter, with slope `a ∈ ℝᵈ` and bias `b ∈ ℝ`.

**`Sᵈ` — retained**
Unit sphere used only in the homogeneous formulation and its linking
paragraph.

**`C₀(Ω)`, `C₀(Ω;L²(D))`, `C₀(Ω;H¹(D))`, and `C₀(Ω;𝓗)` — retained and
extended**
The scalar- and Hilbert-valued continuous functions on `Ω` that vanish in
absolute value or norm at infinity.

**`C_c(U)` — new**
Continuous functions with compact support in an open set `U ⊂ Ω`; in
particular, `C_c(Ω)` is used in the representation theorem for `Φφ`.

**`C_{wₚ⁻¹}(Ω)` — retained**
Continuous scalar functions `f` such that `f/wₚ` belongs to `C₀(Ω)`.

**`C_b(Ω)` — retired from the revised theory**
It is no longer needed after removal of narrow convergence.

## 2. Measures

**`M(Ω)` — retained**
Finite signed Radon measures on `Ω`, with total-variation norm.

**`|μ|` and `‖μ‖TV` — retained**
Total-variation measure and total-variation norm.

**`δ_ω` — retained**
Dirac measure at `ω`.

**`atom(μ)` — retained**
At most countable set of points at which `μ` has nonzero point mass.

**`μ_atom` — retained**
Purely atomic part of `μ`.

**`μ_cont` — retained, canonicalized**
Continuous part of `μ`, meaning the part with no atoms.

**`sign(μ)` — retained**
Polar sign in `μ = sign(μ)|μ|`.

**`⇀*` — changed in role**
Weak-* convergence is the only measure convergence used in the revised
nonhomogeneous theory. For weighted measures it is applied to `μₚ`.

**narrow-convergence arrow — retired**
Removed with the tightness and Prokhorov route.

## 3. Weight and normalized coordinate

**`p > 0` — retained**
Polynomial weight order. The value-level normalization, continuity, and
existence results use `p > s₀`; the gradient-level optimality, support,
local-sufficiency, and insertion results use `p > s₁`.

**`wₚ(ω) = 1 + |ω|ᵖ` — retained**
Positive polynomial parameter weight.

**`Mₚ(Ω)` — retained**
Measures with finite weighted total variation.

**`‖μ‖Mp = ∫Ω wₚ d|μ|` — retained, changed in role**
Norm and local-minimality topology. It is no longer an additive penalty in
the objective.

**`μₚ = wₚ μ` — new**
Normalized measure. The normalization theorem identifies it isometrically
with an element of `M(Ω)`.

**`μ̃` — retired**
Do not use a tilded normalized measure.

**generic normalized measure `ν` — retired**
Conflicts with the population measure.

**generic weight `ψ` or `ψₛ` — retired from Section 3**
Use `wₚ`. The symbol `ψₖ` remains reserved for the homogeneous penalty.

## 4. Network map and fidelities

**`ρ` — retained**
Activation function, required to belong to `C¹(ℝ)` by Assumption 3.1.

**`K(ω)` — retained**
Unnormalized ridge atom, with `K(a,b)(x) = ρ(a·x+b)`.

**`Kₚ(ω) = K(ω)/wₚ(ω)` — new**
Normalized ridge atom. It belongs to `C₀(Ω;L²(D))` for `p > s₀` and to
`C₀(Ω;H¹(D))` for `p > s₁` under the stated growth assumptions.

**`𝒩` — retained**
Network or value operator. The normalization identity is
`𝒩μ = ∫Ω K dμ = ∫Ω Kₚ dμₚ`.

**`N` — retained**
Finite network width. For theoretical width bounds it denotes the number of
distinct nonzero atoms in a reduced representation, after zero coefficients
are discarded and exact repeated locations are merged. Do not use it for a
covering number.

**`ω⃗ = (ωₙ)`, `c = (cₙ)` — retained**
Finite vectors of inner parameters and outer coefficients.

**`μ_{ω⃗,c} = Σₙ cₙδ_{ωₙ}` — retained**
Finite atomic measure representing a shallow network. Its reduced
representation has distinct locations and nonzero coefficients; its width is
`#atom(μ_{ω⃗,c})`.

**`V` — retained**
Target value function.

**`r_μ = 𝒩μ − V` — retained**
Population residual; Theorem 5.1 uses the same symbol for the corresponding
residual in its abstract Hilbert space.

**`L(μ) = ½‖r_μ‖²H¹` — retained**
Population gradient-augmented fidelity, with value `+∞` when the network is
not in `H¹(D)`.

**`M` — retained**
Number of empirical samples, always with `M ≥ 1`.

**`xᵐ` — retained**
Empirical sample points.

**`gᵐ = p^{*,m}(0)` — new**
Costate label in the empirical gradient channel. It equals `∇V(xᵐ)` only
at samples where the value function is differentiable.

**`lᴹ(μ)` — retained**
Empirical gradient-augmented fidelity.

**`Kᴹ(ω)` — new**
Empirical value-gradient feature vector whose Hilbert norm reproduces the
sample average in `lᴹ`.

**`Kᴹₚ(ω) = Kᴹ(ω)/wₚ(ω)` — new**
Normalized empirical feature used in Theorem 5.1 and Algorithm 1.

**value-only substitution — no separate symbol**
When the gradient summand of the fidelity is dropped, the empirical
specialization is read with `ℋ = ℝᴹ` and the gradient entries deleted from
`Kᴹ` and from the target. No separate symbol is introduced for the truncated
feature.

**`s₀` — retained**
Polynomial growth exponent of `ρ`, and hence of `K` in `L²(D)`.

**`s₁` — retained**
One plus the polynomial growth exponent assigned to `ρ′`. Assumption 3.1
requires `s₁ ≥ max{s₀,1}`, and the bound `eq:H1-growth` records growth of
`K` in `H¹(D)` of order `s₁`.

**`Cρ`, `C_D`, `C_K` — retained**
Common activation-growth constant in Assumption 3.1 and the resulting `L²`
and `H¹` atom-growth constants, respectively.

## 5. Scalar and measure penalties

**`φ` — retained**
Generic increasing, concave, coercive scalar penalty in the finite-slope
nonhomogeneous theory.

**`Lφ = φ′(0+)` — new**
Finite positive right slope at zero. The subscript prevents collision with
the fidelity `L`.

**`φ′`, `φ″` — retained with clarified domains**
First and second derivatives on `(0,∞)`; `φ′(0+)` is used at zero.

**`φ⁻¹` — new**
Inverse of the strictly increasing coercive function `φ`, whose existence is
established immediately after Assumption 3.3.

**`γ₁` and `ẑ₁` — retained**
Local lower-curvature-control constants used in the sufficient condition for
local optimality.

**`γ_A = −sup{φ″(t): 0 < t ≤ A}` — new**
Positive strict-curvature constant on a bounded coefficient interval, defined
for each fixed `A > 0`.
For the paper's `φγ` with `γ > 0`, it equals
`γ/(1+2γA)²`; it vanishes in the linear case `γ = 0`.

**`Φφ(ξ)` — changed**
Measure penalty
`Lφ ‖ξ_cont‖TV + Σω∈atom(ξ) φ(|ξ({ω})|)`.
Here `ξ` denotes a generic finite signed Radon measure; the argument in the
objective is normally the normalized measure `μₚ`.

**`α > 0` — retained**
Single global coefficient multiplying `Φφ(μₚ)`.

**`β` — retired from the revised nonhomogeneous theory and Algorithm 1**
No separate moment coefficient is present in the current manuscript.

**`J₀(μ) = L(μ) + αΦφ(μ)` — retained with restricted role**
Unnormalized comparison functional used only in Examples 3.11 and 3.12 and
their immediate discussion.

**`J(μ) = L(μ) + αΦφ(μₚ)` — changed**
Adopted nonhomogeneous population objective.

**`J_{ω⃗}(c)` — new**
Reduced coefficient objective obtained by fixing distinct inner parameters
and varying only their outer coefficients.

**`J_𝓗(μ)` — new, theorem-local abstraction**
Abstract normalized Hilbert-space objective in Theorem 5.1. Its population
and empirical instances are `J` and `Jᴹ`.

**`Jᴹ(ω⃗,c)` — changed**
Adopted empirical finite-network objective
`lᴹ(μ_{ω⃗,c}) + αΣₙφγ(wₚ(ωₙ)|cₙ|)` in Algorithm 1.

**`φγ` — retained with changed argument**
Scalar penalty used in Algorithm 1. Its argument becomes
`wₚ(ωₙ)|cₙ|`.

## 6. Derivative functions and quantitative bounds

**`P̄(ω) = ⟨r_{μ̄},K(ω)⟩H¹` — retained**
Function representing the derivative of the population fidelity at a fixed
candidate `μ̄` in an unnormalized Dirac direction.

**`P̄ₚ(ω) = P̄(ω)/wₚ(ω)` — new**
Weighted representative of the population fidelity derivative. Under
`p > s₁`, it belongs to `C₀(Ω)`.

**`Pᴹ_μ(ω)` — retained**
Derivative of empirical fidelity at `μ` in the direction `δ_ω`.

**`Pᴹₚ,μ(ω) = Pᴹ_μ(ω)/wₚ(ω)` — new**
Weighted representative of the empirical fidelity derivative used by
Theorem 5.1 and Algorithm 1.

**`Pₚ,μ` — new, theorem-local abstraction**
Function representing the weighted fidelity derivative in the abstract
Hilbert formulation of Theorem 5.1. Its population and empirical instances
are the two symbols above.

**`T(μ) = φ⁻¹(J(μ)/α)` — new**
Upper bound on every normalized atom magnitude of a positive-objective local
minimizer.

**`𝒦(μ)` — new**
Compact superlevel set
`{ω : |Pₚ,μ(ω)| ≥ αφ′(T(μ))}`.

**`T₀ = φ⁻¹(‖V‖²H¹/(2α))` — new**
A priori coefficient bound for global minimizers when `V ≠ 0`.

**`𝒦₀` — new**
Known compact set
`{ω : ‖V‖H¹ ‖Kₚ(ω)‖H¹ ≥ αφ′(T₀)}` containing every atom of every global
minimizer.

**`N_cov(A,r)` — new**
For a subset `A` of a metric space and `r > 0`, the least number of balls of
radius `r` needed to cover `A`.

**`#atom(μ)` — retained operation, new theorem role**
Cardinality of the atom set, bounded in the finite-support theorem and its
global-minimizer corollary.

**`Bₚ = supω ‖Kₚ(ω)‖𝓗` — new**
Uniform normalized feature bound in Theorem 5.1.

**`Δ(μ,ω) = max{|Pₚ,μ(ω)| − αLφ,0}` — new**
Nonnegative excess above the insertion threshold at a single candidate. It is
the quantity the one-step decrease of Theorem 5.1 is stated in, so the same
estimate covers a candidate returned by a local search.

**`Δ(μ) = sup_ω Δ(μ,ω) = max{‖Pₚ,μ‖∞ − αLφ,0}` — new**
Its supremum, which drives the rate statement. The two-argument and
one-argument forms are distinguished by arity.

**`ω*` — new stable role**
Parameter attaining the maximum of `|Pₚ,μ|` in Theorem 5.1, so that
`Δ(μ,ω*) = Δ(μ)`.

**`κ` — new**
Free normalized inserted mass, `κ = wₚ(ω)c`, used in the one-step estimate of
Theorem 5.1. It replaces the earlier `q` in this role, which is reserved for
the homogeneous exponent.

**`κ(ω) = −Δ(μ,ω)/B_p² · sign Pₚ,μ(ω)` — new (was `q*`)**
Value of `κ` inserted at the candidate `ω` by Theorem 5.1.

**`c(ω) = κ(ω)/wₚ(ω)` — new (was `c*`)**
Corresponding outer coefficient in the original network coordinate. The symbol
`c*` is left to Section 5.3, where it is the initial outer weight of the
finite-step criterion.

**`R(μ)` — new**
Computable Euclidean search radius from Theorem 5.1.

**`Bᴹₚ = supω ‖Kᴹₚ(ω)‖` — new**
Uniform empirical feature bound used by Algorithm 1.

**`μₖ` — retained with clarified role**
Finite atomic iterate sequence in the rate statement of Theorem 5.1.

**`‖K(ω)‖_M` — new**
Empirical value-gradient seminorm,
`‖K(ω)‖²_M = M⁻¹ Σₘ (|K(ω)(xᵐ)|² + |∇K(ω)(xᵐ)|²)`, used by the finite-step
criterion. It coincides with the Euclidean norm of the scaled feature,
`‖K(ω)‖_M = ‖Kᴹ(ω)‖`; the batch cross-term display uses the pairing
`⟨Kᴹ(ω),Kᴹ(ω′)⟩` directly rather than a second `M`-subscripted symbol.

**`ΔJᴹ_{ψₖ}(c;ω)` — new**
Objective change produced by inserting the single atom `cδ_ω` into the
homogeneous empirical objective, with the surrogate penalty coefficient `α/q`.

**`z_turn`, `z*` — new**
Turning point of the left-hand side of the insertion-weight equation, and the
larger of its positive roots, which is the accepted magnitude of the initial
outer weight.

## 7. Homogeneous link

**`k` — retained**
Positive homogeneity degree. The radial rescaling result allows `k ≥ 1`; the
gradient-augmented homogeneous learning problem and its measure theory assume
`k ≥ 2`.

**`M(Sᵈ)` — new**
Finite signed Radon measures on the unit sphere, with the total-variation
norm `‖·‖_{M(Sᵈ)}`. Identified with the measures in `M(Ω)` vanishing outside
`Sᵈ`. This is the whole domain of the homogeneous theory; “local minimizer”
in Section 4 always means local with respect to `‖·‖_{M(Sᵈ)}`.

**`(P_{ψₖ})` — new**
Tag of the all-measures homogeneous minimization problem
`min_{μ ∈ M(Sᵈ)} J_{ψₖ}(μ)`.

**`ξ, ζ > 0` — new**
Exponents of the separable inner/outer penalty `|c|^ξ/ξ + |ω|^ζ/ζ`, local to
the general rescaling lemma and its remark.

**`C_{k,ξ,ζ} = (1/q) k^{−kξ/(kξ+ζ)}` — new**
Constant induced by minimizing that separable penalty over radial
representations.

**`q` — retained, homogeneous exponent only**
In the homogeneous rescaling lemma, `q = ξζ/(kξ+ζ)`. Its quadratic
specialization at `ξ = ζ = 2` is
`q = 2/(k+1)`, which under the gradient-training assumption `k ≥ 2` lies
strictly between zero and one. Do not use `q` outside these homogeneous
exponents.

**`Cₖ = ((k+1)/2) k^{-k/(k+1)}` — retained and made explicit**
The value `C_{k,2,2}`. Satisfies `C₁ = 1` and `C₂ = (3/2)·2^(−2/3)`. If the
original separable penalty has coefficient `α_orig`, the sphere-normalized
objective has effective coefficient `α = Cₖ α_orig`. Equal nominal values of
`α` across different `k` do not correspond to equal `α_orig`.

**`ψₖ(t) = tᑫ` — retained**
Homogeneous scalar penalty, with infinite right derivative at zero when
`q < 1`.

**`Φ_{ψₖ}` — new**
Extended homogeneous measure penalty on `M(Sᵈ)`. For a purely atomic measure
it is the sum of `ψₖ` applied to the absolute atom weights; it is `+∞` when
the continuous part is nonzero. Repeated locations are merged before its
atomic sum is evaluated.

**`J_{ψₖ}` and `Jᴹ_{ψₖ}` — changed**
Population and empirical homogeneous objectives on `M(Sᵈ)`, obtained by
adding `αΦ_{ψₖ}` to the corresponding fidelity. Their finite-network
restrictions are evaluated through `μ_{ω⃗,c}`.

**`C_S = sup_{ω∈Sᵈ} ‖K(ω)‖_{H¹(D)}` — new**
Uniform dictionary bound on the sphere, finite whenever `K : Sᵈ → H¹(D)` is
continuous. Defined once at the beginning of the homogeneous formulation and
used by every later homogeneous result. It plays the role that `Bₚ` plays in the
nonhomogeneous theory; do not conflate the two.

**`P̄(ω) = ⟨r_μ̄, K(ω)⟩_{H¹(D)}` — retained, restricted to `Sᵈ`**
In Section 4 the function representing the Gâteaux derivative is evaluated on
the sphere and is unweighted; there is no `P̄ₚ` in the homogeneous theory.

**`B(μ̄) = ‖P̄‖_{C(Sᵈ)}` — new**
Sup-norm of that derivative function at the candidate `μ̄`. Finite by
continuity and compactness. `B(μ̄) = 0` forces `μ̄ = 0`.

**`c_min(μ̄) = (αq/B(μ̄))^{1/(1−q)}` — new**
Minimizer-dependent lower bound on every nonzero atom weight of a
total-variation local minimizer. A posteriori: it depends on `μ̄`.

**`c_min,glob = (αq/(‖V‖_{H¹(D)} C_S))^{1/(1−q)}` — new**
A priori counterpart, valid at every global minimizer because
`B(μ̄) ≤ ‖V‖_{H¹(D)} C_S` there.

**`N_max = ‖V‖²_{H¹(D)} / (2α c_min,glob^q)` — new**
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

**`U = L²(t₀,T;ℝ^{m_u})` — new**
Control space of the open-loop problem. The reduced objective is
`u ↦ 𝒞(u;t₀,x)` on `U`; no separate symbol is used for it.

**`prox_{σαφ}` — new**
Scalar proximal operator of `σαφ(|·|)`, acting coordinatewise, together with
the proximal parameter `σ > 0`. It defines the proximal-residual equation
solved by the semismooth Newton correction of both algorithms.

**`T_out` — new**
Number of outer insertion iterations.

**`N_trial` — new**
Number of candidate directions sampled per outer iteration.

**`N_ins` — new**
Maximum number of candidates inserted per outer iteration. The one-candidate
decrease guarantees hold only for `N_ins = 1`.

**`ε_prune` — new**
Pruning tolerance: atoms with `|cₙ| ≤ ε_prune` are removed after the
correction step.

**`θ_merge` — new**
Near-duplicate tolerance: two refined candidates with unit representatives
satisfying `ω̂ᵢ·ω̂ⱼ > 1 − θ_merge` are near-duplicates, and only the first in
the enumeration is kept.

**`Err_{L²}`, `Err_{H¹}` — new**
Relative validation errors of a fit `V̂` against `V`, defined at the opening of
Section 6 and computed on the evaluation set of each experiment.
