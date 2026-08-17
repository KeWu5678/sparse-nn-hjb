# Standalone-thesis specification

## Task and sources

The source thesis is:

`/Users/chaoruiz/Documents/Repos/SparseNNforHJB/papar/archiv/Mthesis.tex`

The standalone deliverable is:

`/Users/chaoruiz/Documents/Repos/SparseNNforHJB/papar/paper_codex.tex`

Use `papar/archiv/Mthesis.tex` for the valuable mathematical ideas, proofs,
remarks, and illustrations that should be retained. Do not use
`papar/paper.tex` as a source unless the user explicitly asks for it in a later
session.

The title is:

> Optimal Feedback Law Recovery by Gradient-Augmented Neural Network with
> Nonconvex Regularization on the Unbounded Parameter Space

This is a full-length thesis with no hard page limit.

## Governing rules

1. The deliverable is a standalone thesis. Its prose may refer to itself as
   “this thesis,” but must not refer to an earlier version, an original paper,
   a revision, a source document, or another TeX file.
2. Keep changes surgical. Preserve correct and useful material from
   `Mthesis.tex`, especially as mathematical remarks or examples.
   Deleting or compressing material is justified only when doing so improves
   understandability without removing a needed argument, dependency, or
   useful explanation. Proofs must retain thesis-level intermediate steps.
3. Use established mathematical terminology. Do not coin labels, metaphors, or
   compressed phrases merely to make the prose sound novel.
4. Define notation clearly and uniformly. Do not use more than one notation
   for the same object, and do not use undefined notation.
5. Keep the prose direct and human. Avoid promotional or AI-like language.
6. State the mathematical object, assumption, or parameter value explicitly
   instead of using vague words such as “configuration” or “operating point.”
7. In the numerical section, follow the pattern:

   `question to be investigated -> result and illustration`.

   Do not narrate the research process when a direct statement of the test and
   result is enough.

## Central framing

The background is gradient-augmented fitting of HJB value functions from PMP
value-gradient samples. The thesis has two equally important contributions:

1. enhancing sparsity with the concave nonconvex functional $Φ_φ$; and
2. expanding the admissible activation functions while treating the unbounded
   inner-parameter domain.

The second contribution has two mathematical cases:

- For nonhomogeneous activations on $Ω = R^{d+1}$, use the moment norm

  $||μ||_{M_p(Ω)} = ∫_Ω (1 + |ω|^p) d|μ|(ω)$

  in the objective with coefficient $β>0$.
- For positively $k$-homogeneous activations, absorb radial scaling into the
  outer coefficient, normalize the inner parameter to the unit sphere, and use
  the induced fractional-power penalty $|c|^{2/(k+1)}$.

Existence, first-order optimality conditions, and finite-network results must
be stated for both cases. The moment term is not the only contribution.

For the nonhomogeneous case, distinguish

$J_0(μ) = L(μ) + α Φ_φ(μ)$

from

$J(μ) = L(μ) + α Φ_φ(μ) + β ||μ||_{M_p(Ω)}$, with $β > 0$.

## Uniform notation

- The activation is $ρ$; do not introduce a second symbol such as $σ$.
- Write $ω=(a,b)$ and define the ridge-function map once by

  $[K(ω)](x)=ρ(a·x+b)$.

  Use $K$ in the theory and proofs; do not introduce a second notation for the
  same ridge function.
- For a finite network, write

  $\boldsymbol\omega=(ω_n)_{n=1}^N$,
  $μ_{\boldsymbol\omega,c}=Σ_n c_nδ_{ω_n}$, and
  $\mathcal N_{\boldsymbol\omega,c}=Σ_n c_nK(ω_n)$.

  After the value operator is defined in Section 3,
  $\mathcal Nμ_{\boldsymbol\omega,c}=\mathcal N_{\boldsymbol\omega,c}$.

  Here $ω$ denotes one inner parameter and $\boldsymbol\omega$ denotes the whole
  parameter vector.
- $L$ is the population fidelity and $l^M$ is the empirical fidelity.
- $J$ and $J^M$ are the corresponding nonhomogeneous objectives.
- $φ$ is a generic scalar concave function, $Φ_φ$ is its functional on
  measures, and $φ_γ$ is the scalar log penalty used computationally. Do not
  write $Φ_γ$.
- $ψ_k(z)=z^{2/(k+1)}$ is the scalar fractional-exponent penalty, and
  $J_{ψ_k}$ and $J^M_{ψ_k}$ are the homogeneous objectives. Do not introduce a
  measure-functional notation for $ψ_k$.
- The moment contribution is written as $β||μ||_{M_p(Ω)}$. Do not introduce
  $Ψ_p$.
- Use the standard Gâteaux derivative notation $D L(μ)[μ']$, where $μ'$ is a
  general signed-measure direction and $δ_ω$ is a Dirac measure.
- At a local minimizer, define only

  $P̄(ω) := D L(μ̄)[δ_ω] = \langle N μ̄ - V, K(ω)\rangle_{H^1(D)}$.

  Say that $P̄$ is the function representing the Gâteaux derivative. Do not
  call it a dual variable, dual field, or dual profile.
- During Algorithm 1, use

  $P^M_{μ_t}(ω):=D l^M(μ_t)[δ_ω]$

  for the empirical directional derivative at the current measure. Do not
  introduce $P_t$, $P_r$, or a generic $P$.
- Reserve lowercase $p$ for the moment order. The PMP costate may remain
  lowercase $p$ in the optimal-control discussion. A Riccati matrix may use
  uppercase $P$ when it is explicitly defined locally.
- $ν$ denotes the data distribution and is Lebesgue measure in the thesis.
- Reserve $β$ for the moment coefficient and $η$ for the control-energy
  coefficient.
- Use $h$ for the running state cost, $C(u;t_0,x)$ for the optimal-control cost,
  and $Ĉ(u)$ for the reduced cost. Do not use the learning objective $J$ for
  the control cost.
- Use $s_0$ for the growth order of the value atom in $L^2(D)$ and $s_1$ for
  the growth order of the full atom in $H^1(D)$.
- Use $λ>0$ for the radial parameter in the theory and algorithm.

## Required mathematical language

- The existence proof uses **the direct method of the calculus of variations
  using narrow convergence**.
- Use **boundedness of the support** and an explicit radius $R_*$; do not
  introduce a separate named “confinement” concept.
- Keep **escaping directions** as the descriptive mathematical phrase for
  directions along which $|ω| \to \infty$. The corollary title is
  “escaping directions in the $H^2$ regime.”
- A PMP characteristic is a **path**. Do not call it a branch.
- For the pendulum, let $V_{2jπ}$ be the fixed-target value for the target
  $(2jπ,0)$ and write

  $V(x) = \min_{j \in Z} V_{2jπ}(x)$.

  In the region where the targets $0$ and $2π$ are minimizing, this reduces to
  $V(x) = \min\{V_0(x), V_{2π}(x)\}$.

- Define the representative adjacent-target switching set, its periodic
  extension to $D$, and its neighborhood by

  $S_{0,2π} := \{x \in R^2 : V_0(x) = V_{2π}(x) = V(x)\}$,

  $S := D ∩ \bigcup_{j \in Z}(S_{0,2π} + (2jπ,0))$,

  and

  $S_δ := \{x \in D : \operatorname{dist}(x,S) \le δ\}$.

  The experiments use $δ = 0.3$.
- At a point of $S$, sampled PMP costates give the corresponding one-sided
  gradients.
- Define

  $R_{0.95}(μ) := \inf\{R \ge 0 : |μ|(B_R) \ge 0.95 |μ|(Ω)\}$.

  Refer to it only by $R_{0.95}$ after the definition.

## Terms and phrasing to avoid

Do not use the following in the thesis:

- “narrow direct method”;
- “parameter-moment penalty”;
- “atomwise fidelity derivative”;
- “dual variable”;
- “dual field”;
- “dual profile”;
- “weighted predual” as a condition for representing the Gâteaux derivative;
- “amplitude penalty” when $α Φ_φ$ or “penalty on the outer coefficients” is
  meant;
- $Ψ_p$, $Φ_γ$, $Ψ_k$, $P_t$, $P_r$, or $σ$ as additional notation for
  objects already denoted above;
- “baseline”;
- “branch” for a PMP path or fixed-target value;
- “lower envelope” when the formula $\min\{V_0,V_{2π}\}$ is clearer;
- “candidate cost associated with a family of trajectories”;
- “switching band,” “switching tube,” or an undefined “switching region”;
- “configuration,” “operating point,” or “cell” in place of explicit
  parameter values or a run;
- “no-escape condition” or “no-escape hypothesis”;
- “confinement lemma,” “confinement axis,” or “clamp-confined”;
- invented labels such as “regularization surface,” “sparsity lever,”
  “gradient-blind,” “stationary-surrogate diagnostic,” or similar prose.

The word “escaping” is permitted in **escaping directions**, because it
describes $|ω| \to \infty$. Standard terms such as narrow convergence,
tightness, coercivity, support, switching set, directional derivative, and
Gâteaux derivative are permitted.

## Section 1: introduction

- Present the traditional sparse shallow network directly: it uses ReLU and an
  $ℓ^1$ penalty on the outer coefficients, with no concave nonconvex penalty and
  no moment regularization.
- Do not call it a baseline.
- Then state its two limitations under gradient-augmented fitting:

  1. a convex $ℓ^1$ penalty may retain redundant clustered neurons;
  2. “The second-order weak derivative of ReLU does not exist as a function
     while it is a distribution, and we cannot use it to approximate
     high-order derivatives.”

- Highlight the extension of the Pieper--Petrosyan measure formulation to
  gradient-augmented fitting on an unbounded parameter domain.
- Give equal prominence to the nonhomogeneous moment-regularized formulation and
  the positively $k$-homogeneous formulation.
- State that the optimality conditions and finite-network results are proved
  in both cases.
- The introductory comparison figure contains exactly four models: traditional
  ReLU with $ℓ^1$, softplus with the log penalty and moment norm, the Gaussian
  with the log penalty and moment norm, and $ReLU^2$ with $ψ_2$.
- Keep the contribution tone strong but mathematical.

## Section 2: general tools only

Section 2 contains only general theory:

- finite Radon measures, total variation, Riesz representation, and the
  atomic/nonatomic decomposition;
- narrow convergence for signed measures;
- uniform tightness;
- the classical Prokhorov theorem for nonnegative finite measures and the
  signed-measure compactness consequence obtained from the Jordan
  decomposition;
- the weighted measure space
  $M_p(Ω)=\{μ:∫_Ω(1+|ω|^p)d|μ|<∞\}$, its norm, and its continuous-function
  pairing;
- the existing Barron-space definitions and cited result statements.

End Section 2 with an unnumbered notation subsection. Centralize the notation
used throughout Sections 3--6 there, including $D$, $ν$, $M$, $Ω$,
$ω=(a,b)$, $K$, $δ_ω$, and $μ'$.

Do not introduce $L$, $N$, $Φ_φ$, the HJB problem, ridge atoms, or any
problem-specific operator in the narrow-convergence subsection. Do not put the
lower-semicontinuity lemmas for $Φ_φ$ or the moment norm in Section 2. The
Barron-space part states cited results; do not add proofs that are not present.

## Section 3: learning problem and proofs

Use a failure-driven structure:

1. value operator, gradient-augmented loss, polynomial growth, and the generic
   concave functional $Φ_φ$, followed by $J_0$;
2. failure of $J_0$ on the unbounded parameter domain;
3. the moment-regularized problem and the definition of local minimizers;
4. existence of a global minimizer;
5. optimality conditions, bounded support, finite support, boundary
   representation, escaping directions, and the sufficient condition for
   local optimality;
6. admissible activations.

Keep the tanh escape and softplus nonattainment examples as illustrations.
Keep the boundary-representation lemma and the escaping-directions corollary.
The full local-optimality theorem stays, with the support estimate reused
rather than proved twice.

Place the lower-semicontinuity statements for $Φ_φ$ and the moment norm here,
under the respective definitions.

For existence, the sufficient condition is $p > s_0$. For bounded activations
such as tanh and the Gaussian, existence alone therefore permits $p > 0$; for
softplus, use $p > 1$.

For bounded support and finite support, use the full $H^1$ growth estimate

$|P̄(ω)| \le C_P (1 + |ω|)^{s_1}$

and require $p > s_1$. The elementary sufficient choice for tanh, softplus,
and the Gaussian is $p > 1$.

Include the comparison remark in the following form, without discussing
necessity:

- for $J_0$, a sufficient asymptotic condition is

  $\limsup_{|ω|\to∞} |P̄(ω)| < α$;

- for $J$, the corresponding condition is

  $\limsup_{|ω|\to∞} [|P̄(ω)| - β(1 + |ω|^p)] < α$.

When $p > s_1$, the expression in brackets tends to $-∞$.

Do not say that $P̄$ must belong to $C_0(Ω)$ in order to represent the Gâteaux
derivative. It is defined pointwise by the directional derivative. If a
weighted $C_0$ statement is useful, keep it separate from the definition of
the derivative.

## Section 4: positively homogeneous formulation

Keep Section 4 close to `Mthesis.tex`. Preserve the sphere normalization, the
fractional-power penalty, the first-order optimality conditions, existence,
and the explicit width bound. Make clear that no moment norm is needed for
compactness after normalization to the sphere.

## Section 5: algorithms

Update Algorithm 1 from the log penalty alone to

$α Φ_φ + β ||μ||_{M_p(Ω)}$.

For a discrete measure $μ=Σ_n c_nδ_{ω_n}$, write the implemented objective as

$J^M(\boldsymbol\omega,c)=l^M(μ_{\boldsymbol\omega,c})+αΣ_n φ_γ(|c_n|)+βΣ_n|c_n|w_p(ω_n)$.

Its insertion condition is

$|P^M_{μ_t}(ω)| > α + β w_p(ω)$, where $w_p(ω)=1+|ω|^p$,

and candidates are ranked using $|P^M_{μ_t}(ω)| - β w_p(ω)$.

For nonhomogeneous activations, optimize the radial scale with

$λ \to |P^M_{μ_t}(λ ω̂)| - β w_p(λ ω̂)$.

For $β>0$ and $p>s_1$, this function is coercive. State the implementation
bound simply:

> The algorithm restricts $λ$ to $λ \le e^5$.

Algorithm 2 remains the positively homogeneous $ReLU^k$ formulation with
$q=2/(k+1)$, sphere normalization, the finite-step decrease criterion, and
$β=0$. The implemented insertion criterion uses

$ΔJ^M_{ψ_k}(c;ω)=cP^M_{μ_t}(ω)+(1/2)c^2||K(ω)||_M^2+(α/q)|c|^q$,

whereas the subsequent coefficient solve minimizes the objective with
$αΣ_n|c_n|^q$. State that $α/q>α$ for $q<1$, so every accepted insertion also
decreases the latter objective, and that the inserted coefficient is an
initial value for the coefficient solve.

## Section 6: numerical examples

Keep the original thesis pattern: state the question, then state the result
and show the figure or table. Avoid long descriptions of experimental
bookkeeping.

### Van der Pol oscillator

- Begin with a dedicated subsection “Effect of $β$ and $p$.”
- Use the headings:

  - “Dependence on $α$ and $β$”;
  - “Effect of $β$ on validation error, support size, and $R_{0.95}$”;
  - “Effect of $p$ on $R_{0.95}$”;
  - “Upper bound on the radial parameter.”

- Report the $α$--$β$ comparison directly.
- If tanh returns the zero measure at $β=10^{-1}$, say exactly that the
  algorithm returns the zero measure; do not say that the model “collapses.”
- Use the $p$ comparison to relate the theoretical radius $R_*$ and the
  observed $R_{0.95}$. Do not invent categories such as “clamp-confined.”
- Algorithm 1 is demonstrated with softplus, tanh, and the Gaussian.
- Algorithm 2 is demonstrated separately with $ReLU^k$.
- Keep gradient augmentation and sparsity/activation choice as distinct
  questions.

### Pendulum swing-up

- Define $V_0$, $V_{2π}$, $V=\min\{V_0,V_{2π}\}$, $S$, and $S_{0.3}$ before using
  regional errors, with the adjacent-target and periodic qualifications stated
  above.
- Use “path,” “fixed-target value,” and “one-sided gradient”; do not use
  “branch.”
- State the shifted-sample acceptance rule in terms of the corresponding
  $V_j$ and the first-order extrapolation of the other value. Do not introduce
  “pad,” “collar,” “band,” or “envelope-certified” as thesis terminology.
- Report errors on $S_{0.3}$ and $D \setminus S_{0.3}$.
- For each model and training set in the sampling-density table, report the
  result with the smallest error on $S_{0.3}$ among the tested $α$ values.
  Take every other entry in that row from the same selected run. Put this
  selection statement once, in the table caption.
- State that the Gaussian sampling-density runs use
  $(β,p,γ)=(10^{-4},2.01,0)$ and
  $α \in \{10^{-3},10^{-4},10^{-5}\}$, while the $ReLU^2$ runs use $ψ_2$,
  $β=0$, and $α \in \{10^{-4},10^{-5},10^{-6}\}$.
- Disclose that the 23 pendulum runs placing at least 95% of their total
  variation at $λ=e^5$ are retained in the numerical record but excluded when
  selecting the reported nonhomogeneous networks.
- Compare feedback from

  $A=(0.71,0.68)$ and $B=(0.23,0.53)$.

  The path from $A$ uses $V_{2π}$ and the path from $B$ uses $V_0$.
- Include the matched softplus comparison with

  $α=10^{-5}$, $γ=10$, $p=2.01$,

  and $β \in \{0,10^{-10},10^{-5},10^{-2}\}$.

  State the observed costs and success outcomes directly. Since none of these
  matched runs reaches the upright, do not attribute the successful softplus
  run with different $α$ and $γ$ to $β$ alone.
- The main Algorithm 1 activations are softplus, tanh, and the Gaussian.
- $ReLU^2$ carries the principal homogeneous comparison for the nonsmooth
  target.

## Conclusion

Rewrite the conclusion around both contributions:

- sparse gradient-augmented fitting with the concave nonconvex functional;
- treatment of the unbounded parameter domain through the moment norm for
  nonhomogeneous activations and sphere normalization for positively
  homogeneous activations.

Summarize the existence, optimality, bounded-support, finite-network, and width
results with formulas rather than slogans. Report the numerical findings
without causal claims not supported by matched runs.

## Validation before handoff

Before declaring the thesis complete:

1. remove all `CHECK`, TODO, or drafting comments from the thesis manuscript;
2. search for every avoided term listed above;
3. confirm that the thesis never refers to another version or source file;
4. check for duplicate labels and unresolved references;
5. compile with

   `latexmk -pdf -interaction=nonstopmode -halt-on-error paper_codex.tex`

   from the `papar/` directory;
6. inspect the final log for LaTeX warnings, undefined citations, and
   overfull boxes.

## TBD

- Reconcile the empirical-loss normalization used for the reported runs with
  the thesis convention. The theoretical empirical fidelity is
  $l^M=(1/(2M))Σ_m(|r(x^m)|²+|∇r(x^m)|²)$, whereas the recorded runs used
  $1/(2Md)$. Before final numerical reporting, convert the regularization
  parameters by $α_thesis=d α_run$ and $β_thesis=d β_run$ ($d=2$ in the
  current experiments) and update table and figure labels if necessary. Do
  not rescale the reported values until this item is explicitly approved.
- Decide how to state the comparability of the nominal parameter $α$ across
  the homogeneous powers $k$. After sphere normalization,
  $q_k=2/(k+1)$ and
  $C_k=(k+1)k^{-k/(k+1)}/2$, while the implementation uses
  $αΣ_n|c_n|^{q_k}$ and absorbs $C_k$ into $α$. Thus equal nominal values of
  $α$ do not give identical regularization for different $k$, because both
  $q_k$ and the absorbed constant depend on $k$. Do not add the warning,
  rescale $α$, or rerun the experiments until the wording and treatment are
  explicitly approved.
- Decide whether to replace the two Van der Pol tables that separately report
  validation errors and support sizes with one table reporting
  $N$, $\mathrm{Err}_{L^2}$, and $\mathrm{Err}_{H^1}$ for each
  activation and training loss. Do not make this change until it is confirmed.
- Decide whether the pendulum surface figure should show only the Gaussian,
  softplus, $\tanh$, and $\mathrm{ReLU}^2$, while retaining
  $\mathrm{ReLU}^3$ and $\mathrm{ReLU}^5$ only in the numerical
  comparison or accuracy--support frontier. Do not make this change until it
  is confirmed.
