# Mathematical wording audit of `paper_0805.tex`

Originally reviewed on 12 October 2026; reassessed and reorganized on 16 October 2026. This revision updates the review document only. Confirmed manuscript changes are marked as applied below.

The original audit covered the abstract, all six sections, theorem statements and proofs, algorithms, captions, and conclusion (source lines 75–3824). It checked figure numbering against the existing `.aux` file, inspected the Figure 12 control panel, and consulted implementation details where needed. The present revision checks the findings against the current manuscript and relevant source code. It is a review of wording and the identified mathematical issues, not an independent certification of every proof or numerical result.

**Locations:** Manuscript line ranges and page numbers retain the original audit's locations; nearby lines may have shifted. Links now use the current repository path. References to implementation details have been refreshed where checked in this revision.

**Version records:** Original manuscript SHA-256: `4d31eaeff939ccfd3c7d018f3e937d90cb522c811880369993538e233f7868d6`. Manuscript used for the 16 October reassessment: `44cc4a94bc170efa419728297795b591101f8e1dec2ebc94a5ab025e32a6dad0`.

**Reading the findings:** Each main item contains separately numbered points, an immediate **Proposed fix** for each point, and an **Overall fix** combining the remaining actions. **Correction** identifies a concrete misstatement or mismatch. **Proof-gap repair** identifies a missing step needed by an argument. **Clarification** makes an intended object, scope, or convention explicit. **Optional** marks an editorial improvement rather than a demonstrated mathematical defect. **Verify** identifies a convention or hypothesis that must be settled before choosing a correction. Mixed items distinguish these assessments point by point.

**Status flags:** `pending` means a proposed change or check remains open; optional proposals can also be pending. `applied` means the change is confirmed in the manuscript. `withdrawn` means the recommendation is no longer requested. Item flags summarize the recommendation; flags beside numbered points record partial completion. The smaller wording items have a Status column.

The revision retains concrete findings, narrows optional wording objections, and withdraws I1's proposed optimality disclaimer and I8's proposed weakening of the clustering claim. The agreed A2 residual wording and A5 “discontinuous gradient” wording are used below. Section 9 summarizes the remaining work.

## 1. The professor's comments

### P1 — Existence does not assert uniqueness

**Status:** `applied`

**Article correction applied; optional scope clarification · [line 131](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:131), page 2.**

1. `applied` — **Existence wording.** The original “the minimizer” was less precise before any minimizer had been selected. The current manuscript already says “We assume a minimizer exists.”

   **Proposed fix:** Retain the current sentence. No further replacement is needed.

2. `withdrawn` — **Scope of the assumption.** Existence for a fixed initial pair and existence for every initial pair are different assumptions. This is only an issue if the intended scope is unclear in context.

   **Proposed fix:** If needed, add the quantifier “for every initial pair under consideration.” Do not add uniqueness or change later uses of “the” after an object has been selected.

**Overall fix:** Keep “We assume a minimizer exists.” An explicit initial-pair quantifier is optional; no blanket article replacement is warranted.

### P2 — Define the derivative and duality in TPBVP

**Status:** `pending`

**Clarification · [lines 133–150](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:133), especially line 142, page 2.**

1. `pending` — **Jacobian and gradient convention.** In the displayed costate equation, the placement of the transpose leaves the derivative convention unclear.

   **Proposed fix:** Set $F(y,u)=f(y)+g(y)u$, use column gradients, define $(D_yF)_{ij}=\partial F_i/\partial y_j$, and write

   $$
   -\dot p^*(t)=D_yF(y^*(t),u^*(t))^\top p^*(t)+\nabla h(y^*(t)).
   $$

2. `pending` — **Costate and duality convention.** A column costate implicitly uses the Euclidean identification of $(\mathbb R^d)^*$ with $\mathbb R^d$.

   **Proposed fix:** State this identification once. Then $D_yF^\top$ is the Euclidean adjoint. The identity $\langle D_yFz,p\rangle=\langle z,D_yF^\top p\rangle$ may be included if the professor's requested explanation calls for it; a separate duality discussion is unnecessary.

3. `withdrawn` — **Variables held fixed.** The state derivative in PMP holds the control fixed, even when the optimal control is later expressed through the state and costate.

   **Proposed fix:** Say that $D_yF$ is taken with $u$ fixed. If the Hamiltonian $H(y,u,p)=h(y)+\eta|u|^2+p^\top F(y,u)$ is introduced, say that $\nabla_yH$ holds both $u$ and $p$ fixed.

4. `withdrawn` — **Meaning of the differential equations.** With $L^2$ controls, the state and costate equations are understood almost everywhere.

   **Proposed fix:** Add “the differential equations hold almost everywhere on $(t_0,T)$.”

**Overall fix:** Give one column-vector/Jacobian convention, use the displayed $D_yF^\top p+\nabla h$ equation, and state that the control is held fixed and the equations hold almost everywhere. Introducing $H$ is optional.

### P3 — Figure 12 must identify the saturation of the applied control

**Status:** `pending`

**Caption clarification · [lines 3646–3650](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3646) and [3722](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3722), Figure 12, page 40.**

1. `pending` — **The applied control is saturated.** The body gives $|u|\le30$, but Figure 12's caption does not explain the plateaus at $\pm30$.

   **Proposed fix:** Add: “The learned controls are saturated at $\pm30$ before being applied to the dynamics; the long $\mathrm{ReLU}^3$ plateaus reflect this saturation.”

2. `pending` — **The implemented law and its costs.** The displayed gradient formula gives the raw feedback; the rollouts and their running costs use its saturated version.

   **Proposed fix:** Identify the implemented law once near the experiment description, for example

   $$
   u_{\rm applied}(x)=\operatorname{clip}_{[-30,30]}
   \left(-\frac{1}{2\eta m\ell^2}\partial_{\dot\theta}\widehat V(x)\right),
   $$

   and say that the reported costs use this law. A sentence stating the same operation is sufficient if an extra display is undesirable.

3. `withdrawn` — **The same convention in related figures.** Figure 13 uses the same bound; the Van der Pol rollout uses $50$.

   **Proposed fix:** Include the relevant bound in those control captions as well. Keep this distinct from clipping a value surface for display.

**Overall fix:** State the saturation operation once in the body and its bound in the control captions. For Figure 12, combine that caption addition with the axis-scale clarification in N9.

### P4 — Figure 12 hides the initial transient

**Status:** `pending`

**Presentation · [lines 3709–3723](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3709), Figure 12.**

1. `pending` — **The initial transient is compressed.** The full $[0,40]$ axis makes early peaks and sign changes difficult to compare.

   **Proposed fix:** Add an inset or a separate initial-time panel. Use $[0,2]$ provisionally and choose the final interval from the existing traces.

2. `pending` — **The long-time behavior remains informative.** The full interval shows sustained Gaussian oscillations and $\mathrm{ReLU}^3$ saturation.

   **Proposed fix:** Retain the full-time panel and identify the zoom interval in the caption.

**Overall fix:** Show both the initial transient and the full rollout, with the zoom interval stated explicitly.

## 2. Abstract and introductory claims

### A1 — The abstract calls the measure functional a concave scalar penalty

**Status:** `applied`

**Correction already applied · [line 92](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:92).**

1. `applied` — **The scalar penalty and the measure functional are different objects.** The scalar function is $\phi$; $\Phi_\phi$ acts on signed measures. In general, $\Phi_\phi$ is not concave on that signed-measure space: opposite nonzero measures have positive penalty while their midpoint has zero penalty.

   **Proposed fix:** Attribute scalar concavity to $\phi$. The current abstract already does so.

2. `applied` — **The finite right derivative belongs to $\phi$.** A slope at zero in this statement is a property of the scalar penalty.

   **Proposed fix:** Retain the current attribution of the finite positive right slope to $\phi$.

**Overall fix:** No further change. The current sentence, “where $\Phi_\phi$ is induced by a concave scalar penalty $\phi$ with finite positive right slope at zero,” addresses both points.

### A2 — Specify what converges at the claimed rate, and for which algorithm

**Status:** `applied`

**Terminology alignment applied · [lines 98–100](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:98), [289–291](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:289), and [3793–3794](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3793).**

1. `applied` — **The rate concerns a specified residual.** Corollary 5.2 bounds $\min_{0\le k<n}\Delta(\mu_k)$, where $\Delta(\mu)=(\|P_{p,\mu}\|_\infty-\alpha L_\phi)_+$. The manuscript now calls this the first-order optimality residual for atom insertion and defines that name alongside $\Delta$. Its vanishing enforces the necessary insertion inequality, not the full stationarity system. The [terminology research](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/review/insertion_residual_terminology_20261016.md) records the primary-source comparison supporting this scoped description.

   **Proposed fix:** Retain the agreed wording: “an $O(n^{-1/2})$ best-iterate bound on the first-order optimality residual for atom insertion.” It is now applied in the abstract, contribution paragraph, and conclusion.

2. `applied` — **The algorithmic hypotheses determine the scope.** The corollary assumes exact global candidate maximization, single-atom insertion, and correction that does not increase the post-insertion objective.

   **Proposed fix:** Retain “For the single-atom PDAP scheme with exact search and monotone correction.” These qualifications are already present in the current summaries. Keep their precise definitions in the corollary.

3. `applied` — **The acronym.** The original abstract used PDPA.

   **Proposed fix:** Retain PDAP; this is already corrected.

**Overall fix:** Applied consistently in the abstract, contributions, and conclusion: “For the single-atom PDAP scheme with exact search and monotone correction, we prove an $O(n^{-1/2})$ best-iterate bound on the first-order optimality residual for atom insertion.” The residual is named at its definition; no further change under A2 is needed.

### A3 — Clarify the normalization and local-minimality scopes in the abstract

**Status:** `applied`

**Narrowed; optional summary clarification · [lines 93–104](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:93).**

1. `applied` — **The moment condition depends on the result.** Defining $w_p$ requires $p>0$; decay uses $p>s_0$ for the value dictionary and $p>s_1$ for the full $H^1$ dictionary. The current abstract already says “choosing $p$ sufficiently large,” so it no longer suggests that every $p>0$ gives decay.

   **Proposed fix:** Retain the current abstract wording. If greater precision is wanted, use “For $p$ exceeding the relevant dictionary growth exponent.” Keep the distinct $L^2$ and $H^1$ assumptions in the theorem statements.

2. `withdrawn` — **Local minimality uses a topology.** The body defines weighted-variation local minimality for the nonhomogeneous problem and total-variation local minimality on the sphere.

   **Proposed fix:** Retain those definitions. Naming both topologies again in the abstract is optional, not a correction required by an identified false claim.

**Overall fix:** No mandatory abstract expansion. Preserve the existing growth qualification and the precise theorem-level assumptions and topology definitions.

### A4 — State the range of the homogeneous sparsity result

**Status:** `pending`

**Scope clarification; abstract range already applied · [lines 101–104](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:101), [263–267](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:263), and [293–304](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:293).**

1. `pending` — **The sparsity results use $k\ge2$.** At $k=1$, quadratic rescaling yields $q=1$; the singular-slope argument uses $0<q<1$. The current abstract includes $k\ge2$, while the contribution paragraph should make the scope of its sparsity claims equally clear.

   **Proposed fix:** State “For $k\ge2$” before the finite-support and width conclusions in the contribution paragraph. Keep the broader range of the radial rescaling lemma itself.

2. `withdrawn` — **The equivalence concerns a particular regularized problem.** The sphere penalty comes from optimizing quadratic inner- and outer-parameter regularization over radial rescalings. The contribution paragraph already identifies this source; the abstract's “equivalent measure problem” can refer to it more clearly.

   **Proposed fix:** If revising that abstract sentence, write: “For positively $k$-homogeneous activations, radial optimization of quadratic parameter regularization yields a measure problem on the unit sphere with penalty proportional to $|c|^{2/(k+1)}$.”

**Overall fix:** Retain $k\ge2$ for the sparsity claims and identify quadratic parameter regularization as the source of the sphere formulation. Do not narrow the rescaling lemma unnecessarily.

### A5 — Attribute the gradient discontinuity to the intended benchmark

**Status:** `applied`

**Correction already applied · [lines 107–109](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:107).**

1. `applied` — **The gradient discontinuity belongs to the pendulum benchmark.** The original relative clause could appear to describe both benchmarks.

   **Proposed fix:** Retain the wording already agreed and applied: “We study the resulting algorithms on the Van der Pol oscillator and pendulum swing-up benchmarks; the latter has a value function with a discontinuous gradient.”

**Overall fix:** No further change; retain “discontinuous gradient” without introducing a switching set in this sentence.

### I1 — Numerical recovery wording and data qualifications — withdrawn

**Status:** `withdrawn`

**The earlier recommendation overreached · [lines 81–85](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:81), [133–135](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:133), [180–181](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:180), [214–225](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:214), and [3100–3104](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3100).**

1. `withdrawn` — **“Recover” does not by itself assert certified global optimality.** In this numerical-method context, the word describes the reconstruction being pursued. The review did not identify an argument that deduces optimality merely from solving the TPBVP.

   **Proposed fix:** Retain “recover.” Withdraw the proposed replacement by “approximate.” That replacement would not address PMP sufficiency in any event.

2. `withdrawn` — **The status of the data is already distinguished.** The current text describes numerical cost and initial-costate labels and conditions the gradient identity on an exact optimal trajectory at a differentiability point. The experiments use “numerical open-loop reference.”

   **Proposed fix:** Retain these qualifications. Add no general global-optimality disclaimer.

3. `applied` — **The previously missing value-label definition is present.** The current data paragraph defines $v^m$ as accumulated cost and $g^m$ as the initial costate.

   **Proposed fix:** Retain these definitions; no additional label definition is needed under I1.

**Overall fix:** No manuscript change under I1. The recommendation is withdrawn; retain the numerical recovery terminology and existing conditional gradient identity.

### I2 — Make the control problem's well-posedness assumptions visible before using them

**Status:** `pending`

**Narrowed; assumption-scope clarification · [lines 119–131](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:119) and [154–174](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:154).**

1. `withdrawn` — **Existence of a minimizer is already assumed.** The review must account for this assumption rather than discuss $C^1$ regularity as if it were the only hypothesis.

   **Proposed fix:** Keep “We assume a minimizer exists.” No replacement of that sentence is proposed.

2. `withdrawn` — **Full-horizon state solvability is a separate issue.** $C^1$ coefficients provide local regularity, but do not alone rule out finite-time blow-up. The algorithms section separately assumes full-horizon solvability and differentiability of the control-to-state map.

   **Proposed fix:** If that assumption is intended throughout the paper, make its scope explicit near (P) or cross-reference the existing assumption. Alternatively, define admissible controls by full-horizon state existence. Choose the formulation matching the intended control class; neither option follows merely from changing the minimizer sentence.

3. `pending` — **The DPP/HJB assertion needs its applicable framework.** Full-horizon solvability and existence of a minimizer do not, by themselves, state every hypothesis of a dynamic-programming/viscosity theorem.

   **Proposed fix:** Identify an applicable result and its assumptions for the unbounded-control setting, or state the intended DPP/HJB characterization as an assumption. Avoid the unsupported placeholder “under standard hypotheses.” This is a scope check, not a request to prove a new control theorem.

**Overall fix:** Preserve the minimizer-existence sentence. Clarify the scope of the already stated state-solvability assumption and identify the framework supporting the HJB characterization; do not insert an unspecified package of “standard” assumptions.

### I3 — Specify the viscosity sign convention

**Status:** `applied`

**Sign-convention correction applied · [lines 158–170](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:158).**

1. `applied` — **The viscosity interpretation depends on the sign convention.** The original displayed equation was $V_t+\inf_u\{\nabla_xV\cdot F+h+\eta|u|^2\}=0$. Under the convention that an upper test function gives an operator value $\le0$, the backward terminal-value operator is its negative. Classical equivalence alone does not settle the viscosity interpretation. [Liberzon, Section 5.3.3](https://liberzon.csl.illinois.edu/teaching/cvoc/node104.html) explicitly discusses this sign choice.

   **Proposed fix:** Either specify the convention used for the displayed equation, or adopt that convention's backward operator:

   $$
   -V_t-\inf_u\{\nabla_xV\cdot F+h+\eta|u|^2\}=0.
   $$

**Overall fix:** Applied: the displayed HJB now uses the backward operator with $-V_t$. The feedback formula is unchanged apart from explicitly identifying the spatial gradient.

### I4 — Say “pointwise” at differentiability points, and identify the spatial gradient

**Status:** `applied`

**Optional clarity improvement · [lines 164–172](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:164).**

1. `withdrawn` — **“Holds classically” at a point has a defensible reading.** In context, it can mean that the derivatives exist and satisfy the equality at that point. It need not assert a classical solution on an open neighborhood.

   **Proposed fix:** Optionally use the more direct wording “At points of differentiability in $(t,x)$, the HJB equality holds pointwise.” Do not classify the existing sentence as a mathematical error on this wording alone.

2. `applied` — **The feedback uses the spatial gradient.** Since $V$ depends on both time and state, an explicit convention for $\nabla V$ would help.

   **Proposed fix:** State once that $\nabla V$ denotes $\nabla_xV$, or use $\nabla_xV$ consistently in the HJB and feedback formulas.

**Overall fix:** If editing this passage, use “holds pointwise” and identify the spatial gradient once. No additional global-optimality qualification is needed here.

### I5 — Learning the initial-time slice does not recover the full finite-horizon feedback

**Status:** `pending`

**Clarification of the implemented feedback · [line 214](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:214), [lines 3412–3420](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3412), and [3801–3802](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3801).**

1. `withdrawn` — **The learned finite-horizon target is an initial-time slice.** $V(x):=V(0,x)$ records a fixed remaining horizon. The full finite-horizon feedback generally depends on time as well as state.

   **Proposed fix:** State that the Van der Pol network is fitted to the initial-time value slice.

2. `pending` — **The rollout uses that slice at every time.** The [implemented feedback](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/scripts/paper/vdp_summary_figures.py:104) evaluates the fitted gradient as a function of the current state, producing a stationary law.

   **Proposed fix:** Add: “For the finite-horizon benchmark, we learn the initial-time value slice and apply its gradient as a stationary feedback throughout the rollout.”

**Overall fix:** Add the single implementation sentence above. It explains the actual controller without adding a general optimality disclaimer. Keep the stationary infinite-horizon pendulum interpretation separate, as in N4.

### I6 — A feedback law does not require solving HJB

**Status:** `applied`

**Correction of an unnecessarily restrictive claim · [line 176](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:176).**

1. `applied` — **Feedback synthesis does not require solving HJB.** “One usually seeks a feedback law, which requires solving the HJB equation” presents one route as a universal requirement.

   **Proposed fix:** Retain the focus on feedback computation for nonlinear control. The applied wording is: “For nonlinear control problems, one usually seeks an optimal feedback law, which can be constructed from the value function obtained by solving the HJB equation.”

**Overall fix:** Applied: preserve the motivation of computing feedback laws for nonlinear control and describe HJB as a way to obtain the value function used in their construction.

### I7 — Avoid universal robustness and necessity claims

**Status:** `applied`

**Mixed: two clarifications and one correction · [line 153](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:153), [173–174](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:173), and [247](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:247).**

1. `applied` — **Open-loop response and robustness are different properties.** “The open-loop control is not robust” is broader than the subsequent explanation that it does not respond to measured deviations.

   **Proposed fix:** Use “An open-loop control does not adjust in response to state disturbances.” This states the intended mechanism directly.

2. `applied` — **Feedback response does not specify its effectiveness.** “Corrects deviations” is understandable informally, but can be made more precise without invoking a stability claim.

   **Proposed fix:** Use “State feedback adapts the input to the current state.”

3. `applied` — **Sparsity is not a prerequisite for knowledge distillation.** The claim that it is necessary for that application is too strong.

   **Proposed fix:** Describe sparsity as useful for computational efficiency and compact representations. If distillation is retained, describe sparse models as one possible choice rather than a requirement.

**Overall fix:** Applied: the manuscript now states that open-loop control does not adjust to state disturbances, state feedback adapts the input to the current state, and sparsity is useful for the listed applications.

### I8 — Proposed weakening of the clustering claim — withdrawn

**Status:** `withdrawn`

**The earlier recommendation overread the claim · [lines 253–255](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:253).**

1. `withdrawn` — **The clustering claim is supported by the cited paper.** Section 5.1, Figure 3, and Table 1(a) show dense clusters in the $\ell^1$ solution: 342 neurons versus 15 for the nonconvex solution with $\gamma=1$, with approximation errors 0.0045 and 0.0038. The discussion on pages 3–4 explains that splitting a coefficient into coefficients of the same sign preserves the $\ell^1$ penalty. The ability to remove individual connections does not rebut this failure to eliminate clustered redundancy. [Primary source, Section 5.1 and the preceding explanation](https://arxiv.org/pdf/2004.11515).

   **Proposed fix:** Withdraw the required replacement of “cannot effectively eliminate” by “need not eliminate.” Retain the clustering argument. Specifying “$\ell^1$ regularization” instead of “convex penalties” is optional precision, not a required correction.

**Overall fix:** No manuscript change is required under I8. The recommendation to weaken the clustering claim is withdrawn.

### I9 — ReLU can be used for first-gradient fitting

**Status:** `applied`

**Correction · [lines 256–259](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:256).**

1. `applied` — **First-gradient fitting does not require a function-valued second derivative.** ReLU has a bounded weak first derivative, and finite ReLU networks belong to $H^1(D)$ on bounded domains. The manuscript also uses a gradient-trained ReLU baseline.

   **Proposed fix:** Replace “does not extend to the gradient fitting” with “poses difficulties for gradient fitting”.

2. `withdrawn` — **The stronger regularity restriction needs its actual purpose.** ReLU is not $C^1$, and its second distributional derivative in one dimension is a Dirac measure. Those facts concern stronger regularity requirements, not the existence of its weak first derivative.

   **Proposed fix:** The additional explanation is withdrawn. Retain the existing sentence about second-order weak derivatives and higher-order approximation.

**Overall fix:** Applied only the phrase replacement in I9.1. The rest of the manuscript passage is unchanged; no explanatory sentence is added.

### I10 — Clarify the cited paper's focus on ReLU

**Status:** `applied`

**Minor wording clarification · [line 279](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:279), also [256](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:256).**

1. `applied` — **“Only” overstates the paper's focus on ReLU.** Pieper and Petrosyan focus on ReLU, while their general theory allows other activations on a compact parameter domain. This does not cover the unrestricted nonhomogeneous setting treated here. [Primary source, page 2](https://arxiv.org/pdf/2004.11515); [general framework, Section 2.1](https://arxiv.org/pdf/2004.11515#page=7).

   **Proposed fix:** Replace “only considers the ReLU case” with “focuses on the ReLU case”.

**Overall fix:** Applied the phrase replacement. Retain the existing comparison concerning nonhomogeneous activations on the unbounded parameter domain; no broader rewrite is needed.

### I11 — Name the representation and the particular assumptions that fail

**Status:** `applied`

**Clarification · [lines 266–267](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:266), [280–285](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:280), and [1990–1992](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1990).**

1. `withdrawn` — **Infinite slope excludes specific hypotheses.** “The usual functional tools … are not applicable” is broader than the finite-slope obstruction actually used.

   **Proposed fix:** Use “The infinite right derivative at zero falls outside the finite-slope assumptions of Section 3 and requires a separate argument.” Use the same explanation at the start of Section 4.

2. `applied` — **The representation has a specific object and consequence.** “A weak representation of the formulation” leaves both unspecified.

   **Proposed fix:** Replace “the formulation” with “the penalty $\Phi_\phi$”.

3. `applied` — **The activation domain and parameter domain are different.** The activation $\rho$ acts on $\mathbb R$; the dictionary is indexed by $\Omega=\mathbb R^{d+1}$.

   **Proposed fix:** Replace “continuous activations on $\Omega$” with “continuous activations $\rho:\mathbb R\to\mathbb R$ satisfying the stated growth bound, with inner parameters in $\Omega$.”

**Overall fix:** Applied the penalty and domain clarifications in I11.2 and I11.3; I11.1 is withdrawn. The continuous-activation assumptions of the existence theorem are preserved.

## 3. Measure theory and the nonhomogeneous formulation

### M1 — Atomic does not mean finitely atomic

**Status:** `withdrawn`

**Correction of terminology · [lines 420–421](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:420).**

1. `withdrawn` — **Atomic measures need not have finite support.** “Agrees exactly with the finite sum on atomic measures” includes countably infinite atomic measures.

   **Proposed fix:** Use “agrees exactly with the finite-network sum on finitely supported measures.”

**Overall fix:** Replace “atomic” by “finitely supported” where the claim concerns a finite network sum.

### M2 — The test-function construction requires disjoint closures

**Status:** `applied`

**Proof-gap repair in the neighborhood choice · [lines 741–758](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:741).**

1. `applied` — **Disjoint open sets can have intersecting closures.** The displayed cutoff is positive throughout $U_i$, so its support is $\overline{U_i}$. Pairwise disjoint open neighborhoods alone do not justify the later assertion of disjoint supports or ensure that those supports lie inside $U$.

   **Proposed fix:** Choose the finitely many neighborhoods with pairwise disjoint compact closures contained in $U$. Then use outer regularity to shrink them while retaining the compact sets $C_i$; the closure properties are preserved.

**Overall fix:** Strengthen the neighborhood choice to “open neighborhoods with pairwise disjoint compact closures contained in $U$.” The existing cutoff construction then has the required support properties.

### M3 — Allow finite and empty atom index sets

**Status:** `withdrawn`

**Minor indexing correction · [lines 820–824](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:820), also [957–959](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:957).**

1. `withdrawn` — **The enumeration should include finite and empty atomic parts.** Writing $\sum_{j\ge1}c_j\delta_{\omega_j}$ while requiring every $c_j\ne0$ literally selects an infinite atom sequence.

   **Proposed fix:** Use $\sum_{j\in I}c_j\delta_{\omega_j}$ with $I$ at most countable, possibly empty.

2. `withdrawn` — **The limiting argument should use the same indexing convention.** “The first $N$ atoms” needs a convention when there are finitely many or no atoms.`z

   **Proposed fix:** Pass through finite subsets exhausting $I$, taking an eventually constant sequence of subsets when $I$ is finite.

**Overall fix:** Use an at-most-countable index set and finite-subset exhaustion throughout these two passages. This repairs notation without changing the theorem.

### M4 — Concentration on a set is different from topological support

**Status:** `withdrawn`

**Optional terminology clarification · [lines 971–974](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:971).**

1. `withdrawn` — **“Supported on” can mean concentration in measure-theoretic usage.** Under that reading the sentence is valid. It can nevertheless be confused with topological support, which contains the closure of the atom set and can include points with no point mass.

   **Proposed fix:** For consistency with the earlier distinction in the paper, optionally write: “The atomic part is concentrated on the countable atom set, while the nonatomic part assigns that set zero total variation.”

**Overall fix:** “Concentrated on” is clearer here. The original wording is not, by itself, a false mutual-singularity argument.

### M5 — An example of escape is not a statement about every escaping sequence

**Status:** `applied`

**Correction of the scope of an example · [lines 1009–1010](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1009).**

1. `applied` — **Failure of dictionary decay allows escape effects but does not force them for every sequence.** The sentence “Mass escaping to infinity carries a non-vanishing value contribution unless $K(\omega)$ tends to zero” is stronger than the example establishes.

   **Proposed fix:** Use “This example shows that mass escaping to infinity can retain a nonzero contribution when the dictionary does not vanish at parameter infinity.”

**Overall fix:** State the possibility demonstrated by the example, using “can retain.”

### M6 — Say precisely what normalization prevents

**Status:** `withdrawn`

**Clarification of what normalization achieves · [lines 1031–1032](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1031), also [3790](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3790).**

1. `withdrawn` — **Normalization does not forbid parameter escape.** If $|\omega_n|\to\infty$ and $\mu_n=w_p(\omega_n)^{-1}\delta_{\omega_n}$, then $(\mu_n)_p=\delta_{\omega_n}$ escapes while its penalty remains $\phi(1)$. The growth gap makes the represented contribution vanish.

   **Proposed fix:** Replace “prevent the escape” with a statement about controlling the effect of escape: “Under the growth gap, escaping normalized mass has a vanishing represented contribution.”

2. `withdrawn` — **The function-space norm depends on the growth hypothesis.** The existence argument uses $L^2(D)$ decay; $H^1(D)$ decay requires the stronger gradient-growth condition.

   **Proposed fix:** Name $L^2(D)$ in the existence discussion and use $H^1(D)$ only where its hypotheses have been imposed.

**Overall fix:** Explain that the location-weighted penalty and normalization address the loss of weak-* lower semicontinuity by making escaping contributions vanish in the appropriate norm. They need not prevent atom locations from escaping.

### M7 — The problem admits a minimizer; its infimum is attained

**Status:** `withdrawn`

**Optional editorial precision · [lines 1233–1236](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1233).**

1. `withdrawn` — **The displayed infimum denotes a scalar, although its use as shorthand for the minimization problem is understandable.** “The infimum … admits a minimizer” can be written more directly without changing the mathematical claim.

   **Proposed fix:** Use “The infimum … is attained by at least one $\bar\mu\in\mathcal M_p(\Omega)$.”

**Overall fix:** Optionally replace “admits at least one global minimizer” after the displayed infimum with “is attained by at least one …”. Retain the existing nonuniqueness-compatible qualification.

### M8 — Population integration and empirical averaging have different normalization

**Status:** `withdrawn`

**No identified error; optional normalization note · [lines 400–402](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:400) and [498–507](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:498).**

1. `withdrawn` — **The two losses have different normalizations.** The population loss uses Lebesgue measure; the empirical loss uses a sample mean. For uniform samples with exact labels, the latter has expectation $L/|D|$. The paper does not assert that it is an unbiased estimator of $L$.

   **Proposed fix:** No correction is required solely because it is called the “corresponding empirical fidelity.” Optionally call it the “empirical mean loss.”

2. `withdrawn` — **A quantitative population-to-sample comparison would need the normalization.** A relation between regularization parameters would also depend on the sampling distribution and loss scaling.

   **Proposed fix:** Specify those details if such a comparison is made. Do not add assumptions or rescale the current loss merely to answer a comparison the paper does not make.

**Overall fix:** Keep the current definitions. A short normalization note is optional; no equality or unbiasedness claim has been found to correct.

### M9 — Define the space of admissible derivative directions

**Status:** `applied`

**Clarification of the derivative domain · [lines 1302–1307](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1302).**

1. `applied` — **“Admissible direction” is not defined at the derivative formula.** The fidelity is extended-valued under the weaker existence assumptions, so the formula needs a domain where the $H^1$ residual and directional output are defined.

   **Applied fix:** The sentence now reads: “Under Assumption 3.1 and $p>s_1$, the Gâteaux derivative at $\mu\in\mathcal M_p(\Omega)$ in any direction $\mu'\in\mathcal M_p(\Omega)$ is …”.

**Overall fix:** State the derivative's domain before using the formula. Keep the existing measure/function pairing convention and preserve the weaker assumptions of the separate existence theorem.

### M10 — Clarify the necessary conditions and the compact-set remark

**Status:** `applied`

**Mixed: optional title and clarification of the remark · [line 1363](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1363) and [1465–1469](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1465).**

1. `withdrawn` — **The theorem states necessary conditions.** Its hypotheses and conclusion already make that logical direction clear; “The optimality condition” does not by itself assert sufficiency.

   **Proposed fix:** Optionally use the more descriptive title “Necessary conditions for local optimality.”

2. `applied` — **The compact-set remark originally used the profile of a local minimizer.** For that profile, the theorem already excludes any threshold exceedance, so the remark's intended algorithmic observation was vacuous.

   **Applied fix:** The remark now considers any $\mu\in\mathcal M_p(\Omega)$ under Assumption 3.1 and $p>s_1$, and states that all points with $|P_{p,\mu}(\omega)|>\alpha L_\phi$ lie in the compact set $\{|P_{p,\mu}|\ge\alpha L_\phi\}$. The final sentence still refers to the lemma localizing the minimizer's support.

**Overall fix:** Applied the compact-set observation to an arbitrary measure. The optional theorem-title change remains withdrawn.

### M11 — A lost guarantee is not a counterexample

**Status:** `applied`

**Correction of the example title · [line 1831](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1831), with [1865–1873](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1865).**

1. `applied` — **The example establishes degeneration of a guarantee.** The body shows that the curvature hypothesis fails at the convex endpoint and the guaranteed separation scale tends to zero. It does not construct minimizers violating separation.

   **Proposed fix:** Replace “separation fails for convex penalty” with “Degeneration of the separation bound at the convex endpoint.”

**Overall fix:** Change the title to match the conclusion already stated in the body.

### M12 — Give the ReLU-power row its regularity range

**Status:** `withdrawn`

**Clarification of the table scope · [line 1969](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:1969).**

1. `withdrawn` — **The growth exponents and the full regularity assumptions are different claims.** The ReLU-power row's growth bounds can be useful at $k=1$, but the surrounding $C^1$ setting excludes ReLU itself. The comparison with Section 4 uses $k\ge2$.

   **Proposed fix:** Add “$k\ge2$” to the row if it is intended to list activations satisfying all the standing hypotheses. If it lists growth exponents only, state that limitation explicitly.

**Overall fix:** Make the table's intended scope explicit. The shortest consistent choice is to label the row $(z_+)^k$, $k\ge2$.

## 4. Homogeneous formulation

### H1 — “Every atom” and “any finite representation” include zero-feature neurons

**Status:** `applied`

**Correction of the rescaling statement · [single-neuron statement](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2079) and [finite-network statement](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2099).**

1. `applied` — **The normalization is undefined at a zero inner parameter.** “Any finite representation” includes $\omega_n=0$, whereas $\omega_n/|\omega_n|$ is then undefined.

   **Applied fix:** Restrict the finite-network statement to representations with nonzero inner parameters before defining the normalized parameters.

2. `applied` — **Define the parameter rescaling explicitly.** The original function-equality constraint required $K(\widehat\omega)\ne0$ to deduce the coefficient relation. That restriction is unnecessary when the rescaling itself is prescribed.

   **Applied fix:** Define $\omega=\lambda\widehat\omega$ and $c=\widetilde c\lambda^{-k}$ for $\lambda>0$, remove the condition $K(\widehat\omega)\ne0$ and its use in the proof, and restrict the finite-network conclusion to independent radial rescalings of this form. The proposed exclusion of all zero-function neurons is superseded by this formulation.

3. `withdrawn` — **The introductory sentence should use the same scope.** “Every atom can be moved onto the unit sphere” includes the degenerate cases just excluded.

   **Proposed fix:** Use “Every nonzero atom function can be represented with a unit inner parameter after rescaling its outer coefficient.”

**Overall fix:** Prescribe the parameter rescaling directly and require nonzero inner parameters for normalization. The introductory wording change in H1.3 remains withdrawn.

### H2 — The exponent is $q$, not the function $\psi_k$

**Status:** `applied`

**Minor wording correction · [lines 2160–2161](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2160).**

1. `applied` — **The exponent is $q$.** After defining $\psi_k(t)=t^q$, “which is the exponent” had an ambiguous antecedent.

   **Applied fix:** The sentence now reads: “Here $q$ is the exponent obtained in Lemma 4.4 when $\xi=\zeta=2$.”

**Overall fix:** Named $q$ explicitly in the explanatory sentence.

### H3 — Continuity into $H^1$ does not imply continuity of sampled gradients

**Status:** `withdrawn`

**Correction of the generality of the sampled extension · [lines 2239–2247](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2239).**

1. `withdrawn` — **$H^1$ continuity does not provide pointwise gradient evaluation.** For a general dictionary continuous into $H^1(D)$, sampled values and gradients need not be well defined or continuous in the parameter. Thus that assumption alone does not justify the extension to $l^M$.

   **Proposed fix:** For the general statement, assume that the sampled value-and-gradient dictionary is well defined and continuous on the sphere.

2. `withdrawn` — **The intended ReLU-power case already has the required property.** The following sentence establishes it for $k\ge2$.

   **Proposed fix:** Move that fact before the sampled-extension claim and scope the claim to it: “For $\mathrm{ReLU}^k$ with $k\ge2$, the sampled value-and-gradient dictionary is well defined and continuous on the compact sphere. Consequently, the following results also hold with $L$ replaced by $l^M$, using the corresponding sampled norms and bounds.”

**Overall fix:** State sampled continuity as the reason for the empirical extension. Either restrict to the established ReLU-power case or add the explicit assumption for a general dictionary.

### H4 — Explain the passage from a parameter minimizer to a merged measure

**Status:** `withdrawn`

**Proof-exposition clarification · [lines 2530–2532](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2530).**

1. `withdrawn` — **The parameter penalty and measure penalty treat duplicate locations differently.** $F_N$ sums penalties before merging, whereas $J_{\psi_k}$ uses the merged measure.

   **Proposed fix:** Explain that two nonzero coefficients at the same location cannot occur at a global minimizer of $F_N$: replacing them by their sum and a zero coefficient preserves the width allowance and output, and strictly decreases the $q$-penalty for $0<q<1$.

2. `withdrawn` — **The atomwise condition then follows from coefficient variations.** After duplicate nonzero coefficients are excluded and zero coefficients are discarded, the two objective values agree.

   **Proposed fix:** Add: “After discarding zero coefficients, $F_N=J_{\psi_k}$, and independent coefficient variations give the atomwise optimality condition.”

**Overall fix:** Insert the duplicate-merging explanation before applying the measure-level width bound to minimizers of $F_N$. The argument supports the stated conclusion without a new theorem.

## 5. Algorithms

### G1 — Do not silently reuse population certificates for sampled ones

**Status:** `withdrawn`

**Clarification of quantities used in the empirical algorithm · [lines 2834–2868](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2834).**

1. `withdrawn` — **The insertion residual changes when the loss changes.** The earlier $\Delta$ uses the population profile, while the algorithm tests the empirical profile $P^M_{p,\mu}$.

   **Proposed fix:** Define $\Delta^M(\mu,\omega)=(|P^M_{p,\mu}(\omega)|-\alpha L_\phi)_+$ and use it consistently, or explicitly announce that $\Delta$ denotes this sampled quantity throughout the algorithm subsection.

2. `withdrawn` — **The feature bound and search radius also change.** The population $B_p$ and $R(\mu)$ depend on the population feature and residual norms.

   **Proposed fix:** Define their sampled counterparts, or include them in the same convention: all profiles, residuals, feature bounds, and radii in this subsection are computed with $l^M$, its supplied labels, and the corresponding sampled growth constants.

3. `withdrawn` — **The insertion proof transfers through its Hilbert-space structure.** The empirical loss is one half of a squared Euclidean norm of the sampled value-and-gradient residual. The needed sampled feature bounds must also hold.

   **Proposed fix:** Add one sentence explaining this substitution and state the sampled bounds used. This supplies the link to the theorem without duplicating its proof.

**Overall fix:** Adopt one explicit sampled convention for all insertion quantities and explain the finite-dimensional version of the theorem. A global renaming of every symbol is unnecessary if the convention is clear.

### G2 — Pruning is outside the stated decrease guarantee

**Status:** `withdrawn`

**Clarification of the decrease guarantee · [lines 2622–2627](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2622), [2919–2920](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2919), [3032–3033](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3032), and [3119–3120](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3119).**

1. `withdrawn` — **The stated acceptance rule covers correction before pruning.** The pseudocode then removes small nonzero coefficients. Penalty reduction alone does not guarantee that the accompanying fidelity increase is smaller.

   **Proposed fix:** State that the one-candidate decrease guarantee applies before tolerance-based pruning, unless an objective check for pruning is included.

2. `withdrawn` — **A claim about the whole outer iteration would need to include pruning.** This issue concerns the scope of the guarantee; it does not show that the recorded numerical iterates increased.

   **Proposed fix:** If full-iteration monotonicity is intended, require the pruning step to preserve the decrease and document its acceptance test. Otherwise keep the guarantee explicitly attached to insertion and accepted correction.

**Overall fix:** Locate the decrease guarantee before pruning, or include pruning in the objective acceptance rule. Do not infer numerical failure from the absence of that rule.

### G3 — Repair the incomplete explanation of finite-amplitude insertion

**Status:** `applied`

**Correction of an incomplete sentence and its reference · [lines 2898–2905](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2898).**

1. `applied` — **The explanation is grammatically incomplete.** “Since the zero network is a local minimizer and the infinitesimal” leaves the mathematical reason unfinished.

   **Applied fix:** Explain that globally minimizing the objective change over the inserted coefficient permits a finite jump out of the local minimum at zero whenever the chosen atom admits a decrease. Retain the reference to the singular-slope remark.

2. `applied` — **The cited atom-weight theorem does not give the displayed initialization.** It supplies necessary conditions and weight bounds at existing atoms.

   **Applied fix:** Describe the theorem's weight bound as guidance for initialization away from zero, and distinguish it from the implemented approach: globally minimize the objective change over the inserted coefficient and accept the insertion only if this change is negative.

**Overall fix:** Distinguish initialization guided by the theorem from the insertion method used in the algorithm, and explain why the latter can escape the local minimum at zero.

### G4 — Normalize inside the sphere search objective

**Status:** `pending`

**Correction of the algorithm description · [lines 3020–3021](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3020).**

1. `pending` — **Normalizing only after optimization describes a different search.** The unnormalized homogeneous profile satisfies $P(r\omega)=r^kP(\omega)$ and is unbounded in magnitude if it is nonzero somewhere. The [implemented closure](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/src/PDAP/insertion.py:285) normalizes before feature evaluation.

   **Proposed fix:** Write “Run L-BFGS on $z\mapsto|P^M_{\mu_t}(z/|z|)|$ for nonzero $z$, then return $\omega=z/|z|$.” State the numerical small-norm safeguard only if needed to specify the implementation fully.

**Overall fix:** Put the normalization inside the search objective so that the pseudocode describes the implemented sphere search.

### G5 — A finite search finds no accepted candidate, not necessarily no descent direction

**Status:** `withdrawn`

**No required correction; optional stopping clarification · [lines 2911–2912](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2911), [3023–3025](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3023), and [3123](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3123).**

1. `withdrawn` — **The pseudocode already refers to the returned candidates.** “Keep the candidates satisfying …; terminate if there is none” follows the multistart search explicitly. In that context it does not claim that every possible direction was searched.

   **Proposed fix:** Retain the pseudocode. If a standalone stopping description is desired, use “Stop when none of the candidates returned by the multistart search passes the acceptance test.”

2. `withdrawn` — **The search limitation is already stated elsewhere.** The algorithm discussion and conclusion distinguish multistart search from the exact maximization used in the rate theorem.

   **Proposed fix:** Retain those existing scope statements. Add no separate global-certification disclaimer under this item.

**Overall fix:** No mandatory edit. The longer stopping sentence is available only as an optional clarification outside the pseudocode's existing context.

### G6 — Local-solver wording — withdrawn

**Status:** `withdrawn`

**Objective descriptions are acceptable in context · [lines 2810–2824](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:2810).**

1. `withdrawn` — **“Minimizes” can describe the optimization task.** “Algorithm … minimizes” and “correction then minimizes” introduce the objectives being solved. They do not alone assert that a finite local-solver run attains a minimum.

   **Proposed fix:** Retain these descriptions. The proposed blanket replacement by “targets” is unnecessary.

2. `withdrawn` — **The actual solver guarantee is already explicit.** The manuscript says that the method is local and accepts correction only when it does not increase the post-insertion objective.

   **Proposed fix:** Retain that acceptance rule. Address the separate pruning issue under G2.

**Overall fix:** No manuscript change under G6. Keep the objective descriptions and the existing statement of the solver's acceptance condition.

## 6. Numerical wording and captions

### N1 — Give sampled error formulas their actual labels and normalization

**Status:** `applied`

**Clarification of the reported metrics · [lines 3020–3030](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3020) and [3119–3121](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3119).**

1. `withdrawn` — **The formulas should identify the actual evaluation labels.** The manuscript already says the norms are empirical, so this is a notation clarification rather than evidence of a claimed exact-error theorem. Using $(v^m,g^m)$ also avoids writing a classical gradient at a nondifferentiable sample.

   **Proposed fix:** For the evaluation index set $E$, write

   $$
   \mathrm{Err}_{L^2}
   =\left(\frac{\sum_{m\in E}|\widehat V(x^m)-v^m|^2}
   {\sum_{m\in E}|v^m|^2}\right)^{1/2},
   \qquad
   \mathrm{Err}_{H^1}
   =\left(\frac{\sum_{m\in E}\bigl(|\widehat V(x^m)-v^m|^2+
   |\nabla\widehat V(x^m)-g^m|^2\bigr)}
   {\sum_{m\in E}\bigl(|v^m|^2+|g^m|^2\bigr)}\right)^{1/2}.
   $$

   Keep the usual $L^2$ and $H^1$ error labels, with the existing explanation that they are evaluated empirically on the original physical scale.

2. `applied` — **The separately reported gradient error lacked a definition.** It was not one of the two preceding metrics.

   **Applied fix:** Replaced the gradient-only error comparison with the already-defined relative $H^1$ errors reported in the gradient-training table: softplus $0.223\to0.0311$, $\tanh$ $0.241\to0.0218$, and Gaussian $0.182\to0.0111$. The comparison uses the same six runs and selected checkpoints as the table.

   The Algorithm 2 discussion now also calls its metric the “relative $H^1$ error” in place of “combined value–gradient error.” Its values already matched the table and are unchanged.

**Overall fix:** Replaced the gradient-only comparison with the table's $H^1$ values. Retained the existing empirical metric definitions and physical-scale convention; N1.1 remains withdrawn.

### N2 — A gradient feature is not just the activation's first derivative

**Status:** `applied`

**Correction of the feature description; narrowed · [lines 3181–3183](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3181).**

1. `withdrawn` — **The gradient feature includes the affine argument and slope vector.** It is $a\rho'(a\cdot x+b)$, not merely $\rho'$.

   **Proposed fix:** Name the full vector feature in the caption.

2. `applied` — **The feature family need not be a basis.** No linear independence is established or needed.

   **Applied fix:** Deleted the caption sentence “Under the gradient-augmented loss, the first derivative is the basis that reconstructs $\nabla_x V$.”

3. `withdrawn` — **“Reconstructs” does not itself assert exact recovery.** The earlier review's objection to that verb was unnecessary in this numerical context.

   **Proposed fix:** Withdraw that objection. Choose the verb for readability, not as a substitute for a mathematical qualification.

**Overall fix:** Deleted the caption's basis claim. N2.1 and N2.3 remain withdrawn.

### N3 — Describe the pendulum data as numerical branch data

**Status:** `applied`

**Clarification of the procedure and the source of nonsmoothness · [lines 3483–3484](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3483).**

1. `applied` — **“The open loop algorithm follows from …” leaves the adopted procedure vague.** The cited method integrates PMP characteristics backward from local LQR terminal data. [Han and Yang (2024)](https://proceedings.mlr.press/v242/han24a.html).

   **Applied fix:** Use “We follow the open-loop algorithm of” with the existing citation to Han and Yang. No additional procedural description was needed.

2. `applied` — **A finite data triple does not itself have a discontinuous gradient.** The discontinuity is a property of the underlying value function; the labels contain costs and costates.

   **Applied fix:** Use “The state-value-gradient triple is generated for a value function with a discontinuous gradient.”

**Overall fix:** Clarified the attribution of the open-loop algorithm and attributed the discontinuous gradient to the value function.

### N4 — The pendulum switches from finite to infinite horizon

**Status:** `withdrawn`

**Optional clarification of horizon conventions · [lines 3498–3503](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3498), [3642–3650](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3642), and [3703–3707](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3703).**

1. `withdrawn` — **The pendulum problem explicitly changes to infinite horizon.** The integral to infinity already defines that change, so reusing $V$ and $\mathcal C$ is not inherently an error.

   **Proposed fix:** For clarity, add “In this subsection, $V$ denotes the stationary infinite-horizon value function.”

2. `withdrawn` — **The rollouts and the comparison solve use a finite horizon.** The text already specifies $T=40$, a free terminal state, and zero terminal cost for the numerical open-loop reference.

   **Proposed fix:** Retain those details. Optionally connect the two roles with “We assess the learned feedback using truncated running costs over a finite rollout horizon.”

**Overall fix:** A short statement identifying the stationary target is sufficient. Keep the existing finite-horizon reference description; no additional optimality warning is called for.

### N5 — The pendulum gradient feedback formula needs its differentiability qualifier

**Status:** `withdrawn`

**Local differentiability clarification · [lines 3642–3646](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3642).**

1. `withdrawn` — **The displayed classical derivative is defined at differentiability points.** The introductory formula already carries this qualification, but repeating it beside the pendulum formula makes the local meaning explicit in the nonsmooth benchmark.

   **Proposed fix:** Begin “At states where $V$ is differentiable, Hamiltonian minimization gives …”.

2. `withdrawn` — **The rollout evaluates the fitted function's gradient.** The next sentence already replaces $V$ with $\widehat V$. This alone does not require a new branch-selection rule for the exact value function.

   **Proposed fix:** Retain that implementation description. Specify a branch or generalized-gradient rule only if the experiment actually evaluates such a rule for the nonsmooth target; none is required merely to use a smooth fit.

**Overall fix:** Add the local differentiability qualifier to the exact formula and retain the current fitted-feedback description. Do not invent a branch-selection requirement for the learned controller.

### N6 — The parameter cross-references do not lead to parameter values

**Status:** `applied`

**Correction of two cross-references · [frontier caption](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3550) and [surface caption](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3596).**

1. `applied` — **The frontier caption points to a table without hyperparameters.** The referenced pendulum feedback table lists widths, costs, and stabilization outcomes.

   **Applied fix:** Deleted the sentence claiming that the selected runs and parameters are reported in the feedback table.

2. `applied` — **The surface caption points to a figure without parameter values.** Figure 1(b)'s caption does not provide them either.

   **Applied fix:** Deleted the sentence claiming that the parameter values are reported in Figure 1(b).

**Overall fix:** Removed both unsupported claims about where the parameter values are reported.

### N7 — Distinguish the two different pendulum references

**Status:** `withdrawn`

**Optional reference-label clarification · [lines 3681–3697](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3681) and [3701–3707](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3701).**

1. `withdrawn` — **The table and figures use different reference constructions.** The table's caption identifies a sampled-PMP feedback heuristic; Figures 12–13 use separately solved finite-horizon open-loop trajectories. The body already explains both constructions.

   **Proposed fix:** Make the short labels equally specific: “sampled-PMP feedback” in the table and “finite-horizon PMP open-loop reference” in the figures.

2. `withdrawn` — **The selection rule for the open-loop reference is already disclosed.** It is the least-cost converged solve among the tested initializations.

   **Proposed fix:** Retain that statement. Do not append an additional global-optimality disclaimer.

**Overall fix:** Distinguish the reference types in their labels while preserving the existing construction and selection descriptions.

### N8 — Define the running-cost symbol and distinguish it from the training objective

**Status:** `applied`

**Definition clarification; optional notation change · [line 3413](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3413).**

1. `applied` — **The integrand $\ell(y,u)$ is not defined locally.** The running cost has already been given as $h(y)+\eta|u|^2$, and $\ell$ also denotes pendulum length.

   **Applied fix:** Replaced $\ell(y(s),u(s))$ with $h(y(s))+\eta|u(s)|^2$ in the definition of $J(t)$.

2. `withdrawn` — **$J(t)$ is reused for accumulated control cost.** Its argument distinguishes it from the training functional, but a local definition would make that convention clear. Renaming it is optional.

   **Proposed fix:** State “In the numerical figures, $J(t)=\int_0^t[h(y(s))+\eta|u(s)|^2]\,ds$ denotes accumulated running cost.” Alternatively, use $\mathcal C_{[0,t]}$ consistently if a notation change is preferred.

**Overall fix:** Defined the plotted cost directly using the known running-cost integrand. N8.2 remains withdrawn.

### N9 — Disclose Figure 12's cost-axis scale

**Status:** `withdrawn`

**Optional caption clarification · [line 3688](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3688), Figure 12(b).**

1. `withdrawn` — **Figure 12(b) uses a symmetric-log cost axis.** The [plotting code](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/scripts/paper/control_comparison.py:48) sets threshold $1$; the [caller](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/scripts/paper/pendulum_figures.py:332) enables it only for start A. Figure 13 therefore uses a different cost scale.

   **Proposed fix:** Add “The cost axis is linear up to $J=1$ and logarithmic above that threshold.”

**Overall assessment:** Withdrawn because the axis tick labels already make the logarithmic scale clear. Describing the linear portion below $J=1$ is optional and adds little to this comparison.

### N10 — Identify the reconstructed switching set as a numerical approximation

**Status:** `applied`

**Clarification of the evaluation region · [lines 3747–3764](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3747).**

1. `applied` — **The errors are evaluated near the discontinuity.** The title “Validation errors on gradient discontinuity” does not match the neighborhood used in the experiment.

   **Proposed fix:** Use “Validation errors near the gradient discontinuity.”

2. `withdrawn` — **The numerical distances use reconstructed points.** The body already mentions reconstruction, but “points on $S$” can suggest exact knowledge of the theoretical boundary.

   **Proposed fix:** Say “Numerical distances are computed to a reconstructed point set approximating $S$, together with its periodic copies.” Introducing a separate symbol $\widehat S$ is optional.

3. `withdrawn` — **The near/far split is within the evaluation set.** That is already implicit in the reported evaluation procedure.

   **Proposed fix:** If spelling out the subsets, define the near subset as evaluation points within distance $0.3$ of the reconstructed set and the far subset as the remaining evaluation points. No claim about evaluating the entire state domain is needed.

**Overall fix:** Change “on” to “near” in the title and state what point set is used for numerical distances. Keep the split tied to the specified evaluation set.

### N11 — The conclusion drops the strong-concavity hypothesis

**Status:** `withdrawn`

**Clarification of conclusion-level assumptions · [lines 3789–3792](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3789).**

1. `withdrawn` — **Concavity alone does not give all the listed sparsity conclusions.** Existence uses weaker assumptions, while finite support and separation use the stronger curvature hypothesis. The current abstract already distinguishes them, but the conclusion groups them under “the concave penalty.”

   **Proposed fix:** Use “Under the respective growth and penalty assumptions, normalization gives existence; strong concavity on bounded intervals yields finite support, separation, and width bounds.”

2. `applied` — **The rate wording matches A2.** The conclusion now uses “first-order optimality residual for atom insertion” and retains the single-atom, exact-search, and monotone-correction hypotheses.

   **Proposed fix:** Retain the agreed residual wording and hypotheses from A2.

**Overall fix:** Separate the assumptions for existence and sparsity in the conclusion. The rate sentence is already aligned with the abstract under A2. No stronger existence assumptions are needed.

### N12 — Define what “global” means in the validation discussion

**Status:** `withdrawn`

**Optional definition of a comparison term · [lines 3559–3574](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3559) and [3804](/Users/chaoruiz/Documents/Repos/SparseNNforHJB/paper/paper_0805.tex:3804).**

1. `withdrawn` — **“Global” refers to the full validation set.** The discussion contrasts this error with the errors on subsets near and far from the gradient discontinuity. The intended meaning is recoverable from context.

   **Proposed fix:** Define that usage once, or write “relative $H^1$ error on the full validation set.”

**Overall fix:** Prefer “full validation set” when it shortens the explanation. No separate statement about global optimization or the entire state space is needed.

## 7. Smaller wording and grammar corrections

Each row is a separate, single-point item. Its final column gives both the proposed fix and the complete fix for that point. Where a mathematical issue is handled above, the row refers to that item rather than adding another requirement. The source locations are those of the original audit.

| Point | Status | Source lines | Wording or issue | Proposed and overall fix |
| --- | --- | --- | --- | --- |
| W1 | `applied` | 83–85 | The earlier review requested a replacement of “Value and gradient are generated.” | **Applied:** retain the current “Numerical estimates of the value and gradient.” No additional replacement under I1. |
| W2 | `applied` | 88 | “, We study” | Use “, we study”. |
| W3 | `applied` | 98 | The original acronym was PDPA. | **Applied:** retain PDAP. Lowercasing the expanded algorithm name is optional. |
| W4 | `withdrawn` | 115 | “the dynamic optimal control problem in the space-time domain” | Prefer “the following finite-horizon optimal-control problem”. |
| W5 | `withdrawn` | 205 | “sufficiently smooth” does not identify the regularity being invoked. | Refer to the applicable dictionary assumption; do not strengthen the separate continuous-activation existence theorem. |
| W6 | `applied` | 220 | The earlier review requested identification of the initial costate. | **Applied:** the current data paragraph defines $g^m$ as the initial costate. Retain it. |
| W7 | `applied` | 256–259 | “the gradient fitting” | Use “gradient fitting”, together with the substantive correction in I9. |
| W8 | `applied` | 256–259 | “the smooth activation functions” | **Applied:** removed the article to read “smooth activation functions”. The additional regularity explanation remains withdrawn under I9.2. |
| W9 | `applied` | 262 | “The difficulty is, such extension requires” | Use “Such an extension requires”. |
| W10 | `applied` | 266 | “admits infinite slope at 0” | Use “has infinite right derivative at zero”, in the range $k\ge2$ used here. |
| W11 | `applied` | 272 | “adress” | Use “address”. |
| W12 | `withdrawn` | 278 | “finite positive right slope” | Optionally use “finite positive right derivative at zero” to make the evaluation point explicit. |
| W13 | `applied` | 279 | “nonhomogeneous defined on the unbounded domain” | Use “nonhomogeneous, with inner parameters in an unbounded domain”; see I11 for the domain distinction. |
| W14 | `applied` | 284–285 | “of implied polynomial growth condition” | Use “satisfying the stated polynomial growth bound”. |
| W15 | `applied` | 287 | “we proved” in the contributions list | Use “we prove”, consistently with the other contributions. |
| W16 | `withdrawn` | 287 | “finite atomic” as an adjective | Prefer “finitely atomic” or “finitely supported”, consistently. |
| W17 | `withdrawn` | 288 | “the number of neurons of optimal network” does not distinguish the width bounds. | Use “the reduced width of every globally minimizing network” for the a priori global bound; retain the minimizer-dependent nature of the local bound. |
| W18 | `applied` | 290 | The contribution paragraph now names the first-order optimality residual for atom insertion. | **Applied:** retain A2's agreed residual wording and the current algorithmic hypotheses. |
| W19 | `applied` | 666 | “relaxed assmuptions” | Use “weaker assumptions”. |
| W20 | `applied` | 710–711 | “a finite subcover Let” | Use “a finite subcover. Let”. |
| W21 | `applied` | 1032 | “The space of measure of finite moment” | Use “The space of measures with finite $p$-th moment”. |
| W22 | `applied` | 1987 | “widely used in the Sobolev training” | **Applied:** removed the article to read “widely used in Sobolev training”, retaining “widely” and the existing citations. |
| W23 | `withdrawn` | 2615 | “sampled value–gradient norm” on the whole function dictionary | Say “sampled value–gradient seminorm” there, or “Euclidean norm of the sampled feature vector”. Finitely many samples do not define a norm on the whole function space. |
| W24 | `withdrawn` | 2596 | A numeric citation is the grammatical subject of “surveys generation methods”. | Use “Generation methods are surveyed in [citation].” |
| W25 | `withdrawn` | 3002–3003 | “has superlinear convergence rate locally under nonsingularity” | Use “converges locally superlinearly near a root whose Jacobian is nonsingular”, retaining the existing exact, undamped, fixed-smooth-branch qualification. |
| W26 | `applied` | 3117 | “The Algorithm …, Algorithm … use” | Use “Algorithms … and … use”. |
| W27 | `applied` | 3154 | “fixed bound $\lambda\le e^5$” without a local radial definition | Use “fixed radial bound $\lvert\omega\rvert\le e^5$”, or define $\lambda=\lvert\omega\rvert$ there. |
| W28 | `applied` | 3169 | “Effect of the gradient training” | Use “Effect of gradient training”. |
| W29 | `applied` | 3353 | “atom power $k$” | Prefer “activation degree $k$” or “homogeneity degree $k$”. |
| W30 | `withdrawn` | 3460 | “Algorithm 1 leaves them free” for atom norms | Use “Algorithm 1 permits varying atom norms within its search region”. |
| W31 | `applied` | 3494 | “We observe the pendulum example” | Use “We consider the pendulum example”. |
| W32 | `applied` | 3519 | “The open loop algorithm follows from” | **Applied:** “We follow the open-loop algorithm of” with the existing citation, as in N3.1. |
| W33 | `withdrawn` | 3589 | “The network shows limitation in approximating the optimal value function” generalizes from selected fits. | Describe the displayed fits, for example “The displayed fits show visible approximation errors,” and identify the observed feature if discussing it further. |
| W34 | `applied` | 3747 | “Validation errors on gradient discontinuity” | Use “Validation errors near the gradient discontinuity”, as in N10. |

**Overall fix:** Apply the remaining grammar and terminology edits individually. Keep rows marked applied, and treat optional word choices as optional. Do not use this table to reintroduce the withdrawn I1 or G6 recommendations.

## 8. Wording already qualified correctly

These are preservation checks, not additional findings requiring revisions.

1. **“The global minimum” does not assert a unique minimizer.** The minimum value and a minimizing argument are different objects. “Existence of a global minimizer” would emphasize attainment, but the definite article is not a uniqueness error.
2. **The weaker existence theorem is intentional.** It uses continuity of the activation, the value-growth bound, and $p>s_0$. Preserve those assumptions rather than importing the stronger gradient-growth assumptions of later results.
3. **The identified uniqueness statements have specific arguments.** These concern the inverse image under the normalization bijection, the zero minimizer in the zero-target/zero-dictionary cases, and the radial minimizer. P1 does not require weakening them.
4. **Reduced width is defined.** Repeated locations are merged and zero coefficients discarded. Support bounds concern that reduced representation; H1 separately addresses zero-feature neurons in the rescaling lemma.
5. **The detailed rate statement already specifies its scope.** The corollary identifies the residual and its hypotheses, and the conclusion distinguishes exact from multistart search. The summary terminology is aligned under A2; retain it.
6. **The nonconvex proximal map is a selected minimizer.** Membership in the argmin and the tie at zero are already specified; no uniqueness correction is needed.
7. **Numerical stabilization is explicitly defined.** The finite-time sampled criterion and its scope are stated in the body and caption. Retain them.
8. **The selection states are identified as selection states.** The manuscript explains how A and B enter model selection. No new independent-test disclaimer is needed.
9. **Figure 11 already identifies display clipping.** It says that the surfaces are clipped for visualization and that errors and feedback use the unclipped fit. P3 concerns the different operation of saturating applied controls.
10. **The target-region figure identifies its numerical construction.** Its nearest-characteristic coloring is already described as an approximation. Retain that description.
11. **The data paragraph distinguishes exact identities from numerical labels.** It defines accumulated costs and initial costates and gives the gradient identity conditionally. Retain it; I1 is withdrawn.
12. **The correction solver's scope is stated.** The text describes a local method with a nonincrease acceptance rule. Keep this rule; G6 is withdrawn. Handle pruning separately through G2.

**Overall fix:** Preserve these qualifications and definitions. Avoid adding repeated caveats or weakening valid statements merely to make terminology uniform.

## 9. Combined revision plan

1. **Preserve applied and withdrawn points.** P1's article correction, A1, A2, A5, I3, I6, I7, I10, and I4's spatial-gradient notation are already applied. I1, I8, and G6 are withdrawn. M8 and G5 identify no required correction in the current context.
2. **Correct concrete mathematical and descriptive mismatches.** Retain the corrections concerning first-gradient fitting (I9), finite versus countable atoms (M1 and M3), the compact-neighborhood choice (M2), the rescaling scope (H1), and the sampled-dictionary scope (H3). Use the narrower example conclusions in M5 and M11.
3. **Make definitions and dependencies explicit where they matter.** Resolve the TPBVP derivative convention (P2), check the state-solvability/DPP scope (I2), explain the parameter-to-measure step (H4), and define the empirical insertion quantities (G1). Retain the now-explicit derivative domain (M9), the minimizer-existence sentence, and the applied HJB sign correction.
4. **Align the algorithm description with its operations.** Repair the finite-amplitude insertion explanation (G3), put normalization inside the sphere-search objective (G4), and specify the scope of decrease before or after pruning (G2). Preserve the existing descriptions of the local solver and candidate search.
5. **Align summaries and numerical descriptions.** Retain A2's applied residual sentence and A5's “discontinuous gradient”; state the homogeneous sparsity range (A4) and the conclusion's curvature assumption (N11). Clarify the stationary use of the initial-time value slice (I5).
6. **Repair the remaining metrics, references, and captions.** Define the standalone gradient metric (N1), correct the gradient features (N2), repair the parameter cross-references (N6), define the plotted cost (N8), and coordinate Figure 12's saturation, initial-time view, and axis-scale changes (P3, P4, N9). Apply the other numerical clarifications only to the extent described in their individual overall fixes.
7. **Finish with the single-point edits in Section 7.** Optional clarifications remain choices; they are not additional mathematical defects.

**Overall fix:** Make the retained local corrections, preserve the qualifications already present, and keep the abstract and conclusion consistent with the formal results. This revision changes the review document only; manuscript, figure, and implementation changes remain separate actions.
