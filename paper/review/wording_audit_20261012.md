# Mathematical wording audit of `paper_0805.tex`

Reviewed on 12 October 2026. This is an audit and a set of proposed corrections; no changes have been made to the manuscript, PDF, figures, or implementation.

The review covers the abstract, all six sections, theorem statements and proofs, algorithms, captions, and conclusion (source lines 75–3824). Locations refer to the source as reviewed. Figure numbering was checked against the existing `.aux` file. The control panel of Figure 12 was also inspected, and a few implementation details were checked where they determine the meaning of the prose. This is a wording and mathematical-precision review, not an independent certification of all proofs or numerical results.

Source SHA-256: `4d31eaeff939ccfd3c7d018f3e937d90cb522c811880369993538e233f7868d6`.

**Reading the findings:** **Correction** identifies a definite misstatement, missing condition in the displayed claim, or incorrect reference. **Clarification** identifies wording with a defensible interpretation that should be made explicit. **Verify** requires settling a mathematical convention or intended scope. **Presentation** concerns the professor's requested figure improvement. Suggested text is for review, not an applied edit.

The highest priorities are the TPBVP derivative convention, exact versus numerical optimality, the finite-horizon feedback interpretation, the scalar/measure penalty distinction, the scope of the convergence claim, and Figure 12's saturation disclosure.

## 1. The professor's comments

### P1 — Existence does not assert uniqueness

**Correction · [line 131](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:131), page 2.**

> “We assume the minimizer exists.”

No minimizer has been selected and uniqueness has not been established. The appropriate existential formulation is:

> “We assume that a minimizer exists.”

If the assumption is intended to apply to the full domain of the value function, be explicit: “For every initial pair $(t_0,x)$ under consideration, we assume that the optimal-control problem admits at least one minimizer.” Subsequently write “let $u^*$ be an optimal control.” Grammatically, **“a minimizer exists”** is correct; “a minimizer exist” is not.

This is not a reason to replace every occurrence of “the.” Once a particular minimizer is fixed, “the chosen minimizer” is unambiguous. The minimum **value** is also a single number even when several minimizers attain it.

### P2 — Define the derivative and duality in TPBVP

**Clarification · [lines 133–150](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:133), especially line 142, page 2.**

> `-d p*/dt = partial_y (f(y*) + g(y*)u*)^top p* + partial_y h(y*)`

The notation does not specify the Jacobian convention, the gradient convention, the space containing the costate, or what is held fixed during differentiation. In particular, the transpose could appear to apply before differentiation, and a reader could differentiate the feedback dependence of $u^*$ rather than holding the control fixed.

A precise convention is to identify $(\mathbb R^d)^*$ with $\mathbb R^d$ using the Euclidean inner product, represent gradients and costates by columns, and define

$$
F(y,u)=f(y)+g(y)u,\qquad
H(y,u,p)=h(y)+\eta|u|^2+p^\top F(y,u).
$$

Then state

$$
-\dot p^*(t)
=D_yF(y^*(t),u^*(t))^\top p^*(t)+\nabla h(y^*(t))
=\nabla_yH(y^*(t),u^*(t),p^*(t)),
$$

with $u$ and $p$ held fixed in the partial derivative. Explain that $D_yF$ has entries $\partial F_i/\partial y_j$, and its transpose is the Euclidean adjoint:

$$
\langle D_yF\,z,p\rangle_{\mathbb R^d}
=\langle z,D_yF^\top p\rangle_{\mathbb R^d}.
$$

Equivalently, retain a covector costate and use $D_yF^*p+D_yh$, defining the dual map. One explicit convention suffices. The state and costate differential equations should be understood almost everywhere for $L^2$ controls, consistently with line 150.

### P3 — Figure 12 must identify the saturation of the applied control

**Clarification · [lines 3646–3650](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3646) and [3722](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3722), Figure 12, page 40.**

> “Pendulum controls and realized costs from $A=(0.74,0.61)$.”

The body mentions $|u|\le30$, but the caption does not. The $\mathrm{ReLU}^3$ trace has long plateaus at $\pm30$; without the caption qualification these can look like a property of the raw learned feedback. It is the **applied control** that is saturated, not the cubic ReLU activation. This also differs from the value-surface display clipping already disclosed in Figure 11.

Suggested caption addition:

> “The learned controls are saturated at $\pm30$ before being applied to the dynamics. In particular, the long $\mathrm{ReLU}^3$ plateaus reflect saturation of the applied control.”

Define the implemented law near the experiment description:

$$
u_{\rm applied}(x)
=\operatorname{clip}_{[-30,30]}
\left(-\frac{1}{2\eta m\ell^2}\,
\partial_{\dot\theta}\widehat V(x)\right).
$$

Make clear that the associated realized costs use this saturated law. Do not describe saturation as mere plot clipping. The same clarification applies to Figure 13; the analogous Van der Pol bound is $50$ (line 3415). The experiment's unconstrained optimal-control problem and the saturated numerical controller should remain distinguishable.

### P4 — Figure 12 hides the initial transient

**Presentation · [lines 3709–3723](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3709), Figure 12.**

The full $[0,40]$ time axis compresses the initial peaks, sign changes, and convergence of several controls. Add an inset or a separate initial-time panel, provisionally $0\le t\le2$, choosing the final interval from the existing traces. Retain the full-time panel because it reveals the sustained Gaussian oscillations and $\mathrm{ReLU}^3$ saturation. State the zoom interval in the caption. No figure has been changed in this audit.

## 2. Abstract and introductory claims

### A1 — The abstract calls the measure functional a concave scalar penalty

**Correction · [line 92](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:92).**

> “where the concave scalar penalty $\Phi_\phi$ has finite positive right slope at zero”

The scalar function is $\phi:[0,\infty)\to[0,\infty)$; $\Phi_\phi$ acts on signed measures. The finite right derivative belongs to $\phi$. Moreover, $\Phi_\phi$ is generally not concave on the signed-measure space: opposite nonzero measures have positive penalty while their midpoint has zero penalty.

Suggested wording: “where $\Phi_\phi$ is the measure penalty induced by a concave scalar function $\phi$ satisfying $0<\phi'(0+)<\infty$.”

### A2 — Specify what converges at the claimed rate, and for which algorithm

**Correction · [lines 98–99](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:98), [290–291](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:290), and [3791–3792](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3791).**

> “which achieves a best-iterate rate of order $n^{-1/2}$”

Corollary 5.2 bounds the smallest **insertion-condition violation** $\min_{0\le k<n}\Delta(\mu_k)$. It assumes exact global candidate maximization, one-atom insertion, and a correction that does not increase the objective. It does not bound the objective gap, the fitted-function error, or the distance to a minimizer, and it does not directly cover the implemented multistart search. The detailed corollary correctly says this at lines 2787–2788; the summaries should preserve it.

Suggested wording: “For an idealized single-atom insertion scheme with exact global candidate search and a correction that does not increase the post-insertion objective, we prove an $O(n^{-1/2})$ best-iterate bound on the insertion-condition violation.” Also correct **PDPA** to **PDAP** at line 98.

### A3 — Clarify the normalization and local-minimality scopes in the abstract

**Clarification · [lines 93–104](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:93).**

> “with $p>0$”; “makes the dictionary vanish at infinity”; “every local minimizer”

Defining the weight only requires $p>0$, but dictionary decay requires a growth gap. The existence theorem uses $p>s_0$ and decay in $L^2(D)$; the gradient-level support results use $p>s_1$ and decay in $H^1(D)$. “Local” also depends on a topology: the weighted variation norm for the nonhomogeneous problem, and total variation on the sphere for the homogeneous problem.

Suggested wording: “For moment order exceeding the relevant dictionary growth exponent, normalization makes the dictionary vanish at infinity in the required function-space norm.” Identify the local-minimality topologies once in the summary, or refer explicitly to the definitions. Keep the weaker existence assumptions distinct from the stronger support assumptions.

### A4 — State the range of the homogeneous sparsity result

**Clarification · [lines 101–104](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:101), [263–267](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:263), and [293–304](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:293).**

The phrase “positively $k$-homogeneous” omits the range of $k$. At $k=1$, the quadratic rescaling gives $q=1$ and a finite slope, while the singular-penalty finite-support argument in Section 4 uses $k\ge2$ and $0<q<1$. The radial rescaling lemma itself has a broader range.

Suggested wording for the sparsity claims: “For positively $k$-homogeneous activations with $k\ge2$ and the stated dictionary regularity, …”. Specify that the equivalence is with the **radially optimized quadratic-parameter-regularized problem**, rather than suggesting equivalence with the nonhomogeneous objective.

### A5 — Attribute the gradient discontinuity to the intended benchmark

**Clarification · [lines 107–109](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:107).**

> “the Van der Pol oscillator and the pendulum swing-up problems, whose value function has a gradient discontinuity”

The singular relative clause can appear to apply to both problems. The switching-set discussion concerns the pendulum benchmark. Suggested wording: “We study the algorithms on the Van der Pol oscillator and pendulum swing-up benchmarks; the latter has a value function whose gradient jumps across a switching set.”

### I1 — Distinguish exact optimal data from numerical PMP labels

**Correction/clarification · [lines 81–85](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:81), [133–135](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:133), [180–181](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:180), [214–225](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:214), and [3100–3104](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3100).**

> “We recover optimal feedback laws”; “Value and gradient are generated … by the Pontryagin maximum principle”; “The optimal feedback law is then synthesized from the fitted value function”

PMP gives necessary conditions. A converged TPBVP solution is not automatically a global optimum; neither are its accumulated cost and initial costate automatically the exact value and value gradient. Similarly, fitting a value function and validating the resulting controller does not by itself prove optimality. The experiment's later “numerical open-loop reference” terminology is appropriately cautious.

Suggested wording: “We approximate optimal feedback laws using numerical value and costate labels from open-loop solves.” In the mathematical data description, distinguish the ideal case—costate of an optimal trajectory at a differentiability point—from the numerical case. Introduce $v^m$ as the numerical cost label approximating $V(x^m)$; this also supplies the missing definition for $v^m$ in the captions at lines 3112 and 3529. Use “candidate feedback law” for a fitted network. Optimality verification requires additional hypotheses and an argument; see [Liberzon, Section 5.1.4](https://liberzon.csl.illinois.edu/teaching/cvoc/node97.html).

### I2 — Make the control problem's well-posedness assumptions visible before using them

**Clarification · [lines 119–131](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:119) and [154–174](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:154).**

The stated $C^1$ regularity is local regularity; it does not ensure that every $L^2$ control generates a state on the entire interval, that the value is finite everywhere, or that all hypotheses for the dynamic-programming/viscosity statement hold. The stronger state-solvability assumption appears only at lines 2581–2584. For example, a smooth vector field can still have trajectories that blow up in finite time.

Suggested wording: identify admissible controls as those for which the state exists on the full horizon, or state the intended global solvability assumption near (P). Qualify the HJB statement by the hypotheses under which it is used, rather than treating $C^1$ alone as sufficient. This is an assumption-scope issue, not a request to prove a new control theorem. The assumptions must suit the present unbounded-control setting.

### I3 — Specify the viscosity sign convention

**Verify · [lines 158–170](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:158).**

The displayed HJB is $V_t+\inf_u\{\nabla_xV\cdot F+h+\eta|u|^2\}=0$. Classically it is equivalent to its negative. Under the conventional viscosity definition in which an upper test function must give $F\le0$, multiplication of an equation by $-1$ is not innocuous. For the backward, terminal-value minimization problem the usual operator is

$$
-V_t-\inf_u\{\nabla_xV\cdot F+h+\eta|u|^2\}=0.
$$

Specify the convention used for the displayed equation, or write the equation with the standard backward operator before asserting that it is satisfied in the viscosity sense. This matters precisely where the value function is nonsmooth. [Liberzon, Section 5.3.3](https://liberzon.csl.illinois.edu/teaching/cvoc/node104.html) explicitly explains this sign issue.

### I4 — Say “pointwise” at differentiability points, and identify the spatial gradient

**Clarification · [lines 164–172](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:164).**

> “At points where $V$ is differentiable, the equation holds classically”

The intended pointwise statement is different from claiming that $V$ is a classical solution on an open region. Because $V$ depends on both $t$ and $x$, the unqualified $\nabla V$ also needs a convention.

Suggested wording: “At points of differentiability in $(t,x)$, the HJB equality holds pointwise. Here and below, $\nabla_xV$ denotes the spatial gradient.” Use $\nabla_xV$ in the feedback formula. A pointwise Hamiltonian minimizer should not be conflated with a globally verified feedback law across nonsmooth points.

### I5 — Learning the initial-time slice does not recover the full finite-horizon feedback

**Clarification · [line 214](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:214), [lines 3412–3420](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3412), and [3801–3802](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3801).**

> “We write $V(x):=V(0,x)$.”

The finite-horizon feedback generally depends on $\nabla_xV(t,x)$ and on the remaining horizon. A network fitted only to $V(0,x)$ supplies the initial-time feedback. Applying its gradient for all simulation times gives a stationary law based on a fixed horizon, rather than the full time-dependent optimal feedback for a fixed terminal time.

This is the implemented Van der Pol interpretation: [the feedback function](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/scripts/paper/vdp_summary_figures.py:102) takes only the state and evaluates the fitted network's gradient.

Suggested addition: “For the finite-horizon benchmark, we learn the initial-time value slice and apply its gradient as a stationary candidate feedback throughout the rollout. This does not assert recovery of the complete time-dependent optimal feedback.” Treat the pendulum's stationary infinite-horizon value separately.

### I6 — A feedback law does not require solving HJB

**Correction · [line 176](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:176).**

> “one usually seeks a feedback law, which requires solving the HJB equation”

Feedback laws can be designed by many methods, including ones without an optimality claim. Even an optimal-control computation need not explicitly solve HJB over the state domain.

Suggested wording: “One route to an optimal feedback law is to compute the value function through the HJB equation.”

### I7 — Avoid universal robustness and necessity claims

**Clarification/correction · [line 153](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:153), [173–174](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:173), and [247](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:247).**

> “The open-loop control is not robust”; “corrects deviations”; “sparsity is necessary … such as knowledge distillation”

Open-loop control does not react to measured deviations, but actual robustness depends on the dynamics and disturbance model. Feedback reacts to the state, but “corrects” can suggest an unproved stability guarantee. Sparsity is useful for compact representations and can model point objects; knowledge distillation does not in general require a sparse network.

Suggested wording: “An open-loop control does not adjust in response to state disturbances”; “state feedback adapts the input to the current state”; and “sparse representations are useful for computational efficiency and for modeling point-like objects.” If retaining distillation, describe sparsity as one possible compression mechanism.

### I8 — The cited paper does not establish a universal failure of convex penalties

**Correction · [lines 253–255](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:253).**

> “convex penalties cannot effectively eliminate redundant neurons”

The claim should be that convex regularization **need not** remove all redundancy, rather than that it cannot do so. The cited paper describes possible overparameterization and examples with clustered neurons.

Suggested wording: “Pieper and Petrosyan show that convex $\ell^1$ regularization need not eliminate redundant or clustered neurons, and investigate nonconvex penalties to promote sparsity.” [Primary source, introduction and Section 5.1](https://arxiv.org/pdf/2004.11515).

### I9 — ReLU can be used for first-gradient fitting

**Correction · [lines 256–259](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:256).**

> “this formulation does not extend to the gradient fitting … the second-order weak derivative of ReLU does not exist as a function … we cannot use it to approximate high-order derivatives”

The argument confuses first-gradient fitting with second-derivative regularity. ReLU has a bounded weak first derivative, and finite ReLU networks belong to $H^1(D)$ on bounded domains. The paper itself uses a gradient-trained ReLU baseline. Its second distributional derivative is a Dirac measure in one dimension, so a $C^1$ activation assumption or higher-Sobolev regularity is a separate issue.

Suggested wording: “ReLU admits weak first derivatives and can be used in $H^1$ fitting. However, it is not $C^1$, and its second distributional derivative is not represented by a locally integrable function. The present smooth-dictionary analysis therefore requires additional regularity.” Name the particular result or optimization step needing that regularity instead of declaring gradient fitting impossible.

### I10 — The novelty comparison overstates the cited paper's restriction to ReLU

**Correction · [line 279](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:279), also [256](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:256).**

> “the original paper which only considers the ReLU case”

The cited paper focuses on ReLU but explicitly says that much of its theory extends to other activations and kernel methods. The sharper comparison is the move to the unbounded inner-parameter domain and the gradient-augmented fidelity, with the present normalization and growth assumptions.

Suggested wording: “Starting from the ReLU setting emphasized by Pieper and Petrosyan, we treat gradient-augmented fitting on the unbounded inner-parameter domain using a weighted normalization.” [Primary source, page 2](https://arxiv.org/pdf/2004.11515).

### I11 — Name the representation and the particular assumptions that fail

**Clarification · [lines 266–267](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:266), [280–285](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:280), and [1990–1992](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1990).**

> “the usual functional tools … are not applicable”; “a weak representation of the formulation”; “does not extend the result in Section 3”

These expressions leave the affected mathematical object or result unspecified. Infinite slope excludes the finite-slope hypotheses used in Section 3; it does not rule out standard existence methods generally. The representation theorem concerns the measure penalty as a supremum over test-function expressions, not an unspecified weak solution of the learning problem.

Suggested wording: “The infinite right derivative at zero falls outside the finite-slope assumptions of Section 3 and requires a separate argument.” For the contribution: “We represent the measure penalty as a supremum of weak-* continuous test-function expressions and deduce weak-* lower semicontinuity.” Also call $\rho$ an activation on $\mathbb R$ and $\Omega$ the inner-parameter domain; “activations on $\Omega$” blurs these two domains.

## 3. Measure theory and the nonhomogeneous formulation

### M1 — Atomic does not mean finitely atomic

**Correction · [lines 420–421](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:420).**

> “agrees exactly with the finite sum on atomic measures”

An atomic measure may have countably infinitely many atoms. Suggested wording: “agrees exactly with the finite-network sum on finitely supported measures.”

### M2 — The test-function construction requires disjoint closures

**Correction · [lines 741–758](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:741).**

> “pairwise disjoint relatively compact open neighborhoods”; later, “Their supports are pairwise disjoint.”

Disjoint open sets can have intersecting closures. The displayed cutoff is positive throughout its open neighborhood, so its support contains that neighborhood's closure. The required stronger choice is available for these finitely many disjoint compact sets.

Suggested wording: “Choose open neighborhoods with pairwise disjoint compact closures contained in $U$. By outer regularity, shrink them while retaining the $C_i$ to obtain the sets $U_i$.” This makes the later support assertion justified without changing the lemma.

### M3 — Allow finite and empty atom index sets

**Correction · [lines 820–824](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:820), also [957–959](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:957).**

> $\mu_{\rm atom}=\sum_{j\ge1}c_j\delta_{\omega_j}$, with distinct $\omega_j$ and $c_j\ne0$

As written, the indexing and nonzero requirement exclude finite and empty atomic parts, both of which the theorem covers. Suggested wording: use $\sum_{j\in I}$, where $I$ is finite or countable, possibly empty, and pass to the limit through finite subsets exhausting $I$. Adjust “the first $N$ atoms” accordingly. This is an indexing correction, not a failure of the theorem.

### M4 — Concentration on a set is different from topological support

**Clarification · [lines 971–974](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:971).**

> “The atomic part is supported on the countable atom set”

The atom set need not be closed; its closure may include points that carry no point mass. The manuscript correctly notes this at lines 361–362. The mutual-singularity argument needs measure concentration, not a topological-support assertion.

Suggested wording: “The atomic part is concentrated on the countable atom set, while the nonatomic part assigns that set zero total variation.”

### M5 — An example of escape is not a statement about every escaping sequence

**Correction · [lines 1009–1010](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1009).**

> “Mass escaping to infinity carries a non-vanishing value contribution unless $K(\omega)$ tends to zero”

Failure of dictionary decay does not mean that every escaping sequence has a nonvanishing contribution. Suggested wording: “This example shows that mass escaping to infinity can retain a nonzero contribution when the dictionary does not vanish at parameter infinity.”

### M6 — Say precisely what normalization prevents

**Clarification · [lines 1031–1032](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1031), also [3790](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3790).**

> “a penalty … aware of the location of the atom to prevent the escape”

Normalization does not prohibit locations, or normalized measure mass, from escaping. For $|\omega_n|\to\infty$, the measure $\mu_n=w_p(\omega_n)^{-1}\delta_{\omega_n}$ has constant normalized penalty $\phi(1)$ while $(\mu_n)_p=\delta_{\omega_n}$ escapes. The growth gap makes its represented contribution vanish.

Suggested wording: “We address the loss of weak-* lower semicontinuity using a location-weighted penalty on measures of finite $p$-th moment. Under the growth gap, escaping normalized mass has a vanishing represented contribution in $L^2(D)$.” Use $H^1(D)$ only under the stronger gradient-growth hypothesis.

### M7 — The problem admits a minimizer; its infimum is attained

**Correction, minor · [lines 1233–1236](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1233).**

> “$\inf_{\mu\in\mathcal M_p(\Omega)}J(\mu)$ admits at least one global minimizer”

An infimum is a scalar. Suggested wording: “The infimum … is attained by at least one $\bar\mu\in\mathcal M_p(\Omega)$.” The existing “at least one” qualification is correct.

### M8 — Population integration and empirical averaging have different normalization

**Clarification · [lines 400–402](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:400) and [498–507](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:498).**

> “the corresponding empirical fidelity”

The population measure is unnormalized Lebesgue measure, whereas $l^M$ is a sample mean. With uniform samples and exact labels, its expectation is $L/|D|$, not $L$. For nonuniform samples the sampling distribution also matters. No unbiasedness claim is presently made, so this is a clarification rather than a false equality.

Suggested wording: “For supplied sample labels, define the empirical mean loss …”. If relating population and empirical regularization parameters, state the measure normalization and sampling distribution explicitly.

### M9 — Define the space of admissible derivative directions

**Clarification · [lines 1302–1307](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1302).**

> “For a measure $\mu$ and an admissible direction $\mu'$”

The preceding fidelity is extended-valued, and the existence theorem uses weaker assumptions than those needed for differentiability everywhere on $\mathcal M_p$. “Admissible” has not been defined here.

Suggested wording: “Throughout this subsection, assume Assumption 3.1 and $p>s_1$. Then for $\mu,\mu'\in\mathcal M_p(\Omega)$, …”. Alternatively define the narrower finite-energy directions explicitly. The derivative's measure/function pairing at lines 1348–1352 is already a useful precise convention.

### M10 — Name necessary conditions as necessary

**Clarification · [line 1363](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1363) and [1465–1469](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1465).**

> “The optimality condition”; “any violation … is confined to a compact set”

The theorem supplies several necessary conditions, not an equivalence with local optimality. A more precise title is “Necessary conditions for local optimality.” In the subsequent compact-set discussion, $\bar P_p$ belongs to an already fixed local minimizer, so the theorem excludes any violation. State the useful algorithmic observation for an arbitrary iterate $\mu$: since $P_{p,\mu}\in C_0(\Omega)$, violations of the positive insertion threshold are contained in a compact superlevel set.

### M11 — A lost guarantee is not a counterexample

**Correction · [line 1831](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1831), with [1865–1873](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1865).**

> “separation fails for convex penalty”

The example shows that the curvature hypothesis fails at the convex endpoint and the guaranteed separation scale tends to zero. It does not exhibit minimizers violating separation.

Suggested title: “Degeneration of the separation bound at the convex endpoint.” The body already largely makes the correct distinction.

### M12 — Give the ReLU-power row its regularity range

**Clarification · [line 1969](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1969).**

> $(z_+)^k$ with $s_0=s_1=k$ and $p>k$

The admissible-activation discussion invokes $C^1$ regularity, which fails at $k=1$. Add “$k\ge2$” to agree with Section 4, or explicitly say that the row reports only growth exponents and does not assert all regularity hypotheses at $k=1$.

## 4. Homogeneous formulation

### H1 — “Every atom” and “any finite representation” include zero-feature neurons

**Correction · [lines 1995–1997](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:1995) and [2124–2132](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2124).**

> “every atom can be moved onto the unit sphere”; “consider any finite representation”

The single-neuron lemma assumes a nonzero coefficient and a nonzero represented atom function, but the finite-network conclusion drops these assumptions. $\omega/|\omega|$ is undefined at zero. A nonzero direction can also yield the zero function, such as $(a,b)=(0,-1)$ for ReLU powers. Preserving a zero function does not force the radial coefficient relation used in the proof. Merging duplicate locations and discarding zero coefficients (lines 1122–1124) does not remove all zero-feature neurons.

Suggested wording: “After discarding neurons representing the zero function, consider a finite representation with $c_n\ne0$ and $K(\omega_n)\ne0$ for every $n$.” Earlier: “Every nonzero atom function can be represented with a unit inner parameter after rescaling its outer coefficient.”

### H2 — The exponent is $q$, not the function $\psi_k$

**Correction, minor · [lines 2176–2184](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2176).**

> “define the scalar penalty … which is the exponent of Lemma …”

Suggested wording: “Here $q$ is the exponent obtained in Lemma 4.4 when $\xi=\zeta=2$.” This removes the ambiguous antecedent.

### H3 — Continuity into $H^1$ does not imply continuity of sampled gradients

**Correction of scope · [lines 2239–2247](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2239).**

> “its dictionary is continuous on the compact sphere, all results below hold with $L$ replaced by $l^M$”

The standing assumption at lines 2091–2093 is continuity of $K:\mathbb S^d\to H^1(D)$. For a general such dictionary, point values and gradient point values need not be well defined or continuous in the parameter. The next sentence establishes the needed property for the specific ReLU-power dictionary.

Suggested wording: “For $\mathrm{ReLU}^k$ with $k\ge2$, the sampled value-and-gradient dictionary is well defined and continuous on the compact sphere. Consequently, the following results also hold with $L$ replaced by $l^M$, using the corresponding sampled norms and bounds.” For a general dictionary, explicitly assume this sampled continuity.

### H4 — Explain the passage from a parameter minimizer to a merged measure

**Clarification, proof exposition · [lines 2530–2532](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2530).**

> “The bound above applies to every minimizer of $F_N$ … its coefficients can be varied independently.”

The compact parameter product used in the existence proof allows repeated locations and zero coefficients, whereas the measure penalty merges repeated locations. The conclusion is justified, but the identification should be stated.

Suggested addition: “At a minimizer of $F_N$, two nonzero coefficients cannot occupy the same location: merging them preserves the output and strictly decreases the $q$-penalty for $0<q<1$. After discarding zero coefficients, $F_N=J_{\psi_k}$, and coefficient variations imply the atomwise optimality condition.” This is an explanatory repair, not a claim that the existence theorem is false.

## 5. Algorithms

### G1 — Do not silently reuse population certificates for sampled ones

**Correction/clarification · [lines 2834–2868](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2834).**

> “$\Delta(\mu_t,\omega)>0$, that is, $|P^M_{p,\mu_t}(\omega)|>\alpha L_\phi$”

The earlier $\Delta$ uses the population profile $P_{p,\mu}$, and $R(\mu)$ uses a population $H^1$ residual and growth constant. These are not literally the same quantities as their empirical counterparts. The earlier convention replacing $L$ inside $J_{\psi_k}$ does not define this replacement for the nonhomogeneous certificates.

Define $\Delta^M(\mu,\omega)=(|P^M_{p,\mu}(\omega)|-\alpha L_\phi)_+$ and the sampled feature bound and search radius, or explicitly state that all certificate quantities in this subsection use sampled norms, labels, and the corresponding growth constants. Explain why the insertion proof transfers to that finite-dimensional feature space.

### G2 — Pruning is outside the stated decrease guarantee

**Clarification · [lines 2622–2627](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2622), [2919–2920](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2919), [3032–3033](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3032), and [3119–3120](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3119).**

The one-atom theorem guarantees decrease at insertion, and the accepted correction preserves that decrease. The algorithms then remove small nonzero coefficients without a displayed objective acceptance test. Pruning reduces the penalty but can increase the fidelity; small coefficient magnitude alone does not prove a net decrease.

Suggested wording: “The one-candidate decrease guarantee applies before tolerance-based pruning. Monotonicity of the complete outer iteration additionally requires pruning to preserve the objective decrease.” If a safeguard is intended, document it explicitly. This does not imply that the recorded numerical iterates actually increased.

### G3 — Repair the incomplete explanation of finite-amplitude insertion

**Correction · [lines 2928–2930](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2928).**

> “since the zero network is a local minimizer and the infinitesimal, we need good initialization method, which is provided by the Theorem …”

The sentence is incomplete. The cited atom-weight theorem gives stationarity and a lower bound at existing atoms; it does not directly give the displayed proximal initialization.

Suggested wording: “For $0<q<1$, the zero measure is a local minimizer and infinitesimal insertion cannot supply the finite-slope descent test. We therefore test finite-amplitude insertions and choose their coefficients by minimizing the one-atom objective increment.” Refer to the subsequent increment/proximal formulas for the initialization, and to Remark 4.9 for the singular-slope obstruction.

### G4 — Normalize inside the sphere search objective

**Correction · [lines 3020–3021](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3020).**

> “run unconstrained L-BFGS w.r.t. $|P|$ and normalize onto $\mathbb S^d$”

This reads as maximizing the unnormalized homogeneous profile and normalizing only afterwards. Such a profile satisfies $P(r\omega)=r^kP(\omega)$ and is unbounded in magnitude if it is nonzero somewhere. The actual [search closure](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/src/PDAP/insertion.py:328) normalizes before each feature evaluation.

Suggested wording: “Run L-BFGS on $z\mapsto |P^M_{\mu_t}(z/|z|)|$ for nonzero $z$, then return $\omega=z/|z|$.” Mention the numerical zero-norm safeguard if needed for reproducibility.

### G5 — A finite search finds no accepted candidate, not necessarily no descent direction

**Clarification · [lines 2911–2912](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2911), [3023–3025](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3023), and [3123](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3123).**

> “terminate if there is none”; “no candidate clears the insertion threshold”

Clarify “none among the candidates returned by the multistart search.” Otherwise stopping can be read as a globally certified insertion condition. The conclusion correctly acknowledges that the search is local; preserve that meaning in the pseudocode and stopping description.

### G6 — A local solver targets a minimization problem without necessarily attaining a minimum

**Clarification · [lines 2810–2824](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:2810).**

> “Algorithm … minimizes”; “correction then minimizes”

In context these can describe the optimization task, but the method is local and only a nonincrease acceptance condition is asserted. A finite solver run is not guaranteed to attain a local or global minimum. Prefer “Algorithm … targets the empirical objective” and “the coefficient correction applies a local solver to …”. Retain the explicit acceptance rule already given at lines 2607–2611.

## 6. Numerical wording and captions

### N1 — Give sampled error formulas their actual labels and normalization

**Clarification · [lines 3046–3056](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3046) and [3170–3172](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3170).**

> “relative errors” in $L^2$ and $H^1$; “all three relative gradient errors”

The text correctly says that norms are evaluated empirically, but the formulas use the exact $V$ and $\nabla V$, even though the actual labels are numerical values and costates. At switching points there is no classical $\nabla V$. The separately quoted relative gradient error is also not defined by either of the two preceding formulas.

Define the empirical errors using $(v^m,g^m)$ and the evaluation set, including

$$
\mathrm{Err}_{\nabla}
=\left(\frac{\sum_{m\in E}|\nabla\widehat V(x^m)-g^m|^2}
{\sum_{m\in E}|g^m|^2}\right)^{1/2},
$$

when the denominator is nonzero. Call $\mathrm{Err}_{H^1}$ a sampled relative value-and-gradient error, with the usual $H^1$ interpretation for appropriate samples and labels. Numerical-reference error is not automatically error against the exact value function.

### N2 — A gradient feature is not just the activation's first derivative

**Correction · [lines 3234–3235](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3234).**

> “the first derivative is the basis that reconstructs $\nabla V$”

The gradient features are the vectors $a\rho'(a\cdot x+b)$, with the affine argument and slope factor; they need not form a linearly independent basis. “Reconstructs” also suggests exact recovery.

Suggested wording: “The gradient channel uses the features $a\rho'(a\cdot x+b)$ to approximate $\nabla V$.”

### N3 — Describe the pendulum data as numerical branch data

**Clarification · [lines 3519–3520](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3519).**

> “The open loop algorithm follows from … The state-value-gradient triple is generated with discontinuous gradient.”

“Follows from” does not explain which procedure is adopted, and a finite data triple does not itself possess a discontinuous gradient. The discontinuity concerns the underlying value function across competing optimal branches.

Suggested wording: “We follow the numerical PMP-based data-generation procedure of Han and Yang. It supplies state, numerical value, and costate labels for a value function with a gradient jump across the switching set.” Retain the distinction between labels on smooth branches and labels assigned at nonsmooth points. The cited procedure uses backward characteristics initialized from a local LQR approximation; see [Han and Yang (2024)](https://proceedings.mlr.press/v242/han24a.html).

### N4 — The pendulum switches from finite to infinite horizon

**Clarification · [lines 3498–3503](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3498), [3642–3650](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3642), and [3703–3707](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3703).**

The introduction defines $\mathcal C$ and $V$ for a finite horizon with zero terminal cost; this benchmark reuses them for an infinite-horizon problem. The rollout and the numerical comparison reference then use $T=40$. These are three distinct roles for the horizon.

Suggested addition: “In this subsection, $V$ denotes the stationary infinite-horizon value function. We evaluate the resulting candidate controllers over a finite rollout horizon and compare their truncated running costs with a finite-horizon numerical PMP reference.” This avoids suggesting that the plotted finite-horizon reference is an independently computed exact infinite-horizon optimum.

### N5 — The pendulum gradient feedback formula needs its differentiability qualifier

**Clarification · [lines 3642–3646](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3642).**

> “gives the optimal control in feedback form”

This is precisely the example with a nondifferentiable value function. Suggested wording: “At states where $V$ is differentiable, Hamiltonian minimization gives …”. At the switching set, specify any branch or generalized-gradient selection if one is used; do not imply that the displayed classical derivative is defined there. The next sentence's “candidate feedback law” for $\widehat V$ is already appropriate.

### N6 — The parameter cross-references do not lead to parameter values

**Correction · [lines 3582–3583](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3582) and [3634–3635](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3634).**

> “parameters in Table …”; “parameter values reported in Figure …”

The referenced pendulum feedback table contains widths, costs, and stabilization outcomes, not the model hyperparameters. Figure 1(b)'s caption does not list those parameter values either. Supply the selected $(\alpha,\gamma,p)$ or $\alpha$ values in an actual table/caption and point to it. Until then, remove the assertion that these locations report the parameters. This is a definite reference error.

### N7 — Distinguish the two different pendulum references

**Clarification · [lines 3681–3697](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3681) and [3701–3707](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3701).**

The table's “reference (PMP)” is the sampled-PMP feedback heuristic, as its caption correctly explains. Figures 12–13 instead use a separately solved finite-horizon open-loop reference. Their different constructions can be missed when both are called “reference.”

Suggested row label: “sampled-PMP feedback.” Suggested figure label/caption term: “finite-horizon PMP open-loop reference.” Preserve the existing statement that the open-loop reference is the least-cost converged solve among the tested initializations; it is not a certified global optimum.

### N8 — Define the running-cost symbol and distinguish it from the training objective

**Clarification · [lines 3446–3454](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3446) and [3679–3694](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3679).**

> $J(t)=\int_0^t\ell(y(s),u(s))\,ds$

The running-cost function $\ell(y,u)$ is not defined in this passage, while $J$ has denoted the regularized learning objective and $\ell$ later denotes pendulum length. Function arguments distinguish some uses, but the accumulation of meanings is avoidable.

Suggested wording: define the accumulated control cost explicitly as $\mathcal C_{[0,t]}=\int_0^t[h(y(s))+\eta|u(s)|^2]ds$, or state clearly that $J(t)$ in the numerical figures denotes accumulated running cost, independently of the training objective. Defining the integrand directly avoids introducing another $\ell$.

### N9 — Disclose Figure 12's cost-axis scale

**Clarification · [line 3722](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3722), Figure 12(b).**

The [plotting code](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/scripts/paper/control_comparison.py:47) uses a symmetric-log scale with threshold $1$ for the cost panel of start A; [the caller](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/scripts/paper/pendulum_figures.py:634) enables it only for A. Thus Figure 13 does not use the same cost scale. This is useful to disclose when readers compare growth and relative distances between curves.

Suggested caption addition: “The cost axis is linear up to $J=1$ and logarithmic above that threshold.” Combine this with P3 rather than adding a disconnected note.

### N10 — Identify the reconstructed switching set as a numerical approximation

**Clarification · [lines 3747–3764](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3747).**

> “Validation errors on gradient discontinuity”; “Let $S$ denote the boundary …”; “Numerically, we use reconstructed points on $S$”

The reported errors are in a neighborhood of a reconstructed set, not on a mathematically known discontinuity boundary. The body already mentions reconstruction; make the notation and title retain that distinction.

Suggested title: “Validation errors near the gradient discontinuity.” Define $S$ as the theoretical switching set and $\widehat S$ as its reconstructed sample representation, or state that distances in the experiment are computed to the reconstructed point set. Specify that the near/far subsets are intersected with the evaluation domain. Do not claim an error directly on $S$.

### N11 — The conclusion drops the strong-concavity hypothesis

**Clarification · [lines 3789–3792](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3789).**

> “together with the concave penalty, yields existence, finite support, separation and width bounds, and a quantitative decay rate”

Mere concavity does not supply all these conclusions. The finite-support and separation theorem uses the stronger curvature assumption, while existence and insertion decrease have weaker hypotheses. The rate also has the algorithmic conditions in A2.

Suggested wording: “Under the respective growth and penalty assumptions, normalization gives existence; strong concavity on bounded intervals yields finite support, separation, and width bounds. An idealized insertion scheme satisfies the stated decay bound for the insertion-condition violation.”

### N12 — Define what “global” means in the validation discussion

**Clarification · [lines 3559–3574](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3559) and [3804](/Users/chaoruiz/Documents/Repos/sparse-nn-hjb/paper/paper_0805.tex:3804).**

> “global relative $H^1$ validation error”; “global validation accuracy”

Here “global” means the error on the full validation set, as opposed to its near-switching or far-switching subsets. It does not mean an error over the entire state space or a globally minimal error among all networks. Define this usage once, or write “relative $H^1$ error on the full validation set.”

## 7. Smaller wording and grammar corrections

These are separate from the mathematical findings above. Some are awkward rather than false, but fixing them makes the argument easier to read.

| Source lines | Current wording | Suggested wording |
| --- | --- | --- |
| 83–85 | “Value and gradient are generated” | “Numerical value and costate labels are generated” (see I1). |
| 88 | “, We study” | “, we study”. |
| 98 | “Primal-Dual-Active-Point (PDPA)” | “primal–dual active point (PDAP)”. |
| 115 | “the dynamic optimal control problem in the space-time domain” | “the following finite-horizon optimal-control problem”. |
| 205 | “sufficiently smooth” | Name the required regularity or refer to Assumption 3.1. Avoid suggesting $C^\infty$ where $C^1$ suffices. |
| 220 | “denote the costate” | “denote the initial costate value”; $g^m=p^{*,m}(0)$ is a vector, not the entire costate trajectory. |
| 256–259 | “the gradient fitting”; “the smooth activation functions” | “gradient fitting”; “smooth activation functions”, with the mathematical correction in I9. |
| 262 | “The difficulty is, such extension requires” | “Such an extension requires”. |
| 266 | “admits infinite slope at 0” | “has infinite right derivative at zero” (for the stated range of $k$). |
| 272 | “adress” | “address”. |
| 278 | “finite positive right slope” | “finite positive right derivative at zero”. |
| 279 | “nonhomogeneous defined on the unbounded domain” | “nonhomogeneous, with inner parameters in an unbounded domain”. |
| 284–285 | “of implied polynomial growth condition” | “satisfying the stated polynomial growth bound”. The continuous-activation existence claim itself is valid. |
| 287 | “we proved”; “finite atomic” | “we prove”; “finitely atomic” or “finitely supported”, consistently. |
| 288 | “the number of neurons of optimal network” | “the reduced width of every globally minimizing network” for the a priori global bound; distinguish the local minimizer-dependent bound. |
| 290 | “with $n^{-1/2}$ best-iterate rate” | “with an $O(n^{-1/2})$ best-iterate bound on …”, identifying the quantity and hypotheses (A2). |
| 666 | “relaxed assmuptions” | “weaker assumptions”. |
| 710–711 | “a finite subcover Let” | “a finite subcover. Let”. |
| 1032 | “The space of measure of finite moment” | “The space of measures with finite $p$-th moment”. |
| 1987 | “widely used in the Sobolev training” | “used in Sobolev training”; retain “widely” only if intended and supported. |
| 2615 | “sampled value–gradient norm” | “sampled value–gradient seminorm” on the function dictionary, or “Euclidean norm of the sampled feature vector”; finitely many samples do not define a norm on the whole function space. |
| 2596 | A numeric citation followed by “surveys generation methods” | “Generation methods are surveyed in [citation].” |
| 3002–3003 | “has superlinear convergence rate locally under nonsingularity” | “converges locally superlinearly near a root on a fixed smooth branch whose Jacobian is nonsingular”. This should remain a local Newton statement, not a guarantee for branch-changing iterations. |
| 3117 | “The Algorithm …, Algorithm … use” | “Algorithms … and … use”. |
| 3154 | “fixed bound $\lambda\le e^5$” | “fixed radial bound $|\omega|\le e^5$”, or explicitly define $\lambda=|\omega|$ here. |
| 3169 | “Effect of the gradient training” | “Effect of gradient training”. |
| 3353 | “atom power $k$” | “activation degree $k$” or “homogeneity degree $k$”. |
| 3460 | “Algorithm 1 leaves them free” | “Algorithm 1 permits varying atom norms within its search region”; its objective and capped numerical search still constrain the choice. |
| 3494 | “We observe the pendulum example” | “We consider the pendulum example”. |
| 3519 | “The open loop algorithm follows from” | “The open-loop data-generation procedure follows”, with the explanation in N3. |
| 3589 | “The network shows limitation in approximating the optimal value function” | State the observed limitation of the displayed fits, e.g. “The displayed fits deviate from the numerical reference near the switching region.” Do not infer a limitation of the network class from these selected runs. |
| 3747 | “Validation errors on gradient discontinuity” | “Validation errors near the gradient discontinuity”. |

## 8. Wording already qualified correctly

These checks matter because indiscriminate changes would introduce new inaccuracies.

- **“The global minimum” is not a uniqueness claim.** A minimum value and a minimizing argument are different objects. At line 95, “existence of a global minimizer” would be more explicit about attainment, but “the global minimum” is not wrong merely because minimizers may be nonunique.
- **The weaker existence theorem is intentional.** Lines 1230–1240 require only continuity of the activation, its value-growth bound, and $p>s_0$. Do not replace this with the full $C^1$/gradient-growth assumptions required for later results.
- **Several uniqueness statements are justified.** Lines 1257–1258 concern the inverse image of a fixed normalized measure under a bijection. The unique zero minimizer in the zero-target/zero-dictionary cases and the unique radial minimizer at line 2154 have explicit arguments. They should not be weakened merely to match P1.
- **Reduced width is defined.** Lines 1122–1124 merge repeated locations and discard zero coefficients. Later support-cardinality bounds are not bounds on every redundant parameter listing of the same network.
- **The detailed rate statement is appropriately limited.** Lines 2787–2788 explicitly deny an objective-gap or global-minimizer convergence conclusion; the conclusion also acknowledges the gap between exact search and multistart search.
- **The nonconvex proximal map is a selected minimizer.** Lines 2936–2941 use membership in an argmin and specify the tie at zero. This already avoids an unjustified uniqueness assertion.
- **Numerical stabilization is not called asymptotic stability.** Lines 3652–3659 give a finite-time sampled criterion and explicitly state its limitation. The table caption repeats the criterion.
- **The selection states are not presented as independent tests.** Lines 3557–3563 disclose how A and B are used in model selection.
- **Figure 11 already discloses display clipping.** Lines 3628–3631 explicitly say that surfaces are clipped to $[0,60]$ for visualization and that errors and feedback use the unclipped fit. The missing Figure 12 disclosure concerns control saturation instead.
- **The target-region figure already limits its claim.** Lines 3546–3551 identify nearest-characteristic coloring as an approximation and say that it does not compare branch costs at every grid point.

## 9. Suggested order for a later revision

1. Resolve P1–P3, I1–I5, A1–A4, G1, and G4, because they affect the meaning of the mathematical formulation or the headline guarantees.
2. Correct theorem/proof language in M1–M7, M11, H1, and H3; retain the distinctions listed in Section 8.
3. Repair numerical labels, parameter references, and captions, including the Figure 12 initial-time view.
4. Apply the smaller language edits and check that the abstract and conclusion use the same assumptions and terminology as the formal results.

No manuscript or figure edits are included in this report.
