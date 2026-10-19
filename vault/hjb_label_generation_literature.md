# Literature on HJB value and gradient label generation

Research date: 2026-10-19. Scope: primary sources on producing value samples `(x,V,gradV)` from nonlinear optimal-control problems. Page numbers below refer to the linked PDFs; arXiv copies use their own pagination. No numerical experiments were run for this note.

Recommendation: retain adaptive PMP collocation for the repository's current smooth, unconstrained VDP and 6/10-state Allen–Cahn datasets, subject to independent value and gradient checks. Compare against direct collocation or multiple shooting with consistent derivatives and a sparse nonlinear optimizer; use that solver to initialize difficult BVPs when needed. This is a recommendation for this implementation and target accuracy, not a literature-established ranking of all direct and indirect methods. The sources reviewed do not supply a matched-accuracy BB-versus-BVP benchmark for these exact repository problems.

## Azmi, Kalise, and Kunisch — gradient-augmented sparse regression (2021)

*Optimal Feedback Law Recovery by Gradient-Augmented Sparse Polynomial Regression*, JMLR 22(48):1–32, January 2021. [Published paper](https://www.jmlr.org/papers/volume22/20-755/20-755.pdf).

Section 2.1, pp. 6–8, explicitly generates cost and initial-costate labels using forward states, backward adjoints, alternating BB steps, and Dai–Zhang nonmonotone line search (Algorithm 1). The initialization is `u[-1]=0, u[0]=-G(0)`. Section 3.1 uses gradient tolerance `1e-5`. Examples include two-state Van der Pol (`dt=1e-4`, §3.2), 18-state Allen–Cahn (`dt=.005`, §3.3), and 80-state consensus (§3.4). This directly supports BB-based label generation in the relevant problem families. It establishes neither superiority over BVP methods nor global optimality: §2.1 describes finding a stationary point. Accuracy depends on state and adjoint integration.

## Nakamura-Zimmerer, Gong, and Kang — adaptive HJB learning (2021)

*Adaptive Deep Learning for High-Dimensional Hamilton–Jacobi–Bellman Equations*, SIAM Journal on Scientific Computing 43:A1221–A1247, 2021. [Author manuscript](https://arxiv.org/pdf/1907.05317), [publication DOI](https://doi.org/10.1137/19M1288802).

Sections 2.1–2.2 generate value/gradient labels through PMP BVPs, assuming the stationary solutions are optimal. Horizon continuation and neural-network warm starts address initialization sensitivity. On deliberately difficult rigid-body initial states, Tables 1–2 (p. 18) report 0.3% convergence without continuation, 98.4% with eight stages, and up to 99.7% with learned warm starts. These are not representative rates for arbitrary problems. Burgers examples have 10, 20, and 30 states (Tables 4–5, p. 23). Reported integration times include successful solves only. This supports initialized BVPs, not unconditional convergence.

## Nakamura-Zimmerer, Gong, and Kang — QRnet (2021)

*QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks*, IEEE Control Systems Letters 5:1303–1308, 2021; accepted manuscript November 2020. [Author manuscript](https://arxiv.org/pdf/2009.05686), [publication DOI](https://doi.org/10.1109/LCSYS.2020.3034415).

Sections II.B–III.A, pp. 2–3, obtain infinite-horizon value/costate data using finite-horizon BVPs. They extend the horizon until terminal running cost is small, reusing solutions, and initialize state trajectories with LQR and costates with `2Px`. Section IV uses a 64-state Burgers discretization, with initial states generated from ten sine modes; this is not full-cube sampling in 64 coordinates. The paper acknowledges multiple stationary solutions and assumes global optimality. It supports LQR warm starts and horizon checks, while illustrating that sampling geometry and stationary-versus-finite-horizon targets matter.

## Nakamura-Zimmerer, Gong, and Kang — locally stable neural control (2022)

*Neural Network Optimal Feedback Control with Guaranteed Local Stability*, IEEE Open Journal of Control Systems 1:210–222, 2022. [Accepted manuscript](https://arxiv.org/pdf/2205.00394), [publication DOI](https://doi.org/10.1109/OJCSYS.2022.3205863).

Section II allows box control constraints. Section IV.A, pp. 5–6, discusses direct and indirect data generation, including costate recovery through covector mapping. The 64-state Burgers example uses indirect BVPs (§V.A); the nonlinear UAV instead uses direct Radau pseudospectral optimization because indirect methods were unreliable (§VI.B, p. 9). Crucially, UAV experiments train only control networks: costate extraction required additional effort. This is evidence for problem-dependent solver choice, not a direct-versus-indirect gradient-accuracy comparison. Tight NLP tolerances and many collocation points are used for the direct data.

## Kunisch, Vásquez-Varas, and Walter — sparse feedback operators (2023)

*Learning Optimal Feedback Operators and their Sparse Polynomial Approximations*, JMLR 24(301):1–38, September 2023. [Published paper](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf).

Section 9 uses gradient descent with backtracking for open-loop reference controls, including the 19-state Allen–Cahn example in §9.3. The BB-based method in §6 optimizes polynomial feedback coefficients. These are different optimization problems. This paper learns feedback through trajectory costs rather than prescribing supervised value/costate label generation; it cannot justify attributing a BB open-loop algorithm to Walter’s §9. See the separate [Allen–Cahn audit](allen_cahn_reference_audit.md).

## QRnet authors’ implementation — direct-to-indirect refinement

The [official QRnet repository](https://github.com/Tenavi/QRnet#the-run-folder), accessed 2026-10-19, documents `qrnet.generate.refine` and `run.data.refine`: generate with a direct solver, then use the resulting trajectories to initialize an indirect solver. The README separately supports LQR or learned-controller initialization. This is a concrete implementation of a hybrid data-generation workflow. It does not establish comparative success rates or guarantee that refinement recovers the global optimum.

## General-purpose trajectory optimization

Kelly's *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*, SIAM Review 59(4), 2017, explicitly rejects a universal best method (§9.11). It discusses initialization difficulties of indirect methods (§9.4), direct-to-indirect initialization, and discretization refinement (§9.11). Its broad accuracy comparison is qualitative; it does not establish that an arbitrary indirect discretization outperforms a modern high-order direct scheme. [Author manuscript, pp. 28–30](https://www.matthewpeterkelly.com/research/MatthewKelly_IntroTrajectoryOptimization_SIAM_Review_2017.pdf#page=28).

Betts' *Practical Methods for Optimal Control Using Nonlinear Programming*, third edition, 2020, develops direct transcription, including implicit Runge–Kutta methods and PDE examples. It documents an established general-purpose approach rather than a BB-versus-PMP comparison. [Publisher description and contents](https://epubs.siam.org/doi/10.1137/1.9781611976199).

The official CasADi documentation, §8, illustrates direct single shooting, multiple shooting, and collocation using constrained Van der Pol dynamics. Its shooting example uses IPOPT and differentiated discrete dynamics. Multiple shooting adds state variables and continuity constraints, permitting state-trajectory initialization and often improving convergence, at the cost of a larger, sparser problem. Thus modern direct optimization includes substantially more than first-order descent on controls. [Official documentation, accessed 2026-10-19](https://web.casadi.org/docs/#optimal-control-with-casadi).

My synthesis: direct collocation or multiple shooting with automatic/consistent derivatives and sparse SQP or interior-point optimization is a defensible starting point for general problems with path, state, and control constraints. This is not a claim about measured market share, and it need not be the fastest choice for this repository's small unconstrained systems.

## What BB theory and costate approximation establish

Azmi and Kunisch, *On the convergence and mesh-independent property of the Barzilai–Borwein method for PDE-constrained optimization*, IMA Journal of Numerical Analysis 42(4):2984–3021, 2022, prove local R-linear convergence under smoothness and positive-definite Hessian assumptions, and study mesh independence. These are meaningful advantages for PDE optimization, not a convergence guarantee from every control initialization of a nonconvex problem. BB uses low memory and scalar curvature approximations; the paper also discusses nonmonotone behavior. [Author manuscript, §§1–3](https://static.uni-graz.at/fileadmin/_files/_homepages/_karl_kunisch/Papers/BBM_Meshind_KK_BA.pdf), [publication record](https://doi.org/10.1093/imanum/drab056).

Hager, *Runge–Kutta Methods in Optimal Control and the Transformed Adjoint System*, Numerische Mathematik 87:247–282, 2000, derives the discrete adjoint associated with a Runge–Kutta control discretization (§3). It generally differs from applying the state integrator unchanged to the continuous costate equation. This explains why independently discretizing the continuous gradient need not produce the exact derivative of the objective used by a discrete line search. [Author PDF, pp. 255–256](https://people.clas.ufl.edu/hager/files/rk.pdf#page=9).

Garg et al., *A unified framework for the numerical solution of optimal control problems using pseudospectral methods*, Automatica 46:1843–1851, 2010, derive transformations from nonlinear-program multipliers to costates. Their comparison shows that costate behavior depends on the transcription; accurate state/control approximations alone do not settle costate convergence. Direct methods can therefore produce gradient labels, provided the multiplier mapping and discretization are appropriate. [Author PDF, §§4–6](https://people.clas.ufl.edu/hager/files/unified.pdf).

## Recommendation and verification for this repository

The evidence supports both reduced-gradient and initialized BVP label generation. Solver choice alone does not certify a Value Sample: convergence to first-order conditions, global optimality, and label accuracy are distinct. In particular, BB failure in one implementation does not contradict Azmi–Kalise–Kunisch, while BVP success on one sample set does not establish universal reliability. A fair comparison must preserve the objective, horizon, sampling domain, and initialization budget, then validate values and costates under discretization refinement. Direct-to-indirect refinement is a documented fallback when both value and gradient labels are needed.

For the current generators, the practical ordering is:

1. Keep adaptive PMP collocation as the main implementation. Its coupled Newton solve, small state dimension, and explicit costates suit this task. This is a repository-specific engineering judgment, not a theorem about PMP.
2. Before judging BB, correct or account for the false rejection of accurate CN roots, check the discrete objective derivative, and implement the intended nonmonotone line search. The existing targeted [diagnostic report](bb_pmp_method_comparison.md) isolates the root-acceptance failure and measures gradient inconsistency. The paper's unguarded initialization `u[0]=-G(0)` should not itself be described as a deviation from Algorithm 1.
3. Use an independent direct collocation/multiple-shooting solver for a representative subset, especially large states and expensive trajectories. For BVP failures, use horizon continuation or the direct trajectory as an initial guess. Compare several starts where distinct stationary branches are plausible.
4. Measure success rate and runtime at comparable **value and initial-costate accuracy**, not equal numerical tolerance settings. Replay controls independently, refine time meshes, and check directional initial-state sensitivities. Small objective differences alone can conceal consequential gradient-label differences.

For substantially larger spatial PDE discretizations, reconsider reduced adjoint optimization instead of extrapolating the performance of a generic small-system BVP solver. BB remains a supported option there. Neither approach certifies global optimality in these nonlinear problems merely by reporting convergence; `p(0)=grad V(x)` as a value-gradient label additionally requires a differentiable optimal branch.
