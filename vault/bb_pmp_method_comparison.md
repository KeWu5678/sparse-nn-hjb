# BB control optimization and Pontryagin collocation

Research date: 2026-10-19. Reduced control-space descent and a coupled Pontryagin boundary-value solve approach the same first-order optimality conditions differently. Faster convergence of the latter can reflect its Newton iteration, adaptive mesh, and stopping criteria. A failure diagnosis additionally needs checks of the actual gradient and integration errors.

## Method and source distinctions

For `y' = f(y) + B u` and running cost `q(y) + beta |u|^2`, the reduced gradient is `g = 2 beta u + B^T p`. A forward state solve and backward costate solve evaluate this gradient. BB then updates `u` along `-g`, with a scalar step estimated from successive iterates and gradients. The original BB paper constructs its step sizes from a secant approximation. [Barzilai and Borwein, 1988](https://doi.org/10.1093/imanum/8.1.141).

Indirect collocation instead substitutes `u = -B^T p/(2 beta)` and solves the state/costate differential equations and endpoint conditions together. SciPy uses fourth-order collocation, mesh refinement driven by residual estimates, and damped Newton iteration with an affine-invariant merit function. [SciPy 1.17 documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_bvp.html).

“Direct versus indirect” and “gradient versus Newton” are separate distinctions. Direct methods discretize the control problem into a finite nonlinear program and can use Newton-type optimization. A reduced-space method using a continuous adjoint is not automatically discretize-then-optimize. [Diehl and Gros, *Numerical Optimal Control*, §§9.4, 13](https://www.syscop.de/files/2024ws/NOC/book-NOCSE.pdf#page=151).

Kunisch, Vásquez-Varas, and Walter specify gradient descent with backtracking for nonlinear **open-loop reference solves** in §9, p. 16. Their BB-based proximal algorithm in §6, pp. 10–11, updates **polynomial feedback coefficients**. The text does not establish that the open-loop references use BB. [JMLR paper](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=16).

## What convergence establishes

SciPy success means the collocation ODE residual meets the requested normalized RMS tolerance on each interval, and endpoint residuals meet an absolute tolerance. Failure can result from a singular collocation Jacobian or exhausting the mesh-node budget. These tolerances measure equation defects, not objective optimality. Different initial guesses can select different BVP solutions. [SciPy documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_bvp.html).

The repository's BB profile instead checks a quadrature approximation to the `L2` reduced-gradient norm. Equal numerical values of its tolerance and the collocation tolerance do not specify equal accuracy. With eliminated control, `2 beta u + B^T p` vanishes algebraically; its smallness supplies no independent evidence for the BVP solve. [VDP solver](../src/OpenLoop/vdp/solver.py), lines 228–329 and 493–516.

First-order necessary conditions alone do not distinguish all stationary solutions from local minima; global optimality requires additional structure or certification. Newton convergence is local and conditional. [Diehl and Gros, §§2–4, 12](https://www.syscop.de/files/2024ws/NOC/book-NOCSE.pdf#page=34).

## Diagnostic interpretation and fair comparison

These principles motivated the diagnostic experiments reported below:

- BB uses a scalar curvature estimate. A coupled Newton solve uses the full linearized system, so strongly unequal curvature scales can favor Newton locally. This does not rank all direct and indirect solvers.
- Verify that the computed gradient differentiates the objective actually passed to line search. For Runge–Kutta discretization, the exact discrete adjoint generally differs from applying the state integrator unchanged to the continuous adjoint. Hager derives the transformed adjoint and discrete cost gradient in §3, pp. 255–256. [Primary paper](https://people.clas.ufl.edu/hager/files/rk.pdf#page=9).
- At several controls, compare `[J_h(u+eps d)-J_h(u-eps d)]/(2 eps)` with the quadrature-weighted inner product `<g,d>_h`, sweeping `eps`. Include endpoint quadrature weights. Check nonlinear integration residuals separately from termination flags.
- Match dynamics, cost, horizon, initial states, control parameterization, and initialization. Refine both time discretizations; independently replay controls and recompute costs. Compare ODE, boundary, and stationarity defects, value labels, and initial-state sensitivities.
- Try both directions of warm-starting: use the BVP control in reduced-space optimization, and its state/costate path to initialize collocation. Use several starts and compare costs. Report failures, wall time, evaluations, and mesh sizes; raw iteration counts are not comparable work units.

For these smooth unconstrained examples, successful independently checked collocation is a reasonable reference generator. Resolving the reduced-gradient or time-integration failure remains necessary before drawing conclusions about BB as a method.

## Measured repository results

Tested current branch `657a36d` using Python 3.14.5, NumPy 2.4.2, and SciPy
1.17.1. Production solver files were not edited. Reproducers and numeric records
are in the ignored local directory
[`rawdata/reports/bb_pmp_comparison_20261019/`](../rawdata/reports/bb_pmp_comparison_20261019/).
These are targeted cases, not a failure-rate survey of the training domain.

### Rejected accurate Crank–Nicolson solves

At `x=(-3,-3)`, `T=3`, and 301 time nodes, the original BB profile fails at
the initial forward solve's step 103, before any BB update. SciPy's root status
is 5 (insufficient progress), but the CN equation residual is
`1.986258379993444e-16`. The code checks only `solved.success` at
`src/OpenLoop/vdp/solver.py:468`; it does not check whether the equation has
already been solved accurately. The same pattern occurs in the adjoint step
at line 488.

A diagnostic wrapper accepts a non-success result only when its finite
equation residual is at most `1e-10`. Changing only this acceptance rule makes
the same case converge in 42 BB iterations. Exactly one such root was accepted.

The default fine grid (`dt=1e-4`, 30,001 nodes) also reproduces the problem:

| Initial state | Configuration | Outcome | Cost | Reduced-gradient norm |
| --- | --- | --- | ---: | ---: |
| `(1,0)` | Original BB, default fine grid | Fails in line search at iteration 36 | 1.358188682726 | 0.02363 |
| `(1,0)` | BB, same grid, residual acceptance only | Converges in 55 iterations | 1.357186924838 | 8.782e-6 |
| `(1,0)` | PMP, 151 initial nodes, adaptive mesh | Converges | 1.357186913651 | Algebraically zero |

The diagnostic BB run accepted 7,764 stagnation-flagged roots; the largest
equation residual among them was `5.56e-17`. Its cost differs from PMP by
`1.12e-8`. Recorded runtimes were approximately 57 seconds for the original
failure, 97 seconds for the diagnostic BB run, and 0.05 seconds for PMP.
These are individual timings with different integration strategies and
instrumentation, not an accuracy-matched performance benchmark.

Reproduction (the first command fails its convergence assertion; the second passes):

```sh
uv run --frozen python rawdata/reports/bb_pmp_comparison_20261019/compare.py --state -3 -3 --nodes 301 --profile paper --require-convergence
uv run --frozen python rawdata/reports/bb_pmp_comparison_20261019/compare.py --state -3 -3 --nodes 301 --profile paper --accept-small-cn-residuals --require-convergence
```

### Gradient versus discretized cost

The code separately applies CN to the continuous state and costate equations
and trapezoidal quadrature to the objective. Its returned control gradient is
not the exact derivative of that discretized objective.

At the converged 301-node diagnostic BB control for `x=(-3,-3)`, perturbing
only the first control coefficient gives:

- Weighted inner product with the implemented gradient: `-2.27328e-8`.
- Centered finite difference, perturbation `1e-4`: `1.19248718e-4`.
- Independent tangent differentiation of the implicit CN steps:
  `1.19248898e-4`.

Finite differences at three perturbation sizes and two additional directions
confirm this discrepancy. The tangent recurrence differentiates the actual
implicit state equations, so this is not an unweighted-versus-weighted gradient
comparison. Details are in `common_validation.json` and `audit_controls.py`.

For a common continuous-time check, both controls were reconstructed by cubic
splines, independently replayed with DOP853 (`rtol=1e-11`, `atol=1e-13`), and
their costates integrated backward. The gradient norm below is computed from
these independent trajectories, not from the stored costates:

| Method, `x=(-3,-3)` | Replayed cost | Replayed L2 control-gradient norm |
| --- | ---: | ---: |
| BB, 301 nodes, residual acceptance | 18.516597129523 | 1.1166e-3 |
| BB, 601 nodes, residual acceptance | 18.516597009679 | 2.7948e-4 |
| PMP, 301 initial nodes | 18.516597001308 | 3.7139e-8 |

The BB norm improves by approximately four on halving the time step, consistent
with second-order time error. Its internal 301-node gradient norm was only
`1.7422e-6`. Objective agreement alone therefore overstates agreement in
first-order optimality and costate accuracy.

### Additional implementation concern

The first BB step has no line search (`solver.py:287`). Later steps accept
`J_trial <= 1.01 J_current` by default (`solver.py:556`), without an Armijo
sufficient-decrease term. This differs from a safeguarded Armijo or standard
nonmonotone Armijo rule. It is a further item to audit, but these experiments
do not isolate it as the cause of the reproduced failures.

The evidence supports using the validated PMP generator for these benchmarks,
while correcting root acceptance, discrete gradient consistency, and step
safeguards before judging BB's performance. Neither set of computations proves
global optimality or guarantees convergence at every initial condition.
