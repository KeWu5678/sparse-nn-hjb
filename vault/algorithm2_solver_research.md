# Algorithm 2 coefficient solver: recommendation

## Bottom line

Use a **sequential global-prox/active-set Newton iteration**. First compute the
exact global $\ell_q$ proximal-gradient point $z_P$ and treat it as the accepted
fallback. Only when this proximal step preserves the support and signs, and the
active reduced Hessian at $z_P$ is positive definite, attempt a sign-preserving
Newton step from $z_P$. Accept that step after Armijo backtracking; otherwise keep
$z_P$.

This is not ordinary Newton applied to the original nonsmooth problem. The
proximal step handles coordinates entering and leaving the support. Ordinary
Newton is used only as a local accelerator on an identified support/sign stratum,
where the objective is smooth. Consequently every iteration retains the genuine
proximal-gradient decrease, while **local quadratic convergence is conditional**
on finite support/sign identification, a strict proximal margin, and a
positive-definite reduced Hessian at the limit.

This sequential method is the leading implementation candidate. Wu--Pan--Yang
HpgSRN is the closest theorem-backed comparator for the exact $\ell_q$ problem;
Bareilles--Iutzeler--Malick supports the prox-then-manifold-Newton architecture
but requires an $\ell_q$ specialization; Gfrerer's semismooth* Newton method is a
more general, more complex alternative. Keep the current stationary-branch SSN
as the production A/B baseline until the new method is benchmarked.

## Exact problem and repository fit

At fixed nodes, Algorithm 2 minimizes

$$
 F(c)=\tfrac12 c^\top Hc-b^\top c+\mathrm{const}
       +\alpha\sum_i |c_i|^q,\qquad q=\frac{2}{k+1}\in(0,1),
$$

where

$$
 H=\frac{w_1}{M}\Phi_v^\top\Phi_v+
   \frac{w_2}{M}\Phi_g^\top\Phi_g\succeq0.
$$

The code already forms this exact, constant data Hessian in
[`src/PDAP/ssn_solve.py`](../src/PDAP/ssn_solve.py#L98). The reported supports
are small: roughly 20--86 atoms in the VDP fractional-penalty runs and up to 131
in the pendulum runs
([VDP results](../experiments/01_vdp/frac_exp_penalty/results.md),
[pendulum results](../experiments/02_pendulum/frac_exp_penalty/results.md)). Dense
reduced factorizations are therefore cheap and simpler than a large-scale
matrix-free framework.

The paper and implementation correctly disclose that the current map is a
*stationary-branch selector*, not the global prox
([`paper_0805.tex`](../paper/paper_0805.tex#L3304),
[`src/SSN/prox.py`](../src/SSN/prox.py#L1)). Its turning threshold is

$$
 t_{\rm turn}=[\nu q(1-q)]^{1/(2-q)},\qquad
 v_{\rm turn}=t_{\rm turn}+\nu q t_{\rm turn}^{q-1}.
$$

Immediately above $v_{\rm turn}$, it returns the larger nonzero stationary
root. The **global** scalar prox of $\nu|\cdot|^q$ instead switches at

$$
 \bar t=[2\nu(1-q)]^{1/(2-q)},\qquad
 \bar v=\bar t+\nu q\bar t^{q-1}.
$$

It chooses zero for $|v|<\bar v$, the larger root of
$t+\nu q t^{q-1}=|v|$ for $|v|>\bar v$, and is set-valued at equality
(choose zero deterministically). Thus, on
$v_{\rm turn}<|v|<\bar v$, the current selector chooses a nonzero local branch
while zero is the global proximal minimizer. LM or Steihaug globalization of that
residual does not repair this mismatch. The outer correction guard only rolls back
a final objective increase
([`src/PDAP/pdap.py`](../src/PDAP/pdap.py#L300)).

## Proposed iteration

Let $L=\lambda_{\max}(H)$ and take $0<\tau<1/L$ (or use monotone
backtracking).

**1. Global support/descent candidate**

$$
 z\in\operatorname{prox}_{\tau\alpha\|\cdot\|_q^q}
       \bigl(c-\tau(Hc-b)\bigr),
$$

using the global threshold $\bar v$ above coordinatewise. This gives

$$
 F(z)\le F(c)-\frac{1-\tau L}{2\tau}\|z-c\|^2.
$$

Call this candidate $z_P=z$.

**2. Support/sign switch test**

Set $A=\operatorname{supp}(z_P)$. Permit a Newton attempt only if

$$
 \operatorname{supp}(z_P)=\operatorname{supp}(c),\qquad
 \operatorname{sign}((z_P)_A)=\operatorname{sign}(c_A).
$$

Thus the proximal map has made no support or orthant change in this iteration.
Also require the active coefficients to be separated from zero and the proximal
inputs to be separated from the switching value $\bar v$, using explicit
numerical margins. If any test fails, set $c^+=z_P$ and begin the next iteration.

These tests do not prove that the limiting support has been identified; they are
algorithmic gates. Finite identification requires a strict proximal margin at the
limit. Requiring stability for more than one consecutive proximal iteration is a
possible conservative implementation choice, not part of the mathematical
definition.

**3. Reduced Newton from the accepted proximal point**

On the fixed support/sign stratum through $z_P$, the objective is smooth and

$$
 g_A=(Hz_P-b)_A+\alpha q\,\operatorname{sign}((z_P)_A)|(z_P)_A|^{q-1},
$$

$$
 B_A=H_{AA}-\alpha q(1-q)
       \operatorname{diag}(|(z_P)_A|^{q-2}).
$$

If Cholesky certifies $B_A\succ0$, solve $B_A d_A=-g_A$. Try the full
step first, then Armijo-backtrack on $F$, with the step capped before any sign
crossing. If Armijo accepts $z_N=z_P+t d$, set $c^+=z_N$; otherwise set
$c^+=z_P$. If Cholesky fails, do not take an ordinary Newton step: keep $z_P$ or,
in an HpgSRN implementation, use its prescribed regularized reduced system.
Any regularization must vanish asymptotically if quadratic convergence of the
ordinary reduced-Newton branch is claimed.

Because the Newton candidate is computed after, and accepted relative to, the
proximal point,

$$
 F(c^+)\le F(z_P)
 \le F(c)-\frac{1-\tau L}{2\tau}\|z_P-c\|^2.
$$

Thus the proximal-gradient sufficient-decrease argument survives whenever Newton
is selected. In contrast, computing an independent Newton candidate from $c$ and
choosing the lower objective is not the algorithm analyzed by
Bareilles--Iutzeler--Malick, Wu--Pan--Yang, or Gfrerer.

**4. Stop and retain outer protection**

Use the proximal-gradient mapping $(c-z_P)/\tau$ plus the active reduced gradient
as stopping diagnostics. Keep the existing post-correction objective guard while
the new solver is experimental.

Near a strict local minimizer $c^\star$, additionally assume the proximal map has
a strict support/sign margin, the iterates identify that stratum in finitely many
steps, and $B_A(c^\star)\succ0$. Active coefficients are then separated from zero
and the restricted Hessian is locally Lipschitz. Eventually Cholesky succeeds,
the full Newton step remains in the orthant and Armijo accepts it. Ordinary
reduced-Newton theory then gives

$$
 \|c^{j+1}-c^\star\|=O(\|c^j-c^\star\|^2).
$$

The coefficient lower bound already proved in the paper helps establish
separation of the nonzero coefficients at a local minimizer. It does **not** by
itself prove finite algorithmic support identification, a strict proximal margin,
or $B_A(c^\star)\succ0$; those remain explicit assumptions in a local-rate
theorem.

## Candidate comparison

| Method | Global safeguard / target | Local rate | Fit here |
|---|---|---:|---|
| Current full-space stationary-branch SSN | Final objective rollback; local branch stationarity only | Can be fast on a consistent branch, but no applicable global theorem | Keep as baseline; branch jump and inactive/active ambiguity are structural |
| **Sequential global prox + active-set Newton** | Accept true PG first; attempt Newton from the PG point only after support/sign and SPD gates | **Quadratic**, conditionally after finite identification | **Recommended tailored candidate:** simple, but needs a short problem-specific convergence proposition |
| Bareilles--Iutzeler--Malick manifold Newton | Proximal-gradient identifies a smooth manifold before Riemannian Newton acceleration | **Quadratic** under their structural and Hessian assumptions | Supports the architecture; applying its theorem to non-Lipschitz $\ell_q$ requires an explicit specialization |
| Wu--Pan--Yang HpgSRN | PG every iteration, sign/support test, regularized reduced Newton; whole-sequence convergence under KL plus their curve-ratio assumptions | Q-order $1+\sigma$, $0<\sigma\le1/2$ | Strongest directly tailored theorem, but more parameters and a weaker stated local order |
| Gfrerer GSSN | Proximal approximation, FBE globalization, and an SCD semismooth* Newton direction | Superlinear under SCD regularity and semismoothness* assumptions | Treats exact $\ell_q$ examples rigorously, but is not ordinary Newton and requires substantially more machinery |
| Independent best-of-two from the same $c$ | PG supplies a descent bound if the accepted objective is no worse, but the Newton sequence need not inherit PG identification | Quadratic only under additional unproved eventual-selection assumptions | **Retired proposal:** useful as an experiment, not supported by the cited algorithms |
| Exact cyclic coordinate descent + Newton | Exact one-dimensional decreases select support | Coordinate phase has no superlinear rate; Newton is fast only after switching | Useful fallback/drop sweep, not the default; sequential updates can redirect the basin |
| IRL1 or IRLS + Newton | Majorization/smoothing and repeated weighted subproblems | Published fast rates often concern smoothed or constrained recovery problems, not this exact penalized objective | Extra tolerances and conditioning problems; no advantage over direct prox + Newton |
| Smooth reduced trust region | Valid only after restricting to a fixed nonzero support; handles negative curvature | Quadratic/superlinear once the trust region is inactive | Serious fallback if non-PD reduced Hessians are common; otherwise more complex than PG fallback |

The [2025 normal-map trust-region SSN of Ouyang--Milzarek](https://doi.org/10.1007/s10107-024-02110-2)
is not applicable here: it assumes the nonsmooth regularizer is convex, whereas
$\sum|c_i|^q$ is not.

## Evidence and decision gate

A preliminary in-memory replay of real post-insertion transitions found that a
pure active-Newton first direction crossed an orthant at iteration zero in 37/39
$q=2/3$ transitions and 18/20 $q=1/3$ transitions. This rules out an
unguarded Newton-first method and supports placing the global proximal step before
Newton.

The same replay tested the now-retired independent best-of-two variant: it took a
median 3.5--4 iterations and matched the archived 20-step SSN objective in 56/59
transitions; the other three objective gaps were $5\times10^{-7}$ to
$4\times10^{-5}$, and all runs converged by the prox-gradient residual. The
current SSN reproduced all archived corrections. These counts do **not** validate
the newly recommended sequential method, because that method was not replayed.

Before changing production, replay identical post-insertion snapshots and compare:

- current SSN, global PG alone, sequential PG-then-Newton, and HpgSRN;
- final objective and prox-gradient residual at equal time/evaluation budgets;
- correction acceptance/failure rate, support size, orthant crossings, and
  sensitivity across $q=2/3,1/2,1/3$.

The archived $q=1/3$ source run is
`rawdata/logs/multirun/vdp/paper_frac_exp_penalty/sequential/37/result_vdppaperfracexppenalty_vdp_20260812_494b.pkl`.
The replay itself was not persisted, so the counts above are a feasibility signal,
not a benchmark result.

## Primary sources

- Bareilles, Iutzeler, and Malick, [*Newton acceleration on manifolds identified by proximal-gradient methods*](https://doi.org/10.1007/s10107-022-01873-w), **Mathematical Programming** 200 (2023), 37--70. Algorithm 1 supplies the sequential PG-then-manifold-Newton architecture and conditional local quadratic/order-$1+\theta$ rates. It does not directly certify the retired independent best-of-two method, and its assumptions must be specialized to the non-Lipschitz $\ell_q$ penalty used here.
- Gfrerer, [*On a globally convergent semismooth\* Newton method in nonsmooth nonconvex optimization*](https://doi.org/10.1007/s10589-025-00658-z), **Computational Optimization and Applications** 91 (2025), 67--124. Section 6.2 treats least squares plus $\ell_q^q$ exactly; the method uses proximal/FBE globalization and SCD semismooth* Newton. It supports a more sophisticated alternative, not an independent ordinary-Newton candidate from the current support.
- Wu, Pan, and Yang, [*A Regularized Newton Method for $\ell_q$-Norm Composite Optimization Problems*](https://doi.org/10.1137/22M1482822), **SIAM Journal on Optimization** 33 (2023), 1676--1706. Exact problem class; finite support identification, whole-sequence convergence under additional assumptions, and Q-order $1+\sigma$.
- Themelis, Stella, and Patrinos, [*Forward-Backward Envelope for the Sum of Two Nonconvex Functions*](https://doi.org/10.1137/16M1080240), **SIAM Journal on Optimization** 28 (2018), 2274--2303. Global nonconvex prox/FBE framework with Dennis--More superlinear quasi-Newton acceleration.
- Liu and Lin, [*A bisection method for computing the proximal operator of the $\ell_p$-norm for any $0<p<1$*](https://doi.org/10.1016/j.cam.2024.115897), **Journal of Computational and Applied Mathematics** 447 (2024), 115897. A recent robust scalar-prox implementation reference.
- Marjanovic and Solo, [*$\ell_q$ Sparsity Penalized Linear Regression With Cyclic Descent*](https://doi.org/10.1109/TSP.2014.2302740), **IEEE Transactions on Signal Processing** 62 (2014), 1464--1475. Exact scalar threshold and coordinatewise convergence, but no superlinear rate.
