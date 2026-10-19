# Higher-dimensional benchmarks from Daniel Walter's papers

Research date: 2026-10-12. Recommendation: use one coupled nonlinear Allen–Cahn control family at both 6 and 10 state dimensions. These are new finite-volume variants of a published problem, not reproductions of the published 19-dimensional experiment.

## Primary source and provenance

Karl Kunisch, Donato Vásquez-Varas, and Daniel Walter, **Learning Optimal Feedback Operators and their Sparse Polynomial Approximations**, *Journal of Machine Learning Research* 24(301):1–38, 2023. [Publication record](https://jmlr.org/papers/v24/22-1010.html); [published PDF, §9.3, Eq. (58), pp. 20–22](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=20).

The published example has

\[
Y_t=\nu Y_{\xi\xi}+Y(1-Y^2)+\sum_{k=1}^{3}\mathbf 1_{\omega_k}(\xi)u_k,
\qquad \xi\in(-1,1),
\]

homogeneous Neumann boundaries, \(\nu=0.1\), \(T=3\), and

\[
\omega_1=(-0.7,-0.4),\quad\omega_2=(-0.2,0.2),\quad\omega_3=(0.4,0.7).
\]

Its objective is \(\int_0^T\int_{-1}^1Y^2\,d\xi\,dt+\beta\int_0^T|u|^2\,dt\). The article uses 19 Chebyshev collocation degrees of freedom, Clenshaw–Curtis quadrature, and uniform Monte Carlo sampling on \((-10,10)^{19}\). These facts come from [§§9 and 9.3](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=16).

**Cost coefficient:** no numerical Allen–Cahn \(\beta\) was found in §§9/9.3. The \(\beta=0.9\) in §9 is the backtracking parameter; \(\beta=10^{-2}\) in §9.4 belongs to Cucker–Smale. Set the repository's control-cost parameter **\(\eta=0.01\)** explicitly as a new benchmark choice. Eq. (58) has no factors of \(1/2\); its feedback therefore uses \(1/(2\eta)\). [Published source](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=17)

**Version discrepancy:** the [October 2022 arXiv version, §9.3](https://arxiv.org/html/2208.14120#S9.SS3) instead gives \(\nu=0.5,T=4\). Use the published \(\nu=0.1,T=3\) consistently in code, data metadata, and experiments.

## Explicit 6D and 10D adaptations

The following discretization and sampling choices are our construction, not claims about the authors' implementation. They keep the published PDE, actuator intervals, diffusion, time horizon, and objective structure.

For \(d\in\{6,10\}\), use equal cells

\[
\Delta\xi=2/d,\qquad C_j=[-1+(j-1)\Delta\xi,-1+j\Delta\xi],\qquad
\xi_j=-1+(j-\tfrac12)\Delta\xi.
\]

The state \(y_j\) represents a cell value. Approximate the reaction at that value and impose zero boundary flux through

\[
(L_dy)_1=(y_2-y_1)/(\Delta\xi)^2,
\]

\[
(L_dy)_j=(y_{j-1}-2y_j+y_{j+1})/(\Delta\xi)^2\quad(2\le j\le d-1),
\]

\[
(L_dy)_d=(y_{d-1}-y_d)/(\Delta\xi)^2.
\]

Use exact cell averages of the control indicators,

\[
B_{jk}=\frac{|C_j\cap\omega_k|}{\Delta\xi},\qquad
\dot y=\nu L_dy+y-y^{\odot3}+Bu.
\]

This preserves all three actuator channels at both dimensions. For ascending spatial order, an independent implementation can check

\[
B_6=\begin{pmatrix}
0.1&0&0\\0.8&0&0\\0&0.6&0\\0&0.6&0\\0&0&0.8\\0&0&0.1
\end{pmatrix},
\]

\[
B_{10}=\begin{pmatrix}
0&0&0\\0.5&0&0\\1&0&0\\0&0&0\\0&1&0\\0&1&0\\0&0&0\\0&0&1\\0&0&0.5\\0&0&0
\end{pmatrix}.
\]

Both matrices have rank three and satisfy \(\Delta\xi\sum_j B_{jk}=|\omega_k|\). Floating-point interval arithmetic may leave negligible entries at touching endpoints; those are zero mathematically.

The finite-dimensional running cost is

\[
h_d(y)+\eta|u|^2=\Delta\xi\sum_{j=1}^d y_j^2+\eta\sum_{k=1}^3u_k^2.
\]

The spatial weight matters: replacing it by an unweighted sum changes the relative state/control penalty when dimension changes.

Sample the full cube \([-1,1]^d\) with scrambled Sobol points and a recorded seed. This changes both the article's sampling method and its larger domain. Independent held-out points should use a separate scramble/seed. Sampling all coordinates avoids presenting a low-dimensional manifold of smooth profiles as a full-dimensional regression test. Use the same sample counts and training settings at both dimensions when comparing algorithm behavior.

Naively retaining six interior Chebyshev nodes would miss the central actuator: their two central nodes are approximately \(\pm0.222521\), outside \((-0.2,0.2)\). Cell averaging avoids that discretization artifact. Six and ten cells remain coarse PDE models; these experiments measure regression behavior on these ODEs and do not establish PDE mesh convergence.

## Open-loop equations and label meaning

This benchmark solves a fixed finite-horizon problem with free terminal state and zero terminal cost. It produces \(x\mapsto V(0,x)\), consistent with the VDP dataset convention in `paper/paper_0805.tex` and the existing open-loop machinery. It does not automatically produce the stationary infinite-horizon value function.

With \(A(y)=\nu L_d+\operatorname{diag}(1-3y_j^2)\), the Hamiltonian first-order conditions give

\[
u^*=-\frac{B^\top p}{2\eta},\qquad
\dot y=\nu L_dy+y-y^{\odot3}-\frac{BB^\top p}{2\eta},
\]

\[
\dot p=-2\Delta\xi\,y-A(y)^\top p,\qquad
y(0)=x,\quad p(T)=0.
\]

Store the cost integral as the value label and \(p(0)\) as the gradient label, in the same unnormalized state coordinates as \(x\). If a cumulative-cost variable is integrated, its derivative is the running cost above. Do not multiply the stored costate by another spatial weight: the weight is already included in its differential equation.

For scientific validation, check dynamics/Jacobians, terminal and initial residuals, solver convergence, and finite differences of the resulting value against \(p(0)\). Multiple successful initial guesses and selection of the lowest-cost candidate improve numerical reliability. A converged Pontryagin boundary-value solution is a stationary candidate and is not by itself a certificate of global optimality.

Do not pool intermediate trajectory rows into a state-only \(V(0,\cdot)\) dataset: those rows correspond to different remaining horizons. A finite-horizon optimal feedback generally depends on time. Using a learned fixed-horizon slice repeatedly is a receding-horizon policy, with a separate interpretation and validation requirement.

## Other suitable source examples

- **An actual 10D bilinear heat-control example:** Kunisch and Walter, *Optimal feedback control of dynamical systems via value-function approximation*, §9, Eqs. (9.1)–(9.2), approximates heat dynamics on \((0,2\pi)\) with ten Dirichlet eigenfunctions and three multiplicative controls. It uses \(T=2\), control coefficient \(\beta=0.01\), terminal coefficient \(\alpha=0.25\), and target \(Y_d(\xi)=\xi^2/10\). This would require state-dependent control matrices and nonzero terminal tracking cost. It is a good later benchmark but changes more of the current dataset contract. [Primary paper, pp. 29–32](https://arxiv.org/pdf/2302.13122#page=29)
- **Cucker–Smale consensus:** the JMLR paper's §9.4 uses ten planar agents, so **40 state coordinates**, not ten. Three or five agents moving on a line would yield 6D/10D adaptations. The position-translation symmetry and unpenalized common velocity produce redundant directions for this regression study; Allen–Cahn is a clearer first dimension comparison. [Primary source, §9.4](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=24)
- **Burgers-type stabilization:** Kunisch and Walter's *Semiglobal optimal feedback stabilization of autonomous systems via deep neural network approximation* uses a scalar distributed control, 13 interior Chebyshev states, and quadratic running cost. Changing spatial resolution to 6/10 is feasible. Its destabilizing reaction variants and documented blow-up behavior make it a harder follow-up for robust open-loop data generation. [Primary preprint, numerical example, pp. 36–39](https://arxiv.org/pdf/2002.08625#page=36)

## Suggested manuscript attribution

“We consider six- and ten-dimensional finite-volume discretizations of the controlled Allen–Cahn equation from Kunisch, Vásquez-Varas, and Walter (2023, §9.3). We retain the published diffusion coefficient, actuator intervals, and horizon, and use cell-averaged actuators, control weight \(\eta=0.01\), and Sobol initial states in \([-1,1]^d\).”

Report sample counts, successful/failed solves, label validation, and held-out value/gradient errors from the actual generated artifacts. Do not import the source paper's errors or claim its 19D numerical results were reproduced.
