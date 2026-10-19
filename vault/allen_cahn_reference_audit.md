# Allen Cahn reference audit

Audit date: 2026-10-19. The dataset generator uses **6 or 10 input coordinates**, each representing one spatial cell value. It approximates the fixed-horizon value `x -> V(0,x)`; time is fixed at zero and is not an input. The source experiment uses 19 state coordinates. The physical PDE has one spatial coordinate.

## Published example

Karl Kunisch, Donato Vásquez-Varas, and Daniel Walter, *Learning Optimal Feedback Operators and their Sparse Polynomial Approximations*, JMLR 24(301):1–38, 2023, published September 2023. [Publication record](https://jmlr.org/papers/v24/22-1010.html).

Section 9.3, Eq. (58), p. 20, with parameters and discretization on p. 22, gives the controlled Allen–Cahn equation with Neumann boundaries on `(-1,1)`, `nu=0.1`, `T=3`, and three actuator intervals. The discretization has 19 Chebyshev degrees of freedom and Clenshaw–Curtis state-cost quadrature. Uniform Monte Carlo sampling, `l=10`, and the domain definition in §2 give `(-10,10)^19`. Section 9.3 leaves the control weight symbolic. [Published paper, §§2, 9, 9.3](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=20).

The October 25, 2022 preprint has a different title ending *their Polynomial Approximation* and gives `nu=0.5, T=4` in §9.3. The generator correctly follows the published parameters. [arXiv version 2](https://arxiv.org/html/2208.14120v2#S9.SS3), [version history](https://arxiv.org/abs/2208.14120).

## Comparison with the generator

| Item | Current implementation | Assessment |
| --- | --- | --- |
| State dimension | CLI accepts `6` or `10`; each sample is a vector of that length | Explicit adaptation |
| Spatial discretization | Uniform cell-centered finite volumes, `h=2/d`, zero-flux Laplacian | Explicit adaptation |
| Control weight | Default `beta=0.01` | Repository choice |
| Initial states | 2,048 scrambled Sobol points in `[-1,1]^d`, seed 42 | Different domain and sampling design |
| Labels | Cost integral and initial costate `p(0)` | Fixed-horizon value and gradient references |

Code evidence: [CLI defaults and metadata](../scripts/generate_allen_cahn_data.py), lines 55–101; [dynamics, cost, spatial matrices](../src/OpenLoop/allen_cahn/problem.py), lines 62–128; [sampling](../src/OpenLoop/allen_cahn/sampling.py), lines 9–24; [boundary conditions and labels](../src/OpenLoop/allen_cahn/solver.py), lines 199–229 and 258–287.

The implemented dynamics, actuator intervals, spatially weighted cost, and Pontryagin equations are consistent with the documented adaptation. In particular, `u = -B^T p/(2 beta)` matches the cost without factors of one half in Eq. (58). The solver imposes `y(0)=x` and `p(T)=0`; `x,dv` have shape `(N,d)` and `v` has shape `(N,)`.

The article learns polynomial feedback through trajectory costs, motivated by the infinite-horizon value, with finite horizons numerically. [Published paper, §§2–3](https://jmlr.org/papers/volume24/22-1010/22-1010.pdf#page=3). The generator's supervised fixed-horizon labels therefore have a different learning interpretation. General `V(t,x)` approximation would require time or remaining horizon as an additional input.

Math/specification and coding-standards review found no material issues. The command `uv run --frozen pytest -q tests/test_allen_cahn_openloop.py tests/test_allen_cahn_reference.py tests/test_allen_cahn_generation.py tests/test_higher_dimensional_samples.py` passed all 47 tests in 11.95 seconds. No production Allen–Cahn directories were present in `rawdata/data`, so the previously documented 4,096-row audit was not repeated. See the [existing research note](higher_dimensional_benchmarks_research.md) for equations and the [experiment README](../experiments/00_openloop/allen_cahn/README.md) for the earlier generation audit.
