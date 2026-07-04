# NOTATION — agreed terms & notation (theory program)

Glossary only. Goals → `OVERVIEW.md`; claims → `CLAIMS.md`. Tags: [repo] in code,
[ref] reference paper, [domain] standard OC/HJB, [session] agreed here.

## Models
- **signed model** [repo `src/models/signed.py`]: $V=\sum_i c_i\,\sigma(w_i\!\cdot\!x+b_i)^p$,
  $c_i$ free sign. Default $p=2.1$.
- **semiconcave model** (a.k.a. **cone model**) [repo `src/models/semiconcave.py`]:
  $V=\tfrac12 C\|x\|^2-g$, $g=\sum_i c_i\,\sigma(w_i\!\cdot\!x+b_i)^p+a\!\cdot\!x+b_0$, with
  $c_i\ge0$, $C\ge0$, $a,b_0$ free. Default $p=1.0$.
- **quadratic head** [repo]: the $\tfrac12 C\|x\|^2$ term ($C$ its single parameter).
- **affine** [repo]: the $a\!\cdot\!x+b_0$ term (unpenalized).
- **atom** / **neuron** [repo]: one term $c_i\,\sigma(w_i\!\cdot\!x+b_i)^p$; $(w_i,b_i)$ = frozen
  inner weights (support), $c_i$ = penalized outer coefficient.
- **power** $p$ [repo]: exponent in $\sigma^p$; induces penalty exponent $q=2/(p+1)$.

## Target
- **value function** $V$ [domain]: the HJB value function being approximated.
- **$C$-semiconcave** [ref Cannarsa]: $D^2V\preceq C I$, i.e. $V=\tfrac{C}{2}\|x\|^2-g$ with $g$ convex.
- **$g$** [session]: the convex correction (the part the cone's atoms carry).
- **switching set** [domain]: where $\nabla V$ is discontinuous ($V$ has a kink).

## Training / algorithm
- **gradient-augmented** / **H1** [repo, session]: fit both $V$ and $\nabla V$ (loss weights
  $[1,1]$); **value-only** / **L2** $=[1,0]$.
- **PDAP** [repo]: primal-dual active-point solver (greedy atom insertion + outer SSN).
  **SSN**: semismooth-Newton outer solve.
- **penalty** [repo, session]: regularizer on $c$ only; **log / nonconvex penalty**,
  strength $\alpha$, nonconvexity $\gamma$ ($\gamma=0$ = $\ell_1$).

## Metrics / complexity
- **model class** $M$ [session]: one of {signed model, semiconcave model} viewed as a function
  class. **$M_n$** = all functions that model realizes with $\le n$ **atoms** (under its sign
  constraint on the $c_i$; for the cone the quadratic head + affine are always included and
  are **not** counted as atoms). So $M_n\subseteq M_{n+1}$.
- **neuron count** $N(M,V,\varepsilon)$ [session]: fewest atoms for class $M$ to reach relative
  gradient error $\varepsilon$ on $V$,
  $$N(M,V,\varepsilon)=\min\{\,n:\ \inf_{f\in M_n}\|\nabla f-\nabla V\|_{L^2(\Omega)}\le\varepsilon\,\|\nabla V\|_{L^2(\Omega)}\,\}.$$
  Metric = **relative gradient-$L^2(\Omega)$**; the inner $\inf$ over all of $M_n$ makes it
  **best-$n$-term** (solver/penalty-independent; PDAP's count is an upper bound on it).
- **$N^+(g,\varepsilon)$** [session]: neuron count for $g$ using **nonnegative** atoms (the cone's
  atom cost; by `head_reduction`, $N(\text{semiconcave},V,\varepsilon)=N^+(g,\varepsilon)$).
- **variation norm** / **$\mathcal R$-norm** [ref Ongie]: $\inf\sum_i|c_i|$ (infinite width);
  ReLU = Radon $\mathcal R$-norm. Distinct from neuron count.
- **budget** $\mathrm{bud}(g)$ [session]: $\|D^2g\|_{\mathcal M}=\int_\Omega d(\Delta g)$ ($g$ convex); $=\|g''\|$ in 1-D.
- **nonneg variation norm** $\gamma^+(g)$ [session]: $\inf\{\|\mu\|:g=\int\sigma(w\!\cdot\!x-b)^k\,d\mu,\ \mu\ge0\}$.
  ($\mathrm{bud}\le c(\Omega)\gamma^+$, equal in 1-D; $\gamma^+$ can be $\infty$ while $\mathrm{bud}<\infty$.)

## Notation
$d$ dimension · $\Omega$ bounded domain · $\varepsilon$ relative gradient error · $\sigma$ activation ·
$\kappa_d,c_d$ dimensional constants · $K$ number of pieces of a piecewise-linear $g$.

## Retired (do NOT use) → see `CLAIMS.md` for why
*bulk*, $Q$ (write $\tfrac12 C\|x\|^2$) · *Tier 1/2* · *capacity vs selection* · *(C1) atom
convexity* · $\kappa_\sigma$ (measured diagnostic only, not a theory quantity).
