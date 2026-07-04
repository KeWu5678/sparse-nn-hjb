# correction_cost

**status:** proved (the two cases we use: flat, curved); smooth-$g$ rate = open remark

**statement.** Cost $N^+(g,\varepsilon)$ = fewest nonnegative ReLU atoms for $\nabla g$ to error $\varepsilon$.
1. **(flat)** $g$ piecewise-linear with $K$ pieces ⟹ $N^+(g,\varepsilon)=K$ ($\varepsilon$-independent).
2. **(curved, $d=2$)** $\exists$ convex $g$ with a curved switching set (e.g. $(\|x\|-1)_+$, or
   $v_d$) with $\gamma^+(g)=\infty$ ⟹ $N^+(g,\varepsilon)\to\infty$ (unbounded as $\varepsilon\to0$).
3. *(smooth, remark/open)* $g$ smooth ⟹ $N^+(g,\varepsilon)\sim1/\varepsilon$ (Barron upper bound; exact
   rate not nailed) — same order as the linear field, so the head's worth is then a
   **constant** factor, not divergent.

**proof.** (flat) $g=\sum_{i\le K}c_i(w_i\!\cdot\!x-b_i)_+$ is exactly $K$ nonneg atoms; $\nabla g$
piecewise-constant with $K$ jumps, no $\varepsilon$-dependence. (curved) $\gamma^+(g)\ge\|g\|_{\mathcal R}=\infty$
by `curved_switching_rnorm`: a curved kink has non-integrable Radon derivative; nonneg
is more constrained than signed, so $\gamma^+\ge$ signed $\mathcal R$-norm $=\infty$ ⟹ no finite-rate
ridge representation. Confirmed empirically: `../../scripts/curved_vs_flat_switching.py`
($(\|x\|-1)_+$ plateaus at relGrad 0.14, 124 atoms). $\square$

**uses.** `budget_1d` (the finite-curvature-mass bound); `curved_switching_rnorm`.

**attempts.** 1. "budget→rate in all $d$" overstated — true only $d=1$ ($\mathrm{bud}=\gamma^+$ there);
$d\ge2$ they differ (bud finite, $\gamma^+$ can be $\infty$) — that gap **is** case 2. 2. Curved
case escape hatch = min architecture → open `../../D4_max_plus/claims/minplus_curved.md`.
