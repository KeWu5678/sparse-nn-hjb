# cone_free_convex  (T3)

**status:** **proved** in the curvature-mass ($\int\Delta$) sense (the count-governing cost, all
$d$); variation-norm version ($\gamma^+=\gamma$) proved $d=1$, $d=2$ numerics $1.00$.

**statement.** For $g$ convex, the cone constraint $c_i\ge0$ does not raise the representation
cost: among all ReLU representations $g=\int(w\!\cdot\!x-b)_+\,d\mu+\text{affine}$, the **nonnegative**
one is optimal. (Combined with `head_reduction` and `signed_lower_bound`, this lifts
`separation_1d_convex` toward $d\ge2$.)

## Main lemma (PROVED, all $d$): nonneg minimizes curvature mass
For any signed representation, Jordan-split $\mu=\mu^+-\mu^-$ ($\mu^\pm\ge0$ mutually singular),
giving $g=h^+-h^-$ with $h^\pm=\int(w\!\cdot\!x-b)_+\,d\mu^\pm$ **convex**. Since $\Delta h=\int\delta(w\!\cdot\!x-b)\,d\mu\ge0$
for $\mu\ge0$,
$$
\int_\Omega \Delta h^+ + \int_\Omega \Delta h^- \;\ge\; \int_\Omega(\Delta h^+-\Delta h^-)=\int_\Omega\Delta g=\mathrm{bud}(g),
\qquad\text{equality}\iff \mu^-=0.
$$
So **signed cancellation strictly increases the total curvature mass** (by $2\int\Delta h^-$); the
nonneg representation uniquely attains the minimum $\mathrm{bud}(g)=\int_\Omega\Delta g$. $\square$
(Verified `../../scripts/cone_free_curvature_mass.py`: excess $=2\int\Delta h^-$ exactly.)

**Why this is the relevant cost.** The neuron count is governed by curvature mass $\int|D^2\cdot|$
— exactly the quantity in the verified curvature-ratio law (`separation_1d_convex`, proved $d=1$;
$d=2$ PDAP confirmed ratio $=\int|D^2g|/\int|D^2V|=0.35$). For convex $g$, $\int|D^2g|=\int\Delta g=
\mathrm{bud}(g)$. So the lemma says: **the cone cannot be beaten by signed cancellation in the cost
that sets the count** — cone-free for the count, *given* count $\sim$ curvature-mass.
Crucially this argument **avoids the Radon ramp-filter obstruction** that blocks the
variation-norm route.

## Supporting: Radon-convexity (PROVED, verified)
$g$ convex $\Rightarrow \partial_b^2\mathcal R\{g\}(w,b)=\int_{\{w\cdot x=b\}}w^\top D^2g\,w\ge0$ ($\mathcal R\{g\}$ convex
in $b$) — the $d\ge2$ generalization of $g''\ge0$. Verified `../../scripts/radon_convexity_check.py`
(4 convex g $\ge0$; nonconvex control $-47.9$). Closes $d=1$ ($\mathcal R$ measure $=g''$); does not by
itself close the *variation-norm* $d\ge2$ case (representing measure needs $\partial_b^{d+1}\mathcal R$).

## Variation-norm version ($\gamma^+=\gamma$)
$d=1$: proved ($g(x)=\text{affine}+\int(x-t)_+g''(t)dt$, $g''\ge0$). $d=2$: $\gamma^+/\gamma=1.00$ exact
for 7/7 smooth convex targets incl. anisotropic/rotated/varying-Hessian
(`../../scripts/t3_cone_free_2d.py`). $d\ge2$ analytic: open (the chord-weighting makes $\gamma^+\ne
\int\Delta$, and the ramp filter blocks the simple argument) — but the **curvature-mass version
above is proved and is what the count needs.**

## remaining gap for the full $d\ge2$ separation theorem
Only the **$d\ge2$ $n$-term rate**: count $\sim \int|D^2\cdot|/\varepsilon$ (does the convex-gradient ReLU
approximation achieve the curvature-mass rate, matching the moment-of-inertia lower bound?). A
polytopal-approximation question (Gruber, D1). Empirically yes ($d=2$ ratio matches).

**uses.** —  **used by** `separation_general`.

**attempts.** Variation-norm route → ramp-filter obstruction (open $d\ge2$). **Curvature-mass
route → PROVED** (the Jordan-split / $\int\Delta$ subadditivity argument). Radon-convexity lemma
proved as a by-product.
