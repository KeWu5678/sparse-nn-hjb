# Overnight progress — 2026-06-18 (read this first)

Worked the current goal ("semiconcave more sparse for general convex $g$") across
directions. **Net: the current goal is now a precise, verified law in $d=1$ (proved) and
$d=2$ (numerically confirmed), with one analytic gap for general $d$.** No churn — every
claim below is either proved or explicitly labeled numerical/open.

## The headline result (new)
**Curvature-ratio law.** For $\sigma=\mathrm{ReLU}$, $V=\tfrac{C}{2}\|x\|^2-g$ ($g$ convex), both
models converge at rate $n^{-1}$ and
$$
\frac{N(\text{semiconcave},V,\varepsilon)}{N(\text{signed},V,\varepsilon)}\ \longrightarrow\ \frac{\int_\Omega|D^2 g|}{\int_\Omega|D^2 V|}.
$$
So **semiconcave is more sparse $\iff \int|D^2g|<\int|D^2V|$** — "the convex correction $g$ is
less curved than $V$ itself", i.e. $V$ has a dominant quadratic bulk. This *refines* "more
sparse" into a checkable curvature condition; it is **not unconditional** (weakly-curved $V$ →
signed wins, because it exploits the cancellation $Cx-g'$ the fixed nonneg head cannot).

- **$d=1$: PROVED** (`D1_geometric/claims/separation_1d_convex.md`). Signed deriv = any $n$-step; semi deriv
  $=Cx+n$-step. Ratio $=\int|g''|/\int|C-g''|$, DP-exact-verified (uniform $g''{=}C/2$→$1.00$,
  $g''{=}0.9C$→$9.00$, flat $g$→$0$). `scripts/sep_1d_convex_check.py`.
- **$d=2$ smooth convex $g$: CONFIRMED** (faithful PDAP). Measured ratio $0.34$/$0.31$ vs
  curvature predictor $0.35$. `scripts/sep_2d_smooth_convex.py`.
- **flat/polyhedral $g$:** the divergent sub-case ($e_{\text{semi}}=0$ at finite $K$) =
  `separation_flat` (already proved).

## Cone constraint is FREE for convex g (T3) — new support
`D1_geometric/claims/cone_free_convex.md`: $\gamma^+(g)=\gamma(g)$ for convex $g$. **Proved $d=1$**; **$d=2$
numerics give ratio exactly $1.00$ for 7/7 smooth convex targets** including anisotropic, rotated,
and varying-Hessian ($x^4$) stress cases (`scripts/t3_cone_free_2d.py`). This removes one of the
two $d\ge2$ obstacles — the cone (nonneg) costs nothing for convex $g$.

## What's PROVED now (legs)
$d=1$ current goal (general convex $g$); `head_reduction`, `signed_lower_bound` (general curved
$V$, all $d$), `correction_cost`, `activation_quadratic_cost`, `separation_flat`, `cone_free_convex`
($d=1$), T1, T2, `certificate_dichotomy` ($d=1$).

## Cone-free (T3) — NOW PROVED (curvature-mass sense), gap shrunk to one rate
Attacked `cone_free_convex`. **Variation-norm route hit the Radon ramp-filter obstruction**
(open $d\ge2$). **Curvature-mass route succeeds — PROVED, all $d$:** Jordan-split any signed
rep $\mu=\mu^+-\mu^-$, $g=h^+-h^-$ ($h^\pm$ convex); since $\Delta h\ge0$ for nonneg,
$\int\Delta h^++\int\Delta h^-\ge\int\Delta g$, equality iff $\mu^-=0$. So **signed cancellation
strictly raises the curvature mass $\int\Delta$ — the nonneg representation is optimal in the
cost that governs the count.** (3-line proof, verified `scripts/cone_free_curvature_mass.py`;
avoids the ramp filter entirely.) By-product: **Radon-convexity lemma** $\partial_b^2\mathcal R\{g\}\ge0$
for convex $g$ (verified), the $d\ge2$ analogue of $g''\ge0$.

**So the full $d\ge2$ curvature-ratio theorem now needs only ONE thing:** the matched $n^{-1}$
rate (count $\sim\int|D^2\cdot|/\varepsilon$ in $d\ge2$ — a polytopal/Gruber approximation question).
Cone-free and head-reduction and the signed lower bound are all in hand. Empirically the rate
(hence the whole law) holds ($d=2$ ratio $0.34$ vs predicted $0.35$).

## Also found / used
- **Gradient ($H^1$) Barron rate** (Li–Lu–Mathé–Pereverzev, `PDE/Barron.pdf`): ReLU$^k$ gives
  $\|f-f_n\|_{H^m}\le C\|f\|_{B_1^k}n^{-1/2}$ ($m\le k$), $n^{-1/2-1/d}$ ($m<k$); $k\ge2$ required
  for derivative approx (justifies repo power $p\approx2$). Refs added to OVERVIEW by direction.

## Suggested next (your call)
- **Close the gap:** prove $\partial_b^{d+1}\mathcal R\{g\}\ge0$ for convex $g$ (the cone-free Radon
  lemma) → upgrades the curvature-ratio law to a $d\ge2$ theorem.
- **Curved switching** (the part where ridge fails): D4 `minplus_curved`.
- Files: `OVERVIEW.md` (goals+refs), `CLAIMS.md` (registry), `D1_geometric/claims/` (the action).
