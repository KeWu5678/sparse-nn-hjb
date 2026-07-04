# separation_general  (ULTIMATE GOAL)

**status:** open

**statement (the semiconcave model is more sparse).** For general $C$-semiconcave
$V=\tfrac{C}{2}\|x\|^2-g$ ($g$ convex, arbitrary — including curved switching sets) on bounded
$\Omega\subset\mathbb{R}^d$, under a condition on $\sigma$,
$$N(\text{semiconcave},V,\varepsilon)\ \le\ N(\text{signed},V,\varepsilon),\quad\text{strictly for small }\varepsilon.$$

**status of the pieces (updated 2026-06-18).**
- **$d=1$: PROVED for all convex $g$** — `separation_1d_convex`: ratio $=\int|g''|/\int|V''|$,
  more sparse $\iff \int|g''|<\int|V''|$ ($V$ more curved than $g$).
- signed side $\Omega(1/\varepsilon)$: **proved** any curved $V$, all $d$ (`signed_lower_bound`).
- semiconcave side: $N(\text{semiconcave})=N^+(g)$ (`head_reduction`); cone constraint **free**
  for convex $g$ — `cone_free_convex` now **PROVED in the curvature-mass sense** (the
  count-governing cost): signed cancellation strictly raises $\int\Delta$, so nonneg is optimal
  (avoids the ramp-filter obstruction). [variation-norm $\gamma^+=\gamma$: $d=1$ proved, $d=2$ num. $1.00$.]
- **$d=2$ smooth convex $g$: NUMERICALLY CONFIRMED** (`../../scripts/sep_2d_smooth_convex.py`,
  faithful PDAP, $g=\tfrac12\mathrm{logsumexp}$, $C=2$): measured $N(\text{semi})/N(\text{signed})
  =0.34$ (relGrad 0.10), $0.31$ (0.07) — **matches the curvature predictor
  $\int|D^2g|/\int|D^2V|=0.35$.** So the $d=1$ curvature-ratio law extends to $d=2$; semiconcave
  $\sim3\times$ sparser. The exact rate ($n^{-1}$ vs $n^{-2/d}$) is moot for "more sparse" — the
  *ratio* is the curvature ratio. **With cone-free now proved (curvature-mass sense), the
  general-$d$ analytic proof needs only ONE thing: the matched $n^{-1}$ rate (count $\sim
  \int|D^2\cdot|/\varepsilon$ in $d\ge2$).**
- **curved switching set:** $N^+(g)=\infty$ for the **ridge** model (`correction_cost`) ⟹
  needs the min architecture (D4 `minplus_curved`).
- activation: best-$n$-term needs $\sigma=$ReLU; carry to ReLU²/smooth = penalized regime (D2).

**the rate piece (the last gap) — bracketed, $d=2$ essentially closed.**
- lower bound $n^{-1}$: moment-of-inertia (`signed_lower_bound`; proved for the quadratic).
- upper bound: Li–Lu–Mathé–Pereverzev Thm 9 gives $\|g-g_n\|_{H^1}\le C\|g\|_{B_1^k}n^{-1/2-1/d}$
  for $m{=}1<k$ (**needs $k\ge2$**); in **$d=2$ this is $n^{-1}$, matching the lower bound.** So for
  the repo's setting (power $\approx2$, $d=2$, gradient training) the rate is pinned to $n^{-1}$
  ⟹ the curvature-ratio law is an analytic theorem there (modulo the $k{=}2$ lower bound, which
  the $\sigma''$-Hessian argument should give). General $d$ / $k{=}1$: rate bracketed
  $[n^{-1},n^{-1/2}]$, empirically $n^{-1}$.

**what's needed (open sub-claims).**
1. `minplus_curved` (D4): a non-ridge (min-of-paraboloids) semiconcave model with finite count
   on curved switching sets — then re-compose with `signed_lower_bound`.
2. `achievability` (D2): penalized PDAP attains the cone count, $d\ge2$.
3. activation carry: the penalized/variation-cost statement for ReLU²/smooth $\sigma$.

**uses (when closed).** `head_reduction`, `signed_lower_bound`, `minplus_curved`, `achievability`.

**attempts.** Reduced to the three open sub-claims above; `separation_flat` is the proved
special case (flat switching, ReLU).
