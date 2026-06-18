# D1 — Geometric machinery (CURRENT direction)

**Machinery.** Polytopal approximation of convex bodies (Gruber; McClure–Vitale)
**×** ReLU linear regions (Montúfar et al.; Yarotsky lower bounds). Engine: a ReLU
net's gradient is piecewise-constant on a hyperplane arrangement ($\le\kappa_d n^d$ cells);
the moment-of-inertia inequality bounds the $L^2$ error per cell. **Solver-independent**
— it bounds what *any* network can represent, so a lower bound here holds for whatever
the solver outputs (optimality conditions cannot strengthen a lower bound).

**Role.** Proves the **separation** (current goal): lower bound on signed + constructive
upper bound on cone.

## Claims → `claims/`  (uniform format; registry in `../CLAIMS.md`)
**Head-reduction separation.** ratio $=$ cost$_\sigma(\nabla V)/$cost$_\sigma(\nabla g)$; the cone's
head supplies $Cx$ free, so it pays only for $\nabla g$.

| claim | statement | status |
|---|---|---|
| `head_reduction` | $N(\text{semiconcave},V,\varepsilon)=N^+(g,\varepsilon)$ | **proved** all $d$ |
| `signed_lower_bound` | $N(\text{signed},V,\varepsilon)\ge\kappa_d c\,\mathrm{vol}(B)^{1/2+1/d}/\varepsilon$ | **proved** general curved $V$ |
| `correction_cost` | $N^+(g)$: flat $K$ / curved $\infty$ | **proved** (flat, curved); smooth = remark |
| `activation_quadratic_cost` | $\sigma''=0$ a.e. $\iff$ quadratic $\Theta(1/\varepsilon)$-costly | **proved** (best-$n$-term) |
| `separation_flat` (current goal, proved instance) | curved bulk + flat $g$ ⟹ more sparse (in fact ratio $\to\infty$) | **proved** all $d$ |
| `separation_general` (ULTIMATE) | general convex $g$ ⟹ $N(\text{semiconcave})\le N(\text{signed})$ | **open** |
| `capacity_selection_split` | additive head+cone decomposition | **refuted** |
| `single_property_activation` | one $\sigma$-property ⟺ advantage | **refuted** |

**Current goal proved:** `head_reduction` + `signed_lower_bound`(curved) + `correction_cost`(flat)
⟹ ratio $=\Theta(1/(K\varepsilon))\to\infty$, general curved bulk + flat switching, all $d$. Frontier =
`separation_general` (curved switching → D4; $d\ge2$ achievability → D2).

## Cross-direction dependencies
- `correction_cost` (curved) uses **D3** `curved_switching_rnorm` (T2).
- achievability of these counts by penalized PDAP is **D2** `certificate_dichotomy`.

## Refs → `refs/`
`theorem-A-d-and-history.md` (`signed_lower_bound` flat case + retracted history),
`semiconcave-rate.md` (`head_reduction`, `correction_cost`),
`sparsity-statements-and-structure.md` (templates), `direction2-relu2.md` (ReLU² endpoint
for `activation_quadratic_cost`). External: Gruber, Montúfar, Yarotsky, Yang–Zhou.
