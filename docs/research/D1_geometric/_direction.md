# D1 — Geometric machinery (CURRENT direction)

**Machinery.** Polytopal approximation of convex bodies (Gruber; McClure–Vitale)
**×** ReLU linear regions (Montúfar et al.; Yarotsky lower bounds). Engine: a ReLU
net's gradient is piecewise-constant on a hyperplane arrangement ($\le\kappa_d n^d$ cells);
the moment-of-inertia inequality bounds the $L^2$ error per cell. **Solver-independent**
— it bounds what *any* network can represent, so a lower bound here holds for whatever
the solver outputs (optimality conditions cannot strengthen a lower bound).

**Role.** Proves the **separation**: lower bound on signed + constructive upper bound on cone.

## Claims (full statements in `claims/`; registry in `../CLAIMS.md`)
Guiding idea: the cone's head represents $\tfrac{C}{2}\|x\|^2$ for free, so the semiconcave model
only pays to approximate the convex correction $g$, while the signed model must build all of $V$.

name: `head_reduction`\
status: proved (all $d$)\
statement: The semiconcave model represents the quadratic $\tfrac{C}{2}\|x\|^2$ exactly with its head (zero atoms), so its neuron count equals $N^+(g,\varepsilon)$ — the count needed to approximate the convex correction $g$ alone with nonnegative atoms — independent of $C$.

---

name: `signed_lower_bound`\
status: proved (general curved $V$, all $d$)\
statement: If $\sigma=$ReLU and $V$ is positive homogeneous on some ball $B$ ($D^2V\succeq cI>0$ there), then every signed network needs at least $\kappa_d\,c\,\mathrm{vol}(B)^{1/2+1/d}/\varepsilon$ atoms to reach gradient error $\varepsilon$; that is, $N(\text{signed},V,\varepsilon)=\Omega(1/\varepsilon)$.

---

name: `correction_cost`\
status: proved (flat and curved cases)\
statement: The cost $N^+(g,\varepsilon)$ of approximating the convex correction $g$ with nonnegative ReLU atoms is $O(1)$ (exactly $K$ atoms, independent of $\varepsilon$) when $g$ is piecewise-linear with $K$ pieces, but becomes infinite when $g$ has a curved switching set (its ridge variation norm diverges).

---

name: `activation_quadratic_cost`\
status: proved (best-$n$-term)\
statement: A signed network needs $\Theta(1/\varepsilon)$ atoms to approximate the quadratic $\tfrac{C}{2}\|x\|^2$ in gradient norm if and only if $\sigma''=0$ almost everywhere (ReLU-type); for $\sigma''\not\equiv0$ (smooth or polynomial activations) only $O(d)$ atoms suffice.

---

name: `separation_1d_convex`\
status: proved ($d=1$, any convex $g$)\
statement: In one dimension both models converge at rate $n^{-1}$ and $N(\text{semiconcave})/N(\text{signed})=\int|g''|/\int|V''|$; so the semiconcave model is more sparse exactly when $g$ carries less total curvature than $V$.

---

name: `cone_free_convex`\
status: proved (curvature-mass sense, all $d$)\
statement: For convex $g$ the nonnegativity constraint $c_i\ge0$ costs nothing in the curvature-mass cost that governs the neuron count: any signed representation that uses cancellation has strictly larger total curvature mass $\int\Delta$ than the nonnegative one.

---

name: `separation_flat`\
status: proved (all $d$; a special case of `separation_general`)\
statement: When $V$ has a curved quadratic bulk and $g$ is piecewise-linear (flat switching set), the semiconcave model needs $K$ atoms while the signed model needs $\Theta(1/\varepsilon)$, so the ratio $N(\text{signed})/N(\text{semiconcave})=\Theta(1/(K\varepsilon))$ diverges.

---

name: `separation_general`\
status: open (the ultimate goal)\
statement: For a general convex correction $g$, the semiconcave model is more sparse than the signed model: $N(\text{semiconcave},V,\varepsilon)\le N(\text{signed},V,\varepsilon)$ for small $\varepsilon$. Proved in $d=1$; confirmed numerically in $d=2$; the general-$d$ rate is the remaining gap.

---

name: `capacity_selection_split`\
status: refuted\
statement: The cone advantage does **not** split additively into a "head/capacity" term plus a "cone/selection" term — the subtraction test meant to measure the two is confounded and gives contradictory answers.

---

name: `single_property_activation`\
status: refuted\
statement: No single qualitative property of $\sigma$ (neither smoothness nor convexity of the atom) decides whether the cone is more sparse; what matters is the relative curvature cost, not one property.

---

**Frontier** (not a claim — roadmap): `separation_general` is the open ultimate goal; its two
remaining pieces are the curved switching set (→ D4 `minplus_curved`) and $d\ge2$ achievability
(→ D2 `certificate_dichotomy`).

## Cross-direction dependencies
- `correction_cost` (curved) uses **D3** `curved_switching_rnorm`.
- achievability of these counts by penalized PDAP is **D2** `certificate_dichotomy`.

## Refs → `refs/`
`theorem-A-d-and-history.md` (`signed_lower_bound` flat case + retracted history),
`semiconcave-rate.md` (`head_reduction`, `correction_cost`),
`sparsity-statements-and-structure.md` (templates), `direction2-relu2.md` (ReLU² endpoint
for `activation_quadratic_cost`). External: Gruber, Montúfar, Yarotsky, Yang–Zhou.
