# signed_lower_bound

**status:** proved

**statement.** Let $\sigma=\mathrm{ReLU}$ and let $V$ have a ball $B\subseteq\Omega$ with
$D^2V\succeq cI>0$ on $B$ (genuinely curved). Then every signed ReLU net
$f=\sum_{i\le n}c_i(w_i\!\cdot\!x-b_i)_++\text{affine}$ satisfies
$$\|\nabla f-\nabla V\|_{L^2(B)}\ \ge\ \kappa_d\,c\,\mathrm{vol}(B)^{\frac12+\frac1d}\,n^{-1},
\qquad\text{hence}\qquad N(\text{signed},V,\varepsilon)=\Omega(1/\varepsilon).$$
For flat $g$ ($D^2V=CI$) a matching $O(1/\varepsilon)$ ReLU-staircase upper bound makes it tight:
$N(\text{signed})=\Theta(1/\varepsilon)$.

**proof.** $\nabla f$ is constant on each of $N\le\kappa_d n^d$ cells $K_\ell$ of the arrangement
of $f$'s $n$ hyperplanes (ReLU gradient is piecewise-constant). On $K_\ell$, minimizing over
the constant value, $\int_{K_\ell}|\nabla f-\nabla V|^2\ge\int_{K_\ell}|\nabla V-\langle\nabla V\rangle|^2$.
*Curvature step:* on $B$ write $V=\tfrac{c}{2}\|x\|^2+h$, $h$ convex ($D^2h=D^2V-cI\succeq0$);
then $\int_{K_\ell}|\nabla V-\langle\nabla V\rangle|^2\ge c^2\int_{K_\ell}|x-\bar x|^2$, since the cross term
$\int(x-\bar x)\!\cdot\!(\nabla h(x)-\nabla h(\bar x))\ge0$ by monotonicity of $\nabla h$. Moment-of-
inertia $\int_{K_\ell}|x-\bar x|^2\ge c_d\mathrm{vol}(K_\ell)^{1+2/d}$, then sum + power-mean
($1+2/d>1$) gives $\|\nabla f-\nabla V\|_{L^2(B)}^2\ge c^2c_d\kappa_d^{-2/d}\mathrm{vol}(B)^{1+2/d}n^{-2}$.
$\square$ (Curvature step checked: `../../scripts/curved_lowerbound_check.py`, worst ratio
$3.15\ge1$. Flat case = Theorem A-d, `../refs/theorem-A-d-and-history.md`.)

**uses.** —  (foundational; generalizes Theorem A-d from flat $g$ to general curved $V$)

**caveat.** Proved for $\sigma=$ReLU. Pure-ReLU gradient training is not the repo setting
(repo: $p=1$/smooth/ReLU²) — this is a **machinery/leg**, carried to real activations by
`activation_quadratic_cost`.

**attempts.** 1. Early draft had rate $\varepsilon^{-d/2}$ — **wrong**; corrected to tight $\Theta(1/\varepsilon)$.
2. Reading the gap as "capacity" in an additive split → refuted `capacity_selection_split`.
