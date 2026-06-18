# separation_flat  (proved instance of the current goal)

**status:** proved (all $d$, 2026-06-17)

The current goal (OVERVIEW) is the divergence for **general convex $g$**, which holds iff
$N^+(g,\varepsilon)=o(1/\varepsilon)$. This claim is the proved instance: $g$ flat/polyhedral ⟹ $N^+=K=O(1)$
⟹ $\Theta(1/(K\varepsilon))$. (Smooth $g$: constant ratio. Curved switching: `separation_general`.)

**statement.** Let $\sigma=\mathrm{ReLU}$; $V=\tfrac{C}{2}\|x\|^2-g$ with a ball $B$ where
$D^2V\succeq cI>0$ (curved bulk) and $g$ piecewise-linear with $K$ pieces (flat switching set).
Then
$$N(\text{signed},V,\varepsilon)=\Theta(1/\varepsilon),\quad N(\text{semiconcave},V,\varepsilon)=K,\quad\Rightarrow\quad
\frac{N(\text{signed})}{N(\text{semiconcave})}=\Theta\!\Big(\frac1{K\varepsilon}\Big)\to\infty.$$

**proof.** Compose proved legs: $N(\text{semiconcave})=N^+(g)=K$ by `head_reduction` + `correction_cost`
(flat); $N(\text{signed})=\Omega(1/\varepsilon)$ by `signed_lower_bound` (the curved-$B$ part), with the
matching $O(1/\varepsilon)$ staircase upper bound giving $\Theta$. $\square$

**uses.** `head_reduction`, `correction_cost` (flat), `signed_lower_bound`.

**caveat.** $\sigma=$ReLU (machinery). The carry to the practical activation is
`activation_quadratic_cost`; the curved-switching generalization is `separation_general`.

**attempts.** Assembled once `signed_lower_bound` was extended from flat $g$ (Theorem A-d) to
general curved $V$ (2026-06-17).
