# budget_1d  (T1)

**status:** proved

**statement.** Let $V$ be $C$-semiconcave and $L$-Lipschitz on bounded convex $\Omega\subset\mathbb{R}^d$,
$g=\tfrac{C}{2}\|x\|^2-V$ convex. Then the curvature mass (budget) is finite, independent of the
number/curvature of switching surfaces:
$$\mathrm{bud}(g):=\|D^2g\|_{\mathcal M}=\int_\Omega d(\Delta g)=\int_{\partial\Omega}\partial_n g\,dS\ \le\ \mathrm{Lip}(g)\,|\partial\Omega|.$$
In $d=1$: $\mathrm{bud}(g)=\|g''\|=\mathrm{TV}(V'')\le 2C|\Omega|+2L$, and $\mathrm{bud}=\gamma^+(g)$ (the nonneg
ridge norm) exactly.

**proof.** $g$ convex ⟹ $D^2g\succeq0$ (Alexandrov), so $\Delta g\ge0$ as a measure;
Gauss–Green on convex $\Omega$ gives $\int_\Omega d(\Delta g)=\int_{\partial\Omega}\partial_n g$, and
$\partial_n g\le|\nabla g|\le\mathrm{Lip}(g)$. In $d=1$, $g(x)=\text{affine}+\int(x-t)_+\,dg''(t)$ with
$g''\ge0$, so the nonneg ReLU measure is $g''$ and $\gamma^+=\|g''\|=\mathrm{bud}$. $\square$

**uses.** —  (foundational; the $d$-dim form of the 1-D semiconcavity budget)

**attempts.** The $d=1$ identity $\mathrm{bud}=\gamma^+$ is special — in $d\ge2$ they differ
(`correction_cost`: $\mathrm{bud}$ finite but $\gamma^+$ can be $\infty$). Detail:
`../refs/t1-1d-budget-lemma.md`.
