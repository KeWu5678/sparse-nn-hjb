# head_reduction

**status:** proved

**statement.** Let $V=\tfrac{C}{2}\|x\|^2-g$ on bounded $\Omega\subset\mathbb{R}^d$ with $g$ convex;
cone (semiconcave) model with quadratic head; cost $N(M,V,\varepsilon)$ = gradient-$L^2$
best-$n$-term (CONTEXT.md). Then
$$N(\text{semiconcave},V,\varepsilon)=N^+(g,\varepsilon),\quad\text{independent of }C,$$
where $N^+(g,\varepsilon)$ = fewest **nonnegative** atoms approximating $g$ to gradient error $\varepsilon$.

**proof.** Set $\tilde C=C$, affine $=0$. The head reproduces $\tfrac{C}{2}\|x\|^2$ exactly at
0 atoms, so $\nabla(f-V)=-\nabla(\sum c_i\sigma^p-g)$; the head contributes 0 to error and 0
atoms, and the constraint $c_i\ge0$ is exactly nonnegativity of $g$'s representing measure. $\square$

**uses.** —  (foundational)

**attempts.** Clean from the start; the earlier entanglement in an additive
head+selection split is the refuted `capacity_selection_split` — `head_reduction` is its
correct residue. Detail: `../refs/semiconcave-rate.md` (Prop A).
