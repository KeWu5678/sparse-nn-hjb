# activation_quadratic_cost

**status:** proved (best-$n$-term)

**statement.** Cost = gradient-$L^2$ best-$n$-term. For the quadratic $Q=\tfrac{C}{2}\|x\|^2$:
1. $\sigma''=0$ a.e. (piecewise-linear $\sigma$: ReLU, leaky-ReLU, hard-tanh) ⟹
   $N(\text{signed},Q,\varepsilon)=\Theta(1/\varepsilon)$.
2. $\sigma''\not\equiv0$ ($\exists t_0:\sigma''(t_0)\ne0$; smooth, or polynomial ReLU$^k$, $k\ge2$) ⟹
   $N(\text{signed},Q,\varepsilon)=O(d)$, $\varepsilon$-independent.

Hence the head's divergent advantage on the quadratic occurs **iff $\sigma''=0$ a.e.**

**proof.** (1) is `signed_lower_bound` specialized to $Q$ (constant Hessian $CI$) + staircase
upper bound. (2) **broad-atom construction:** for axis $e_k$, $c[\sigma(a\,e_k\!\cdot\!x)+\sigma(-a\,e_k\!\cdot\!x)]
=\text{const}+\tfrac{a^2\sigma''(0)}{1}(e_k\!\cdot\!x)^2+O(a^4)$; choosing $c=C/(a^2\sigma''(0))$ and summing
$d$ axes gives $\nabla Q$ to error $O(a)\to0$ with $2d$ atoms (weights $\sim1/a^2$). Exact for
ReLU²: $(w\!\cdot\!x)_+^2+(-w\!\cdot\!x)_+^2=(w\!\cdot\!x)^2$, $2d$ atoms (`../refs/direction2-relu2.md`).
Verified: `../../scripts/quadratic_cost_by_activation.py`. $\square$

**uses.** `signed_lower_bound` (case 1).

**caveat (important).** This is the **best-$n$-term** condition. In the **penalized** regime
the broad-atom shortcut (case 2) is forbidden (weights $\sim1/a^2$ exceed the penalty
budget), so smooth $\sigma$ also shows a large cone advantage in practice — a separate
*variation-cost* statement, D2, not this claim. So: for the clean best-$n$-term theorem fix
$\sigma=$ReLU; the practical smooth/ReLU² advantage is penalized-regime.

**attempts.** 1. "smoothness ⟺ advantage" — refuted (`single_property_activation`). 2. "atom
convexity C1 ⟺ advantage" — refuted (same). The correct invariant is $\sigma''$ + the metric.
