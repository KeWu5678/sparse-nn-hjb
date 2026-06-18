# Direction 2 — ReLU² (squared activation): the semiconcave advantage is *algorithmic*, not representational

Status: 2026-06-16. Setting per the locked formulation (`sparsity-gap-real-setting.md`):
target $V=\tfrac{C}{2}\|x\|^2-g$, $g$ convex, gradient-$L^2$ metric $N(\text{model},\varepsilon)$,
$d\ge2$. Activation $\sigma_2(t)=(t)_+^2$ (ReLU²). Reference: Yang–Zhou,
*Optimal Rates of Approximation by Shallow ReLUᵏ NN* (`optimal_rate_Relu^k.pdf`).

## Why ReLU² is the clean case

The Hessian of a ReLU² atom is trivial:
$$
D^2\big[(w\!\cdot\!x-b)_+^2\big] = 2\,H(w\!\cdot\!x-b)\,ww^{\mathsf T}
\quad(\text{rank-1, constant } 2ww^{\mathsf T}\text{ on } \{w\!\cdot\!x>b\},\ 0 \text{ else}).
$$
So a model $V=\tfrac{C}{2}\|x\|^2-\sum_i c_i(w_i\!\cdot\!x-b_i)_+^2$ has
$D^2V = C\,I - 2\sum_i c_i H(w_i\!\cdot\!x-b_i)\,w_iw_i^{\mathsf T}$, piecewise-constant, and
$$
\boxed{\,c_i\ge 0 \iff D^2V \preceq C\,I \iff V \text{ is } C\text{-semiconcave.}\,}
$$
**The cone constraint is *identical* to semiconcavity.** Cone class $=$ exactly
$\{C\text{-semiconcave ReLU² functions}\}$; signed class $=$ all ReLU² functions
(affine head only). This removes the capacity confound and makes the Hessian algebra exact.

## Result 1 — no diverging capacity gap ($\kappa_\sigma = 0$, PROVED)

A quadratic is a degree-$2$ ($=2k$) polynomial, exactly representable in the ReLU²
dictionary:
$$
(w\!\cdot\!x)^2 = (w\!\cdot\!x)_+^2 + (-w\!\cdot\!x)_+^2,
\qquad
\tfrac{C}{2}\|x\|^2 = \tfrac{C}{2}\sum_{k=1}^d\big[(e_k\!\cdot\!x)_+^2+(-e_k\!\cdot\!x)_+^2\big]
\ \ (2d \text{ atoms, exact}).
$$
(Consistent with Yang–Zhou Thm 2.1: $\mathcal H^\alpha\subseteq\mathcal F_{\sigma_2}(M)$ for
$\alpha>(d+5)/2$, so a $C^\infty$ quadratic has finite variation norm; here it is
moreover *exact*.) Hence, for $g=\sum_{i\le m}c_i(\cdot)_+^2$ ($m$ ReLU² ridges):
$$
N(\text{cone},0)=m, \qquad N(\text{signed},0)\le m+2d,
\qquad\Rightarrow\qquad
N(\text{signed})-N(\text{cone}) \le 2d = O(d).
$$
**The best-$n$-term gap is bounded by a constant $2d$ — not diverging.** Contrast
ReLU ($k=1$): $\Theta(1/\varepsilon)$. So for squared activations the semiconcave model has
**no fundamental (representational) sparsity advantage.** $\kappa_\sigma=0$.

## Result 2 — the observed gap is greedy emulation-failure (CONFIRMED)

Yet empirically the gap is large (gelu² on VDP: cone $0$ vs signed $28$). It is an
**optimization** effect, not representation. Measured on a pure quadratic
($n_{\mathrm{cone}}=0$, so $n_{\mathrm{signed}}=N_{\sigma_2}(Q,\varepsilon)$ as realized by PDAP;
`../../scripts/quadratic_cost_by_activation.py`, true ReLU²):

| relGrad ε | 0.15 | 0.10 | 0.05 | 0.02 |
|---|---|---|---|---|
| greedy $n_{\mathrm{signed}}$ | 6 | 7 | 11 | 16 |

$\kappa_\sigma^{\mathrm{greedy}}\approx +0.50$ — **growing**, although the best-$n$-term cost is
$2d=4$ flat (at $n=4$ PDAP reaches only $\varepsilon=0.53$; the exact 4-atom emulation is
never discovered). The mechanism: PDAP inserts the single atom maximising residual
correlation, but the emulation is a *symmetric pair* $(w\!\cdot\!x)_+^2+(-w\!\cdot\!x)_+^2$
whose halves are one-sided convex bumps — each individually a poor match to the
symmetric quadratic, so greedy lays down many one-sided bumps instead of the pair.
The quadratic head **hard-codes** the emulation, bypassing the discovery problem.

## Conclusion (honest)

For ReLU² / squared activations:
- **Representationally:** cone and signed are equivalent up to $+2d$ atoms — *no
  diverging advantage* ($\kappa_\sigma=0$). The earlier "head = capacity" story is
  **false here**.
- **Algorithmically:** the cone wins ($\kappa^{\mathrm{greedy}}\approx0.5$) purely because
  its head encodes the bulk emulation that greedy insertion fails to find.
- **Design implication:** if one wants the *fundamental* semiconcave inductive-bias
  advantage, squared activations are the *wrong* choice — there the cone only fixes a
  solver pathology (which a better signed solver could also fix). The fundamental
  (capacity, diverging) advantage lives in sub-quadratic-growth activations:
  ReLU ($\kappa=1$, proved) and the smooth/sigmoidal families (Directions 1 & 3,
  $\kappa>0$ measured).

This is itself a useful separation: **ReLU² cleanly isolates "selection" from
"capacity," and shows the semiconcave model's *representational* edge is null there.**

## Open / next within Direction 2 (optional)

Prove $\kappa_\sigma^{\mathrm{greedy}}>0$ rigorously: that PDAP/greedy insertion against a
symmetric quadratic residual cannot realize the $O(1)$-atom emulation, forcing
$\omega(1)$ atoms. Uses the one-sided-correlation profile of a single ReLU² ridge
vs the symmetric target (the "paired-atom low marginal gain" obstruction). This
would convert the measured $\kappa\approx0.5$ into a theorem about greedy on the
squared dictionary.
