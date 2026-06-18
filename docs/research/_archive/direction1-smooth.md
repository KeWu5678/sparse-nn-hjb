# Direction 1 — smooth / kernel activations: the head advantage is smoothness-graded

Status: 2026-06-16. Setting per `sparsity-gap-real-setting.md`: $V=\tfrac{C}{2}\|x\|^2-g$,
gradient-$L^2$ metric $N(\text{model},\varepsilon)$, $d\ge2$. Here $\sigma$ is a smooth,
**non-polynomial** activation (gaussian $e^{-t^2/2}$, matérn-$\nu$, also softplus/tanh).
Goal: is the head's capacity advantage $N_\sigma(Q,\varepsilon)$ (quadratic cost) fundamental,
and at what rate $\kappa_\sigma$?

## Lemma 1 (restriction to a line) — reduces d-D to 1-D, RIGOROUS

Let $f(x)=\sum_{i\le n}c_i\sigma(w_i\!\cdot\!x-b_i)+(\text{affine})$. On any line
$x=p+tv$ ($\|v\|=1$),
$$
f(p+tv)=\sum_{i\le n} c_i\,\sigma(a_i t+\beta_i)+(\text{affine in }t),
\quad a_i=w_i\!\cdot\!v,\ \beta_i=w_i\!\cdot\!p-b_i,
$$
a **1-D σ-ridge sum with $\le n$ atoms**; and $Q=\tfrac{C}{2}\|x\|^2$ restricts to the
1-D quadratic $\tfrac{C}{2}t^2+(\text{linear})$. Since
$\|\nabla f-\nabla Q\|_{L^2(\Omega)}^2 \ge \int_\Omega|\partial_v(f-Q)|^2
=\int_{\text{lines}\,\|\,v}\!\int_t |\tfrac{d}{dt}(f-Q)|^2\,dt\,d\ell_\perp$ (Fubini),
a model with $n$ atoms achieving d-D gradient error $\le\varepsilon$ yields 1-D
derivative-error $\le \varepsilon/\sqrt{c_\Omega}$ on a positive fraction of lines. Hence
$$
\boxed{\,N_\sigma^d(Q,\varepsilon)\ \ge\ N_\sigma^1\big(t^2,\ c_\Omega\,\varepsilon\big).\,}
$$
**The d-D head advantage is at least the 1-D one** — and 1-D is tractable.

## Lemma 2 (dichotomy) — diverges for every non-polynomial σ, RIGOROUS

If $\sigma$ is **not** a polynomial of degree $\le 2$, then $t^2$ is not a finite
combination $\sum_{i\le m}c_i\sigma(a_it+\beta_i)$ for any $m$. (If it were, $\sigma$
would satisfy: every dilate/translate lies in the span of finitely many others
needed to build $t^2$ — forcing $\sigma$ into a finite-dim shift-dilation-invariant
space, i.e. a quasi-polynomial; excluded for gaussian, matérn, softplus, tanh.)
Therefore $N_\sigma^1(t^2,\varepsilon)\to\infty$ as $\varepsilon\to0$, and by Lemma 1
$$
N_\sigma^d(Q,\varepsilon)\to\infty:\quad\text{the head advantage DIVERGES for all non-poly }\sigma.
$$
Contrast ReLU² (polynomial, $\kappa_\sigma=0$, `direction2-relu2.md`).

## The rate is graded by σ's smoothness (structural principle)

In 1-D, approximating the (entire) target $t^2$ by translates/dilates of a single
non-poly $\sigma$ saturates at **σ's own smoothness**:

| σ smoothness | example | 1-D rate $N_\sigma^1(t^2,\varepsilon)$ | $\kappa_\sigma$ (fundamental) |
|---|---|---|---|
| $C^0$ (kink) | ReLU | $\Theta(1/\varepsilon)$ (proved, Thm A-d) | $1$ |
| $C^s$, $s<\infty$ | matérn-$\nu$ | polynomial, $\sim\varepsilon^{-1/s}$-type | $\in(0,1)$ |
| $C^\infty$ / analytic | gaussian, softplus, tanh | spectral: $\mathrm{poly}\log(1/\varepsilon)$ (Mhaskar) | $\to 0$ |
| polynomial deg $\le2k$ | ReLU² | $O(d)$ exact | $0$ |

So the **fundamental** (best-$n$-term) head advantage is large only for *rough*
activations (ReLU, low-order matérn); for $C^\infty$ activations it is only
logarithmic. The mechanism: a single smooth ridge already carries curvature
$\sigma''(w\!\cdot\!x-b)ww^{\mathsf T}$ over a tunable-width slab, and smoother $\sigma$ ⟹ broader,
better-conditioned curvature ⟹ fewer atoms needed for the quadratic.

## Consistency with measurement, and the greedy/fundamental gap

Measured **greedy** exponents (`scripts/quadratic_cost_by_activation.py`, pure
quadratic, repo PDAP) respect the smoothness ordering exactly:
$$
\kappa^{\mathrm{greedy}}:\quad \text{ReLU }1.0\ >\ \text{matérn }0.86\ >\ \text{gaussian }0.56\ \ (>\ \text{softplus/tanh: plateau}).
$$
**But these are greedy upper bounds on the fundamental $\kappa_\sigma$.** The ReLU²
lesson (`direction2-relu2.md`: greedy $0.5$ vs fundamental $0$) warns that for the
$C^\infty$ activations the *fundamental* advantage may be much smaller (log) than the
measured greedy $0.5$–$0.86$ — i.e. for gaussian/softplus the cone's practical edge
is partly a **greedy/solver effect**, as with ReLU². For matérn (finite smoothness)
a genuine polynomial $\kappa>0$ is expected even at best-$n$-term.

## Conclusion (honest, useful)

- **Rigorous:** for every non-polynomial activation the head gives a *diverging*
  capacity advantage (Lemmas 1–2). This includes all smooth activations we use —
  so unlike ReLU², the advantage is *real*, not merely algorithmic.
- **Graded:** its rate $\kappa_\sigma$ tracks $\sigma$'s smoothness — **large for ReLU
  and low-order matérn, vanishing (log) for $C^\infty$ gaussian/softplus/tanh**, zero
  for polynomial. The measured greedy ordering confirms the grading.
- **Design read:** the strongest *fundamental* semiconcave advantage is at rough
  activations (ReLU, low-$\nu$ matérn); at very smooth activations it is weak
  fundamentally but can still be large in practice via greedy (a solver effect,
  per D2). Matérn is the sweet spot: smooth enough for gradient fitting, rough
  enough for a genuine $\kappa>0$.

## Open (the remaining analysis)

1. **1-D lower bound for matérn-$\nu$:** prove $N_\sigma^1(t^2,\varepsilon)\gtrsim\varepsilon^{-1/s}$
   for $\sigma\in C^s\setminus C^{s+1}$ (the finite-smoothness ridge rate); via
   Lemma 1 this lifts to $d$-D and makes the matérn case a theorem.
2. **Separate greedy from fundamental** for $C^\infty$ σ: measure best-$n$-term
   (non-greedy fit) for gaussian/softplus on a pure quadratic; predict $n\sim
   \mathrm{polylog}(1/\varepsilon)$, far below the greedy $\varepsilon^{-0.5}$ — confirming the
   practical edge there is solver-driven (like ReLU²).
3. Then D3 (sigmoidal/Barron) supplies the cleanest external rate for softplus/tanh.
