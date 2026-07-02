# OVERVIEW — semiconcave sparsity program (hawk-eye)

Entry point. Goals first, then the **directions = mathematical machineries**.
Definitions: `NOTATION.md`. **All claims (proved/refuted/open): `CLAIMS.md`.** Keep short;
detail in the pointed folders.

> **Working draft (current writeup):** `/Users/chaoruiz/Documents/Repos/tex-files/Thesis/Mthesis.tex`
> — the LaTeX paper this program feeds into.

---

## Ultimate goal
> **Conditions.** Target $V = \tfrac{C}{2}\|x\|^2 - g$, general $C$-semiconcave ($g$ convex),
> bounded $\Omega\subset\mathbb{R}^d$, $d\ge2$; gradient-$L^2$ best-$n$-term cost $N(M,V,\varepsilon)$
> (NOTATION.md); models = **signed** vs **semiconcave**.
> **Statement (the semiconcave model is more sparse).**
> $$N(\text{semiconcave},V,\varepsilon)\ \le\ N(\text{signed},V,\varepsilon),\quad\text{strictly for small }\varepsilon,$$
> under a condition on $\sigma$.

---

## Directions = the 4 mathematical machineries
A direction is a **named mathematical tool**, not a metric/aspect. Each folder has a
`_direction.md` (machinery + status + its claims) and `claims/`, `refs/` (detail).

| dir | machinery | role now | folder |
|---|---|---|---|
| **D1** | geometric: polytopal approx of convex bodies (Gruber) × ReLU linear regions (Montúfar/Yarotsky) | **CURRENT** — proves the separation (solver-independent) | `D1_geometric/` |
| **D2** | measure-theoretic optimization: Tikhonov over measures, dual certificate, source condition (Bredies, Parhi–Nowak, ADCG) | current-but-stalled — **achievability** half; delivered $d=1$ only | `D2_measure_optimization/` |
| **D3** | harmonic analysis: Radon/ridgelet, $\mathcal R$-norm (Ongie; Helgason) | supplies the norm facts (`budget_1d`, `curved_switching_rnorm`) used by D1 | `D3_harmonic_analysis/` |
| **D4** | max-plus / tropical algebra (McEneaney, Gaubert) | parked — for curved switching where ridges fail | `D4_max_plus/` |

**The separation has two halves needing different machineries:** a lower bound on
the signed model is solver-independent ⟹ **D1** (optimality conditions can't
strengthen a lower bound); achievability of the cone's count by penalized PDAP needs
optimality ⟹ **D2**. So D1 is the active engine; D2/D3 supply complementary pieces.

## Where the goal stands → **all claims in `CLAIMS.md`** (proved/refuted/open registry)
Proved legs (D1, `D1_geometric/claims/`): `head_reduction`, `signed_lower_bound` (general
curved $V$), `correction_cost` (flat & curved), `activation_quadratic_cost`, `separation_flat`,
`separation_1d_convex`, `cone_free_convex`. Achievability:
`D2_measure_optimization/claims/certificate_dichotomy.md` ($d=1$). Open frontier:
`separation_general` (the ultimate goal), `achievability` ($d\ge2$), `minplus_curved` (D4).

## Refuted claims (recorded so we don't recircle)
- `D1_geometric/claims/capacity_selection_split.md`, `single_property_activation.md`.
- `D2_measure_optimization/claims/source_condition_separation_free.md`.
- `_archive/`: legacy-theory-ladder, direction1-smooth (false lemma), full program-log.

## The goal as a convergence-rate comparison (equivalent framing)
"More sparse" $\iff$ "faster rate": $N(M,V,\varepsilon)=e_M^{-1}(\varepsilon)$ where $e_M(n)$ = best $n$-atom
gradient-$L^2$ error. The known **ridge/signed baselines** give $e(n)\le\gamma(\text{target})\,n^{-s}$,
$\gamma$ = the target's variation/Barron norm. **The directly relevant one is in our gradient
($H^1$) norm:**
- **Li–Lu–Mathé–Pereverzev 2024** (ReLU$^k$, extended Barron $B_1^k$, *derivative* approx;
  `PDE/Barron.pdf`): $B_1^k\subset H^k$, and $\|f-f_n\|_{H^m(\Omega)}\le C\|f\|_{B_1^k}\,n^{-1/2}$ for
  $0\le m\le k$ (Thm 8); improved to $\,N^{-1/2-1/d}$ when $m<k$ (Thm 9). **So gradient ($m=1$)
  error decays $n^{-1/2}$ (or $n^{-1/2-1/d}$, needs $k\ge2$); constant = $\|V\|_{B_1^k}$.** Also:
  $k\ge2$ is *required* for stable derivative approximation (for ReLU $k{=}1$ the 2nd derivative
  is a distribution) — the theory reason the repo uses power $p\approx2$.
- Barron 1993: $n^{-1/2}$ (value-$L^2$). Bach 2017: $n^{-1/2}$ on $F_1$, adaptive. Yang–Zhou 2024:
  $n^{-1/2-(2k+1)/2d}$ (variation), $n^{-\alpha/d}$ (Hölder) — value/$L^p$/sup norms.

**What we must beat (now in $H^1$).** Signed fits $V=\tfrac{C}{2}\|x\|^2+g$: $e_{\text{signed}}(n)\le
C\|V\|_{B_1^k}n^{-1/2}$. Semiconcave gets the quadratic free (head): $e_{\text{semi}}(n)\le
C\|g\|_{B_1^k}n^{-1/2}$. Two regimes:
- **constant-factor gain** (smooth $g$): same exponent $n^{-1/2}$, constant $\|g\|_{B_1^k}<\|V\|_{B_1^k}$
  (head removes the quadratic's Barron norm). This is "more sparse" by a constant.
- **faster exponent** (flat/finitely-representable $g$): $g$ exact at finite $K$ while the
  quadratic is *not* a finite ReLU sum ($e_{\text{signed}}\ge c/n>0$ always, `signed_lower_bound`)
  ⟹ semiconcave exact, signed never — `separation_flat`. Condition = `correction_cost`'s
  $N^+(g,\varepsilon)=o(1/\varepsilon)$.

## References (one entry per paper)

`vault:` = file under `~/Documents/NotePaper/MasterThesis/`. ★ = directly load-bearing.
Links verified where marked; a few not-in-vault arXiv ids are best-known (title+author suffice to find).

### A. Approximation rates — the signed-model baselines to beat
- ★ **Y. Li, S. Lu, P. Mathé, S. V. Pereverzev (2024).** *Two-layer networks with the ReLU$^k$
  activation function: Barron spaces and derivative approximation.* Numerische Mathematik 156, 319–360.
  https://doi.org/10.1007/s00211-023-01384-6 — **Gives:** the gradient/$H^m$ rates we use:
  $\|f-f_n\|_{H^m}\le C\|f\|_{B_1^k}n^{-1/2}$ ($m\le k$), improved to $n^{-1/2-1/d}$ for $m<k$ (needs
  $k\ge2$); and that $k\ge2$ is required for derivative approximation (why the repo uses power
  $p\approx2$). `vault: PDE/Barron.pdf`
- **A. R. Barron (1993).** *Universal approximation bounds for superpositions of a sigmoidal
  function.* IEEE Trans. Information Theory 39(3), 930–945. https://doi.org/10.1109/18.256500 —
  **Gives:** the classic $n^{-1/2}$ value-$L^2$ rate, constant = Barron norm $\int|\omega||\hat f(\omega)|d\omega$.
  `vault: Barron_93.pdf`
- **F. Bach (2017).** *Breaking the Curse of Dimensionality with Convex Neural Networks.* JMLR
  18(19), 1–53. https://jmlr.org/papers/v18/14-546.html (arXiv:1412.8690) — **Gives:** the $F_1$
  variation-space framework; $n^{-1/2}$ rate adaptive to low-dimensional linear structure.
  `vault: breaking the curse of dimensionality.pdf`
- **Y. Yang, D.-X. Zhou (2025).** *Optimal Rates of Approximation by Shallow ReLU$^k$ Neural
  Networks and Applications to Nonparametric Regression.* Constructive Approximation 62, 329–360.
  https://doi.org/10.1007/s00365-024-09679-z — **Gives:** optimal variation-space rate
  $n^{-1/2-(2k+1)/2d}$ and Hölder $n^{-\alpha/d}$; matching $n$-width / pseudo-dimension lower bounds.
  `vault: optimal_rate_Relu^k.pdf`

### B. Sparse representation & optimality (D2 — why the solution is atomic; the certificate)
- **R. Parhi, R. D. Nowak (2021).** *Banach Space Representer Theorems for Neural Networks and
  Ridge Splines.* JMLR 22(43), 1–40. https://jmlr.org/papers/v22/20-583.html — **Gives:** the
  representer theorem — the minimizer of [data fit + Radon-domain TV norm] **is** a finite-width
  ReLU net ($m=2$ ridge spline). `vault: Banach space representer theorem.pdf`
- **K. Bredies, H. K. Pikkarainen (2013).** *Inverse problems in spaces of measures.* ESAIM:COCV
  19(1), 190–218. https://doi.org/10.1051/cocv/2011205 — **Gives:** Tikhonov over Radon measures —
  dual certificate (support on the contact set $\{|q|=\alpha\}$), source condition, $O(\delta)$ Bregman
  convergence rate. `vault: Inverse problem in the space of measures.pdf`
- **N. Boyd, G. Schiebinger, B. Recht (2017).** *The Alternating Descent Conditional Gradient
  Method for Sparse Inverse Problems.* SIAM J. Optimization 27(2), 616–639. arXiv:1507.01562 —
  **Gives:** ADCG = **PDAP's ancestor** (conditional gradient over measures, nonnegative by default).
  `vault: CG/The Alternating Descent Conditional Gradient Method….pdf`
- **E. J. Candès, C. Fernández-Granda (2014).** *Towards a Mathematical Theory of Super-resolution.*
  Comm. Pure Appl. Math. 67(6), 906–956. https://doi.org/10.1002/cpa.21455 — **Gives:** the
  minimal-separation certificate for **signed** spike recovery (the "signed needs separation" side).
  *(not in vault)*
- **Y. de Castro, F. Gamboa (2012).** *Exact reconstruction using Beurling minimal extrapolation.*
  J. Math. Anal. Appl. 395(1), 336–354. https://doi.org/10.1016/j.jmaa.2012.05.011 — **Gives:**
  **nonnegative** spike recovery **without** a separation condition (the cone advantage, 1-D). *(not in vault)*
- **V. Duval, G. Peyré (2015).** *Exact Support Recovery for Sparse Spikes Deconvolution.* Found.
  Comput. Math. 15, 1315–1355. arXiv:1306.6909 — **Gives:** support-stability / certificate analysis
  for spike deconvolution. *(not in vault)*

### C. Harmonic analysis / $\mathcal R$-norm (D3 — the variation-norm machinery)
- **G. Ongie, R. Willett, D. Soudry, N. Srebro (2020).** *A Function Space View of Bounded Norm
  Infinite Width ReLU Nets: The Multivariate Case.* ICLR 2020. arXiv:1910.01635 — **Gives:** ReLU
  variation norm $=$ Radon $\mathcal R$-norm $=\|\partial_b^{d+1}\Lambda^{d-1}\mathcal R f\|$; finite for $W^{d+1,1}$.
  (The ramp filter $\Lambda^{d-1}$ here is the obstruction to the variation-norm cone-free route.)
  `vault: Relu(33)_FuncSpace.pdf` (= `1910.01635v1.pdf`)
- **A. Petrosyan, A. Dereventsov, C. G. Webster (2020).** *Neural network integral representations
  with the ReLU activation function.* PMLR 107, 128–143 (MSML 2020).
  https://proceedings.mlr.press/v107/petrosyan20a.html — **Gives:** explicit ReLU integral
  representation and the least-$\ell_1$ representing measure. `vault: Relu(15)_IntgRep.pdf`
- **S. Helgason (1999).** *The Radon Transform* (2nd ed.), Birkhäuser. — **Gives:** the Radon
  transform, inversion, and the ramp filter $\Lambda^{d-1}$. `vault: Radon transform.pdf`
- **W. E, S. Wojtowytsch (2022).** *Representation formulas and pointwise properties for Barron
  functions.* Calc. Var. PDE 61:46. arXiv:2006.05982 — **Gives:** structure theory of Barron-space
  functions. *(not in vault — related)*

### D. Max-plus / tropical (D4 — the curved-switching escape hatch)
- **W. M. McEneaney (2007).** *A Curse-of-Dimensionality-Free Numerical Method for Solution of
  Certain HJB PDEs.* SIAM J. Control Optim. 46(4), 1239–1276. https://doi.org/10.1137/040610830 —
  **Gives:** the max-plus value-function method; value as a min/max of quadratics.
  `vault: curse-of-dimensionality-free_McEeany.pdf`, `curse-free-max-plus_McEneay.pdf`
- **S. Gaubert, W. McEneaney, Z. Qu (2011).** *Curse of dimensionality reduction in max-plus based
  approximation methods: theoretical estimates and improved pruning algorithms.* IEEE CDC 2011.
  arXiv:1109.5241 — **Gives:** error estimates / pruning — how many max-plus basis functions for
  accuracy $\varepsilon$ (the min-architecture atom count). `vault: Gaubert–McEneaney–Qu-2011.pdf`
- **L. Zhang, G. Naitzat, L.-H. Lim (2018).** *Tropical Geometry of Deep Neural Networks.* ICML
  2018. arXiv:1805.07091 — **Gives:** the tropical-algebra view of ReLU networks. *(not in vault)*

### E. Geometric approximation (D1 — our main machinery)
- **P. M. Gruber (1993).** *Asymptotic estimates for best and stepwise approximation of convex
  bodies, I & II.* Forum Mathematicum 5, 281–297 & 521–538. — **Gives:** polytopal approximation
  rates of convex bodies (the moment-of-inertia / facet-count theory behind our $n^{-1}$ bounds).
  *(not in vault)*
- **D. E. McClure, R. A. Vitale (1975).** *Polygonal approximation of plane convex bodies.* J. Math.
  Anal. Appl. 51(2), 326–358. — **Gives:** the 2-D polygonal-approximation rate. *(not in vault)*
- **G. Montúfar, R. Pascanu, K. Cho, Y. Bengio (2014).** *On the Number of Linear Regions of Deep
  Neural Networks.* NeurIPS 2014. arXiv:1402.1869 — **Gives:** counting the hyperplane-arrangement
  cells (linear regions) of ReLU nets. *(not in vault)*
- **D. Yarotsky (2017).** *Error bounds for approximations with deep ReLU networks.* Neural Networks
  94, 103–114. arXiv:1610.01145 — **Gives:** ReLU approximation lower bounds for smooth functions.
  *(not in vault)*
- **R. DeVore, B. Hanin, G. Petrova (2021).** *Neural network approximation.* Acta Numerica 30,
  327–444. https://doi.org/10.1017/S0962492921000052 — **Gives:** survey of NN approximation rates.
  *(not in vault)*

### F. Problem / target
- ★ **K. Kunisch, D. Vásquez-Varas (2026).** *Structure Preserving Approximation of Semiconcave
  Functions.* arXiv:2602.07770 — **Gives:** the repo's semiconcave (min-of-$C^2$) model;
  structure-preserving approximation + gradient convergence. `vault: semiconcave_approximaton.pdf`
- **P. Cannarsa, C. Sinestrari (2004).** *Semiconcave Functions, Hamilton–Jacobi Equations, and
  Optimal Control.* Birkhäuser. — **Gives:** semiconcave-function theory ($D^2V\preceq CI$,
  switching sets, viscosity solutions). `vault: semiconcave-functions_Cannarsa.pdf`
- **X. Li, J. Yong (1995).** *Optimal Control Theory for Infinite Dimensional Systems.* Birkhäuser.
  — **Gives:** optimal-control background. `vault: Zotero`

## Don't get lost
Goals here → defs in NOTATION.md → claims in CLAIMS.md → per-direction `_direction.md`. **Source
of truth = these files.** Do not change agreed structure/definitions without asking
(see memory: dont-change-agreed-things).
