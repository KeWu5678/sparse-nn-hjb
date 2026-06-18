# OVERVIEW — semiconcave sparsity program (hawk-eye)

Entry point. Goals first, then the **directions = mathematical machineries**.
Definitions: `CONTEXT.md`. **All claims (proved/refuted/open): `CLAIMS.md`.** Keep short;
detail in the pointed folders.

---

## Ultimate goal
> **Conditions.** Target $V = \tfrac{C}{2}\|x\|^2 - g$, general $C$-semiconcave ($g$ convex),
> bounded $\Omega\subset\mathbb{R}^d$, $d\ge2$; gradient-$L^2$ best-$n$-term cost $N(M,V,\varepsilon)$
> (CONTEXT.md); models = **signed** vs **semiconcave**.
> **Statement (the semiconcave model is more sparse).**
> $$N(\text{semiconcave},V,\varepsilon)\ \le\ N(\text{signed},V,\varepsilon),\quad\text{strictly for small }\varepsilon,$$
> under a condition on $\sigma$.

## Current goal
> **Conditions.** $\sigma=\mathrm{ReLU}$; $V=\tfrac{C}{2}\|x\|^2-g$ with $g$ **general convex**;
> gradient-$L^2$ best-$n$-term cost.
> **Statement (more sparse — curvature comparison).** Both models converge at rate $n^{-1}$,
> with $N(M,V,\varepsilon)\sim K_M/\varepsilon$, and
> $$\frac{N(\text{semiconcave},V,\varepsilon)}{N(\text{signed},V,\varepsilon)}=\frac{\int_\Omega|D^2 g|}{\int_\Omega|D^2 V|},
> \qquad\text{so semiconcave more sparse}\iff \int|D^2 g|<\int|D^2 V|.$$
> ("$g$ carries less total curvature than $V$" — i.e. $V$ has a dominant quadratic bulk.)
>
> **STATUS (2026-06-18, overnight).** **PROVED in $d=1$** (`separation_1d_convex`, DP-verified):
> ratio $=\int|g''|/\int|C-g''|$ exactly; more sparse $\iff \int|g''|<\int|V''|$; divergent when
> $g$ piecewise-linear (`separation_flat`). **Honest refinement:** "more sparse" is *not*
> unconditional — for weakly-curved $V$ (large $\int|g''|$ vs $C$) the **signed** model wins
> (it exploits the cancellation $Cx-g'$ the fixed nonneg head cannot). The program's premise
> (HJB value functions have a dominant curved bulk) is exactly the winning regime.
> **$d\ge2$:** same structure ($\int|D^2g|$ vs $\int|D^2V|$). Two pieces — **(1) cone-free: now
> PROVED** in the curvature-mass sense (`cone_free_convex`: signed cancellation strictly raises
> $\int\Delta$; avoids the ramp-filter obstruction). **(2) the $n^{-1}$ rate:** bracketed
> $[n^{-1},n^{-1/2}]$; in **$d=2$ with power $\ge2$ pinned to $n^{-1}$** (Li et al. Thm 9 upper +
> moment-of-inertia lower) ⟹ the law is a theorem there modulo the $k{=}2$ lower bound.
> $d{=}2$ confirmed by PDAP (ratio $0.34$ vs $0.35$). See `D1_geometric/claims/separation_general.md`.

---

## Directions = the 4 mathematical machineries
A direction is a **named mathematical tool**, not a metric/aspect. Each folder has a
`_direction.md` (machinery + status + its claims) and `claims/`, `refs/` (detail).

| dir | machinery | role now | folder |
|---|---|---|---|
| **D1** | geometric: polytopal approx of convex bodies (Gruber) × ReLU linear regions (Montúfar/Yarotsky) | **CURRENT** — proves the separation (solver-independent) | `D1_geometric/` |
| **D2** | measure-theoretic optimization: Tikhonov over measures, dual certificate, source condition (Bredies, Parhi–Nowak, ADCG) | current-but-stalled — **achievability** half; delivered $d=1$ only | `D2_measure_optimization/` |
| **D3** | harmonic analysis: Radon/ridgelet, $\mathcal R$-norm (Ongie; Helgason) | supplies norm facts (T1, T2) used by D1 | `D3_harmonic_analysis/` |
| **D4** | max-plus / tropical algebra (McEneaney, Gaubert) | parked — for curved switching where ridges fail | `D4_max_plus/` |

**The separation has two halves needing different machineries:** a lower bound on
the signed model is solver-independent ⟹ **D1** (optimality conditions can't
strengthen a lower bound); achievability of the cone's count by penalized PDAP needs
optimality ⟹ **D2**. So D1 is the active engine; D2/D3 supply complementary pieces.

## Where the goal stands → **all claims in `CLAIMS.md`** (proved/refuted/open registry)
Proved legs (D1, `D1_geometric/claims/`): `head_reduction`, `signed_lower_bound` (general
curved $V$), `correction_cost` (flat & curved), `activation_quadratic_cost`, `separation_flat`
(= current goal). Achievability: `D2_measure_optimization/claims/certificate_dichotomy.md`
($d=1$). Open frontier: `separation_general` (ultimate), `achievability` ($d\ge2$),
`minplus_curved` (D4).

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

## References by direction
**D1 geometric** (polytopal approx × ReLU linear regions): Gruber, *Asymptotic estimates for
best/stepwise approximation of convex bodies*; McClure–Vitale; Montúfar–Pascanu–Cho–Bengio
2014 (linear regions); Yarotsky 2017 (ReLU lower bounds); DeVore–Hanin–Petrova 2021 (survey);
Yang–Zhou 2024 `optimal_rate_Relu^k` ($n$-width / pseudo-dim lower bounds, **rates**).
**D2 measure-theoretic optimization**: Bredies–Pikkarainen `Inverse problem in the space of
measures` (source condition, $O(\delta)$ rate); Parhi–Nowak `Banach space representer theorem`;
Boyd–Schiebinger–Recht `…Sparse Inverse Problems` (ADCG = PDAP ancestor); Bach
`convex-neural-networks` / `breaking the curse of dimensionality` (**$F_1$ rates**);
Candès–Fernández-Granda (super-resolution certificate); de Castro–Gamboa; Duval–Peyré.
**D3 harmonic analysis** (Radon/ridgelet/$\mathcal R$-norm, **the Barron rates**):
**Li–Lu–Mathé–Pereverzev 2024 `PDE/Barron.pdf`** (ReLU$^k$ extended Barron $B_1^k$, **gradient/
$H^m$ rates $n^{-1/2}$, $n^{-1/2-1/d}$ for $m<k$** — the directly relevant gradient-error rate);
Barron 1993 `Barron_93` ($n^{-1/2}$, value, Barron norm); Ongie–Willett–Soudry–Srebro
`Relu(33)`=`1910.01635` ($\mathcal R$-norm, $W^{d+1,1}$); Petrosyan–Dereventsov–Webster `Relu(15)`
(integral representation); Parhi–Nowak (Radon-domain TV); Helgason `Radon transform`;
E–Wojtowytsch (Barron spaces); Candès (ridgelets).
**D4 max-plus / tropical**: McEneaney `curse-of-dimensionality-free`, `curse-free-max-plus`;
Gaubert–McEneaney–Qu 2011; `CDC-max-plus-complexity bounds`; Zhang–Naitzat–Lim (tropical
geometry of NNs).
**Target / cross-cutting**: Kunisch–Vásquez-Varas 2026 `semiconcave_approximaton` (structure-
preserving approx, the repo's model); Cannarsa–Sinestrari `semiconcave-functions` (theory);
Li–Yong (optimal control).

## Don't get lost
Goals here → defs in CONTEXT.md → claims in CLAIMS.md → per-direction `_direction.md`. **Source
of truth = these files.** Do not change agreed structure/definitions without asking
(see memory: dont-change-agreed-things).
