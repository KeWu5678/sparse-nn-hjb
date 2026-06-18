# CLAIMS — registry (categorization + format)

Every claim is a formal mathematical statement in a **uniform format**. A *proved*
claim is a usable **leg** for other claims. Definitions: `CONTEXT.md`. Goals: `OVERVIEW.md`.

## Claim format (each `claims/<name>.md`)
```
# <name>
status:  proved | refuted | open
statement:  <assumptions> ⟹ <result>   (precise; like a theorem)
proof / refutation:  <argument, or pointer to refs/ detail>
attempts:  1. … 2. …   (what was tried; for open/refuted especially)
uses:  <proved claims this leg depends on>
```

## Categorization

### PROVED (usable legs)
| claim | one-line | dir |
|---|---|---|
| `head_reduction` | $N(\text{semiconcave},V,\varepsilon)=N^+(g,\varepsilon)$, indep. of $C$ | D1 |
| `signed_lower_bound` | $\sigma=$ReLU, $D^2V\succeq cI$ on $B$ ⟹ $N(\text{signed})=\Omega(1/\varepsilon)$ | D1 |
| `correction_cost` (flat part) | $g$ piecewise-linear ($K$ pieces) ⟹ $N^+(g)=K$ | D1 |
| `correction_cost` (curved part) | curved switching ⟹ $N^+(g)=\infty$ ($d=2$) | D1 |
| `activation_quadratic_cost` | $\sigma''=0$ a.e. ⟺ quadratic $\Theta(1/\varepsilon)$-costly (else $O(d)$) | D1 |
| `separation_flat` (current goal, proved instance) | curved bulk + flat/polyhedral $g$ ⟹ ratio $\Theta(1/(K\varepsilon))\to\infty$ | D1 |
| `separation_1d_convex` ($d=1$, general convex $g$) | ratio $=\int|g''|/\int|V''|$; more sparse $\iff \int|g''|<\int|V''|$ | D1 |
| `budget_1d` (T1) | $\mathrm{bud}(g)\le\mathrm{Lip}(g)|\partial\Omega|$, indep. of #shocks | D3 |
| `curved_switching_rnorm` (T2) | curved switching ⟹ ridge $\mathcal R$-norm $=\infty$ ($d=2$) | D3 |
| `cone_free_convex` (T3) | $g$ convex ⟹ cone free in **curvature-mass** sense (signed cancel. raises $\int\Delta$), all $d$ | D1 |
| `certificate_dichotomy` (d=1 only) | cone one-sided ⟹ contact-region ratio $\to 2$ | D2 |

### REFUTED (do not reuse; recorded so we don't recircle)
| claim | one-line | dir |
|---|---|---|
| `capacity_selection_split` | additive head+cone decomposition (subtraction-confounded) | D1 |
| `single_property_activation` | one qualitative $\sigma$-property ⟺ advantage | D1 |
| `source_condition_separation_free` | nonneg ⟹ separation-free recovery in general $d$ | D2 |

### OPEN (frontier)
| claim | one-line | dir |
|---|---|---|
| `separation_general` (ULTIMATE GOAL) | general semiconcave $V$ ⟹ $N(\text{semiconcave})\le N(\text{signed})$ (more sparse); $d\ge2$ gap now = the $n^{-1}$ rate only | D1 |
| `achievability` | penalized PDAP attains the cone count, $d\ge2$ | D2 |
| `certificate_dichotomy` ($d\ge2$) | lift the $d=1$ dichotomy to $d\ge2$ | D2 |
| `minplus_curved` | min-architecture attains budget→rate on curved switching | D4 |

## Reading note (the ReLU caveat)
`signed_lower_bound` is proved for $\sigma=$ReLU and is a rigorous **machinery / leg**,
but pure-ReLU gradient training is not what the repo runs (repo uses $p=1$/smooth /
ReLU²). So ReLU claims are *building blocks*; the **ultimate goal** must carry them to the
practical activation — that carry is `activation_quadratic_cost` (which says the
best-$n$-term divergence is ReLU-specific; smooth/ReLU² need the penalized/variation-cost
regime, D2). Keep this distinction when composing legs.
