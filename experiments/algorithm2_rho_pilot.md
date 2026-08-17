# Algorithm 2 warm-start scale pilot

Run on 2026-08-17 with the final Algorithm 2 implementation: selected global
prox, actual one-atom increment, distinct-new-node filter, sequential insertion,
H1 fitting, seed 42, powers 2 and 3, and endpoint values `alpha=1e-3` and
`alpha=1e-6`. The 24 pilot cells are not manuscript results.

The tested rule was

`prox_scale = min(alpha, rho * min_nonzero_abs_c^(2-q) / (2*(1-q)))`.

| rho | problem | power | alpha | train objective | val H1 | neurons | failed SSN steps |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | VDP | 2 | 1e-3 | 0.030655 | 0.117046 | 13 | 0 |
| 0.1 | VDP | 2 | 1e-6 | 0.012930 | 0.098564 | 83 | 0 |
| 0.1 | VDP | 3 | 1e-3 | 0.029546 | 0.127196 | 11 | 19 |
| 0.1 | VDP | 3 | 1e-6 | 0.012892 | 0.096854 | 40 | 0 |
| 0.1 | pendulum | 2 | 1e-3 | 1.304969 | 0.505316 | 27 | 0 |
| 0.1 | pendulum | 2 | 1e-6 | 0.575288 | 0.378961 | 133 | 0 |
| 0.1 | pendulum | 3 | 1e-3 | 1.977910 | 0.581635 | 20 | 0 |
| 0.1 | pendulum | 3 | 1e-6 | 1.025725 | 0.469241 | 97 | 0 |
| 0.5 | VDP | 2 | 1e-3 | 0.030655 | 0.117046 | 13 | 0 |
| 0.5 | VDP | 2 | 1e-6 | 0.012970 | 0.098533 | 78 | 0 |
| 0.5 | VDP | 3 | 1e-3 | 0.029099 | 0.126708 | 11 | 38 |
| 0.5 | VDP | 3 | 1e-6 | 0.012892 | 0.096854 | 40 | 0 |
| 0.5 | pendulum | 2 | 1e-3 | 1.304969 | 0.505316 | 27 | 0 |
| 0.5 | pendulum | 2 | 1e-6 | 0.579753 | 0.375168 | 136 | 0 |
| 0.5 | pendulum | 3 | 1e-3 | 1.977910 | 0.581635 | 20 | 0 |
| 0.5 | pendulum | 3 | 1e-6 | 0.935548 | 0.455761 | 103 | 0 |
| 0.9 | VDP | 2 | 1e-3 | 0.030655 | 0.117046 | 13 | 0 |
| 0.9 | VDP | 2 | 1e-6 | 0.012970 | 0.098533 | 78 | 0 |
| 0.9 | VDP | 3 | 1e-3 | 0.029210 | 0.127098 | 11 | 36 |
| 0.9 | VDP | 3 | 1e-6 | 0.012892 | 0.096854 | 40 | 0 |
| 0.9 | pendulum | 2 | 1e-3 | 1.304969 | 0.505316 | 27 | 0 |
| 0.9 | pendulum | 2 | 1e-6 | 0.579753 | 0.375168 | 136 | 0 |
| 0.9 | pendulum | 3 | 1e-3 | 1.977910 | 0.581635 | 20 | 0 |
| 0.9 | pendulum | 3 | 1e-6 | 0.935548 | 0.455761 | 103 | 0 |

Relative to the best training objective in each of the eight
problem/power/alpha cells:

| rho | cells tied for best | mean relative regret | worst relative regret | failed SSN steps |
|---:|---:|---:|---:|---:|
| 0.1 | 6/8 | 1.40% | 9.64% | 19 |
| 0.5 | 6/8 | 0.14% | 0.78% | 38 |
| 0.9 | 5/8 | 0.18% | 0.78% | 36 |

No pilot correction was rejected by the outer objective guard. `rho=0.5` is
selected: it has the lowest mean regret, shares the lowest worst-case regret,
and improves the difficult pendulum power-3 endpoint materially. `rho=0.9`
does not improve that endpoint and is slightly worse on the VDP power-3 cell.
