# Algorithm 2 — Van der Pol

Fresh sequential runs using the actual one-atom increment and the global-prox warm-start-scaled correction.

| power | loss | alpha | N | rel L2 | rel H1 |
|---:|---|---:|---:|---:|---:|
| 2 | l2 | 1e-05 | 13 | 0.0278 | 0.4087 |
| 3 | l2 | 1e-05 | 12 | 0.0367 | 0.4526 |
| 2 | h1 | 1e-05 | 45 | 0.4172 | 0.0996 |
| 3 | h1 | 1e-05 | 27 | 0.4150 | 0.0987 |

## Fixed-alpha value surfaces

- ReLU squared: `/Users/chaoruiz/Documents/Repos/SparseNNforHJB/experiments/01_vdp/paper_frac_exp_penalty/figures/value_surface_p2.png`
- ReLU cubed: `/Users/chaoruiz/Documents/Repos/SparseNNforHJB/experiments/01_vdp/paper_frac_exp_penalty/figures/value_surface_p3.png`
