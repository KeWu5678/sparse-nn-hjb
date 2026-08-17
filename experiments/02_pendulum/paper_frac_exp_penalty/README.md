# Algorithm 2 — pendulum swing-up

This study applies Algorithm 2 of `paper/paper_0805.tex` to the two-sided
pendulum switching-set data. The inner parameters lie on the unit sphere and the
outer penalty is `alpha * sum |c_n|^q`, with `q = 2/(k+1)`.

Candidate nodes are refined by multistart L-BFGS on the sphere. For each
candidate, the code minimizes the actual one-atom objective increment with the
selected global scalar proximal map; a candidate is inserted only when that
increment is negative. The same coefficient initializes the outer-weight
correction. For `k > 1`, the correction uses the global-prox normal map with a
fixed scale derived from the warm start (`rho = 0.5`) and retains the result only
when it does not increase the post-insertion objective.

## Grid

| axis | values |
|---|---|
| activation | ReLU |
| power `k` | 2, 3 |
| exponent `q` | 2/3, 1/2 |
| `alpha` | 1e-3, 1e-4, 1e-5, 1e-6 |
| loss weights | [1,0], [1,1] |
| insertion mode | batch, sequential |
| seed | 42 |

The ReLU--L1 endpoint (`k=1`, `q=1`) is run separately over six values of
`alpha`. The oversampling study reruns the ReLU-squared model on four training
sets and three values of `alpha`. Every fractional-power record carries the
solver identifier and `rho`; the L1 records identify the soft-threshold solver.

Use `make paper-algorithm2-refresh` to stage, validate, archive, and replace the
complete current Algorithm 2 record set. `make paper-artifacts` rescores the
region metrics and regenerates the figures and reports consumed by the
manuscript.
