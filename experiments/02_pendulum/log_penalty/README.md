# Log-penalty activation search — pendulum swing-up

This Hydra study compares activation functions on the two-sided pendulum data
with signed profile insertion. For every nonhomogeneous activation, PDAP uses
the current normalized-measure Algorithm 1 objective

`l^M + alpha * sum phi_gamma(w_p(omega_n) * |c_n|)`.

The retained `leaky_relu` cell is positively homogeneous and sphere-gauged, so
it is an unnormalized profile comparator rather than an Algorithm 1 cell.

## Sweep axes

| axis | values |
|---|---|
| `model.activation` | leaky_relu, softplus, tanh, gaussian, gausscent_1, matern52, gelu_squared |
| `model.alpha` | 1e-2, 1e-3, 1e-4, 1e-5 |
| `model.gamma` | 0, 0.1, 1, 10 |
| `model.loss_weights` | [1,0] (value only), [1,1] (value and gradient) |

Fixed settings include `model.power=1`, normalized data, the pendulum region
evaluation, and seed 42. This sweep's config (`conf/experiment/pendulum/log_penalty.yaml`) was retired on 2026-09-10;
only the `paper_*` experiments are runnable now. The archived records under
`rawdata/logs/multirun/pendulum/log_penalty/` remain readable by `analysis.py`.

The former report and figures used the retired unweighted objective and are not
current evidence; they remain available from Git history.
