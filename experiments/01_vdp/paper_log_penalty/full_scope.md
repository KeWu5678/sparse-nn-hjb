# Van der Pol full evidence scope

All Algorithm 1 rows use shared parameters (alpha=1e-04, gamma=10, p=2.01), so the cross-activation comparison is not confounded with per-activation tuning.
Algorithm 2 rows use the sphere formulation.

## Algorithm 1: gradient augmentation

| activation | loss | gamma | rel L2 | rel H1 | N | R95 |
|---|---|---:|---:|---:|---:|---:|
| tanh | l2 | 10 | 0.0958 | 0.5973 | 10 | 3.19 |
| tanh | h1 | 10 | 0.4174 | 0.1008 | 38 | 3.49 |
| softplus | l2 | 10 | 0.1041 | 0.5819 | 6 | 2.19 |
| softplus | h1 | 10 | 0.4241 | 0.1028 | 16 | 11.2 |
| gaussian | l2 | 10 | 0.0716 | 0.5373 | 10 | 2.34 |
| gaussian | h1 | 10 | 0.4168 | 0.0983 | 34 | 3.48 |

![representative activation shapes](figures/shape_softplus_tanh_gaussian.png)

| softplus | tanh | gaussian |
|---|---|---|
| ![softplus](figures/value_surface_softplus.png) | ![tanh](figures/value_surface_tanh.png) | ![gaussian](figures/value_surface_gaussian.png) |

| tanh | softplus | gaussian |
|---|---|---|
| ![tanh](figures/derivative_distribution_tanh.png) | ![softplus](figures/derivative_distribution_softplus.png) | ![gaussian](figures/derivative_distribution_gaussian.png) |

### Sequential insertion and pruning

Each outer iteration retains at most one candidate satisfying `|P_p(ω)| > α L_φ`. The guarded coefficient correction and pruning then determine the recorded support, so its size may increase by one, stay unchanged, or decrease; a negative change is pruning, not a negative insertion. At the shared operating point, Gaussian uses 34 atoms and softplus 16 atoms at relative H1 0.0983 and 0.1028, respectively.

## Algorithm 1: synthesized feedback

![Algorithm 1 feedback](figures/control_synthesis.png)

| controller | N | stabilizes | closed-loop cost |
|---|---:|:---:|---:|
| true | — | yes | 6.48 |
| softplus | 16 | yes | 6.48 |
| tanh | 38 | yes | 6.50 |
| gaussian | 34 | yes | 6.50 |

## Algorithm 1 versus Algorithm 2

![error-support frontier](figures/frontier.png)

| state norm | control magnitude |
|---|---|
| ![state](figures/feedback_state.png) | ![control](figures/feedback_control.png) |

| controller | N | stabilizes | closed-loop cost |
|---|---:|:---:|---:|
| true | — | yes | 6.48 |
| softplus | 16 | yes | 6.48 |
| gaussian | 34 | yes | 6.50 |
| relu5 | 29 | yes | 6.50 |

| gaussian | softplus | ReLU5 |
|---|---|---|
| ![gaussian](figures/weights_raw3d_gaussian.png) | ![softplus](figures/weights_raw3d_softplus.png) | ![relu5](figures/weights_raw3d_relu5.png) |
