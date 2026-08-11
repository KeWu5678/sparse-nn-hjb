# Pendulum moment-penalty results

**Status: complete first-pass screen.** The report contains 16 deduplicated beta=0 baselines and 256 positive-beta runs, all at seed 42.

## Headline

The raw lowest positive-beta switching-tube H1 error is 0.5124 for `softplus` at alpha=1e-04, beta=1e-10, p=3, but its R95=148 hits the exp(5) radial search ceiling. It is therefore evidence that this tiny beta has not resolved parameter escape, not the selected operating point. The lowest error away from that ceiling is 0.5217 for `softplus` at alpha=1e-04, beta=1e-10, p=2.01, with N=81 and R95=3.83.

## Screen-level observations

- At softplus, p=2.01, alpha=1e-4, beta=1e-10 improves switching-tube H1 from 0.5635 to 0.5217; N changes from 72 to 81 and R95 from 1.00 to 3.83. At beta=1e-5 the error is already 0.6944 with N=20. The five-decade gap therefore needs beta=1e-9 through 1e-6.
- Tanh shows a second, later transition at p=3 and alpha=1e-4: beta=1e-5 changes switching-tube H1/N/R95 from 0.5904/99/6.89 to 0.5404/43/2.15, whereas beta=1e-2 gives 0.8941/13/1.21. Beta=1e-4 and 1e-3 resolve that gap.
- 23 cells place at least 95% of their amplitude mass at the radial search ceiling; they are censored for model selection. 32 cells return the exact zero measure (all tanh or softplus at beta=1e-1), marking the collapse boundary rather than a failed run.

## Best interior positive-beta switching-tube H1 by activation

| activation | p | alpha | beta | val H1 | switching H1 | rest H1 | N | R95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tanh | 3 | 1e-04 | 1e-05 | 0.5611 | 0.5404 | 0.4795 | 43 | 2.15 |
| softplus | 2.01 | 1e-04 | 1e-10 | 0.5272 | 0.5217 | 0.4405 | 81 | 3.83 |
| gaussian | 2.5 | 1e-04 | 1e-05 | 0.5803 | 0.5911 | 0.5534 | 83 | 2.66 |
| gelu_squared | 2.01 | 1e-05 | 1e-10 | 0.5923 | 0.5993 | 0.5675 | 99 | 1.36 |

## Interior positive-beta Pareto frontier: switching-tube H1, support, and R95

| activation | p | alpha | beta | val H1 | switching H1 | rest H1 | N | R95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| softplus | 2.01 | 1e-04 | 1e-10 | 0.5272 | 0.5217 | 0.4405 | 81 | 3.83 |
| softplus | 2.5 | 1e-05 | 1e-10 | 0.5152 | 0.5244 | 0.4194 | 88 | 1 |
| tanh | 3 | 1e-04 | 1e-05 | 0.5611 | 0.5404 | 0.4795 | 43 | 2.15 |
| softplus | 3 | 1e-05 | 1e-10 | 0.5450 | 0.5438 | 0.4767 | 50 | 1 |
| softplus | 3 | 1e-03 | 1e-10 | 0.6432 | 0.6141 | 0.6324 | 28 | 4.54 |
| gaussian | 4 | 1e-02 | 1e-10 | 0.6372 | 0.6462 | 0.6655 | 20 | 4.82 |
| softplus | 2.5 | 1e-03 | 1e-10 | 0.6780 | 0.6510 | 0.6964 | 24 | 1 |
| gelu_squared | 2.01 | 1e-02 | 1e-05 | 0.6579 | 0.6755 | 0.6900 | 22 | 2.9 |
| gaussian | 3 | 1e-02 | 1e-05 | 0.6888 | 0.6957 | 0.7256 | 16 | 5.18 |
| softplus | 4 | 1e-05 | 1e-10 | 0.7641 | 0.7082 | 0.8236 | 14 | 1 |
| tanh | 4 | 1e-04 | 1e-05 | 0.7351 | 0.7498 | 0.7948 | 12 | 6.15 |
| gelu_squared | 2.5 | 1e-03 | 1e-02 | 0.7862 | 0.8178 | 0.8507 | 4 | 8.74 |
| gelu_squared | 2.01 | 1e-03 | 1e-02 | 0.7840 | 0.8186 | 0.8450 | 11 | 2.53 |
| gelu_squared | 2.01 | 1e-05 | 1e-02 | 0.7935 | 0.8231 | 0.8708 | 7 | 2.53 |
| gelu_squared | 3 | 1e-03 | 1e-02 | 0.7837 | 0.8302 | 0.8334 | 5 | 3.03 |
| gelu_squared | 4 | 1e-05 | 1e-02 | 0.8062 | 0.8425 | 0.8652 | 9 | 1.55 |
| gelu_squared | 4 | 1e-03 | 1e-02 | 0.8093 | 0.8454 | 0.8695 | 7 | 1.5 |
| softplus | 3 | 1e-05 | 1e-05 | 0.8204 | 0.8579 | 0.8821 | 7 | 0.0507 |
| gelu_squared | 2.5 | 1e-02 | 1e-02 | 0.8354 | 0.8693 | 0.8950 | 6 | 2.73 |
| softplus | 2.01 | 1e-03 | 1e-02 | 0.8579 | 0.8897 | 0.9060 | 5 | 2.21 |
| softplus | 2.5 | 1e-05 | 1e-02 | 0.8619 | 0.8927 | 0.9043 | 6 | 1.65 |
| softplus | 2.5 | 1e-04 | 1e-02 | 0.8621 | 0.8928 | 0.9044 | 4 | 1.66 |
| tanh | 2.5 | 1e-03 | 1e-02 | 0.8698 | 0.8958 | 0.9199 | 6 | 1.37 |
| softplus | 3 | 1e-05 | 1e-02 | 0.8661 | 0.8964 | 0.9050 | 5 | 1.34 |
| softplus | 3 | 1e-04 | 1e-02 | 0.8664 | 0.8972 | 0.9055 | 4 | 1.33 |
| softplus | 4 | 1e-05 | 1e-02 | 0.8692 | 0.8987 | 0.9055 | 5 | 1.09 |
| softplus | 4 | 1e-04 | 1e-02 | 0.8693 | 0.8990 | 0.9052 | 4 | 1.09 |
| gelu_squared | 4 | 1e-05 | 1e-01 | 0.9258 | 0.9450 | 0.9448 | 2 | 1.08 |
| gaussian | 4 | 1e-05 | 1e-01 | 0.9596 | 0.9697 | 0.9704 | 2 | 0.876 |
| gaussian | 2.01 | 1e-03 | 1e-01 | 0.9796 | 0.9842 | 0.9889 | 1 | 1.04 |
| gaussian | 2.5 | 1e-02 | 1e-01 | 0.9810 | 0.9853 | 0.9894 | 1 | 0.994 |
| softplus | 2.01 | 1e-04 | 1e-01 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

## Alpha-beta grids and fixed slices

Alpha and beta remain independent axes. The heatmaps show global and switching-tube error together; each companion figure reads both error arrays along fixed-alpha rows and fixed-beta columns.

### tanh, p=2.01

![alpha-beta grid](figures/grid_tanh_p2p01.png)

![fixed row and column slices](figures/h1_slices_tanh_p2p01.png)

### tanh, p=2.5

![alpha-beta grid](figures/grid_tanh_p2p5.png)

![fixed row and column slices](figures/h1_slices_tanh_p2p5.png)

### tanh, p=3

![alpha-beta grid](figures/grid_tanh_p3.png)

![fixed row and column slices](figures/h1_slices_tanh_p3.png)

### tanh, p=4

![alpha-beta grid](figures/grid_tanh_p4.png)

![fixed row and column slices](figures/h1_slices_tanh_p4.png)

### softplus, p=2.01

![alpha-beta grid](figures/grid_softplus_p2p01.png)

![fixed row and column slices](figures/h1_slices_softplus_p2p01.png)

### softplus, p=2.5

![alpha-beta grid](figures/grid_softplus_p2p5.png)

![fixed row and column slices](figures/h1_slices_softplus_p2p5.png)

### softplus, p=3

![alpha-beta grid](figures/grid_softplus_p3.png)

![fixed row and column slices](figures/h1_slices_softplus_p3.png)

### softplus, p=4

![alpha-beta grid](figures/grid_softplus_p4.png)

![fixed row and column slices](figures/h1_slices_softplus_p4.png)

### gaussian, p=2.01

![alpha-beta grid](figures/grid_gaussian_p2p01.png)

![fixed row and column slices](figures/h1_slices_gaussian_p2p01.png)

### gaussian, p=2.5

![alpha-beta grid](figures/grid_gaussian_p2p5.png)

![fixed row and column slices](figures/h1_slices_gaussian_p2p5.png)

### gaussian, p=3

![alpha-beta grid](figures/grid_gaussian_p3.png)

![fixed row and column slices](figures/h1_slices_gaussian_p3.png)

### gaussian, p=4

![alpha-beta grid](figures/grid_gaussian_p4.png)

![fixed row and column slices](figures/h1_slices_gaussian_p4.png)

### gelu_squared, p=2.01

![alpha-beta grid](figures/grid_gelu_squared_p2p01.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p2p01.png)

### gelu_squared, p=2.5

![alpha-beta grid](figures/grid_gelu_squared_p2p5.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p2p5.png)

### gelu_squared, p=3

![alpha-beta grid](figures/grid_gelu_squared_p3.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p3.png)

### gelu_squared, p=4

![alpha-beta grid](figures/grid_gelu_squared_p4.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p4.png)
