# Van der Pol moment-penalty results

**Status: complete first-pass screen.** The report contains 16 deduplicated beta=0 baselines and 256 positive-beta runs, all at seed 42.

## Headline

The lowest validation H1 error among positive-beta cells is 0.0989 for `gaussian` at alpha=1e-05, beta=1e-10, p=3. This is an accuracy optimum, not by itself the selected operating point; support size and R95 are independent Pareto objectives.

## Screen-level observations

- At Gaussian, p=3, alpha=1e-5, moving from beta=0 to beta=1e-5 changes validation H1 from 0.0994 to 0.1053, while support falls from 113 to 82 and R95 from 2.49 to 2.01.
- The next screened value beta=1e-2 is already on the strong-penalty side at that point: H1=0.3556, N=9, R95=0.82. The transition between 1e-5 and 1e-2 therefore needs intermediate decades before the full follow-up.
- 16 cells return the exact zero measure. They are all tanh at beta=1e-1, across every screened alpha and p; this is the collapse boundary of the regularization surface, not a failed run.
- 24 positive-beta cells place at least 95% of their amplitude mass at the radial search ceiling. They remain in the raw grid but are censored from the model-selection tables.

## Best interior positive-beta validation H1 by activation

| activation | p | alpha | beta | val H1 | N | R95 | Phi1 | Psi_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tanh | 2.01 | 1e-5 | 1e-10 | 0.1665 | 76 | 5.8 | 356 | 2.99e+04 |
| softplus | 2.01 | 1e-5 | 1e-10 | 0.1247 | 35 | 1 | 561 | 1.02e+05 |
| gaussian | 3 | 1e-5 | 1e-10 | 0.0989 | 112 | 4.82 | 97.5 | 4.33e+03 |
| gelu_squared | 3 | 1e-3 | 1e-10 | 0.1184 | 34 | 1.26 | 5.12 | 2.41e+03 |

## Interior positive-beta Pareto frontier: validation H1, support, and R95

| activation | p | alpha | beta | val H1 | N | R95 | Phi1 | Psi_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 3 | 1e-5 | 1e-10 | 0.0989 | 112 | 4.82 | 97.5 | 4.33e+03 |
| gaussian | 4 | 1e-5 | 1e-10 | 0.0989 | 112 | 2.01 | 62.7 | 4.09e+03 |
| gaussian | 2.01 | 1e-4 | 1e-10 | 0.1052 | 95 | 3.02 | 26.3 | 233 |
| gaussian | 3 | 1e-4 | 1e-10 | 0.1053 | 89 | 2.42 | 23.8 | 5.59e+03 |
| gaussian | 3 | 1e-5 | 1e-5 | 0.1053 | 82 | 2.01 | 29.4 | 248 |
| gaussian | 2.5 | 1e-5 | 1e-5 | 0.1079 | 78 | 2.01 | 25.4 | 177 |
| gaussian | 2.01 | 1e-4 | 1e-5 | 0.1083 | 61 | 2.76 | 18.3 | 122 |
| gaussian | 4 | 1e-5 | 1e-5 | 0.1091 | 63 | 2.01 | 20.6 | 318 |
| gaussian | 3 | 1e-4 | 1e-5 | 0.1110 | 62 | 2.01 | 20.1 | 184 |
| gelu_squared | 3 | 1e-3 | 1e-10 | 0.1184 | 34 | 1.26 | 5.12 | 2.41e+03 |
| gelu_squared | 2.5 | 1e-3 | 1e-10 | 0.1191 | 30 | 1.26 | 5.12 | 202 |
| gelu_squared | 4 | 1e-5 | 1e-10 | 0.1197 | 45 | 1.26 | 22.1 | 5.07e+05 |
| gelu_squared | 2.01 | 1e-3 | 1e-5 | 0.1209 | 27 | 1.18 | 9.06 | 40.7 |
| softplus | 2.01 | 1e-5 | 1e-10 | 0.1247 | 35 | 1 | 561 | 1.02e+05 |
| gelu_squared | 4 | 1e-3 | 1e-10 | 0.1336 | 26 | 0.753 | 4.34 | 2.57e+05 |
| gelu_squared | 2.5 | 1e-3 | 1e-5 | 0.1490 | 31 | 0.752 | 3.18 | 132 |
| gelu_squared | 3 | 1e-3 | 1e-5 | 0.1571 | 33 | 0.752 | 0.033 | 1.13e+03 |
| gelu_squared | 2.01 | 1e-3 | 1e-2 | 0.1926 | 10 | 2.71 | 0.309 | 4.06 |
| gaussian | 4 | 1e-2 | 1e-10 | 0.2007 | 21 | 2.01 | 5.69 | 160 |
| gaussian | 2.01 | 1e-2 | 1e-10 | 0.2021 | 14 | 2.01 | 5.7 | 34.5 |
| gelu_squared | 2.01 | 1e-4 | 1e-2 | 0.2034 | 9 | 2.8 | 0.412 | 4.04 |
| gaussian | 4 | 1e-2 | 1e-5 | 0.2096 | 18 | 2.01 | 5.26 | 115 |
| gaussian | 3 | 1e-2 | 1e-5 | 0.2125 | 15 | 2.01 | 5.32 | 57.5 |
| gelu_squared | 4 | 1e-3 | 1e-5 | 0.2194 | 22 | 1.78 | 46.7 | 2.87e+03 |
| gelu_squared | 4 | 1e-5 | 1e-2 | 0.2486 | 8 | 1.57 | 1.13 | 6.49 |
| gelu_squared | 4 | 1e-2 | 1e-2 | 0.2882 | 6 | 1.87 | 0.811 | 6.23 |
| softplus | 4 | 1e-5 | 1e-10 | 0.3329 | 15 | 1 | 9.95 | 1.96e+07 |
| gaussian | 4 | 1e-4 | 1e-2 | 0.3530 | 14 | 0.804 | 5.8 | 8.84 |
| gaussian | 3 | 1e-5 | 1e-2 | 0.3556 | 9 | 0.82 | 5.47 | 9.03 |
| gelu_squared | 2.5 | 1e-5 | 1e-2 | 0.3589 | 4 | 2.76 | 0.0483 | 12.2 |
| softplus | 2.01 | 1e-3 | 1e-2 | 0.3726 | 5 | 2.51 | 2.75 | 15 |
| softplus | 2.01 | 1e-5 | 1e-2 | 0.3802 | 5 | 2.5 | 2.41 | 14.7 |
| softplus | 2.5 | 1e-5 | 1e-2 | 0.3828 | 6 | 1.74 | 4.14 | 17.7 |
| softplus | 2.5 | 1e-4 | 1e-2 | 0.3831 | 5 | 1.74 | 4.08 | 17.7 |
| softplus | 2.5 | 1e-3 | 1e-2 | 0.3870 | 4 | 1.76 | 3.61 | 17.6 |
| softplus | 3 | 1e-4 | 1e-2 | 0.3889 | 5 | 1.43 | 5.02 | 19.9 |
| softplus | 3 | 1e-5 | 1e-2 | 0.3906 | 4 | 1.43 | 4.97 | 19.8 |
| softplus | 3 | 1e-3 | 1e-2 | 0.3932 | 3 | 1.42 | 4.78 | 19.5 |
| softplus | 4 | 1e-5 | 1e-2 | 0.3941 | 5 | 1.12 | 7.3 | 21.1 |
| gaussian | 2.01 | 1e-2 | 1e-2 | 0.3987 | 8 | 0.971 | 4.15 | 7.75 |
| gaussian | 3 | 1e-2 | 1e-2 | 0.4172 | 7 | 0.963 | 3.67 | 6.77 |
| tanh | 4 | 1e-3 | 1e-2 | 0.4239 | 5 | 1.05 | 6.92 | 17.3 |
| softplus | 2.5 | 1e-3 | 1e-5 | 0.4404 | 11 | 0.0498 | 1.1 | 5.91e+03 |
| gelu_squared | 3 | 1e-5 | 1e-1 | 0.4686 | 4 | 1.28 | 0.662 | 2.24 |
| gelu_squared | 4 | 1e-5 | 1e-1 | 0.4719 | 4 | 1.06 | 0.938 | 2.42 |
| gelu_squared | 4 | 1e-4 | 1e-1 | 0.4720 | 3 | 1.06 | 0.929 | 2.42 |
| gelu_squared | 3 | 1e-2 | 1e-1 | 0.4757 | 2 | 1.33 | 0.578 | 2.2 |
| gelu_squared | 4 | 1e-2 | 1e-1 | 0.4828 | 2 | 1.14 | 0.807 | 2.38 |
| gaussian | 4 | 1e-5 | 1e-1 | 0.6856 | 3 | 0.878 | 1.59 | 2.58 |
| gaussian | 3 | 1e-5 | 1e-1 | 0.7005 | 2 | 0.916 | 1.4 | 2.52 |
| softplus | 4 | 1e-5 | 1e-1 | 0.9846 | 2 | 0.756 | 0.137 | 0.187 |
| tanh | 2.01 | 1e-2 | 1e-1 | 1.0000 | 0 | 0 | 0 | 0 |

## Alpha-beta grids and fixed slices

Each heatmap uses alpha as rows and beta as columns. The companion plots read those same arrays in both directions: the left panel fixes one alpha per curve, while the right panel fixes one beta per curve. Beta/alpha is not used as an experimental coordinate.

### tanh, p=2.01

![alpha-beta grid](figures/grid_tanh_p2p01.png)

![fixed row and column slices](figures/h1_slices_tanh_p2p01.png)

### tanh, p=2.5

![alpha-beta grid](figures/grid_tanh_p2p5.png)

![fixed row and column slices](figures/h1_slices_tanh_p2p5.png)

### tanh, p=3

![alpha-beta grid](figures/grid_tanh_p3p0.png)

![fixed row and column slices](figures/h1_slices_tanh_p3p0.png)

### tanh, p=4

![alpha-beta grid](figures/grid_tanh_p4p0.png)

![fixed row and column slices](figures/h1_slices_tanh_p4p0.png)

### softplus, p=2.01

![alpha-beta grid](figures/grid_softplus_p2p01.png)

![fixed row and column slices](figures/h1_slices_softplus_p2p01.png)

### softplus, p=2.5

![alpha-beta grid](figures/grid_softplus_p2p5.png)

![fixed row and column slices](figures/h1_slices_softplus_p2p5.png)

### softplus, p=3

![alpha-beta grid](figures/grid_softplus_p3p0.png)

![fixed row and column slices](figures/h1_slices_softplus_p3p0.png)

### softplus, p=4

![alpha-beta grid](figures/grid_softplus_p4p0.png)

![fixed row and column slices](figures/h1_slices_softplus_p4p0.png)

### gaussian, p=2.01

![alpha-beta grid](figures/grid_gaussian_p2p01.png)

![fixed row and column slices](figures/h1_slices_gaussian_p2p01.png)

### gaussian, p=2.5

![alpha-beta grid](figures/grid_gaussian_p2p5.png)

![fixed row and column slices](figures/h1_slices_gaussian_p2p5.png)

### gaussian, p=3

![alpha-beta grid](figures/grid_gaussian_p3p0.png)

![fixed row and column slices](figures/h1_slices_gaussian_p3p0.png)

### gaussian, p=4

![alpha-beta grid](figures/grid_gaussian_p4p0.png)

![fixed row and column slices](figures/h1_slices_gaussian_p4p0.png)

### gelu_squared, p=2.01

![alpha-beta grid](figures/grid_gelu_squared_p2p01.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p2p01.png)

### gelu_squared, p=2.5

![alpha-beta grid](figures/grid_gelu_squared_p2p5.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p2p5.png)

### gelu_squared, p=3

![alpha-beta grid](figures/grid_gelu_squared_p3p0.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p3p0.png)

### gelu_squared, p=4

![alpha-beta grid](figures/grid_gelu_squared_p4p0.png)

![fixed row and column slices](figures/h1_slices_gelu_squared_p4p0.png)
