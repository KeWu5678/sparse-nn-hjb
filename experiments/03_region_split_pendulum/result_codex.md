# Pendulum Switching-Set Model Comparison

Comparison set: signed H1-trained `gaussian`, `softplus` with the
log penalty, and signed H1-trained `ReLU^2`, `ReLU^5` with the fractional
penalty `|c|^q`, `q = 2/(p+1)`. Each row uses the best run in its cell by far
L1 error. All rows are trained on the **two-sided** dataset (3,900 samples =
3,000 in-basin + 900 envelope-certified pad/collar samples straddling the
switching arms), so the gradient jump is in-sample; numbers are not comparable
to the earlier one-sided (3,000-sample) report.

| model | penalty | neurons | near L1 | far L1 | near/far |
| --- | --- | ---: | ---: | ---: | ---: |
| `ReLU^2` | fractional, `q=2/3` | 114 | 1.01e+0 | 1.20e-1 | 8.39 |
| `ReLU^5` | fractional, `q=1/3` | 79 | 1.78e+0 | 4.38e-1 | 4.07 |
| gaussian | log | 113 | 1.81e+0 | 4.38e-1 | 4.14 |
| softplus | log | 82 | 1.77e+0 | 3.49e-1 | 5.09 |

Main conclusion: `ReLU^2` remains the best fit in both the near-switching band
and the smooth far region — by ~3× on far error. The picture away from ReLU²
changed relative to the one-sided data: with the kink in-sample the near band
is comparably hard for all other models (near L1 1.77–1.81), and every model's
absolute error grew, because the band dominates the unweighted H1 objective
(23% of the samples, ~75% of the squared value mass). The near/far ratio is now
largest for `ReLU^2` precisely because its far error is the smallest.

## Geometry

| Value samples | Value surface | Switching set |
| --- | --- | --- |
| ![value samples](../00_openloop/pendulum/figures/value_scatter.png) | ![value surface](../00_openloop/pendulum/figures/value_surface.png) | ![switching set](../00_openloop/pendulum/figures/regions_of_attraction.png) |

The pendulum value is continuous, but the gradient changes branch across the
switching spirals. This is why global H1 alone is not enough: the fit must be
checked by distance to the switching set and by the induced feedback law.

## Learned Value Surfaces

All learned surfaces use the same physical window as the true value surface,
`theta, theta_dot in [-8, 8]`, and the same displayed value range, `V in [0, 60]`.

| gaussian | softplus |
| --- | --- |
| ![gaussian learned value surface](figures/surface_gaussian.png) | ![softplus learned value surface](figures/surface_softplus.png) |

| `ReLU^2` | `ReLU^5` |
| --- | --- |
| ![ReLU2 learned value surface](figures/surface_relu2.png) | ![ReLU5 learned value surface](figures/surface_relu5.png) |

With the band in the training data the models now shape the full multi-well
landscape rather than a single basin. `ReLU^2` raises the sharpest diagonal
walls along the switching arms between the `2πk` wells. Gaussian reproduces the
wells but rounds the ridge off. Softplus smears the structure, matching its
poor feedback results below.

## Insertion Frontier

![insertion frontier](figures/frontier.png)

The frontier shows the running best relative `H1` validation error reached as
neurons are inserted for the selected run in each model family. `ReLU^2`
separates from the field almost immediately and reaches `~3.6e-1` at 113
neurons; softplus plateaus near `5.4e-1` from ~50 neurons; `ReLU^5` and
gaussian sit near `5.5e-1`–`5.9e-1`. Errors are uniformly higher than on the
one-sided data — the two-sided target is genuinely harder — but the per-atom
accuracy advantage of the low-power ReLU atom is unchanged.

## Near/Far Error

![near/far error](figures/near_far_dumbbell.png)

Relative `H1` error inside a fixed geometric tube around the switching set
(`d <= 0.3`, filled) versus the rest (open). `ReLU^2` is best in both regions
(rest `~0.20`, tube `~0.31`). `ReLU^5` is the only model *better* inside the
tube than outside: its stiff atoms seat the band and pay everywhere else.
Gaussian and softplus sit in between, 2–3× above `ReLU^2`.

## Error By Distance

| Value error | Gradient error |
| --- | --- |
| ![value error by distance](figures/error_vs_distance_value.png) | ![gradient error by distance](figures/error_vs_distance_gradient.png) |

Per-bin relative error against distance to the switching set (grey bars =
samples per bin). The profile inverted relative to the one-sided data: the
switching set itself (`d < 0.3`) is no longer the peak — the pad/collar band
anchors the fit there and `|V|` is large — while the peak moved to `d ≈ 0.65`,
the dense sample band around the upright equilibrium where `|V| → 0` inflates
the relative error and where two-sided training costs interior accuracy.
`ReLU^2` keeps the lowest profile at every distance; `ReLU^5` pays the largest
interior penalty.

## Switching-Set Transect

| True value branches | True normal-gradient branches |
| --- | --- |
| ![true value branches](figures/transect_true_branches_value.png) | ![true normal-gradient branches](figures/transect_true_branches_gradient.png) |

The blue and teal curves are the two candidate PMP branches before taking the
minimum: one goes to the upright at `0`, the other to the upright at `2π`. The
true value is the dashed lower envelope. In the value plot the lower envelope is
continuous; in the gradient plot the selected branch changes at the switching
set, so `n · grad V` jumps.

| Value along normal direction | Normal gradient along normal direction |
| --- | --- |
| ![value transect](figures/transect_value.png) | ![normal gradient transect](figures/transect_normal_gradient.png) |

Unlike the one-sided data, the models now saw samples on **both** sides of
`s = 0`, and the jump being in-sample is visible: `ReLU^2` is the only model
that develops a kink at `s ≈ 0` and tracks the true value level best on both
sides. The smooth activations interpolate a gentle slope through the
discontinuity, as their atoms must. No model reproduces the jump's full
magnitude (true `n·grad V ≈ −100` just before the curve vs fitted `−20` to
`−58`): a representation limit at a *seen* discontinuity, not a data-coverage
artifact.

## Feedback Reliability

One phase panel per feedback law; start A (blue) and start B (red) straddle the
switching curve, which here separates two optimal behaviours: from A the true
law swings over the top to the `2π` upright, from B it brakes directly to the
`θ = 0` upright. Both starts are in-sample now.

| true PMP | gaussian | softplus |
| --- | --- | --- |
| ![true PMP](figures/feedback_true_pmp.png) | ![gaussian](figures/feedback_gaussian.png) | ![softplus](figures/feedback_softplus.png) |

| `ReLU^2` | `ReLU^5` |
| --- | --- |
| ![ReLU2](figures/feedback_relu2.png) | ![ReLU5](figures/feedback_relu5.png) |

| Control from start B |
| --- |
| ![feedback control B](figures/feedback_control_b.png) |

| controller | cost A | upright A | cost B | upright B |
| --- | ---: | --- | ---: | --- |
| true PMP | 26.2 | yes | 10.2 | yes |
| gaussian | 1221.9 | no | 1199.2 | no |
| softplus | 168.0 | no | 190.7 | no |
| `ReLU^2` | 331.6 | yes | 10.1 | yes |
| `ReLU^5` | 631.4 | no | 352.9 | no |

The branch decision at the curve is now learnable — and only `ReLU^2` learns
it. From B it brakes to the `θ = 0` upright at the true cost (10.1 vs 10.2);
from A it correctly swings over to the `2π` upright, though with an
over-energetic arc (331.6 vs 26.2). Every smooth model fails from **both**
starts, and the failure tracks global fit quality rather than proximity to the
curve: gaussian saturates the actuator and overshoots past `2π`; softplus
settles at a spurious equilibrium; `ReLU^5` under-rotates and stalls. On the
one-sided data every model mis-branched from beyond the curve; the data fix
moved the bottleneck from coverage to fit quality.

## Summary

With two-sided data the switching set is an interior kink of the training set,
and the study measures representation rather than extrapolation. `ReLU^2` is
the best approximation choice on both sides of the split and the only model
that yields a correct cross-switching feedback law; the cost is a several-fold
loss of interior accuracy for every model, because the high-value band
dominates the unweighted H1 objective. Rebalancing the objective (per-sample
weighting or band share) is the open follow-up.
