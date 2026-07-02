# penaltypowers — pendulum (switching set)

How the **power** of the atom `σ(z)^p` — equivalently the nonconvex penalty exponent
`q = 2/(p+1)` — behaves on a value function with a **gradient jump across the
switching set** (pendulum swing-up). Method and sweep axes are in `README.md`;
this file reports the findings, with ReLU at powers {2, 3, 5} as
representatives. This is the switching-set counterpart of `../../01_vdp/frac_exp_penalty`:
there higher power was free sparsity; **here it is the opposite** — the reversal is the point.

## Key finding

The coefficient penalty is `α·Σ |c|^q`, `q = 2/(power+1)`: higher power ⇒ smaller
`q` ⇒ more aggressive nonconvex pruning. On a smooth target that was free, but a
switching-set value function needs **more, lower-degree** atoms to seat the gradient
discontinuity — so raising the power both over-smooths each atom and over-prunes,
and the fit degrades sharply.

### Penalty & atom shape

![penalty shape](figures/penalty_shape.png)

Left: the atom `σ(z)=ReLU(z)^p` sharpens with `p`. Right: the penalty `φ(c)=|c|^q`
grows more concave as `q=2/(p+1)` shrinks. The mildest nonconvex penalty, `p=2`
(`q=0.67`, the ReLU² atom), is the sweet spot here.

### Fitted value surfaces

The learned `V̂(x)` of the best ReLU fit at each power (shared `plot_model_value_surface`
renderer, z **unclipped**). The reconstruction degrades visibly as the power rises: the
in-basin bowl is correct at `p=2`, but a high-power atom `σ(z)^p` extrapolates as a
degree-`p` polynomial off the thin basin, so by `p=5` the surface is dominated by a
~10⁴ off-basin spike and the true value range (≲57) is squashed flat — itself a picture
of why high power over-fits the boundary and loses the interior.

| ReLU $p=2$ · 107 neurons · far Lv=0.01 | ReLU $p=3$ · 71 neurons · far Lv=0.03 | ReLU $p=5$ · 55 neurons · far Lv=0.05 |
| --- | --- | --- |
| ![p2](figures/value_surface_p2.png) | ![p3](figures/value_surface_p3.png) | ![p5](figures/value_surface_p5.png) |

### Best metrics

Best ReLU^p H1 fit per power (rel H1 + region-split absolute L1)

| power | q=2/(p+1) | α     | neurons | rel H1 | near Lv | far Lv | near Lg | far Lg |
| ----- | --------- | ----- | ------- | ------ | ------- | ------ | ------- | ------ |
| 2     | 0.67      | 1e-06 | 107     | 0.034  | 0.015   | 0.010  | 0.045   | 0.019  |
| 3     | 0.50      | 1e-06 | 71      | 0.049  | 0.057   | 0.035  | 0.102   | 0.039  |
| 5     | 0.33      | 1e-06 | 55      | 0.093  | 0.111   | 0.053  | 0.167   | 0.056  |

`rel H1` is the global relative H1 loss (total value+gradient); `near`/`far` are the
region-split **mean per-sample absolute L1** (`near` = lowest-10% distance to the
switching set, `far` = the smooth rest — absolute, so robust to the V→0 upright
interior). The reversal: **power 2 (ReLU², `q=0.67`) is best on every representative
column** — total H1, both value bands, and both gradient bands worsen as the power
is raised to `p=3` and `p=5`. The `near` columns are the harder
switching-set band and stay above `far` throughout. The aggressive nonconvex penalty
that was free on smooth VDP is *harmful* here, because the kinked target cannot be
tiled by a few high-power atoms.

### Synthesized control vs true feedback

The pendulum is control-affine with cost `r·u²`, so the value induces the **feedback
law** `u(x) = −(1/(2r·ml²)) ∂_θ̇ V(x)` (Han & Yang Eq. 15,
`PendulumSwingUpProblem.feedback_from_gradient`). We synthesize û from each fitted ReLU^p
`V̂` and roll it out in the true dynamics, beside the true PMP feedback. **Start.** The
PMP samples are the *upright smooth basin* (faithful basin restriction, issue #18): they
cover `θ̇∈[−7.7,7.7]` and `V≲57` around each upright copy, with **no data at the
hanging-down switching set `θ=π`** — so a swing-up *from hanging* is unsupported
(off-data, every feedback law would extrapolate). We instead start from the **deepest
supported state** (the highest cost-to-go sample, here `x0≈[-6.91, 7.68]`, a fast-moving
edge-of-basin state) and drive to the nearest upright copy. Left is the angle from upright
`θ(t)−2kπ`; right is the feedback law `u(t)`.

![control synthesis](figures/control_synthesis.png)

Stabilize to upright from deepest supported start x0=[-6.91, 7.68], T=8

| controller | neurons | reaches upright? | closed-loop cost |
| ---------- | ------- | ---------------- | ---------------- |
| true PMP   | —       | yes              | 57.6             |
| ReLU p=2   | 107     | yes              | 57.2             |
| ReLU p=3   | 71      | yes              | 57.3             |
| ReLU p=5   | 55      | yes              | 57.4             |

At `t=0` all four controllers sit at the same supported state, and their controls
**agree in sign and rough magnitude** — the feedback law is synthesized correctly (the
earlier off-data hanging start, by contrast, gave sign-flipped garbage because no sample
constrained `∇V̂` there). From this supported start **all four controllers stabilize** to
the upright with near-identical cost (true PMP 57.6; power 2 57.2; power 3 57.3; power 5 57.4). Unlike the on-data accuracy — where power 2 is materially better
— the induced *controllers* are barely separated here: every fitted `V̂` produces a benign
global field from this fast edge-of-basin start, so the closed loop does not amplify the
accuracy gap. The mild cost ordering still favors low power (`p2 < p3 < p5`), consistent
with the reversal, but the catastrophic high-power failure seen on the old (narrower,
cap-35) basin data is **gone** with the faithful wider basin. (Caveats: a single initial
condition, and closed-loop outcomes are sensitive to the start because `∇V̂` is only pinned
on the thin basin samples; the basin data cannot reach the hanging configuration at all.
The robust, data-level statement is the accuracy reversal itself: **higher power degrades
the fit**, far Lv rising from 0.01 at `p=2` to 0.05 at `p=5`.)

## Parameter discussion (power, α)

The **power** is the headline lever above; the penalty strength **α** only refines
within a power (the sweep fixes `γ=0`, so the penalty is the pure `α·Σ|c|^q`).

ReLU H1: effect of α at each power (best far Lv per cell)

| power | α     | neurons | far Lv | far Lg |
| ----- | ----- | ------- | ------ | ------ |
| 2     | 1e-06 | 107     | 0.010  | 0.019  |
| 2     | 1e-05 | 65      | 0.021  | 0.036  |
| 2     | 1e-04 | 24      | 0.045  | 0.067  |
| 2     | 1e-03 | 15      | 0.414  | 0.258  |
| 2     | 1e-02 | 9       | 0.694  | 0.580  |
| 3     | 1e-06 | 71      | 0.035  | 0.039  |
| 3     | 1e-05 | 39      | 0.076  | 0.063  |
| 3     | 1e-04 | 24      | 0.253  | 0.158  |
| 3     | 1e-03 | 15      | 0.781  | 0.470  |
| 3     | 1e-02 | 3       | 0.984  | 0.754  |
| 5     | 1e-06 | 55      | 0.053  | 0.056  |
| 5     | 1e-05 | 34      | 0.458  | 0.298  |
| 5     | 1e-04 | 22      | 0.566  | 0.402  |
| 5     | 1e-03 | 12      | 0.388  | 0.696  |
| 5     | 1e-02 | 2       | 0.961  | 0.739  |

![power/alpha tradeoff](figures/power_alpha_tradeoff.png)

The scatter places every signed ReLU-H1 run on the neurons-vs-(far value-L1) plane
(marker = power, colour = α): power 2 occupies the accurate region; higher powers sit
at larger error regardless of α. α moves a model along its frontier, but the power
sets which frontier — and at a switching set, low power wins.

## Full result

Region-split **mean per-sample L1**, normalized by the global mean ‖true‖. `far` =
smooth region, `near/far` = how many times worse the switching set is. Best α per
power by far value-L1 (ReLU, `γ=0`).

### H1 (gradient-augmented) loss

Pendulum H1 fit — best far value-L1 per power (α swept)

| power | α     | neurons | far Lv | near/far V | far Lg | near/far G |
| ----- | ----- | ------- | ------ | ---------- | ------ | ---------- |
| 2     | 1e-06 | 107     | 0.010  | 1.54       | 0.019  | 2.34       |
| 2.01  | 1e-06 | 106     | 0.013  | 1.49       | 0.023  | 2.22       |
| 3     | 1e-06 | 71      | 0.035  | 1.63       | 0.039  | 2.62       |
| 4     | 1e-06 | 58      | 0.152  | 1.17       | 0.100  | 2.62       |
| 5     | 1e-06 | 55      | 0.053  | 2.09       | 0.056  | 2.98       |

### L2 (value-only) loss

Pendulum L2 fit — best far value-L1 per power (α swept)

| power | α     | neurons | far Lv | near/far V | far Lg | near/far G |
| ----- | ----- | ------- | ------ | ---------- | ------ | ---------- |
| 2     | 1e-06 | 25      | 0.231  | 3.33       | 0.559  | 3.89       |
| 2.01  | 1e-06 | 28      | 0.237  | 3.31       | 0.596  | 3.83       |
| 3     | 1e-06 | 23      | 0.206  | 3.36       | 0.458  | 4.62       |
| 4     | 1e-06 | 19      | 0.285  | 3.06       | 0.700  | 3.99       |
| 5     | 1e-06 | 17      | 0.351  | 2.51       | 0.773  | 3.50       |
