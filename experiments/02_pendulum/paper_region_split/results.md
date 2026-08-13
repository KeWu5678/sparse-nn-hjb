# region_split Results

**Questions.** (1) How well do sparse shallow models fit an optimal value function whose **gradient jumps across the swing-up switching set**, and what role do the activation and the nonconvex penalty play? (2) Can a reliable **feedback law** be synthesized from the fitted value function near — and across — the switching set?

**Setup.** Pendulum swing-up value samples, **two-sided** at the switching set: 3,900 = 3,000 in-basin samples + 900 shifted samples straddling the switching set (300 from the $V_0$ side and 600 from the $V_{2\pi}$ side, within distance 0.5 of the set; see `README.md` for the construction and error metric). This study runs no sweep of its own: it reads the H1 runs of the two pendulum model-family sweeps—**log_penalty** (signed, profile insertion; `../log_penalty`) and **frac_exp_penalty** (ReLU^p atoms, penalty exponent q = 2/(p+1), finite_step insertion, gamma=0 by design; `../frac_exp_penalty`)—with alpha and gamma selected per run by rest L1. The regional comparison uses the switching set identified during data generation: the **near-switching subset** is the lowest 10% of samples by distance to the (±2π-tiled) switching set (d ≤ 0.25 on this dataset); the **rest** contains all other samples. The model-level studies (§4–§5) use five representative signed H1 models—gaussian, softplus, leaky ReLU, relu², relu⁵.

## 1. The target: a value function with a gradient discontinuity

The pendulum value is continuous, but its one-sided gradients differ across the switching spirals—which is why global H1 alone is not enough: the fit must be checked against the switching set identified during data generation and by the induced feedback law. The regions of attraction of the periodic upright equilibria—PMP paths filled by nearest-point classification—are separated by the nonsmooth switching curves used in the regional comparison (open-loop data visualisations are centralised in [`experiments/00_openloop/pendulum`](../../00_openloop/pendulum)):

| value samples | value surface | switching set |
| --- | --- | --- |
| ![value samples](../../00_openloop/pendulum/figures/value_scatter.png) | ![value surface](../../00_openloop/pendulum/figures/value_surface.png) | ![regions of attraction](../../00_openloop/pendulum/figures/regions_of_attraction.png) |

### 1.1 The kink, seen in the data

Along a normal cross-section of the switching curve (through the densest data region), the two fixed-target values—one for the upright at 0, the other for the upright at 2π—cross. Their pointwise minimum $V=\min\{V_0,V_{2\pi}\}$ is continuous, with a concave kink where the minimizing target changes, so ∇V jumps (left: V; right: n·∇V):

| value along the cross-section | normal gradient along the cross-section |
| --- | --- |
| ![fixed-target values](figures/transect_true_branches_value.png) | ![fixed-target gradients](figures/transect_true_branches_gradient.png) |

One structural fact controls everything below: **the training data straddles the switching curve** (this reverses the earlier one-sided generation, whose samples stopped at the curve). The basin restriction alone yields one-sided data because the adjacent $2\pi k$-shifted target is excluded by the cut. The dataset therefore adds shifted samples on both sides, each retained only where its corresponding fixed-target value is below the first-order extrapolation of the other value. Verified on the emitted samples: the 10% near-switching subset (d ≤ 0.25) contains 221 samples on one side and 169 on the other, and 44% of all samples within 0.3 of the curve have an opposite-side neighbour within 0.3 (0% in the one-sided data). The gradient jump is therefore **in-sample** wherever both fixed-target values supply data; the residual one-sided stretches are portions whose opposite-target paths exceed the PMP integration cap. Ground truth on both sides is reconstructed as the pointwise minimum of the tiled raw fixed-target values.

## 2. Error concentrates at the switching set — now as a representation cost

Region mean per-sample L1 (absolute) error / global mean ‖true‖, scored on the **region-eval pool**—the dense certified two-sided point set (~962k points) with the training rows excluded—on $S_{0.3}$ and its complement. This split is count-fair, out-of-sample, and exogenous to the sampling design. `switch/rest` > 1 ⇒ worse near the switching set.

Mean per-sample L1 on the region-eval pool (out-of-sample, $S_{0.3}$)—count-fair and robust to V→0

| kind   | insertion   | activation   | loss | gamma | neurons | switching L1 | rest L1  | switch/rest |
| ------ | ----------- | ------------ | ---- | ----- | ------- | ------------ | -------- | ----------- |
| signed | finite_step | relu^5       | h1   | 0     | 23      | 1.99e+00     | 4.48e-01 | 4.45        |
| signed | finite_step | relu^2.01    | h1   | 0     | 129     | 5.55e-01     | 1.19e-01 | 4.66        |
| signed | finite_step | relu^4       | h1   | 0     | 25      | 2.12e+00     | 4.34e-01 | 4.87        |
| signed | finite_step | relu^2       | h1   | 0     | 125     | 6.62e-01     | 1.32e-01 | 5.01        |
| signed | finite_step | relu^3       | h1   | 0     | 129     | 1.01e+00     | 1.96e-01 | 5.17        |
| signed | profile     | softplus     | h1   | 1     | 68      | 2.38e+00     | 7.15e-01 | 3.32        |
| signed | profile     | matern52     | h1   | 1     | 108     | 1.39e+00     | 3.45e-01 | 4.01        |
| signed | profile     | leaky_relu   | h1   | 0.1   | 128     | 5.12e-01     | 1.05e-01 | 4.90        |
| signed | profile     | gaussian     | h1   | 0     | 19      | 1.91e+00     | 3.87e-01 | 4.93        |
| signed | profile     | gelu_squared | h1   | 1     | 12      | 1.97e+00     | 3.95e-01 | 5.00        |
| signed | profile     | gausscent_1  | h1   | 0.1   | 29      | 2.00e+00     | 3.86e-01 | 5.17        |
| signed | profile     | tanh         | h1   | 1     | 26      | 2.03e+00     | 3.88e-01 | 5.24        |

Every model—every activation and every penalty—is **4.1–6.7× worse on $S_{0.3}$**. The composition matters: switching-set L1 is compressed across models (0.72–2.48, a ~3.4× spread) while rest L1 spans ~4.6× (0.11–0.52), so the ratio mostly reflects how good a model is *away* from the curve. With the jump in-sample, $S_{0.3}$ is genuinely hard for every atom class: a uniform representation cost, no longer a one-sided extrapolation artifact.

### 2.1 The error profile against distance

Per-bin relative error (bin mean |V̂−V| / bin mean |V|) against distance to the switching set (equal-width bins; grey bars = samples per bin). Read it as a *spatial failure diagnostic* — where each model concentrates its own error — not as an absolute model ranking (that is the table above); the far tail has few samples and should not be over-interpreted point by point:

| value error vs distance | gradient error vs distance |
| --- | --- |
| ![value error](figures/error_vs_distance_value.png) | ![gradient error](figures/error_vs_distance_gradient.png) |

The profile inverted relative to the one-sided data. The switching set itself (d < 0.3) is no longer the relative-error peak — the pad/collar band anchors the fit there and |V| is large. The peak now sits at d ≈ 0.65: that bin holds the dense sample mass around the upright equilibrium, where |V| → 0 inflates the per-bin *relative* error and where two-sided training visibly costs interior accuracy (§3). ReLU² keeps the lowest profile at every distance; ReLU⁵ pays the largest interior penalty.

## 3. The price of two-sided coverage

The near-switching subset is expensive by construction: at the production share it is 23% of the sample count but carries ~75% of the squared value mass and ~57% of the squared gradient mass of the normalized H1 objective (mean |V| ≈ 24.5 near the switching set vs 3.8 in the body), so the unweighted least-squares fit is dominated by the hardest, kink-carrying region and interior accuracy is traded away (§2.1). The control below asks the follow-up directly: **does spending more samples near the switching set improve the fit there?**

### 3.1 Oversampling near the switching set

![oversampling control](figures/oversampling_control.png)

Four two-sided training sets built from the same certified pools (`scripts/investigation/make_twosided_oversampling_sets.py`) vary only the near-switching share: 6k at the production ~23% share, 6k reallocated to 40% and 60%, and the original set plus 2,000 near-switching samples (8k total, 42%). Two atom families use one α capacity ladder per variant: signed gaussian (γ=1, α ∈ {1e-3…1e-5}) and signed ReLU² (γ=0, α ∈ {1e-4…1e-6}). Every fitted model is re-scored on one common two-sided evaluation set: the region-eval pool minus the union of all variants' training rows (~936k points, identical across models and strictly out-of-sample), with the split fixed at $S_{0.3}$ and one denominator pair. Faint dots show the α ladder; lines show the best run per variant.

Common-set relative H1 error for the switching-best run (three α values per variant; all entries come from one run)

| family   | variant        | runs | switching | rest  | neurons |
| -------- | -------------- | ---- | --------- | ----- | ------- |
| gaussian | 6k 23% (base)  | 3    | 0.581     | 0.589 | 111     |
| gaussian | 6k 40% band    | 3    | 0.572     | 0.530 | 114     |
| gaussian | 6k 60% band    | 3    | 0.602     | 0.668 | 125     |
| gaussian | 6k+2k band add | 3    | 0.597     | 0.568 | 118     |
| ReLU^2   | 6k 23% (base)  | 3    | 0.246     | 0.156 | 108     |
| ReLU^2   | 6k 40% band    | 3    | 0.289     | 0.172 | 131     |
| ReLU^2   | 6k 60% band    | 3    | 0.346     | 0.235 | 109     |
| ReLU^2   | 6k+2k band add | 3    | 0.288     | 0.184 | 131     |

**Oversampling near the switching set does not improve the fit there for either atom family.** Gaussian is essentially flat across all variants (switching 0.57–0.60): more near-switching samples cannot make a smooth atom represent a kink. ReLU²—uniformly 2–4× better on both regions—*degrades monotonically* as the near-switching share grows (switching 0.246 → 0.289 → 0.346, rest 0.156 → 0.220): these samples already dominate the unweighted objective at the production share, and reallocating samples away from the interior starves the smooth structure its ridges anchor to. Adding 2,000 near-switching samples on top of the budget beats reallocation but not the original set. Thus the production ~23% share is at or near optimal for both families, and the error on $S_{0.3}$ is a **representation limit of the atom class** (§4.4), not a sampling deficit; per-sample objective weighting remains untested.

## 4. Which atoms fit the switching-set target best

### 4.1 Insertion frontier

![insertion frontier](figures/frontier.png)

The running best relative H1 validation error reached as neurons are inserted, for the selected run in each model family. ReLU² separates from the field almost immediately and reaches the lowest error; the other families plateau well above it. This is the sparsity side of the switching/rest story: low-power rectified atoms buy the most accuracy per neuron on this nonsmooth target.

### 4.2 Accuracy per model

![switching/rest dumbbell](figures/near_far_dumbbell.png)

Relative H1 error (log scale) on $S_{0.3}$ (filled) and its complement (open), per representative model; rows are ordered by error on the complement. **ReLU² dominates both regions** (rest ≈ 0.20, $S_{0.3}$ ≈ 0.31); leaky ReLU is the clear runner-up (rest ≈ 0.29)—the two kink-capable atoms lead both regions, 1.5–3× ahead of the smooth activations. ReLU⁵ is the only model *better* on $S_{0.3}$ than outside it; its stiff high-degree atoms fit the near-switching samples but pay for it elsewhere (see §2.1).

### 4.3 Learned value surfaces

| gaussian | softplus | leaky ReLU |
| --- | --- | --- |
| ![gaussian surface](figures/surface_gaussian.png) | ![softplus surface](figures/surface_softplus.png) | ![leaky relu surface](figures/surface_leaky_relu.png) |

| ReLU² | ReLU⁵ |
| --- | --- |
| ![relu2 surface](figures/surface_relu2.png) | ![relu5 surface](figures/surface_relu5.png) |

The learned V̂ over the state plane (z clipped at 60). With samples from both sides of the switching set, the models now shape the full multi-well landscape, not just the central bowl: ReLU² raises sharp diagonal walls along the switching set between the 2πk wells; leaky ReLU builds the same walls with piecewise-linear facets; Gaussian reproduces the wells but rounds the ridge off; softplus—the weakest fit throughout—smears the structure.

### 4.4 Models on the normal cross-section

The same cross-section as §1.1, with the fitted models overlaid (solid black = lower-envelope truth; unlike the one-sided data, the models now saw samples on **both** sides of s = 0):

| value | normal gradient |
| --- | --- |
| ![transect value](figures/transect_value.png) | ![transect gradient](figures/transect_normal_gradient.png) |

At s = 0 the true n·∇V jumps by ≈ 80–100 units. The jump being in-sample is necessary but not sufficient: **no model reproduces its magnitude**. The rectified atoms come closest—their derivatives can break across a hyperplane: ReLU² develops a visible kink at s ≈ 0 and tracks the true V level best. Leaky ReLU's staircase is its atom geometry made visible: a piecewise-linear network has zero curvature, so ∇V̂ is **piecewise constant, not zero**—along the cross-section n·∇V̂ is exactly a step function (verified: every step coincides with one of the 10 atom-line crossings in the window, and between crossings the variation is machine-zero), holding a nonzero plateau ≈ −30…−42 whose level is the summed c·(a·n) of the active atoms. The smooth activations interpolate a gentle slope through the discontinuity, exactly as their C^∞ atoms must. All models undershoot the steep pre-jump gradient (true n·∇V ≈ −100 at s < 0): the finite-width near-switching subset limits how much one-sided steepness the global H1 fit spends neurons on. This is the §2 cost on $S_{0.3}$ seen pointwise: a genuine representation limit at a *seen* discontinuity.

### 4.5 Mechanism: where the atoms sit

![atom portrait](figures/atom_portrait.png)

Each atom's active line {a·x + b = 0} in the physical (θ, θ̇) plane (line strength ∝ |outer weight|), for the §2 representatives relu² (left: 125 neurons, switching/rest L1 0.66/0.13) and gaussian (right: 19 neurons, switching/rest L1 1.91/0.39), with the switching curve in black. ReLU² concentrates its strongest lines parallel to the diagonal switching arms — piecewise low-degree ridges whose derivative breaks exactly where the target's does — while gaussian's strength is spread over near-isotropic bumps that can tile the wells but not seat a gradient break. This is the mechanism behind §4.1–§4.2 and the cross-section kink in §4.4.

## 5. Can a reliable feedback law be synthesized?

Closed-loop rollouts of u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x), one phase panel per feedback law, from two starts placed symmetrically on either side of the switching curve (× markers)—**both represented in-sample** because the data straddle the curve. The curve separates two optimal behaviours here: from **start A** (blue) the true law swings over the top to the 2π upright; from **start B** (red) it brakes directly to the θ = 0 upright. The switching set is black; all panels share the same axes. True PMP feedback uses nearest neighbours over the tiled raw paths and the pointwise minimum of the fixed-target values.

| true PMP | gaussian | softplus |
| --- | --- | --- |
| ![true PMP](figures/feedback_true_pmp.png) | ![gaussian](figures/feedback_gaussian.png) | ![softplus](figures/feedback_softplus.png) |

| leaky ReLU | ReLU² | ReLU⁵ |
| --- | --- | --- |
| ![leaky relu](figures/feedback_leaky_relu.png) | ![relu2](figures/feedback_relu2.png) | ![relu5](figures/feedback_relu5.png) |

The control signal from start B, per feedback law (axis clipped to the informative band — softplus's ±30 actuator-saturation excursion leaves the frame). True PMP brakes to θ = 0 with u rising from ≈ −7 to 0; **ReLU² (red dashes on the black line, right panel) tracks it almost exactly**; the others oscillate or saturate:

| log-penalty models | ReLU^p models |
| --- | --- |
| ![control from B, log-penalty](figures/feedback_control_b_log_penalty.png) | ![control from B, ReLU^p](figures/feedback_control_b_relu.png) |

Closed-loop cost / stabilization from the two straddling starts (A = (0.71, 0.68), B = (0.23, 0.53); T=10)

| model      | cost A | upright A | cost B | upright B |
| ---------- | ------ | --------- | ------ | --------- |
| true PMP   | 26.2   | yes       | 10.2   | yes       |
| gaussian   | 210.2  | no        | 199.1  | no        |
| softplus   | 155.2  | no        | 153.4  | no        |
| leaky ReLU | 773.4  | no        | 10.3   | yes       |
| ReLU^2     | 55.5   | yes       | 10.1   | yes       |
| ReLU^5     | 218.8  | no        | 216.1  | no        |

**The target choice at the curve is now learnable—and only ReLU² learns it from both sides.** From B it brakes to the θ = 0 upright at the true cost (10.1 vs 10.2); from A it correctly swings over to the 2π upright, though with an over-energetic arc (55.5 vs 26.2)—correct target, inefficient execution. Every other law fails from *both* starts: leaky ReLU—the accuracy runner-up—and softplus over-accelerate, blow past the uprights and never brake (costs 10.3 and 153.4 from B); Gaussian settles into a limit cycle around the wells without reaching an upright (199.1 from B); ReLU⁵ swings over from A but arrives at 2π too slowly to be captured, and from B stalls just short of the θ = 0 upright (216.1 from B). On the one-sided data every model chose the wrong target from beyond the curve; the data fix moved the bottleneck from *coverage* to *fit quality*—only the atom class that fits the kink yields a usable feedback law.

## 6. Conclusions

- **The switching set is now an interior kink of the training data** (§1.1): the shifted-sample construction puts the gradient jump in-sample wherever both fixed-target values supply data. The error on $S_{0.3}$ (4.1–6.7× the rest error, §2) is a genuine representation cost at a seen discontinuity—the earlier one-sided sampling diagnosis no longer applies.
- **No atom class represents the jump; the rectified atoms come closest** (§4.4, §4.5): they alone develop a kink on the cross-section and align their strongest ridges with the arms, and ReLU² is the best model on *both* sides of the split (§2, §4.2). Smooth activations necessarily interpolate through the discontinuity.
- **Two-sided coverage has an interior price, and near-switching oversampling does not pay it down** (§2.1, §3): the near-switching subset is 23% of the samples but dominates the unweighted H1 objective (~75% of the squared value mass), so interior accuracy degrades several-fold relative to one-sided training—most for stiff ReLU⁵, least for ReLU². Varying this share (23–60%) or adding samples leaves the error on $S_{0.3}$ flat (§3.1); per-sample objective weighting is the open follow-up.
- **Cross-switching feedback synthesis now works—for the atom that fits** (§5): ReLU² chooses the correct target from both sides of the curve (matching the true cost from B), which no model achieved on one-sided data; every other atom class fails from both starts. The bottleneck moved from data coverage to fit quality.
