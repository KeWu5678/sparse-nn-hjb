# region_split_pendulum Results

**Questions.** (1) How well do sparse shallow models fit an optimal value function whose **gradient jumps across the swing-up switching set**, and what role do the activation and the nonconvex penalty play? (2) Can a reliable **feedback law** be synthesized from the fitted value function near — and across — the switching set?

**Setup.** Pendulum swing-up value samples, **two-sided** at the switching set: 3,900 = 3,000 in-basin body + a 900-sample envelope-certified band straddling the switching arms (300 near-side pad + 600 far-side collar, within 0.5 of the arms; see `README.md` for the construction and the error-metric rationale). Two sweeps on the same dataset and `eval=region_split` hook: smooth activations (profile insertion, gamma selected per cell) and **ReLU^p atoms with the fractional-exponent penalty** q = 2/(p+1) (finite_step insertion, gamma=0 by design, alpha selected per cell) — see `../02_pendulum/frac_exp_penalty` — plus the kink specialist `leaky_relu` from the activationsearch sweep (`../02_pendulum/log_penalty`, H1 runs). The region split uses the switching set identified during data generation: the **switching band** = the lowest 10% of samples by distance to the (±2π-tiled) switching set (d ≤ 0.25 on this dataset); the **rest** = all other samples. The model-level studies (§4–§5) use five representative signed H1 models — gaussian, softplus, leaky ReLU, relu², relu⁵; semiconcave models are excluded there (they do not round-trip through the fit artifact, #19).

## 1. The target: a value function with a gradient discontinuity

The pendulum value is continuous, but the gradient changes branch across the switching spirals — which is why global H1 alone is not enough: the fit must be checked against the switching set identified during data generation, and by the induced feedback law. The regions of attraction of the (periodic) upright equilibria — PMP characteristics filled by nearest-point classification — are separated by the nonsmooth switching curves the region split is built on (open-loop data visualisations are centralised in [`experiments/00_openloop/pendulum`](../00_openloop/pendulum)):

| value samples | value surface | switching set |
| --- | --- | --- |
| ![value samples](../00_openloop/pendulum/figures/value_scatter.png) | ![value surface](../00_openloop/pendulum/figures/value_surface.png) | ![regions of attraction](../00_openloop/pendulum/figures/regions_of_attraction.png) |

### 1.1 The kink, seen in the data

Along a transect normal to the switching curve (through the densest data region), the two candidate PMP branches — one driving to the upright at 0, the other to the upright at 2π — cross; the optimal V is their lower envelope — continuous, with a concave kink where the branches exchange optimality, so ∇V jumps (left: V; right: n·∇V):

| value along the transect | normal gradient along the transect |
| --- | --- |
| ![true branches, value](figures/transect_true_branches_value.png) | ![true branches, gradient](figures/transect_true_branches_gradient.png) |

One structural fact controls everything below: **the training data straddles the switching curve** (this reverses the earlier one-sided generation, whose samples stopped AT the curve). The basin restriction alone yields one-sided data — the neighbouring branch is the 2πk-shifted basin, excluded by the cut — so the dataset adds an envelope-certified band: near-side pad points (the central branch between the basin's conservative trim and the true arm) and far-side collar points (the ±2π branch across the arm), each kept only where its branch value beats the competing branch's locally extrapolated value. Verified on the emitted samples: the 10% switching band (d ≤ 0.25) contains 221 near-side and 169 far-side points, and 44% of all samples within 0.3 of the curve have an opposite-side neighbour within 0.3 (0% in the one-sided data). The gradient jump is therefore **in-sample** wherever both branches carry data; the residual one-sided stretches are arms whose far branch lies beyond the PMP integration cap. Ground truth on both sides is reconstructed as the lower envelope of the raw (unrestricted) PMP trajectories tiled by 2πk in θ.

## 2. Error concentrates at the switching set — now as a representation cost

Region mean per-sample L1 (absolute) error / global mean ‖true‖ — count-fair and robust to the V→0 interior; `switch/rest` > 1 ⇒ worse at the switching set. (On the two-sided data the relative-H1 appendix table agrees in direction; it no longer flips.)

Mean per-sample L1 over the full dataset — count-fair, robust to V→0

| kind        | insertion   | activation   | loss | gamma | neurons | switching L1 | rest L1  | switch/rest |
| ----------- | ----------- | ------------ | ---- | ----- | ------- | ------------ | -------- | ----------- |
| semiconcave | profile     | tanh         | h1   | 0     | 25      | 2.39e+00     | 7.90e-01 | 3.02        |
| semiconcave | profile     | softplus     | h1   | 1     | 8       | 2.52e+00     | 8.33e-01 | 3.03        |
| semiconcave | profile     | leaky_relu   | h1   | 0.1   | 118     | 1.69e+00     | 5.03e-01 | 3.36        |
| semiconcave | profile     | gaussian     | h1   | 0     | 23      | 2.15e+00     | 5.65e-01 | 3.81        |
| semiconcave | profile     | gelu_squared | h1   | 1     | 19      | 2.02e+00     | 5.27e-01 | 3.83        |
| semiconcave | profile     | matern52     | h1   | 0     | 21      | 1.94e+00     | 4.79e-01 | 4.05        |
| signed      | finite_step | relu^5       | h1   | 0     | 79      | 1.78e+00     | 4.38e-01 | 4.07        |
| signed      | finite_step | relu^4       | h1   | 0     | 75      | 1.70e+00     | 3.98e-01 | 4.28        |
| signed      | finite_step | relu^3       | h1   | 0     | 99      | 1.33e+00     | 1.97e-01 | 6.77        |
| signed      | finite_step | relu^2.01    | h1   | 0     | 132     | 9.23e-01     | 1.27e-01 | 7.24        |
| signed      | finite_step | relu^2       | h1   | 0     | 114     | 1.01e+00     | 1.20e-01 | 8.39        |
| signed      | profile     | gelu_squared | h1   | 0     | 39      | 2.13e+00     | 5.72e-01 | 3.72        |
| signed      | profile     | gaussian     | h1   | 1     | 113     | 1.81e+00     | 4.38e-01 | 4.14        |
| signed      | profile     | matern52     | h1   | 0     | 125     | 1.74e+00     | 4.09e-01 | 4.24        |
| signed      | profile     | tanh         | h1   | 1     | 99      | 1.88e+00     | 4.42e-01 | 4.26        |
| signed      | profile     | leaky_relu   | h1   | 0     | 135     | 1.20e+00     | 2.73e-01 | 4.41        |
| signed      | profile     | softplus     | h1   | 0     | 82      | 1.77e+00     | 3.49e-01 | 5.09        |

Every model — both kinds, every activation, every penalty — is **3.0–8.4× worse in the switching band**, a wider spread than the 2.2–3.7× measured on the earlier one-sided data. The composition changed too: switching L1 is compressed across models (0.92–2.52, a ~2.7× spread) while rest L1 spans ~7× (0.12–0.83), so the ratio mostly reflects how good a model is *away* from the curve — ReLU² has the largest ratio (8.4) precisely because its rest error is the smallest. With the jump in-sample, the switching band is genuinely hard for every atom class: a uniform representation cost, no longer a one-sided extrapolation artifact.

### 2.1 The error profile against distance

Per-bin relative error (bin mean |V̂−V| / bin mean |V|) against distance to the switching set (equal-width bins; grey bars = samples per bin). Read it as a *spatial failure diagnostic* — where each model concentrates its own error — not as an absolute model ranking (that is the table above); the far tail has few samples and should not be over-interpreted point by point:

| value error vs distance | gradient error vs distance |
| --- | --- |
| ![value error](figures/error_vs_distance_value.png) | ![gradient error](figures/error_vs_distance_gradient.png) |

The profile inverted relative to the one-sided data. The switching set itself (d < 0.3) is no longer the relative-error peak — the pad/collar band anchors the fit there and |V| is large. The peak now sits at d ≈ 0.65: that bin holds the dense sample mass around the upright equilibrium, where |V| → 0 inflates the per-bin *relative* error and where two-sided training visibly costs interior accuracy (§3). ReLU² keeps the lowest profile at every distance; ReLU⁵ pays the largest interior penalty.

## 3. The price of two-sided coverage

The band is expensive by construction: at the production share it is 23% of the sample count but carries ~75% of the squared value mass and ~57% of the squared gradient mass of the normalized H1 objective (mean |V| ≈ 24.5 in the band vs 3.8 in the body), so the unweighted least-squares fit is dominated by the hardest, kink-carrying region and interior accuracy is traded away (§2.1). The control below asks the follow-up directly: **does spending more samples on the switching band buy the switching fit anything?**

### 3.1 Oversampling the switching band

![oversampling control](figures/oversampling_control.png)

Four two-sided training sets built from the same certified pools (`scripts/investigation/make_twosided_oversampling_sets.py`), varying only the band share: 6k at the production ~23% share (base), 6k reallocated to a 40% and a 60% band, and base + 2,000 *added* band samples (8k total, 42% band). Two atom families, one α capacity ladder each per variant: signed gaussian (γ=1, α ∈ {1e-3…1e-5}) and signed ReLU² (γ=0, α ∈ {1e-4…1e-6}). Every fitted model is re-scored on ONE common two-sided evaluation set — the full certified pool (restricted in-basin points + the envelope-certified band, ~966k points), one switching tube (d ≤ 0.3 to the ±2π-tiled ridge), one denominator pair — since each variant's own recorded metrics use its own band and denominator. Faint dots = the α ladder, lines = the best run per variant.

Best common-set relative H1 error per variant and family (min over the α ladder; neurons = size of the switching-best run)

| family   | variant        | runs | switching | rest  | neurons |
| -------- | -------------- | ---- | --------- | ----- | ------- |
| gaussian | 6k 23% (base)  | 3    | 0.581     | 0.588 | 111     |
| gaussian | 6k 40% band    | 3    | 0.570     | 0.527 | 114     |
| gaussian | 6k 60% band    | 3    | 0.598     | 0.662 | 125     |
| gaussian | 6k+2k band add | 3    | 0.595     | 0.565 | 118     |
| ReLU^2   | 6k 23% (base)  | 3    | 0.250     | 0.156 | 108     |
| ReLU^2   | 6k 40% band    | 3    | 0.289     | 0.172 | 131     |
| ReLU^2   | 6k 60% band    | 3    | 0.343     | 0.218 | 109     |
| ReLU^2   | 6k+2k band add | 3    | 0.288     | 0.184 | 131     |

**Band oversampling does not buy the switching fit for either atom family.** gaussian is essentially flat across all variants (switching 0.57–0.60): more band samples cannot teach a smooth atom a kink. ReLU² — uniformly 2–4× better on both regions — *degrades monotonically* as the band share grows (switching 0.250 → 0.289 → 0.343, rest 0.156 → 0.218): the band already dominates the unweighted objective at the production share, and reallocating samples away from the interior starves the smooth structure its ridges anchor to. Adding 2,000 band samples on top of the budget beats reallocation but not the base. So the production ~23% share is at or near optimal for both families, and the switching-band error is a **representation limit of the atom class** (§4.4), not a sampling deficit; per-sample objective weighting remains the untried lever.

## 4. Which atoms fit the switching-set target best

### 4.1 Insertion frontier

![insertion frontier](figures/frontier.png)

The running best relative H1 validation error reached as neurons are inserted, for the selected run in each model family. ReLU² separates from the field almost immediately and reaches the lowest error; the other families plateau well above it. This is the sparsity side of the switching/rest story: low-power rectified atoms buy the most accuracy per neuron on this nonsmooth target.

### 4.2 Accuracy per model

![switching/rest dumbbell](figures/near_far_dumbbell.png)

Relative H1 error (log scale) in a fixed geometric tube around the switching set (d ≤ 0.3, filled — well posed there now that the band makes |V| large) and in the rest of the domain (open), per representative model; rows ordered by rest error. **ReLU² dominates both regions** (rest ≈ 0.20, tube ≈ 0.31); leaky ReLU is the clear runner-up (rest ≈ 0.29) — the two kink-capable atoms lead both regions, 1.5–3× ahead of the smooth activations. ReLU⁵ is the only model *better* inside the tube than outside — its stiff high-degree atoms seat the band but pay for it everywhere else (see §2.1).

### 4.3 Learned value surfaces

| gaussian | softplus | leaky ReLU |
| --- | --- | --- |
| ![gaussian surface](figures/surface_gaussian.png) | ![softplus surface](figures/surface_softplus.png) | ![leaky relu surface](figures/surface_leaky_relu.png) |

| ReLU² | ReLU⁵ |
| --- | --- |
| ![relu2 surface](figures/surface_relu2.png) | ![relu5 surface](figures/surface_relu5.png) |

The learned V̂ over the state plane (z clipped at 60). With the band in the training data the models now shape the full multi-well landscape, not just the central bowl: ReLU² raises sharp diagonal walls along the switching arms between the 2πk wells; leaky ReLU builds the same walls with piecewise-linear facets; gaussian reproduces the wells but rounds the ridge off; softplus — the weakest fit throughout — smears the structure.

### 4.4 Models on the transect

The same transect as §1.1, with the fitted models overlaid (solid black = lower-envelope truth; unlike the one-sided data, the models now saw samples on **both** sides of s = 0):

| value | normal gradient |
| --- | --- |
| ![transect value](figures/transect_value.png) | ![transect gradient](figures/transect_normal_gradient.png) |

At s = 0 the true n·∇V jumps by ≈ 80–100 units. The jump being in-sample is necessary but not sufficient: **no model reproduces its magnitude**. The rectified atoms come closest — their derivatives can break across a hyperplane: ReLU² develops a visible kink at s ≈ 0 and tracks the true V level best. leaky ReLU's staircase is its atom geometry made visible: a piecewise-linear network has zero curvature, so ∇V̂ is **piecewise constant, not zero** — along the transect n·∇V̂ is exactly a step function (verified: every step coincides with one of the 10 atom-line crossings in the window, and between crossings the variation is machine-zero), holding a nonzero plateau ≈ −30…−42 whose level is the summed c·(a·n) of the active atoms. The smooth activations interpolate a gentle slope through the discontinuity, exactly as their C^∞ atoms must. All models undershoot the steep pre-jump gradient (true n·∇V ≈ −100 at s < 0): the finite-width band bounds how much one-sided steepness the global H1 fit will spend neurons on. This is the §2 switching-band cost seen pointwise: a genuine representation limit at a *seen* discontinuity.

### 4.5 Mechanism: where the atoms sit

![atom portrait](figures/atom_portrait.png)

Each atom's active line {a·x + b = 0} in the physical (θ, θ̇) plane (line strength ∝ |outer weight|), for the §2 representatives relu² (left: 114 neurons, switching/rest L1 1.01/0.12) and gaussian (right: 113 neurons, switching/rest L1 1.81/0.44), with the switching curve in black. ReLU² concentrates its strongest lines parallel to the diagonal switching arms — piecewise low-degree ridges whose derivative breaks exactly where the target's does — while gaussian's strength is spread over near-isotropic bumps that can tile the wells but not seat a gradient break. This is the mechanism behind §4.1–§4.2 and the transect kink in §4.4.

## 5. Can a reliable feedback law be synthesized?

Closed-loop rollouts of u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x), one phase panel per feedback law, from two starts placed symmetrically either side of the switching curve (× markers) — **both in-sample now** that the band straddles the curve. The curve separates two optimal behaviours here: from **start A** (blue) the true law swings over the top to the 2π upright; from **start B** (red) it brakes directly to the θ = 0 upright. Switching set in black; all panels share the same axes. True PMP feedback = envelope nearest-neighbour over the tiled raw trajectories.

| true PMP | gaussian | softplus |
| --- | --- | --- |
| ![true PMP](figures/feedback_true_pmp.png) | ![gaussian](figures/feedback_gaussian.png) | ![softplus](figures/feedback_softplus.png) |

| leaky ReLU | ReLU² | ReLU⁵ |
| --- | --- | --- |
| ![leaky relu](figures/feedback_leaky_relu.png) | ![relu2](figures/feedback_relu2.png) | ![relu5](figures/feedback_relu5.png) |

The control signal from start B, per feedback law (true PMP brakes to θ = 0 with u rising from ≈ −7 to 0; ReLU² tracks it almost exactly; softplus settles at a spurious equilibrium with u ≈ −4; gaussian saturates at ±30):

![control from B](figures/feedback_control_b.png)

Closed-loop cost / stabilization from the two straddling starts (A = (0.71, 0.68), B = (0.23, 0.53); T=10)

| model      | cost A | upright A | cost B | upright B |
| ---------- | ------ | --------- | ------ | --------- |
| true PMP   | 26.2   | yes       | 10.2   | yes       |
| gaussian   | 1221.9 | no        | 1199.2 | no        |
| softplus   | 168.0  | no        | 190.7  | no        |
| leaky ReLU | 1345.3 | no        | 14.8   | yes       |
| ReLU^2     | 331.6  | yes       | 10.1   | yes       |
| ReLU^5     | 631.4  | no        | 352.9  | no        |

**The branch decision at the curve is now learnable — and only ReLU² learns it from both sides.** From B it brakes to the θ = 0 upright at the true cost (10.1 vs 10.2); from A it correctly swings over to the 2π upright, though with an over-energetic arc (331.6 vs 26.2) — right branch, inefficient execution. leaky ReLU, the accuracy runner-up, gets the braking side right (14.8 from B) but fails from the swing-over side. Every smooth model fails on *both* sides, and the failure tracks the global fit quality of §2, not proximity to the curve: gaussian's degraded interior gradient field saturates the actuator and overshoots past 2π (cost ≈ 1200); softplus never reaches an upright (spurious equilibrium); ReLU⁵ under-rotates and stalls near the origin. On the one-sided data every model mis-branched from beyond the curve; the data fix moved the bottleneck from *coverage* to *fit quality*.

## 6. Conclusions

- **The switching set is now an interior kink of the training data** (§1.1): the envelope-certified pad+collar band puts the gradient jump in-sample wherever both branches carry data. The switching-band error (3.0–8.4× the rest error, §2) is a genuine representation cost at a seen discontinuity — the one-sided era's 'sampling artifact' diagnosis no longer applies.
- **No atom class represents the jump; the rectified atoms come closest** (§4.4, §4.5): they alone develop a kink on the transect and align their strongest ridges with the arms, and ReLU² is the best model on *both* sides of the split (§2, §4.2). Smooth activations necessarily interpolate through the discontinuity.
- **Two-sided coverage has an interior price, and band oversampling does not pay it down** (§2.1, §3): the band is 23% of the samples but dominates the unweighted H1 objective (~75% of the squared value mass), so interior accuracy degrades several-fold relative to one-sided training — most for stiff ReLU⁵, least for ReLU². Varying the band share (23–60%) or adding band samples leaves the switching fit flat (§3.1); per-sample objective weighting is the open follow-up.
- **Cross-switching feedback synthesis now works — for the atom that fits** (§5): ReLU² makes the correct branch decision from both sides of the curve (matching the true cost from B), which no model achieved on one-sided data; leaky ReLU gets the braking side only. The smooth models fail globally; the bottleneck moved from data coverage to fit quality.

## Appendix: the same comparison under region-local relative H1

The table repeats the §2 comparison (same rows, same best-per-cell runs) under the *historical* metric of this study: relative H1 computed **per region with that region's own denominator**, ‖pred−true‖_H1 / ‖true‖_H1 restricted to the switching band and to the rest separately. It exists because the metric choice used to change the story: on the one-sided data the rest region's denominator was dominated by the V→0 upright interior, which inflated the rest relative error and made every model look *better* at the switching set (switch/rest < 1 for 14 of 15 rows) — the opposite of the absolute-L1 table. On the two-sided data the band contributes large-|V| pad/collar samples to the switching-band denominator, so the confound is gone: switch/rest ≥ 0.99 for every row and the ranking agrees with §2. The table is kept to document that the conclusion is now metric-robust; the count-fair absolute mean-L1 of §2 remains the primary metric (see `README.md` for the rationale).

Region-local relative H1 (historical metric — agrees with §2 on the two-sided data)

| kind        | insertion   | activation   | loss | gamma | neurons | switching H1 | rest H1  | switch/rest |
| ----------- | ----------- | ------------ | ---- | ----- | ------- | ------------ | -------- | ----------- |
| semiconcave | profile     | softplus     | h1   | 1     | 8       | 8.31e-01     | 8.39e-01 | 0.99        |
| semiconcave | profile     | tanh         | h1   | 0     | 25      | 8.21e-01     | 8.25e-01 | 0.99        |
| semiconcave | profile     | leaky_relu   | h1   | 0.1   | 118     | 6.22e-01     | 5.95e-01 | 1.05        |
| semiconcave | profile     | gaussian     | h1   | 0     | 23      | 7.15e-01     | 6.66e-01 | 1.07        |
| semiconcave | profile     | gelu_squared | h1   | 1     | 19      | 6.91e-01     | 6.36e-01 | 1.09        |
| semiconcave | profile     | matern52     | h1   | 0     | 21      | 6.46e-01     | 5.75e-01 | 1.12        |
| signed      | finite_step | relu^5       | h1   | 0     | 79      | 5.94e-01     | 5.03e-01 | 1.18        |
| signed      | finite_step | relu^4       | h1   | 0     | 75      | 5.90e-01     | 4.79e-01 | 1.23        |
| signed      | finite_step | relu^3       | h1   | 0     | 99      | 4.91e-01     | 2.71e-01 | 1.81        |
| signed      | finite_step | relu^2.01    | h1   | 0     | 132     | 3.74e-01     | 1.79e-01 | 2.09        |
| signed      | finite_step | relu^2       | h1   | 0     | 114     | 4.05e-01     | 1.84e-01 | 2.20        |
| signed      | profile     | gelu_squared | h1   | 0     | 39      | 7.26e-01     | 6.79e-01 | 1.07        |
| signed      | profile     | gaussian     | h1   | 1     | 113     | 6.22e-01     | 5.35e-01 | 1.16        |
| signed      | profile     | tanh         | h1   | 1     | 99      | 6.50e-01     | 5.44e-01 | 1.20        |
| signed      | profile     | matern52     | h1   | 0     | 125     | 6.18e-01     | 5.15e-01 | 1.20        |
| signed      | profile     | softplus     | h1   | 0     | 82      | 6.13e-01     | 4.47e-01 | 1.37        |
| signed      | profile     | leaky_relu   | h1   | 0     | 135     | 4.83e-01     | 3.24e-01 | 1.49        |
