# region_split Results

**Questions.** (1) How well do sparse shallow models fit an optimal value function whose **gradient jumps across the swing-up switching set**, and what role do the activation and the nonconvex penalty play? (2) Can a reliable **feedback law** be synthesized from the fitted value function near — and across — the switching set?

**Setup.** Pendulum swing-up value samples, **two-sided** at the switching set: 3,900 = 3,000 in-basin body + a 900-sample envelope-certified band straddling the switching arms (300 near-side pad + 600 far-side collar, within 0.5 of the arms; see `README.md` for the construction and the error-metric rationale). This study runs no sweep of its own: it reads the H1 runs of the two pendulum model-family sweeps — **log_penalty** (signed, profile insertion; `../log_penalty`) and **frac_exp_penalty** (ReLU^p atoms, penalty exponent q = 2/(p+1), finite_step insertion, gamma=0 by design; `../frac_exp_penalty`) — with alpha and gamma selected per cell by rest L1. The region split uses the switching set identified during data generation: the **switching band** = the lowest 10% of samples by distance to the (±2π-tiled) switching set (d ≤ 0.25 on this dataset); the **rest** = all other samples. The model-level studies (§4–§5) use five representative signed H1 models — gaussian, softplus, leaky ReLU, relu², relu⁵.

## 1. The target: a value function with a gradient discontinuity

The pendulum value is continuous, but the gradient changes branch across the switching spirals — which is why global H1 alone is not enough: the fit must be checked against the switching set identified during data generation, and by the induced feedback law. The regions of attraction of the (periodic) upright equilibria — PMP characteristics filled by nearest-point classification — are separated by the nonsmooth switching curves the region split is built on (open-loop data visualisations are centralised in [`experiments/00_openloop/pendulum`](../../00_openloop/pendulum)):

| value samples | value surface | switching set |
| --- | --- | --- |
| ![value samples](../../00_openloop/pendulum/figures/value_scatter.png) | ![value surface](../../00_openloop/pendulum/figures/value_surface.png) | ![regions of attraction](../../00_openloop/pendulum/figures/regions_of_attraction.png) |

### 1.1 The kink, seen in the data

Along a normal cross-section of the switching curve (through the densest data region), the two candidate PMP branches — one driving to the upright at 0, the other to the upright at 2π — cross; the optimal V is their lower envelope — continuous, with a concave kink where the branches exchange optimality, so ∇V jumps (left: V; right: n·∇V):

| value along the cross-section | normal gradient along the cross-section |
| --- | --- |
| ![true branches, value](figures/transect_true_branches_value.png) | ![true branches, gradient](figures/transect_true_branches_gradient.png) |

One structural fact controls everything below: **the training data straddles the switching curve** (this reverses the earlier one-sided generation, whose samples stopped AT the curve). The basin restriction alone yields one-sided data — the neighbouring branch is the 2πk-shifted basin, excluded by the cut — so the dataset adds an envelope-certified band: near-side pad points (the central branch between the basin's conservative trim and the true arm) and far-side collar points (the ±2π branch across the arm), each kept only where its branch value beats the competing branch's locally extrapolated value. Verified on the emitted samples: the 10% switching band (d ≤ 0.25) contains 221 near-side and 169 far-side points, and 44% of all samples within 0.3 of the curve have an opposite-side neighbour within 0.3 (0% in the one-sided data). The gradient jump is therefore **in-sample** wherever both branches carry data; the residual one-sided stretches are arms whose far branch lies beyond the PMP integration cap. Ground truth on both sides is reconstructed as the lower envelope of the raw (unrestricted) PMP trajectories tiled by 2πk in θ.

## 2. Error concentrates at the switching set — now as a representation cost

Region mean per-sample L1 (absolute) error / global mean ‖true‖, scored on the **region-eval pool** — the dense certified two-sided point set (~962k points) with the training rows excluded, split by the fixed **switching tube** (distance to the ±2π-tiled switching curve ≤ 0.3) — count-fair, out-of-sample, and exogenous to the sampling design (the earlier percentile band moved with the training distribution). `switch/rest` > 1 ⇒ worse at the switching set.

Mean per-sample L1 on the region-eval pool (out-of-sample, switching tube d ≤ 0.3) — count-fair, robust to V→0

| kind   | insertion   | activation   | loss | gamma | neurons | switching L1 | rest L1  | switch/rest |
| ------ | ----------- | ------------ | ---- | ----- | ------- | ------------ | -------- | ----------- |
| signed | finite_step | relu^5       | h1   | 0     | 24      | 1.95e+00     | 4.76e-01 | 4.10        |
| signed | finite_step | relu^4       | h1   | 0     | 46      | 2.09e+00     | 4.46e-01 | 4.68        |
| signed | finite_step | relu^3       | h1   | 0     | 99      | 1.07e+00     | 1.94e-01 | 5.48        |
| signed | finite_step | relu^2       | h1   | 0     | 131     | 7.24e-01     | 1.27e-01 | 5.68        |
| signed | finite_step | relu^2.01    | h1   | 0     | 104     | 7.51e-01     | 1.12e-01 | 6.71        |
| signed | profile     | gelu_squared | h1   | 0.1   | 30      | 2.48e+00     | 5.17e-01 | 4.79        |
| signed | profile     | leaky_relu   | h1   | 0     | 105     | 9.95e-01     | 2.04e-01 | 4.88        |
| signed | profile     | matern52     | h1   | 0.1   | 129     | 1.70e+00     | 3.18e-01 | 5.33        |
| signed | profile     | gausscent_1  | h1   | 0.1   | 123     | 2.04e+00     | 3.68e-01 | 5.54        |
| signed | profile     | gaussian     | h1   | 0     | 114     | 2.07e+00     | 3.73e-01 | 5.56        |
| signed | profile     | tanh         | h1   | 10    | 100     | 1.96e+00     | 3.41e-01 | 5.75        |
| signed | profile     | softplus     | h1   | 10    | 77      | 1.52e+00     | 2.38e-01 | 6.38        |

Every model — every activation, every penalty — is **4.1–6.7× worse in the switching tube**. The composition matters: switching L1 is compressed across models (0.72–2.48, a ~3.4× spread) while rest L1 spans ~4.6× (0.11–0.52), so the ratio mostly reflects how good a model is *away* from the curve. With the jump in-sample, the switching tube is genuinely hard for every atom class: a uniform representation cost, no longer a one-sided extrapolation artifact.

### 2.1 The error profile against distance

Per-bin relative error (bin mean |V̂−V| / bin mean |V|) against distance to the switching set (equal-width bins; grey bars = samples per bin). Read it as a *spatial failure diagnostic* — where each model concentrates its own error — not as an absolute model ranking (that is the table above); the far tail has few samples and should not be over-interpreted point by point:

| value error vs distance | gradient error vs distance |
| --- | --- |
| ![value error](figures/error_vs_distance_value.png) | ![gradient error](figures/error_vs_distance_gradient.png) |

The profile inverted relative to the one-sided data. The switching set itself (d < 0.3) is no longer the relative-error peak — the pad/collar band anchors the fit there and |V| is large. The peak now sits at d ≈ 0.65: that bin holds the dense sample mass around the upright equilibrium, where |V| → 0 inflates the per-bin *relative* error and where two-sided training visibly costs interior accuracy (§3). ReLU² keeps the lowest profile at every distance; ReLU⁵ pays the largest interior penalty.

## 3. The price of two-sided coverage

The switching band is expensive by construction: at the production share it is 23% of the sample count but carries ~75% of the squared value mass and ~57% of the squared gradient mass of the normalized H1 objective (mean |V| ≈ 24.5 in the switching band vs 3.8 in the body), so the unweighted least-squares fit is dominated by the hardest, kink-carrying region and interior accuracy is traded away (§2.1). The control below asks the follow-up directly: **does spending more samples on the switching band buy the switching fit anything?**

### 3.1 Oversampling the switching band

![oversampling control](figures/oversampling_control.png)

Four two-sided training sets built from the same certified pools (`scripts/investigation/make_twosided_oversampling_sets.py`), varying only the switching-band share: 6k at the production ~23% share (base), 6k reallocated to a 40% and a 60% share, and base + 2,000 *added* switching-band samples (8k total, 42%). Two atom families, one α capacity ladder each per variant: signed gaussian (γ=1, α ∈ {1e-3…1e-5}) and signed ReLU² (γ=0, α ∈ {1e-4…1e-6}). Every fitted model is re-scored on ONE common two-sided evaluation set: the region-eval pool minus the union of all variants' training rows (~936k points, identical across models and strictly out-of-sample for every one of them — the same convention as the consolidated per-run metrics), one switching tube (d ≤ 0.3 to the ±2π-tiled switching curve), one denominator pair. Faint dots = the α ladder, lines = the best run per variant.

Best common-set relative H1 error per variant and family (min over the α ladder; neurons = size of the switching-best run)

| family   | variant        | runs | switching | rest  | neurons |
| -------- | -------------- | ---- | --------- | ----- | ------- |
| gaussian | 6k 23% (base)  | 3    | 0.581     | 0.589 | 111     |
| gaussian | 6k 40% band    | 3    | 0.572     | 0.530 | 114     |
| gaussian | 6k 60% band    | 3    | 0.602     | 0.668 | 125     |
| gaussian | 6k+2k band add | 3    | 0.597     | 0.568 | 118     |
| ReLU^2   | 6k 23% (base)  | 3    | 0.246     | 0.156 | 108     |
| ReLU^2   | 6k 40% band    | 3    | 0.289     | 0.172 | 131     |
| ReLU^2   | 6k 60% band    | 3    | 0.346     | 0.220 | 109     |
| ReLU^2   | 6k+2k band add | 3    | 0.288     | 0.184 | 131     |

**Band oversampling does not buy the switching fit for either atom family.** gaussian is essentially flat across all variants (switching 0.57–0.60): more switching-band samples cannot teach a smooth atom a kink. ReLU² — uniformly 2–4× better on both regions — *degrades monotonically* as the switching-band share grows (switching 0.246 → 0.289 → 0.346, rest 0.156 → 0.220): the switching band already dominates the unweighted objective at the production share, and reallocating samples away from the interior starves the smooth structure its ridges anchor to. Adding 2,000 switching-band samples on top of the budget beats reallocation but not the base. So the production ~23% share is at or near optimal for both families, and the switching-band error is a **representation limit of the atom class** (§4.4), not a sampling deficit; per-sample objective weighting remains the untried lever.

## 4. Which atoms fit the switching-set target best

### 4.1 Insertion frontier

![insertion frontier](figures/frontier.png)

The running best relative H1 validation error reached as neurons are inserted, for the selected run in each model family. ReLU² separates from the field almost immediately and reaches the lowest error; the other families plateau well above it. This is the sparsity side of the switching/rest story: low-power rectified atoms buy the most accuracy per neuron on this nonsmooth target.

### 4.2 Accuracy per model

![switching/rest dumbbell](figures/near_far_dumbbell.png)

Relative H1 error (log scale) in a fixed geometric tube around the switching set (d ≤ 0.3, filled — well posed there now that the switching band makes |V| large) and in the rest of the domain (open), per representative model; rows ordered by rest error. **ReLU² dominates both regions** (rest ≈ 0.20, tube ≈ 0.31); leaky ReLU is the clear runner-up (rest ≈ 0.29) — the two kink-capable atoms lead both regions, 1.5–3× ahead of the smooth activations. ReLU⁵ is the only model *better* inside the tube than outside — its stiff high-degree atoms seat the switching band but pay for it everywhere else (see §2.1).

### 4.3 Learned value surfaces

| gaussian | softplus | leaky ReLU |
| --- | --- | --- |
| ![gaussian surface](figures/surface_gaussian.png) | ![softplus surface](figures/surface_softplus.png) | ![leaky relu surface](figures/surface_leaky_relu.png) |

| ReLU² | ReLU⁵ |
| --- | --- |
| ![relu2 surface](figures/surface_relu2.png) | ![relu5 surface](figures/surface_relu5.png) |

The learned V̂ over the state plane (z clipped at 60). With the switching band in the training data the models now shape the full multi-well landscape, not just the central bowl: ReLU² raises sharp diagonal walls along the switching arms between the 2πk wells; leaky ReLU builds the same walls with piecewise-linear facets; gaussian reproduces the wells but rounds the ridge off; softplus — the weakest fit throughout — smears the structure.

### 4.4 Models on the normal cross-section

The same cross-section as §1.1, with the fitted models overlaid (solid black = lower-envelope truth; unlike the one-sided data, the models now saw samples on **both** sides of s = 0):

| value | normal gradient |
| --- | --- |
| ![transect value](figures/transect_value.png) | ![transect gradient](figures/transect_normal_gradient.png) |

At s = 0 the true n·∇V jumps by ≈ 80–100 units. The jump being in-sample is necessary but not sufficient: **no model reproduces its magnitude**. The rectified atoms come closest — their derivatives can break across a hyperplane: ReLU² develops a visible kink at s ≈ 0 and tracks the true V level best. leaky ReLU's staircase is its atom geometry made visible: a piecewise-linear network has zero curvature, so ∇V̂ is **piecewise constant, not zero** — along the cross-section n·∇V̂ is exactly a step function (verified: every step coincides with one of the 10 atom-line crossings in the window, and between crossings the variation is machine-zero), holding a nonzero plateau ≈ −30…−42 whose level is the summed c·(a·n) of the active atoms. The smooth activations interpolate a gentle slope through the discontinuity, exactly as their C^∞ atoms must. All models undershoot the steep pre-jump gradient (true n·∇V ≈ −100 at s < 0): the finite-width switching band bounds how much one-sided steepness the global H1 fit will spend neurons on. This is the §2 switching-band cost seen pointwise: a genuine representation limit at a *seen* discontinuity.

### 4.5 Mechanism: where the atoms sit

![atom portrait](figures/atom_portrait.png)

Each atom's active line {a·x + b = 0} in the physical (θ, θ̇) plane (line strength ∝ |outer weight|), for the §2 representatives relu² (left: 131 neurons, switching/rest L1 0.72/0.13) and gaussian (right: 114 neurons, switching/rest L1 2.07/0.37), with the switching curve in black. ReLU² concentrates its strongest lines parallel to the diagonal switching arms — piecewise low-degree ridges whose derivative breaks exactly where the target's does — while gaussian's strength is spread over near-isotropic bumps that can tile the wells but not seat a gradient break. This is the mechanism behind §4.1–§4.2 and the cross-section kink in §4.4.

## 5. Can a reliable feedback law be synthesized?

Closed-loop rollouts of u(x) = −(1/(2r·ml²)) ∂_θ̇ V̂(x), one phase panel per feedback law, from two starts placed symmetrically either side of the switching curve (× markers) — **both in-sample now** that the switching band straddles the curve. The curve separates two optimal behaviours here: from **start A** (blue) the true law swings over the top to the 2π upright; from **start B** (red) it brakes directly to the θ = 0 upright. Switching set in black; all panels share the same axes. True PMP feedback = envelope nearest-neighbour over the tiled raw trajectories.

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

| model      | cost A  | upright A | cost B  | upright B |
| ---------- | ------- | --------- | ------- | --------- |
| true PMP   | 26.2    | yes       | 10.2    | yes       |
| gaussian   | 298.5   | no        | 278.9   | no        |
| softplus   | 73013.3 | no        | 66996.7 | no        |
| leaky ReLU | 869.1   | no        | 776.9   | no        |
| ReLU^2     | 57.9    | yes       | 10.3    | yes       |
| ReLU^5     | 235.9   | no        | 232.4   | no        |

**The branch decision at the curve is now learnable — and only ReLU² learns it from both sides.** From B it brakes to the θ = 0 upright at the true cost (10.3 vs 10.2); from A it correctly swings over to the 2π upright, though with an over-energetic arc (57.9 vs 26.2) — right branch, inefficient execution. Every other law fails from *both* starts: leaky ReLU — the accuracy runner-up — and softplus over-accelerate, blow past the uprights and never brake (costs 776.9 and 66996.7 from B); gaussian settles into a limit cycle around the wells without reaching an upright (278.9 from B); ReLU⁵ swings over from A but arrives at 2π too slowly to be captured, and from B stalls just short of the θ = 0 upright (232.4 from B). On the one-sided data every model mis-branched from beyond the curve; the data fix moved the bottleneck from *coverage* to *fit quality* — only the atom class that fits the kink yields a usable feedback law.

## 6. Conclusions

- **The switching set is now an interior kink of the training data** (§1.1): the envelope-certified pad+collar band puts the gradient jump in-sample wherever both branches carry data. The switching-band error (4.1–6.7× the rest error, §2) is a genuine representation cost at a seen discontinuity — the one-sided era's 'sampling artifact' diagnosis no longer applies.
- **No atom class represents the jump; the rectified atoms come closest** (§4.4, §4.5): they alone develop a kink on the cross-section and align their strongest ridges with the arms, and ReLU² is the best model on *both* sides of the split (§2, §4.2). Smooth activations necessarily interpolate through the discontinuity.
- **Two-sided coverage has an interior price, and band oversampling does not pay it down** (§2.1, §3): the switching band is 23% of the samples but dominates the unweighted H1 objective (~75% of the squared value mass), so interior accuracy degrades several-fold relative to one-sided training — most for stiff ReLU⁵, least for ReLU². Varying the switching-band share (23–60%) or adding samples leaves the switching fit flat (§3.1); per-sample objective weighting is the open follow-up.
- **Cross-switching feedback synthesis now works — for the atom that fits** (§5): ReLU² makes the correct branch decision from both sides of the curve (matching the true cost from B), which no model achieved on one-sided data; every other atom class fails from both starts. The bottleneck moved from data coverage to fit quality.
