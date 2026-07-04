# activationsearch — pendulum (discontinuous gradient)

Which activations fit a value function whose gradient **jumps across a switching
set** (pendulum swing-up). Method, sweep axes, and the activation list are in
`README.md`; this file reports the findings. Three representatives span the
derivative-regularity ladder — `leaky_relu` (kink), `softplus` (smooth ridge),
`gaussian` (localized RBF) — all under the `signed` model (semiconcave models do
not round-trip through the saved result, issue #19; plain `relu` is the convex L1
baseline, not an H1 candidate — see `../../baseline`).

## Key finding

Under the gradient-augmented (H1) loss each neuron contributes σ to V and
**w·σ′ to ∇V**, so the activation *derivative* σ′ is the basis that reconstructs
the gradient field. On the **faithful** basin data (issue #18: wider basin, V≲57,
θ̇±7.7), the result **reverses** the old narrow-basin (cap-35) finding: the sampled
region is dominated by the **smooth interior** of the upright basin — the switching
set is only its sparse boundary — so the activation that tiles a smooth field most
economically wins. That is the **localized RBF (`gaussian`)**, best on every error
band *and* in closed loop; the kink (`leaky_relu`) is a competent, uniquely
sparse-robust runner-up, and the smooth ridge (`softplus`) is worst.

### Activation shape

![activation shape](figures/shape_leakyrelu_softplus_gaussian.png)

`leaky_relu′` is a step (slope `a`→`1` across its hyperplane — it can seat a finite
gradient jump, and unlike plain `relu` has no dead side); `softplus′` is a smooth
sigmoid (it can only smear a jump over width ≈ 1/β); the Gaussian derivative is a
sign-changing bump (localized, non-monotone). The kink can *localize* a switching-set
discontinuity, but most of the basin is smooth — where the localized bump tiles best.

### Fitted value surfaces

The learned value function V̂(x) of each representative, plotted as a surface over
the state plane (after Han & Yang Fig. 2 left). `gaussian` reproduces the value bowl
most faithfully; `leaky_relu` is competent; `softplus` collapses.

| $\mathrm{leaky\,ReLU}$ · 116 neurons · rel $H^1$=0.21 | $\mathrm{softplus}$ · 38 neurons · rel $H^1$=0.63 | $e^{-x^2}$ · 126 neurons · rel $H^1$=0.05 |
| --- | --- | --- |
| ![leaky_relu](figures/value_surface_leaky_relu.png) | ![softplus](figures/value_surface_softplus.png) | ![gaussian](figures/value_surface_gaussian.png) |

### Best metrics

Best signed H1 fit per representative

| activation | neurons | rel H1 | far Lv | far Lg |
| ---------- | ------- | ------ | ------ | ------ |
| leaky_relu | 116     | 0.211  | 0.095  | 0.246  |
| softplus   | 38      | 0.629  | 0.521  | 0.622  |
| gaussian   | 126     | 0.049  | 0.064  | 0.040  |

These are the runs shown in the surface and control panels (the lowest-rel-H1
signed fit per activation). `rel H1` is the relative H1 error on the validation
split; `far Lv`/`far Lg` are the absolute far-field value/gradient L1 (the robust
metric the rest of the doc uses). On the faithful basin data the two metrics now
**agree**:

- **`gaussian` is best on every column** — lowest rel H1 (0.049), lowest far value
  L1 (0.064) *and* lowest far gradient L1 (0.040). On the old cap-35 basin rel H1
  *inverted* (gaussian posted a low rel H1 but a large absolute error); here the
  V→0 confound does not flip the ranking, and the robust L1 confirms it.
- **`leaky_relu` is the competent runner-up** (rel H1 0.211, far Lv 0.095) — it
  seats the boundary kink but pays on the gradient band (far Lg 0.246, ~6× gaussian).
- **`softplus` is worst on all three** (far Lv 0.521): its smooth, single-signed σ′
  cannot localize structure, so its surface collapses (see the panel).

### Synthesized control vs true feedback

The pendulum is control-affine with cost `r·u²`, so a value function induces the
feedback `u(x) = −(1/(2r·ml²)) ∂_θ̇ V(x)` (`PendulumSwingUpProblem.feedback_from_gradient`).
We synthesize û from each fitted V̂ and **roll it out in the true dynamics** from the
deepest *supported* state (the basin has no data at hanging θ=π, so a swing-up from
hanging is off-data; we start from the highest cost-to-go sample and drive to the
nearest upright copy), beside the true PMP feedback (nearest-neighbour interpolated
from the dataset's costate samples) — the closed-loop test of Han & Yang Fig. 3:
does the induced controller reach upright, and at what cost.

![control synthesis](figures/control_synthesis.png)

Stabilize to upright from deepest supported start x0=[-6.91, 7.68], T=10

| controller | neurons | reaches upright? | closed-loop cost |
| ---------- | ------- | ---------------- | ---------------- |
| true PMP   | —       | yes              | 57.6             |
| leaky_relu | 116     | yes              | 59.4             |
| softplus   | 38      | no               | 494.4            |
| gaussian   | 126     | yes              | 57.5             |

Left is the angle from upright θ(t)−2kπ; right is the feedback law u(t); cost is the
accumulated running cost. From a supported start the closed loop **agrees with the
accuracy ranking**:

- **`gaussian` ≈ true PMP** — it reaches upright at cost 57.5, essentially matching
  the true feedback (57.6) and the best learned controller. Its low-error, smooth
  surface induces a benign global field.
- **`leaky_relu` reaches upright** at a slightly higher cost (59.4) — competent, as
  its piecewise-linear extrapolation stays bounded.
- **`softplus` fails** (cost 494) — its collapsed value surface gives a feedback that
  never reaches upright.

So on the faithful basin data the localized RBF wins *both* on-data accuracy and
closed-loop control; the kink is a safe runner-up; the smooth ridge is unusable.
This **reverses** the old cap-35 conclusion (where the kink was the only stabilizing
controller and gaussian diverged) — with the corrected wider basin, the closed loop
no longer punishes the smooth surface, because the data now covers the region the
controlled trajectory actually traverses.

## Parameter discussion (α, γ)

The nonconvex penalty `α·Σ φ(|c|)` has two knobs: **α** scales the penalty (the
sparsity lever) and **γ** controls the log-term nonconvexity (γ=0 turns it off;
larger γ prunes redundant clustered atoms). The tables take the three signed reps.

Effect of alpha (best gamma per alpha), signed H1

| activation | alpha  | gamma | neurons | rel H1 |
| ---------- | ------ | ----- | ------- | ------ |
| leaky_relu | 1e-05  | 0.1   | 148     | 0.237  |
| leaky_relu | 0.0001 | 1     | 127     | 0.250  |
| leaky_relu | 0.001  | 10    | 116     | 0.211  |
| leaky_relu | 0.01   | 10    | 30      | 0.339  |
| softplus   | 1e-05  | 0.1   | 119     | 0.738  |
| softplus   | 0.0001 | 10    | 83      | 0.644  |
| softplus   | 0.001  | 1     | 38      | 0.629  |
| softplus   | 0.01   | 0     | 17      | 0.801  |
| gaussian   | 1e-05  | 1     | 126     | 0.049  |
| gaussian   | 0.0001 | 10    | 112     | 0.129  |
| gaussian   | 0.001  | 0.1   | 58      | 0.249  |
| gaussian   | 0.01   | 1     | 32      | 0.470  |

**α is the dominant sparsity lever**: raising it from 1e-5 to 1e-2 cuts the neuron
count sharply (leaky_relu 148→30, gaussian 126→32). What that pruning costs depends on
the activation, and the dependence is itself a finding:

- `gaussian` is **most accurate but pruning-sensitive** — rel H1 0.049 at α=1e-5
  (126 neurons) degrades to 0.470 by α=1e-2 (32 neurons). Its low error is bought by
  *tiling* the smooth field with many localized bumps; remove them and it cannot fit.
- `leaky_relu` **stays competent when sparse** — rel H1 holds ≈ 0.21–0.34 all the way
  down to 30 neurons. A kink atom carries irreducible structure, so few are needed —
  this sparse-robustness is its remaining edge over the RBF.
- `softplus` never fits (rel H1 ≈ 0.63–0.80 at every α).

Effect of gamma (best alpha per gamma), signed H1

| activation | gamma | alpha  | neurons | rel H1 |
| ---------- | ----- | ------ | ------- | ------ |
| leaky_relu | 0     | 0.0001 | 133     | 0.257  |
| leaky_relu | 0.1   | 1e-05  | 148     | 0.237  |
| leaky_relu | 1     | 0.0001 | 127     | 0.250  |
| leaky_relu | 10    | 0.001  | 116     | 0.211  |
| softplus   | 0     | 1e-05  | 117     | 0.750  |
| softplus   | 0.1   | 0.001  | 24      | 0.711  |
| softplus   | 1     | 0.001  | 38      | 0.629  |
| softplus   | 10    | 0.0001 | 83      | 0.644  |
| gaussian   | 0     | 1e-05  | 122     | 0.070  |
| gaussian   | 0.1   | 1e-05  | 124     | 0.088  |
| gaussian   | 1     | 1e-05  | 126     | 0.049  |
| gaussian   | 10    | 1e-05  | 129     | 0.111  |

**γ only refines**: the best γ improves `leaky_relu` modestly (0.211 at γ=10 vs 0.257
at γ=0) and `gaussian` slightly (0.049 at γ=1 vs 0.070 at γ=0); it does not change the
ranking.

![alpha/gamma tradeoff](figures/alpha_gamma_tradeoff.png)

The scatter places every signed-H1 run on the neurons-vs-accuracy plane (marker =
activation, colour = γ). `gaussian` reaches the lowest error but only at high neuron
count, rising steeply as it is pruned; `leaky_relu` is a low, flat band — competent
across the whole sparsity range; `softplus` sits high throughout. The penalty moves a
model *along* its frontier, but where that frontier sits — and whether accuracy
survives sparsity — is set by the activation.

## Full result

Region-split **mean per-sample L1**, normalized by the global mean ‖true‖ — robust
to the V→0 interior. `far` = smooth region, `near` = 10% closest to the switching
set, `near/far` = how much worse the fit is at the switching set. Best (α, γ) per
(model, activation) by far value-L1; ranked best-first; both model kinds, all seven
activations.

### H1 (gradient-augmented) loss

Pendulum H1 fit — best far value-L1 per model/activation

| kind        | activation   | gamma | alpha  | neurons | far Lv | near Lv | near/far V | far Lg | near/far G |
| ----------- | ------------ | ----- | ------ | ------- | ------ | ------- | ---------- | ------ | ---------- |
| signed      | gaussian     | 0     | 1e-05  | 122     | 0.052  | 0.032   | 0.62       | 0.054  | 0.77       |
| signed      | leaky_relu   | 10    | 0.0001 | 130     | 0.080  | 0.044   | 0.56       | 0.268  | 0.92       |
| signed      | matern52     | 10    | 1e-05  | 114     | 0.109  | 0.073   | 0.67       | 0.121  | 0.87       |
| signed      | tanh         | 1     | 1e-05  | 102     | 0.134  | 0.073   | 0.54       | 0.118  | 0.79       |
| signed      | gelu_squared | 0.1   | 1e-05  | 100     | 0.144  | 0.056   | 0.38       | 0.171  | 0.67       |
| signed      | gausscent_1  | 1     | 0.0001 | 91      | 0.180  | 0.084   | 0.47       | 0.240  | 0.97       |
| semiconcave | matern52     | 1     | 0.001  | 14      | 0.234  | 0.213   | 0.91       | 0.367  | 1.08       |
| semiconcave | leaky_relu   | 1     | 0.0001 | 104     | 0.284  | 0.223   | 0.78       | 0.653  | 0.95       |
| semiconcave | gaussian     | 0.1   | 0.001  | 12      | 0.375  | 0.222   | 0.59       | 0.322  | 0.87       |
| semiconcave | tanh         | 1     | 0.001  | 15      | 0.481  | 0.310   | 0.65       | 0.792  | 1.13       |
| signed      | softplus     | 1     | 0.0001 | 77      | 0.490  | 0.387   | 0.79       | 0.724  | 1.24       |
| semiconcave | gelu_squared | 0.1   | 0.0001 | 30      | 0.498  | 0.406   | 0.82       | 0.619  | 1.08       |
| semiconcave | gausscent_1  | 1     | 0.001  | 17      | 0.584  | 0.434   | 0.74       | 0.660  | 1.03       |
| semiconcave | softplus     | 1     | 0.0001 | 7       | 0.790  | 0.681   | 0.86       | 0.888  | 1.16       |

### L2 (value-only) loss

Pendulum L2 fit — best far value-L1 per model/activation

| kind        | activation   | gamma | alpha  | neurons | far Lv | near Lv | near/far V | far Lg | near/far G |
| ----------- | ------------ | ----- | ------ | ------- | ------ | ------- | ---------- | ------ | ---------- |
| signed      | leaky_relu   | 1     | 1e-05  | 125     | 0.176  | 0.206   | 1.17       | 0.695  | 1.42       |
| semiconcave | leaky_relu   | 10    | 1e-05  | 20      | 0.385  | 0.393   | 1.02       | 0.959  | 1.28       |
| signed      | gaussian     | 10    | 1e-05  | 46      | 0.396  | 0.384   | 0.97       | 0.930  | 1.21       |
| signed      | gausscent_1  | 0     | 1e-05  | 102     | 0.410  | 0.405   | 0.99       | 0.924  | 1.29       |
| signed      | matern52     | 0.1   | 1e-05  | 70      | 0.418  | 0.413   | 0.99       | 0.916  | 1.35       |
| signed      | tanh         | 10    | 1e-05  | 45      | 0.434  | 0.460   | 1.06       | 0.966  | 1.53       |
| signed      | softplus     | 10    | 1e-05  | 42      | 0.434  | 0.426   | 0.98       | 1.001  | 1.28       |
| semiconcave | gelu_squared | 10    | 1e-05  | 5       | 0.439  | 0.444   | 1.01       | 0.903  | 1.44       |
| semiconcave | tanh         | 10    | 1e-05  | 7       | 0.440  | 0.452   | 1.03       | 0.901  | 1.36       |
| semiconcave | gausscent_1  | 10    | 1e-05  | 8       | 0.442  | 0.447   | 1.01       | 0.911  | 1.39       |
| signed      | gelu_squared | 1     | 0.0001 | 9       | 0.468  | 0.497   | 1.06       | 0.939  | 1.43       |
