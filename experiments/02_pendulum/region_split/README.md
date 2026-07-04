# region_split — pendulum

Profiles how well sparse PDAP models fit the pendulum value function in the
**switching band** (around the nonsmooth swing-up switching set identified
during data generation) vs the **rest** of the domain. Built on the corrected
pendulum dataset (the branch-restriction fix, issue #18 — diagnosis in
`../00_openloop/DIAGNOSIS.md`); without that fix the samples never reached the
switching set.

## Method

- **Data**: `data=pendulum`, which auto-selects `eval=region_split`
  (`conf/data/pendulum.yaml`). The dataset is the open-loop PMP solve
  (`scripts/run_pendulum_pmp_openloop_example.py`): 2000 backward-PMP
  trajectories, basin-restricted (the validated 256-path basin — issue #18
  workaround, since the 2000-path auto-basin came back empty) and **two-sided
  at the switching set**: **3,900** samples = 3,000 in-basin body (adaptive
  level-set thinning, single upright well at θ = 0, no periodic tiling of the
  emitted domain) + a 900-sample envelope-certified band within 0.5 of the
  switching arms — 300 near-side *pad* (central-branch points between the
  basin's conservative trim and the true arm) and 600 far-side *collar*
  (±2π-branch points across the arm), each candidate kept only where its
  branch value beats the competing branch's locally extrapolated value
  (`PendulumPmpSolver.build_collar_samples`). The gradient jump is in-sample
  wherever both branches carry data. Each sample carries a precomputed
  distance to the **±2π-tiled** switching set
  (`scripts/investigation/precompute_region_distances.py`; tiling matters —
  the stored ridge covers one period while the basin's left arm is its −2π
  copy); the switching band (lowest 10%) is distance ≤ 0.25 → 390 band samples
  (221 on the central-branch side + 169 across the arm).
- **Region split**: the **switching tube** = the fixed tubular neighborhood
  {distance to the ±2π-tiled switching curve ≤ 0.3}; the **rest** = everything
  else. Metrics are scored on the **region-eval pool** — the dense certified
  two-sided point set (~962k points) with the training rows excluded, so the
  split is out-of-sample and exogenous to the sampling design
  (`scripts/investigation/build_region_eval_pool.py`; sidecars from
  `scripts/investigation/rescore_region_metrics.py`). The earlier percentile
  band over the emitted samples was endogenous — adding switching-band samples
  shrank the region it was graded on — and was mostly seen data. (Records
  written before the consolidation carry the same metrics under legacy
  `near`/`far` keys.)
- **No sweep of its own** (`make region-split` regenerates the analysis): this
  study reads the H1 runs of the two pendulum model-family sweeps, which record
  the region metrics on every run (`data=pendulum` auto-selects the region
  eval). The former dedicated region_split sweep was a strict subset of the
  log_penalty grid and was retired.
- **log_penalty rows** (`make sweep EXPERIMENT=pendulum/log_penalty`): signed,
  profile insertion, `activation ∈ {leaky_relu, softplus, tanh, gaussian,
  gausscent_1, matern52, gelu_squared}` × `α ∈ {1e-2…1e-5}` ×
  `γ ∈ {0, 0.1, 1, 10}`; α and γ selected per activation by rest L1.
- **frac_exp_penalty rows** (`make sweep EXPERIMENT=pendulum/frac_exp_penalty`):
  signed, finite_step insertion, `power ∈ {2, 2.01, 3, 4, 5}` (penalty exponent
  q = 2/(p+1)), gamma=0 by design, `α ∈ {1e-3…1e-6}` selected per power by
  rest L1.
- **Output**: `results.md` (tables + figures), via `analysis.py`.

## Choice of error metric

Measuring approximation error *near a discontinuity* and *where the reference
field vanishes* are both known to break the default metric. We validated the
choice against the literature rather than picking what fit our hypothesis.

**Region-local relative error is confounded when a region's ‖true‖ collapses.**
Relative error `‖pred−true‖ / ‖true‖` computed per region with that region's own
denominator inflates wherever the reference approaches zero¹ — historically this
flipped the conclusion here (the one-sided data's rest region was dominated by
the `V→0` upright interior). On the two-sided pool the switching tube carries
large `|V|`, so the per-region relative H1 is well posed and is reported for the
tube/rest split; the count-fair absolute L1 below remains the primary metric.

**Near a discontinuity, L1 is the appropriate norm.** For shock/discontinuity
problems the L1 norm gives the error "without significant contamination from
localized deviations," whereas L2 worst-case bounds "cannot be generalized … near
a shock."² So we report the region's **mean per-sample L1 (absolute) error,
normalized by the global mean `‖true‖`**: L1 is discontinuity-appropriate; the
*mean* (not the sum) keeps switching/rest **count-fair** — the switching band is
only ~10% of samples, so a summed L1 would trivially favour it (a sum-based
version reported switch/rest ≈ 0.12 ≈ the 10/90 split, an artifact); and the
shared global denominator
removes the per-region `V→0` confound. This is the **primary** table.

**The primary diagnostic is error vs distance to the switching set.** The standard
way to see behaviour near a kink is the spatial error profile — Gibbs-type error is
"most severe on nodes around the discontinuity, with error decreasing" with
distance.³ `analysis.py` plots per-sample absolute error (value + gradient) in
**equal-width** distance bins, normalized by the model's own mean, so a ratio > 1
marks worse-than-average accuracy at that distance and the curve is comparable
across activations. The bins are equal-width (not equal-count) so the x-axis is a
true spatial coordinate; because samples pile up in the dense band near the
switching set and thin out toward the basin boundary, the per-bin sample count is
overlaid as grey bars so the under-sampled far tail is visibly noisier.

The historical percentile-band metrics and the confounded per-region relative
table were retired with the tube/pool consolidation.

Findings live in `results.md` (generated by `analysis.py`); this file only states
the purpose, method, and metric rationale.

### Sources
1. [Forecast Evaluation for Data Scientists: Pitfalls and Best Practices](https://arxiv.org/pdf/2203.10716);
   [Regression Evaluation Metrics](https://programming-ocean.com/knowledge-hub/regression-evaluation-metrics-ai-atlas.php).
2. [Numerical smoothness and error analysis for RKDG on scalar nonlinear conservation laws](https://arxiv.org/pdf/1105.1393).
3. [Battling Gibbs Phenomenon: On Finite Element approximation of discontinuities](https://arxiv.org/pdf/1907.03429).
