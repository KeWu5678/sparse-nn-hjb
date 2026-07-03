# region_split_pendulum

Profiles how well sparse PDAP models fit the pendulum value function in the
**smooth region** vs the **switching region** (near the nonsmooth swing-up
switching set). Built on the corrected pendulum dataset (the branch-restriction
fix, issue #18 / `experiments/pendulum_branch_restriction/`); without that fix the
samples never reached the switching set.

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
  copy); the near band (lowest 10%) is distance ≤ 0.25 → 390 near samples
  (221 near-side + 169 far-side).
- **Region split**: `near` = the lowest `near_percentile` (default 10) % of
  validation samples by distance to the switching set; `far` = the rest.
- **Sweep** (`make region_split_pendulum`): `model.kind ∈ {signed, semiconcave}` ×
  `activation ∈ {tanh, softplus, matern52, gaussian, gelu_squared}` ×
  `gamma ∈ {0, 1}`, profile insertion, H1 loss, α=1e-4, seed 42.
- **ReLU^p rows** (`make penaltypowers DATA=pendulum`): the analysis additionally
  folds in the H1 runs of the fractional-exponent-penalty sweep
  (`../02_pendulum/frac_exp_penalty`) — signed, finite_step insertion,
  `power ∈ {2, 2.01, 3, 4, 5}` (penalty exponent q = 2/(p+1)), gamma=0 by design,
  `α ∈ {1e-2…1e-6}` selected per power by far L1. Same dataset, same
  `eval=region_split` hook, so the region metrics are directly comparable.
- **Output**: `results.md` (two tables + a diagnostic plot), via `analysis.py`.

## Choice of error metric

Measuring approximation error *near a discontinuity* and *where the reference
field vanishes* are both known to break the default metric. We validated the
choice against the literature rather than picking what fit our hypothesis.

**The default (region-local relative H1) is confounded here.** Relative error
`‖pred−true‖ / ‖true‖` is computed per region with that region's own denominator.
The `far` (smooth interior) region contains the upright equilibrium where `V→0`
and `∇V→0`, so its small `‖true‖` denominator inflates the relative error. Result:
every model looks *better* near the switching set — an artifact of where the small
values live, not of switching-set skill. Relative/percentage error is documented
as unreliable when the reference approaches zero (it can report huge error for an
accurate near-zero prediction); the standard remedies are absolute error there, or
excluding the near-zero region.¹

**Near a discontinuity, L1 is the appropriate norm.** For shock/discontinuity
problems the L1 norm gives the error "without significant contamination from
localized deviations," whereas L2 worst-case bounds "cannot be generalized … near
a shock."² So we report the region's **mean per-sample L1 (absolute) error,
normalized by the global mean `‖true‖`**: L1 is discontinuity-appropriate; the
*mean* (not the sum) keeps `near`/`far` **count-fair** — `near` is only ~10% of
samples, so a summed L1 would trivially favour it (a sum-based version reported
near/far ≈ 0.12 ≈ the 10/90 split, an artifact); and the shared global denominator
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

We keep the relative-H1 table too, clearly labelled as confounded, so the
metric-choice effect is visible rather than hidden.

Findings live in `results.md` (generated by `analysis.py`); this file only states
the purpose, method, and metric rationale.

### Sources
1. [Forecast Evaluation for Data Scientists: Pitfalls and Best Practices](https://arxiv.org/pdf/2203.10716);
   [Regression Evaluation Metrics](https://programming-ocean.com/knowledge-hub/regression-evaluation-metrics-ai-atlas.php).
2. [Numerical smoothness and error analysis for RKDG on scalar nonlinear conservation laws](https://arxiv.org/pdf/1105.1393).
3. [Battling Gibbs Phenomenon: On Finite Element approximation of discontinuities](https://arxiv.org/pdf/1907.03429).
