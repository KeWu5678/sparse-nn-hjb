# Research Directions

This note restores the provenance log that used to live in `CLAUDE.md`. The
curated experiment summaries are now the source of truth; the legacy
`autoresearch/.../SUMMARY.md` files remain historical reference.

1. Smooth VDP activation search.
   Softplus is the best sparse compromise on the smooth VDP value-sample
   benchmark; Matern 5/2 and Gaussian are more accurate on H1 but use many
   more neurons, while tanh remains the weakest gradient-fitting choice.
   Current curated readouts: [01_vdp/log_penalty/results.md](01_vdp/log_penalty/results.md)
   and [02_pendulum/log_penalty/results.md](02_pendulum/log_penalty/results.md)

2. Finite-step penalty powers.
   The power sweep is non-monotone. Powers around 3-4 form the useful tradeoff
   region; `p=5` stops improving the H1/sparsity balance and is the first clear
   degradation case.
   Current curated readouts: [01_vdp/frac_exp_penalty/results.md](01_vdp/frac_exp_penalty/results.md)
   and [02_pendulum/frac_exp_penalty/results.md](02_pendulum/frac_exp_penalty/results.md)

3. Discontinuous-gradient activation search.
   On the analytic discontinuous-gradient study, the best near-jump behavior
   comes from leaky squared-ReLU / squared-ReLU families with spherical
   parameterization. They beat smooth activations in near-discontinuity error
   and preserve the expected near/far localization pattern.
   Legacy summary: [docs/research/D3_harmonic_analysis/refs/legacy-analytical-search.md](../docs/research/D3_harmonic_analysis/refs/legacy-analytical-search.md)

4. Semiconcave versus signed comparison.
   Semiconcavity-aware modeling is competitive on the VDP reference data, but
   it is not a universal win across problems. The pendulum comparison is mixed:
   one dataset favors the signed model, another favors the semiconcave model by
   score, and the later semiconcave-labeled rerun is split again. This belongs
   in the curated research log because the conclusion is about model choice, not
   a single benchmark score.
   Legacy summary: [docs/research/D4_max_plus/refs/legacy-semiconcave-comparison.md](../docs/research/D4_max_plus/refs/legacy-semiconcave-comparison.md)

5. Paper-conforming algorithms (2026-08-12).
   The manuscript's revised Sections 3–5 changed the objective and both
   insertion algorithms: the moment norm moved from an additive `β` term into
   the penalty argument as `φ(w_p(ω)|c|)`, the insertion condition normalizes
   the derivative representative by `w_p`, the accepted atom gets the theorem's
   own coefficient, the correction is rejected when it raises the objective, and
   the loop inserts before correcting. These re-runs sit beside the preserved
   comparators rather than replacing them, so the algorithm change can be read
   off run for run.

   At a **matched neuron budget** — the comparison that is free of both the α
   shift and any capacity difference — sequential insertion under the revised
   algorithm is the best of the three at nearly every budget on both benchmarks,
   substantially so on the pendulum (relative H¹ 0.2271 against the comparator's
   0.4250 at ≤160 neurons). The exception is VDP at ≤20 neurons, where the
   comparator still wins. Both readings are conservative: the comparator draws on
   2–6× more runs.

   Do **not** compare the two studies at equal α. The empirical fidelity now
   divides by `M` rather than `M·d`, so at the same α label a paper run carries
   `1/d` the effective regularization and buys more neurons; a per-α table reads
   that head start as an improvement. The `results.md` files lead with the
   matched-budget table for this reason.
   Current curated readouts:
   [01_vdp/paper_log_penalty/results.md](01_vdp/paper_log_penalty/results.md),
   [02_pendulum/paper_log_penalty/results.md](02_pendulum/paper_log_penalty/results.md),
   [01_vdp/paper_frac_exp_penalty/results.md](01_vdp/paper_frac_exp_penalty/results.md)
   and [02_pendulum/paper_frac_exp_penalty/results.md](02_pendulum/paper_frac_exp_penalty/results.md).
   Decisions behind them: [docs/adr/0007](../docs/adr/0007-per-neuron-curvature-in-the-insertion-step.md),
   [docs/adr/0008](../docs/adr/0008-sequential-insertion-budget.md), and the
   amendment to [docs/adr/0006](../docs/adr/0006-retain-radial-search-clamp.md).

## Migration Note

The old `autoresearch` summaries were consolidated into the curated experiment
tree and the Markdown experiment readouts; the two still-cited legacy summaries
(directions 3–4) were migrated verbatim into `docs/research/*/refs/` and the
remaining `autoresearch/` tree was archived under `outdated/` (2026-07-02).
New work should extend the curated experiment paths.
