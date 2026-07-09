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

## Migration Note

The old `autoresearch` summaries were consolidated into the curated experiment
tree and the Markdown experiment readouts; the two still-cited legacy summaries
(directions 3–4) were migrated verbatim into `docs/research/*/refs/` and the
remaining `autoresearch/` tree was archived under `outdated/` (2026-07-02).
New work should extend the curated experiment paths.
