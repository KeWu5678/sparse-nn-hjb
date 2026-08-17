# Research Directions

This note restores the provenance log that used to live in `CLAUDE.md`. The
curated experiment summaries are now the source of truth; the legacy
`autoresearch/.../SUMMARY.md` files remain historical reference.

1. Normalized-measure activation search.
   Algorithm 1 compares nonhomogeneous activations under the normalized-moment
   objective, using the joint candidate search and guarded coefficient
   correction described in the manuscript.
   Current readouts: [01_vdp/paper_log_penalty/results.md](01_vdp/paper_log_penalty/results.md)
   and [02_pendulum/paper_log_penalty/results.md](02_pendulum/paper_log_penalty/results.md).

2. Finite-step fractional penalties.
   Algorithm 2 currently supports `k=2,3`, hence `q=2/3,1/2`, plus the separate
   `k=1` ReLU--L1 endpoint. Insertion minimizes the actual one-atom increment
   through the selected global scalar prox; the correction uses the same global
   prox with a warm-start-derived fixed scale.
   Current readouts: [01_vdp/paper_frac_exp_penalty/results.md](01_vdp/paper_frac_exp_penalty/results.md)
   and [02_pendulum/paper_frac_exp_penalty/results.md](02_pendulum/paper_frac_exp_penalty/results.md).

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

5. Algorithm 1 studies (2026-08-13).
   The nonhomogeneous studies minimize
   `l^M + alpha * sum phi_gamma(w_p(omega) * |c|)`. Candidate locations are
   obtained by joint multistart optimization of the normalized profile over the
   full parameter vector. Starts are sampled within the certified search radius;
   final points outside it are discarded, Euclidean near-duplicates are removed,
   and accepted atoms receive the candidate-specific warm coefficient before the
   guarded semismooth Newton correction. The main studies report both batch and
   sequential insertion, while the manuscript uses the sequential method covered
   by the one-candidate decrease argument.

   The dedicated moment-order study measures support confinement, and the radius
   ablation isolates the effect of the computable theorem radius from the fixed
   numerical cap. The pendulum study additionally reports errors near and away
   from the switching set and tests whether denser sampling there changes the
   gradient-approximation floor.
   Current curated readouts:
   [01_vdp/paper_log_penalty/results.md](01_vdp/paper_log_penalty/results.md),
   [02_pendulum/paper_log_penalty/results.md](02_pendulum/paper_log_penalty/results.md),
   [01_vdp/paper_frac_exp_penalty/results.md](01_vdp/paper_frac_exp_penalty/results.md)
   and [02_pendulum/paper_frac_exp_penalty/results.md](02_pendulum/paper_frac_exp_penalty/results.md).
   Decisions behind them: [docs/adr/0007](../docs/adr/0007-per-neuron-curvature-in-the-insertion-step.md),
   [docs/adr/0008](../docs/adr/0008-sequential-insertion-budget.md),
   [docs/adr/0009](../docs/adr/0009-use-verified-closed-form-global-proximal-maps.md),
   and the amendment to
   [docs/adr/0006](../docs/adr/0006-retain-radial-search-clamp.md).

## Migration Note

The old `autoresearch` summaries were consolidated into the curated experiment
tree and the Markdown experiment readouts; the two still-cited legacy summaries
(directions 3–4) were migrated verbatim into `docs/research/*/refs/` and the
remaining `autoresearch/` tree was archived under `outdated/` (2026-07-02).
New work should extend the curated experiment paths.
