# Research Directions

Current paper run records, generated reports, and figures are local artifacts
and are not version-controlled. Their Hydra definitions remain under `conf/`,
while `paper/paper_0805.tex` and its compiled PDF are the publication record.
The entries below describe the research directions; links are retained only
for older curated studies already present in the repository.

1. Normalized-measure activation search.
   Algorithm 1 compares nonhomogeneous activations under the normalized-moment
   objective, using the joint candidate search and guarded coefficient
   correction described in the manuscript.

2. Finite-step fractional penalties.
   Algorithm 2 currently supports `k=2,3`, hence `q=2/3,1/2`, plus the separate
   `k=1` ReLU--L1 endpoint. Insertion minimizes the actual one-atom increment
   through the selected global scalar prox; the correction uses the same global
   prox with a warm-start-derived fixed scale.

3. Discontinuous-gradient activation search.
   On the analytic discontinuous-gradient study, the best near-jump behavior
   comes from leaky squared-ReLU / squared-ReLU families with spherical
   parameterization. They beat smooth activations in near-discontinuity error
   and preserve the expected near/far localization pattern.
   Legacy summary: [docs/research/D3_harmonic_analysis/refs/legacy-analytical-search.md](../docs/research/D3_harmonic_analysis/refs/legacy-analytical-search.md)

4. Archived semiconcave-versus-signed comparison.
   This historical study found no consistent advantage from the semiconcave
   parametrization. The implementation was retired by ADR 0012 because it is
   unused by the manuscript and current experiments; Git history preserves it.
   Legacy summary: [docs/research/D4_max_plus/refs/legacy-semiconcave-comparison.md](../docs/research/D4_max_plus/refs/legacy-semiconcave-comparison.md)

## Migration Note

The old `autoresearch` summaries were consolidated into the curated experiment
tree and the Markdown experiment readouts; the two still-cited legacy summaries
(directions 3–4) were migrated verbatim into `docs/research/*/refs/` and the
remaining `autoresearch/` tree was archived under `outdated/` (2026-07-02).
New work should extend the curated experiment paths.
