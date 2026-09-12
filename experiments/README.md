# Research Directions

Current paper run records, generated reports, and figures are local artifacts
and are not version-controlled. Their Hydra definitions remain under `conf/`,
while `paper/paper_0805.tex` and its compiled PDF are the publication record.
The active plotting pipeline is restricted to the PNGs included by that TeX source;
historical reports remain available separately from their archived images.

## Current executable paths

The cleanup on 2026-09-12 audited all 26 Python/shell source files, including
ignored local scripts. Seven entry points remain:

| Path | Responsibility |
| --- | --- |
| `00_openloop/{vdp,pendulum}/generate.py` | Five reference PNGs directly in `paper/plot/`; also called by `make openloop` |
| `{01_vdp,02_pendulum}/paper_log_penalty/analysis.py` | Thin entry points for the benchmark's shared full-scope generator |
| `{01_vdp,02_pendulum}/paper_frac_exp_penalty/analysis.py` | The same full-scope generators; both algorithm families share each benchmark's pipeline |
| `01_vdp/paper_log_penalty/p_study_figure.py` | Joint moment-order support statistics and the single included `p_study.png` |

The neutral VDP and pendulum generators emit 13 and 17 PNGs, respectively.
Together with the five reference images and one p-study image, these are exactly
the manuscript's 36 PNGs. Neither the study wrappers nor `scripts/paper/` generate
surplus diagnostic plots. Both VDP wrappers use alpha=1e-6 for the frontier/feedback
comparison, while the separate homogeneous surface/table comparison stays at 1e-5.

After regeneration, check both the provenance and TeX-derived image allowlist:

```sh
uv run python scripts/paper/preflight.py --require-sidecars --require-figures
```

The figure check rejects missing included images and unreferenced PNGs in
`experiments/` and `paper/plot/`; it does not certify the reference calculations.

Saved-run reconstruction/evaluation belongs to `src/results.py`; rendering and
export belong to `src/plots.py`. Paper-specific selection and feedback logic live
in the local `scripts/paper/` workspace. See [plotting](../docs/plotting.md) and
the [current-paper commands](../scripts/paper/README.md). Those local paper paths
require the original datasets, records and sidecars; they are not a clean-checkout
reproduction guarantee.

Training uses only the shared `conf/experiment/{log_penalty,frac_exp_penalty,
relu_l1_baseline}.yaml` presets through `make sweep EXPERIMENT=<name>`. Dataset-
prefixed presets and the former study-specific Make targets no longer exist.
New generic sweeps do not refresh the separately reviewed paper record trees.

## Historical artifacts and source recovery

The `baseline`, `log_penalty`, `frac_exp_penalty`, `moment_penalty`, VDP `summary`
and pendulum `region_split` directories retain their historical reports. Their
unreferenced PNGs are archived; report image links point there where the image
was available. These directories are not current regeneration entry points.

Nineteen obsolete source files (5,709 lines) were moved, without changing their
bytes, to the local archive
[`outdated/experiment-code-cleanup-2026-09-12/`](../outdated/experiment-code-cleanup-2026-09-12/).
It contains the original paths, SHA-256 manifest, old READMEs and stale bytecode.
The fourteen additive-moment files depend on the retired objective/configuration;
the other five are historical analyzers with obsolete metric/solver contracts,
including two identical baseline analyzers. No current-paper executable depends
on them. Reproduction of retired studies requires the historical implementation,
not restoring compatibility branches to current code.

The subsequent paper-only trim archived `scripts/paper/log_penalty_analysis.py`
and preserved copies of the narrowed generators and shared renderer under
`paper_pipeline_before/`. It also moved 170 unreferenced PNGs unchanged to
`unreferenced_pngs/`, preserving repository-relative paths. Verify the images with
`shasum -a 256 -c PNG_SHA256SUMS` from that archive subdirectory. The 36 included
PNGs, datasets, run records, and historical report numbers were preserved;
historical report changes are limited to image/reference paths.

Figure 12's known reference-selection defect is still a separate scientific
correction. Its retained PNGs were not silently changed by this cleanup.

## Research directions

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
