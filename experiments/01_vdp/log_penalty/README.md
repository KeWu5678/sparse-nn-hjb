# Historical log_penalty — Van der Pol

The old unweighted objective is the retired moment_beta=0 special case described
in [ADR 0010](../../../docs/adr/0010-use-only-the-normalized-algorithm-1-objective.md).
These historical results must not be relabeled as normalized-measure Algorithm 1.

Historical report numbers are unchanged. Their stored relative errors use
normalized training coordinates, not the current paper's original-coordinate
metric, and must not be used as current comparisons. Figures are archived
locally and are not distributed with this repository.

The local
analyzer was already absent at cleanup; its executable source is not in this
archive. The previous README is preserved under
`outdated/experiment-code-cleanup-2026-09-12/experiments/01_vdp/log_penalty/`
(local only, not distributed).
This archive is for recovery, not an alternative executable pipeline.

For current results, see the [tracked paper](../../../paper/paper_0805.pdf).
Shared training presets are in `conf/experiment/`; see the
[repository commands](../../../README.md#reproduce-it).
