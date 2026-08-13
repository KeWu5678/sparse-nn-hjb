# Current-paper artifact pipeline

`make paper-artifacts` runs `scripts/regenerate_algorithm1_paper_artifacts.sh`.
The pipeline uses tracked, paper-specific generators:

1. `preflight.py` validates the exact current Algorithm 1/2 record grids,
   datasets, fit histories, search settings, and reviewed record-tree digests.
2. `rescore_regions.py` creates physical-coordinate region sidecars for the two
   production pendulum roots. A second preflight requires the complete sidecar
   set before any paper figure or report is written.
3. The current study analyses and the neutral generators in this directory
   write the figures and Markdown reports referenced by the manuscript.

Oversampling models are compared on one production region-evaluation pool after
removing the union of all variant training rows. They do not use per-variant
region sidecars.

From an empty record tree, reproduce the complete input set with:

```sh
make paper-sweep EXPERIMENT=vdp/paper_log_penalty
make paper-sweep EXPERIMENT=pendulum/paper_log_penalty
make paper-sweep EXPERIMENT=vdp/paper_frac_exp_penalty
make paper-sweep EXPERIMENT=pendulum/paper_frac_exp_penalty
make paper-l1-study
make paper-p-study EXPERIMENT=vdp/paper_log_penalty
make paper-p-study EXPERIMENT=pendulum/paper_log_penalty
make paper-radius-study
make paper-oversampling-study
.venv/bin/python scripts/paper/preflight.py --write-provenance
make paper-artifacts
```

The provenance command is intentionally separate: run it only after reviewing a
fresh Algorithm 1 sweep. It binds every paper-facing Algorithm 1 JSON record to
the manifest by path and SHA-256 digest.
