# Experiment workflow

Current datasets were generated on September 24, 2026. Their paths are selected
by `conf/data/*.yaml`; training settings and sweep axes live in
`conf/experiment/*.yaml`. All 588 configured fits and the 43 manuscript figures
have been regenerated. Numerical conclusions in the current paper use these runs.

Datasets, run records, fits, reports, and figures are local, ignored artifacts.
The manuscript source and compiled PDF are tracked under `paper/`.

## Generate and configure the data

| Input | Current location under `rawdata/data/` | Generator |
| --- | --- | --- |
| Van der Pol | `VDP_20260924_62305c7ac2534952a7c85d6c4656a491/` | `scripts/generate_vdp_data.py` |
| Pendulum production data, raw paths, and switching curve | `Pendulum_20260924_ebf42e75ab9748aa855f44aa2387b6df/` | `scripts/run_pendulum_pmp_openloop_example.py` |
| Pendulum sampling variants | `Pendulum_2sided_oversample_20260924/` | `scripts/investigation/make_twosided_oversampling_sets.py` |

The [VDP](00_openloop/vdp/README.md) and
[pendulum](00_openloop/pendulum/README.md) guides describe the objectives,
generation commands, and reference-data plots. A new generation creates a new
run directory. Update the data configurations to those exact artifacts before
training; changing a date in a filename does not regenerate data.

For pendulum, build the four sampling variants, their distance caches, and one
shared evaluation pool before training. Exclude the union of the production
set and all four variants from that pool. `conf/eval/region_split.yaml` selects
the common pool; each data configuration selects its own aligned distance
cache. The switching tube is distance at most 0.3 from the tiled switching-set
points.

## Train the current grids

The six presets currently define 588 experiment cells:

| Preset | VDP runs | Pendulum runs, including variants |
| --- | ---: | ---: |
| `log_penalty` | 224 | 224 |
| `frac_exp_penalty` | 16 | 16 |
| `moment_order_study` | 32 | 32 |
| `relu_l1_baseline` | 10 | 10 |
| `oversampling_gaussian` | 0 | 12 |
| `oversampling_relu2` | 0 | 12 |

Start the local [MLflow service](../deploy/README.md), then run each preset once:

```sh
make sweep EXPERIMENT=log_penalty
make sweep EXPERIMENT=frac_exp_penalty
make sweep EXPERIMENT=moment_order_study
make sweep EXPERIMENT=relu_l1_baseline
make sweep EXPERIMENT=oversampling_gaussian
make sweep EXPERIMENT=oversampling_relu2
```

`make sweep` refuses to run when that experiment already has records under
`rawdata/logs/multirun/`. Existing artifacts must be deliberately archived
before a replacement sweep. Records are stored at
`rawdata/logs/multirun/<dataset>/<experiment>/<axis>/<job>/`, with a run-adjacent
`result_<run_id>.pkl` fit. Every run uses seed 42 and a 90% training split.
The runner records training-data and evaluation-input SHA-256 hashes before
fitting and reports errors in physical coordinates.

Algorithms 1 and 2, the moment-order study, and the oversampling studies use
150 outer iterations with sequential insertion. The traditional ReLU–L1
baseline has its own correction-first, batch-insertion settings; its preset
must not be substituted with Algorithm 2 at power one.

## Validate and generate the manuscript figures

The following paper scripts are local/ignored and require the datasets,
completed records, and fit files above:

```sh
uv run python scripts/paper/preflight.py
make openloop
uv run python experiments/01_vdp/paper_log_penalty/p_study_figure.py
uv run python scripts/paper/vdp_full_scope.py --homogeneous-alpha 1e-6
uv run python scripts/paper/pendulum_full_scope.py
uv run python scripts/paper/preflight.py --require-figures --write-manifest
```

Preflight derives the expected cells from the current Hydra presets. It rejects
missing or duplicate cells, obsolete dataset paths, mismatched recorded input
hashes, incomplete runs, missing fit files, and incompatible training settings.
It requires physical-coordinate metrics and the configured shared pendulum
pool. A partial benchmark check is available with `--problem vdp` or
`--problem pendulum`; publication manifest creation requires both benchmarks.

The final command writes `scripts/paper/current_run_manifest.json`, recording
SHA-256 hashes for records, fit files, datasets, evaluation inputs, and included
figures. The figure allowlist is derived from `paper/paper_0805.tex`; missing
included PNGs and unreferenced PNGs in `experiments/` or `paper/plot/` fail the
check. Provenance checks do not establish numerical correctness by themselves.

The two open-loop plotters write five reference PNGs to `paper/plot/`.
`vdp_full_scope.py` and `pendulum_full_scope.py` write the learned-model figures
and reports under their benchmark's `paper_log_penalty/` and
`paper_frac_exp_penalty/` output directories. Those names identify report
locations; the training record roots are `log_penalty` and `frac_exp_penalty`.
The local `analysis.py` wrappers delegate to the same benchmark generators.
The joint moment-order plot reads the two `moment_order_study` record roots.

Signed-control and cumulative-cost comparisons use each controller's own
trajectory: VDP has horizon 3; pendulum has horizon 10 for both starts. The
separate VDP stabilization diagnostic has horizon 12. Rollout arrays and final
costs are preserved beside the reports as NPZ and CSV files. See
[plotting and evaluation](../docs/plotting.md) for the numerical and rendering
contracts.

## Archive boundary

Superseded datasets, run records, reports, figures, obsolete pipeline files,
and the old `00_openloop/DIAGNOSIS.md` are preserved under the ignored
`outdated/experiment-refresh-20260924/` directory. Its JSON manifests record
original paths and SHA-256 hashes. Retired report directories are no longer
active paths under `experiments/`, and the current generators do not read the
archive. Archived artifacts are not distributed with a clean checkout.
