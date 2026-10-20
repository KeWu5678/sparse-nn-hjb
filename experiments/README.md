# Running experiments

An experiment is a preset in `conf/experiment/`: one algorithm, its settings,
and a grid of hyperparameters and datasets. Running a preset trains every cell
of its grid and records each fit in MLflow.

## 1. Prepare

```sh
scripts/init.sh
```

This installs the Python environment with `uv` and starts the local MLflow
server in Docker; Docker Desktop must be running. It is safe to repeat. Nothing
trains without MLflow.

## 2. Run a preset

```sh
make sweep EXPERIMENT=log_penalty
```

`make help` lists the presets. Optional: `JOBS=8` parallel jobs,
`VERBOSE=true` to stream progress.

| Preset | Runs | Study |
| --- | ---: | --- |
| `log_penalty` | 448 | Algorithm 1, VDP and pendulum |
| `frac_exp_penalty` | 32 | Algorithm 2, VDP and pendulum |
| `moment_order_study` | 64 | Algorithm 1 moment order, VDP and pendulum |
| `relu_l1_baseline` | 20 | traditional ReLU–L1 comparator, VDP and pendulum |
| `oversampling_gaussian` | 12 | pendulum sampling variants |
| `oversampling_relu2` | 12 | pendulum sampling variants |
| `higher_dim_*` | 500 | [Allen–Cahn at 6 and 10 dimensions](03_high_dimensional/README.md) |

A preset reads its datasets from `rawdata/data/`, at the paths in
`conf/data/*.yaml`. To generate a missing dataset, follow the open-loop guide
for [VDP](00_openloop/vdp/README.md),
[pendulum](00_openloop/pendulum/README.md), or
[Allen–Cahn](00_openloop/allen_cahn/README.md).

## 3. Read the results

Open the dashboard at <http://127.0.0.1:5000>. Each job is one MLflow run with
its configuration, dataset hashes, errors in physical coordinates, and the fit
as `history.pt`. Every run uses seed 42 and a 90/10 training/validation split.
Hydra configurations and logs are kept under
`rawdata/logs/multirun/<dataset>/<experiment>/<axis>/<job>/`.

To run a preset again, first move its folders out of `rawdata/logs/multirun/`;
`make sweep` refuses to mix new runs with existing ones.

## Analysis and paper figures

Analysis is local and git-ignored. Superseded datasets, runs, and reports are
kept under the ignored
`outdated/experiment-refresh-20260924/`.
