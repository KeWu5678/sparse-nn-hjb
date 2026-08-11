# Moment penalty — pendulum swing-up

This directory is the isolated record and analysis home for the
nonhomogeneous moment-penalty rerun. It preserves `../log_penalty` and all
homogeneous `../frac_exp_penalty` records.

The existing two-sided switching-band open-loop data selected by
`data=pendulum` are reused unchanged; their control-energy coefficient is
denoted by `r` and equals 1. The β in this experiment denotes only the
parameter-moment weight.
The configs target the `model.moment_beta` and `model.moment_order` fields
introduced by the moment-penalty implementation.

## First pass

Run:

```sh
make moment-sweep EXPERIMENT=pendulum/moment_penalty
```

Records are separated into:

- `rawdata/logs/multirun/pendulum/moment_penalty/baseline`: one β=0 run per
  non-p configuration, represented at p=2.01.
- `rawdata/logs/multirun/pendulum/moment_penalty/screen`: all positive-β runs.

The seed-42 screen fixes `signed + profile`, γ=1, and H1 weights `[1,1]`.

| axis | values |
|---|---|
| activation | `tanh`, `softplus`, `gaussian`, `gelu_squared` |
| α | `1e-2`, `1e-3`, `1e-4`, `1e-5` |
| β | `0`, `1e-10`, `1e-5`, `1e-2`, `1e-1` |
| p | `2.01`, `2.5`, `3`, `4` |

Treat α and β as independent axes and inspect fixed rows and columns; use
β/α only as a derived diagnostic. Refine unresolved transitions with
intermediate β decades.

## Follow-up

Run the adaptive beta refinement with:

```sh
make moment-refine EXPERIMENT=pendulum/moment_penalty
```

It writes only to `rawdata/logs/multirun/pendulum/moment_penalty/refine`. The
target fills `1e-9` through `1e-6` on the selected softplus and squared-GELU
rows, fills `1e-4` and `1e-3` on the selected tanh and Gaussian rows, and adds
`matern52` at α in `{1e-4, 1e-5}`. It refuses to run when that stage already
contains records.

The selected γ/loss comparison is reproducible with:

```sh
make moment-followup EXPERIMENT=pendulum/moment_penalty
```

It reuses each selected γ=1, `[1,1]` anchor and writes the other 35 records to
the isolated `followup` stage. It sweeps γ in `{0, 0.1, 1, 10}` and compares
`[1,0]` against `[1,1]`. `gausscent_1` is excluded. Rank configurations using
a Pareto view of region-aware validation H1 error, active support, and
amplitude-mass-weighted `R95`, while recording `Phi_1`, `Psi_p`, and the
complete objective decomposition.

## Matched control: beta at a fixed operating point

The representatives above are selected on their own grid, so they differ from
the earlier log-penalty softplus fit in `alpha`, `gamma`, and `beta` at once —
which means the feedback comparison cannot attribute anything to the moment
term. The control stage holds the earlier study's operating point fixed
(`softplus`, alpha=1e-5, gamma=10, p=2.01, H1 loss) and varies only beta:

```sh
OMP_NUM_THREADS=1 .venv/bin/python scripts/train.py -m \
  +experiment=pendulum/moment_penalty \
  hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  hydra.sweep.dir=rawdata/logs/multirun/pendulum/moment_penalty/control \
  env.verbose=false env.seed=42 \
  model.activation=softplus model.alpha=1e-5 model.gamma=10 \
  model.loss_weights='[1.0,1.0]' \
  model.moment_beta=0,1e-10,1e-5,1e-2 model.moment_order=2.01
```

Then `MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python
experiments/02_pendulum/moment_penalty/control.py`, which rolls all four out
from A and B under the full-scope protocol and writes `control.md`. The
beta=0 row reproduces the earlier fit exactly (77 neurons, rel H1 0.515), so
the rows differ in the moment weight alone.

## Full paper scope

The original four-dataset oversampling control requires a positive-moment
Gaussian arm. Run it once with:

```sh
make moment-oversampling EXPERIMENT=pendulum/moment_penalty JOBS=1
```

This creates 12 fits: four sampling variants times
`alpha ∈ {1e-3, 1e-4, 1e-5}`, with
`beta=1e-4`, `p=2.01`, and `gamma=0`. The homogeneous ReLU² arm is reused
unchanged.

Then regenerate the full model-wise scope with:

```sh
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \
  experiments/02_pendulum/moment_penalty/full_scope.py
```

The report tests Algorithm 1 independently with softplus, tanh, and Gaussian,
including its switching-set feedback figure and table. It tests Algorithm 2
independently with ReLU powers 2, 3, and 5, including its feedback figure and
table. Cross-model transects, frontiers, regional errors, atom portraits, and
the common-pool oversampling figure/table are generated only afterward. The
generated report is `full_scope.md`.
