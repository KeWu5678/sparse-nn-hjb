# Moment penalty — Van der Pol

This is the isolated record and analysis home for the unbounded,
nonhomogeneous moment-penalty rerun. It does not replace
`../log_penalty`; those records remain the β=0 historical experiment.

The open-loop data are reused unchanged from
`VDP_beta_0.1_grid_30x30.npy`. The legacy filename uses `beta` for the
control-energy coefficient; in the paper that coefficient is now denoted
η=0.1. The β in this experiment is exclusively the moment weight.

The configs target the `model.moment_beta` and `model.moment_order` fields
introduced by the moment-penalty implementation.

## First pass

Run:

```sh
make moment-sweep EXPERIMENT=vdp/moment_penalty
```

This creates two disjoint record stages:

- `rawdata/logs/multirun/vdp/moment_penalty/baseline`: β=0, evaluated once at
  p=2.01 because p has no effect when β=0.
- `rawdata/logs/multirun/vdp/moment_penalty/screen`: the positive-β Cartesian
  grid.

The seed-42 screen fixes `signed + profile`, γ=1, and H1 weights `[1,1]`.
Its axes are:

| axis | values |
|---|---|
| activation | `tanh`, `softplus`, `gaussian`, `gelu_squared` |
| α | `1e-2`, `1e-3`, `1e-4`, `1e-5` |
| β | `0`, `1e-10`, `1e-5`, `1e-2`, `1e-1` |
| p | `2.01`, `2.5`, `3`, `4` |

Inspect α and β as independent plot axes. The ratio β/α is a derived
diagnostic, not the experimental coordinate. Add intermediate β decades only
where a row or column leaves a transition unresolved.

## Follow-up

Run the adaptive beta refinement with:

```sh
make moment-refine EXPERIMENT=vdp/moment_penalty
```

It writes only to `rawdata/logs/multirun/vdp/moment_penalty/refine`. The target
fills `1e-9` through `1e-6` on the selected tanh and softplus rows, fills
`1e-4` and `1e-3` on the selected Gaussian and squared-GELU rows, and adds
`matern52` at α in `{1e-4, 1e-5}`. It refuses to run when that stage already
contains records, protecting the completed baseline and screen.

The selected γ/loss comparison is reproducible with:

```sh
make moment-followup EXPERIMENT=vdp/moment_penalty
```

It reuses each selected γ=1, `[1,1]` anchor and writes the other 35 records to
the isolated `followup` stage. It sweeps γ in `{0, 0.1, 1, 10}` and compares
loss weights `[1,0]` and `[1,1]`. `gausscent_1` is retired from this study.
Record validation H1 error, active support, amplitude-mass-weighted `R95`,
`Phi_1`, `Psi_p`, and the full objective decomposition.

## Full paper scope

After the screen, refinement, and follow-up records exist, regenerate the
original downstream evidence scope with:

```sh
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \
  experiments/01_vdp/moment_penalty/full_scope.py
```

This uses softplus, tanh, and Gaussian as the three representative
nonhomogeneous activations. It produces their gradient-training table,
activation/surface/derivative diagnostics, insertion history, and
Algorithm 1 feedback figure and table. It then reuses the unchanged
homogeneous checkpoints for Algorithm 2 and the final cross-model figures.
The generated report is `full_scope.md`.
