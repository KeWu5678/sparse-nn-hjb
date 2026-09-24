# Open-loop data — Van der Pol

The configured dataset is
`rawdata/data/VDP_20260924_62305c7ac2534952a7c85d6c4656a491/VDP_pmp_grid_30x30_20260924.npz`.
`conf/data/vdp.yaml` selects this file for training and reference-data plots.
It contains 900 value/gradient samples on the 30-by-30 grid over `[-3,3]^2`.

## Numerical problem and generation

The objective is the integral of `y1² + y2² + 0.1 u²` over `0 <= t <= 3`,
with a fixed initial state, free terminal state, and zero terminal cost.
`scripts/generate_vdp_data.py` uses the current PMP boundary-value solver with
151 initial mesh points. For each state it retains the least-cost converged
stationary solution from two initial guesses and saves the initial costate as
the value gradient. These are numerical references, without a claim of global
optimality. Generation fails rather than saving an incomplete grid.

From the repository root:

```sh
uv run python scripts/generate_vdp_data.py --grid-size 30 --tag 20260924
```

A new dated run directory contains the sample NPZ, solver metadata, and failure
record. The metadata includes the data hash, generator source hashes, and
collocation residuals. A new generation receives its own directory identifier;
point `conf/data/vdp.yaml` at that exact output before starting a replacement
sweep. The superseded dataset and old results are in the ignored
`outdated/experiment-refresh-20260924/` archive.

## Plot the configured samples

```sh
uv run python experiments/00_openloop/vdp/generate.py
```

`make openloop` also runs this plotter and the pendulum plotter. It renders
existing configured data; it does not solve the control problem or retrain.
Only these manuscript PNGs are written, directly under `paper/plot/`:

| File | Content |
| --- | --- |
| `v.png` | State/value scatter coloured by value |
| `dv.png` | State-plane samples coloured by value, with gradient arrows |

These are data plots. Learned control/cost comparisons are generated separately
by `scripts/paper/vdp_full_scope.py` and use the same benchmark horizon `T=3`.
The longer `T=12` stabilization diagnostic is separate. See the
[experiment workflow](../../README.md) for run and figure validation.
