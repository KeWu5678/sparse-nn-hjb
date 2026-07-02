# Open-loop data — Van der Pol

Visualisations of the Van der Pol open-loop value/gradient training data
(`rawdata/data/VDP_beta_0.1_grid_30x30.npy`, a 30×30 grid over the state plane,
β = 0.1). These show the **data only** — no learned model. Regenerate with
`python generate.py` (or `make openloop`). Figures carry no titles.

| file | what it shows |
| --- | --- |
| `figures/value_scatter.png` | 3D scatter of the samples (x[0], x[1], V) coloured by value |
| `figures/value_gradient.png` | state-plane scatter coloured by V, with ∇V arrows on a grid |
| `figures/value_surface.png` | V(x) cubic-interpolated to a smooth surface |
