# Open-loop data — Van der Pol

Visualisations of the Van der Pol open-loop value/gradient training data
(`rawdata/data/VDP_beta_0.1_grid_30x30.npy`, a 30×30 grid over the state plane,
β = 0.1). These show the **data only** — no learned model. Regenerate with
`python generate.py` from this directory (or `make openloop` from the repository
root). Only the two TeX-included PNGs are generated, directly in `paper/plot/`.
Figures carry no titles.

| file | what it shows |
| --- | --- |
| `paper/plot/v.png` | 3D scatter of the samples (x[0], x[1], V) coloured by value |
| `paper/plot/dv.png` | state-plane scatter coloured by V, with ∇V arrows on a grid |

These generated files are local; the distributed figures are embedded in the
[tracked paper](../../../paper/paper_0805.pdf). The unused interpolated-surface
PNG and old experiment-directory copies are archived locally under
`outdated/experiment-code-cleanup-2026-09-12/unreferenced_pngs/` and are not
distributed or regenerated. The optional local pipeline notes in
`scripts/paper/README.md` describe the final allowlist/provenance check.
