# Open-loop data — pendulum swing-up

Visualisations of the pendulum swing-up open-loop value data and the switching-set
geometry it traces out (backward-PMP, `src/OpenLoop/pendulum`). These follow the
paper's own figure code (Han & Yang, arXiv:2312.17467, `github_main_nosat.m`) and show
the **data only** — no learned model. The dataset path is read from
`conf/data/pendulum.yaml` (currently the 2000-path, 3000-sample set under
`rawdata/data/Pendulum_20260630_.../`); the trajectory figures use the co-located
raw-trajectory pickle (2000 ordered paths). Regenerate with `python generate.py` (or
`make openloop`). Figures carry no titles.

| file | what it shows |
| --- | --- |
| `figures/value_scatter.png` | 3D scatter of the raw samples (θ, θ̇, V), coloured by value |
| `figures/value_surface.png` | V(θ, θ̇) over the state plane (the 3000-sample set, periodic-folded into one cell then tiled) |
| `figures/trajectories.png` | every backward-PMP characteristic plotted whole (paper's `plot(θ, θ̇)`), multicolored — the curves spiral into the centers (paper Fig. 2, right) |
| `figures/regions_of_attraction.png` | each state coloured by the upright it belongs to (nearest basin-cut characteristic, tiled by 2πk); the boundaries are the switching-set spirals winding around the hanging points ±π, ±3π (paper Fig. 2, left) |

The regions figure tracks the switching set deeper (`_REGIONS_CAP = 80`) than the wired
training data (`basin_value_max = 50`, which only resolves ~half a spiral turn); the deeper
cut recovers the multi-winding spiral. This is for visualisation only — it does not change
the training samples.
