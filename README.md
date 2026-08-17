# Sparse neural networks for optimal feedback control

[![CI](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml/badge.svg)](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml)

The current mathematical formulation and numerical study are in
[`paper/paper_0805.tex`](paper/paper_0805.tex) and the tracked
[`paper/paper_0805.pdf`](paper/paper_0805.pdf).

## The problem

Optimal feedback control has a classical answer: solve the
Hamilton–Jacobi–Bellman (HJB) equation for the value function $V(x)$, and the
optimal controller falls out as a function of its gradient, e.g.
\(\hat u(x)=-\partial_{x_2}\hat V(x)/(2\eta)\). The catch is that \(V\) is expensive to compute globally —
and if you learn it from data instead, the controller quality depends on
$\nabla \hat V$, not on $\hat V$. A model with excellent mean-squared fit and a mediocre
gradient field produces a controller that oscillates, saturates, or diverges.

This repository learns $V$ from open-loop trajectory data (value + gradient
samples along optimal trajectories, generated via Pontryagin's principle) with
shallow networks $\sum_k c_k \sigma(a_k \cdot x + b_k)$, fitted in Sobolev ($H^1$) loss so the
gradient is a first-class training target. Sparsity is not post-hoc pruning:
neurons are inserted greedily (a Primal-Dual Active Point method, PDAP, over
the measure-space relaxation of the network) and selected by **non-convex
penalties** — log penalty $\varphi_\gamma$ and fractional powers $|c|^q$ — that, unlike $\ell^1$,
detect and merge redundant neurons clustering in the same direction. The
resulting outer-weight problem is nonsmooth and non-convex. It is corrected
with a guarded **semismooth Newton normal-map method implemented as a native
PyTorch optimizer**. For the fractional penalties used in the paper, the
scalar global proximal maps are evaluated in closed form and the normal map is
scaled from the insertion warm start. The correction is retained only when it
does not increase the objective.

## Numerical studies

The paper compares the normalized nonhomogeneous formulation on the Van der
Pol problem with the positively homogeneous formulation on both Van der Pol
and pendulum swing-up data. Run records, derived reports, and figures remain
local; the manuscript and compiled PDF are the version-controlled publication
record.

## Reproduce it

```bash
uv sync --extra dev          # install (Python ≥ 3.12)
uv run pytest                # test suite
make help                    # list experiment targets
```

The experiment definitions are tracked Hydra configs. For example:

```bash
uv run python scripts/train.py -m +experiment=vdp/paper_log_penalty
uv run python scripts/train.py -m +experiment=pendulum/paper_frac_exp_penalty
```

Their run records and derived artifacts are written only to local ignored
paths.

### Under the hood

- **`src/SSN/` — a semismooth Newton optimizer in PyTorch**: a
  `torch.optim.Optimizer` subclass with matrix-free CG for the Newton system
  and proximal handling of the non-convex penalties. Algorithm 2 uses the
  closed-form global scalar proximal maps for q in {1/2, 2/3, 1}
  ([ADR-0009](docs/adr/0009-use-verified-closed-form-global-proximal-maps.md)),
  while the outer acceptance guard prevents a local coefficient correction
  from increasing the objective
  ([ADR-0004](docs/adr/0004-model-trainer-eval-separation.md)).
- **Golden-output tests** guard the PDAP solver: refactors of the numerical
  core are checked against stored reference solutions, not just unit
  assertions (`tests/`).
- **Runs are records**: each training run writes a JSON run record under
  `rawdata/logs/multirun/`; `make mlflow-backfill` publishes them to an MLflow
  tracking server whose full stack is defined as Terraform in
  [`deploy/`](deploy) — see [docs/adr/mlflow.md](docs/adr/mlflow.md).
- **Publication artifacts are local**: experiment run records, generated
  reports, figures, and paper-support scripts are intentionally not tracked.
  The manuscript source and compiled PDF are the publication record.
- CI runs the test suite and `ruff` on every push.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Library code: signed shallow networks, `PDAP/`, `SSN/`, data/eval/plotting |
| `conf/` | Hydra configs: data, model, eval, experiment sweeps |
| `scripts/` | Training entrypoint (`train.py`), dataset generators, MLflow backfill |
| `experiments/` | Experiment definitions and legacy curated studies; current paper outputs stay local |
| `tests/` | pytest suite incl. golden-output solver tests |
| `docs/` | Research program & claims registry, ADRs, MLflow guide |
| `deploy/` | Terraform for the MLflow tracking server |
| `vault/` | Deeper implementation notes (algorithm map, model internals, benchmarks) |
