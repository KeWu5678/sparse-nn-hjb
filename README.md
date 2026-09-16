# Sparse neural networks for optimal feedback control

[![CI](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml/badge.svg)](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml)

**An 18-neuron softplus network stabilizes the Van der Pol system at a
closed-loop cost of 6.68, against 6.48 for the exact optimal control.**

Learning a value function is easy to score and easy to get wrong. This project
learns one from trajectory data and then *flies* it — the reported result is the
behaviour of the closed loop, not a regression error.

## The problem

Optimal feedback control has a classical answer: solve the
Hamilton–Jacobi–Bellman equation for the value function $V(x)$, and read the
optimal controller off its gradient — for the Van der Pol system,
$\hat u(x) = -\partial_{x_2}\hat V(x)/(2\eta)$.

Two things make that hard. $V$ is expensive to compute globally, and if it is
learned from data instead, **controller quality depends on $\nabla\hat V$, not on
$\hat V$**. A model with an excellent value fit and a mediocre gradient field
produces a controller that oscillates, saturates, or diverges. Most of the
engineering here follows from taking that second point seriously.

The approach: sample value *and* gradient data from open-loop solves via
Pontryagin's principle, then fit a shallow network
$\sum_k c_k\sigma(a_k\cdot x + b_k)$ under an $H^1$ loss, so the gradient is a
first-class training target. Width is not fixed in advance and sparsity is not
post-hoc pruning — neurons are **inserted one at a time** by a Primal–Dual Active
Point method over a measure-space formulation, each insertion certified to
decrease the objective.

## Result

<p align="center">
  <img src="docs/figures/readme/vdp_frontier.png" width="520"
       alt="Relative H1 error against number of neurons on Van der Pol">
</p>

Accuracy against width on Van der Pol. The sparse nonconvex models reach a given
error with fewer neurons at small budgets; the conventional ReLU + $\ell^1$
network overtakes them once the support is allowed to grow. This is a
small-width trade-off, not dominance everywhere.

Representative $H^1$-trained checkpoints, selected by minimum training
objective:

| activation | penalty | neurons | rel. $H^1$ error |
| --- | --- | ---: | ---: |
| softplus | normalized log penalty | **18** | 0.243 |
| tanh | normalized log penalty | 33 | 0.237 |
| Gaussian | normalized log penalty | 39 | 0.236 |
| ReLU<sup>2</sup> | $\|c\|^{2/3}$ | 53 | 0.235 |
| ReLU<sup>3</sup> | $\|c\|^{1/2}$ | 26 | **0.235** |

Errors are relative $H^1$ in original coordinates — the fitted $V$ and its
gradient with respect to the original state variables, after undoing the
training normalization.

Closed-loop rollout from $y_0 = (2,1)$, which is what the value function is for:

| controller | neurons | stabilizes | cost |
| --- | ---: | :---: | ---: |
| exact optimal control | — | yes | 6.48 |
| softplus | 18 | yes | 6.68 |
| Gaussian | 39 | yes | 6.49 |
| ReLU<sup>3</sup> | 36 | yes | 6.50 |

## Engineering

**A semismooth Newton optimizer, written as a native PyTorch optimizer.** The
outer-weight problem is nonsmooth and nonconvex. `src/SSN/` is a
`torch.optim.Optimizer` subclass implementing a normal-map semismooth Newton
method with matrix-free CG for the Newton system, closed-form global proximal
maps for the fractional penalties $q\in\{1/2,\ 2/3,\ 1\}$, and a guard that
discards a correction which increases the objective — the property that keeps
the outer loop monotone.

**Golden-output tests.** Refactors of the numerical core are checked against
stored reference solutions, not just unit assertions. A change in solver
behaviour shows up as a diff in neuron counts and errors, not as a silently
different answer. The suite is 216 tests and runs on every push alongside
`ruff`.

**Runs are records.** Every training run writes a self-describing JSON record —
full resolved config, per-iteration metrics, artifact paths — next to its
checkpoint. Records are the source of truth; the MLflow dashboard is a
projection of them, published live during a run, and rebuildable from disk at
any time. The tracking server is a container in [`deploy/`](deploy).

**Experiments are configs, not scripts.** Each study is one tracked Hydra file
declaring its own sweep axes and its own output layout, so a sweep is a single
command and the record tree is self-organizing:

```bash
make sweep EXPERIMENT=log_penalty        # 448 runs across two datasets
```

Records land under `rawdata/logs/multirun/<dataset>/<experiment>/<axis>/<job>/`.
The target refuses to start if that subtree already holds records, and aborts
immediately if the tracking server is unreachable rather than training for hours
unpublished.

**Reproducibility is checked, not assumed.** Runs are deterministic under a
fixed seed — re-running a completed configuration reproduces its metrics to
seven digits. Solver changes that move results are recorded as ADRs in
[`docs/adr/`](docs/adr) with an explicit instruction to re-run affected studies.

## Reproduce

```bash
uv sync --extra dev          # Python >= 3.12
uv run pytest
make help
```

A single run picks a model family and a dataset; everything else has a default:

```bash
uv run python scripts/train.py +model=profile +data=vdp model.activation=softplus
```

Datasets are generated from the open-loop solves:

```bash
make openloop
```

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Library: shallow networks, `PDAP/` insertion, `SSN/` optimizer, data/eval/plotting |
| `conf/` | Hydra configs — model families, datasets, evaluation, experiment sweeps |
| `scripts/` | `train.py` entry point, dataset generators, MLflow importer |
| `experiments/` | Study definitions and curated results |
| `tests/` | pytest suite, including golden-output solver tests |
| `docs/` | ADRs, MLflow guide, research notes |
| `deploy/` | Containerized MLflow tracking server |
| `vault/` | Deeper implementation notes |

Generated run records, figures, and reports are written to ignored local paths.
