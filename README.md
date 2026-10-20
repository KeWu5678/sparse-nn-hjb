# Sparse neural networks for optimal feedback control

[![CI](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml/badge.svg)](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml)

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
$\sum_k c_k\sigma(a_k\cdot x + b_k)$ under an $H^1$ loss with **nonconvex penalties**,
so the gradient is a first-class training target.
Width is not fixed in advance and sparsity is not post-hoc pruning — neurons are inserted adaptively by a Primal–Dual Active
Point method over a measure-space formulation.

## Result

We tested on Van der Pol system (smooth value function), Pendulum-swing-up system (nonsmooth value function) and Allen-Cahn system (high-dimensional case). Below is an example of Van der Pol.
The displayed sparse nonconvex models reach lower error with fewer neurons.
<p align="center">
  <img src="docs/figures/readme/vdp_frontier.png" width="520"
       alt="Relative H1 error against number of neurons on Van der Pol">
</p>


The assembled feedback laws all stabilized (from $y_0=(2,1)$) over the benchmark horizon $T=3$
(The ReLU<sup>3</sup> controller uses a separate $\alpha=10^{-6}$ checkpoint):

| controller | neurons | terminal state norm | running cost |
| --- | ---: | ---: | ---: |
| numerical open-loop reference | — | 0.3517 | 6.4010 |
| softplus | 14 | 0.2354 | 6.4191 |
| Gaussian | 32 | 0.2178 | 6.4249 |
| ReLU<sup>3</sup> | 30 | 0.1597 | 6.4487 |

The reference solves the finite-horizon PMP boundary-value problem with a free
terminal state and zero terminal cost.

## Engineering
Outer-weight problem is nonsmooth and nonconvex.
A semismooth Newton optimizer, written as a native PyTorch optimizer
is implemented with a normal-map semismooth Newton
method, closed-form global proximal maps for the fractional penalties $q\in\{1/2,\ 2/3,\ 1\}$.

**MLflow** is the system of record. Every training job is one MLflow run holding
its resolved config, dataset hashes, errors in physical units, per-checkpoint
traces, and the fit as `history.pt`.

**Experiments are configs, not scripts.** Each study is one tracked Hydra file
declaring its own sweep axes and its own output layout, so a sweep is a single
command and the record tree is self-organizing:

```bash
make sweep EXPERIMENT=log_penalty        # 448 runs across two datasets
```

Hydra configs and logs land under
`rawdata/logs/multirun/<dataset>/<experiment>/<axis>/<job>/`. The target refuses
to start if that subtree already exists or the tracking server is unreachable.
Runs are deterministic under a fixed seed — re-running a completed configuration reproduces its metrics to
seven digits.

## Reproduce

```bash
scripts/init.sh                             # Python environment + local MLflow server (Docker)
uv run python scripts/generate_vdp_data.py  # generate samples from open-loop solves
make sweep EXPERIMENT=log_penalty           # `make help` lists the presets
```

Results appear in the dashboard at <http://127.0.0.1:5000>. The generator writes
a new dated folder under `rawdata/data/`; point `conf/data/vdp.yaml` at it to
train on it.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Library: shallow networks, `PDAP/` insertion, `SSN/` optimizer, data/eval/plotting |
| `conf/` | Hydra configs — model families, datasets, evaluation, experiment sweeps |
| `scripts/` | `train.py` entry point, dataset generators, `init.sh` |
| `experiments/` | Current workflows, reference-data plotters, and local paper outputs |
| `tests/` | pytest suite, including golden-output solver tests |
| `docs/` | ADRs, research notes |
| `deploy/` | Containerized MLflow tracking server |
