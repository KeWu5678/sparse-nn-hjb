# Sparse neural networks for optimal feedback control

[![CI](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml/badge.svg)](https://github.com/KeWu5678/sparse-nn-hjb/actions/workflows/ci.yml)

**A 16-neuron softplus network stabilizes the Van der Pol system at the
reference rollout cost (6.48), with original-coordinate relative $H^1$
error 0.241.**

Closed-loop rollout of the Van der Pol oscillator from $y_0=(2,1)$ shows that
the fitted feedback laws stabilize the system with different support sizes but
nearly the same cost. The complete tables and figures are in the
[current paper](paper/paper_0805.pdf).

## The problem

Optimal feedback control has a classical answer: solve the
Hamilton–Jacobi–Bellman (HJB) equation for the value function $V(x)$, and the
optimal controller follows from its gradient, for example
$\hat u(x)=-\partial_{x_2}\hat V(x)/(2\eta)$. The catch is that $V$ is
expensive to compute globally—and if it is learned from data instead, the
controller quality depends on $\nabla\hat V$, not only on $\hat V$. A model
with an excellent value fit and a mediocre gradient field can produce a
controller that oscillates, saturates, or diverges.

This repository learns $V$ from open-loop trajectory data: value and gradient
samples generated through Pontryagin's principle. It fits shallow networks
$\sum_k c_k\sigma(a_k\cdot x+b_k)$ in Sobolev ($H^1$) loss so the gradient is
a first-class training target. Sparsity is not post-hoc pruning. Neurons are
inserted by a Primal-Dual Active Point method (PDAP) over the measure-space
formulation and penalized by either a log penalty on the normalized measure or
a fractional power $|c|^q$ for positively homogeneous activations.

The resulting outer-weight problem is nonsmooth and nonconvex. It is corrected
with a guarded **semismooth Newton normal-map method implemented as a native
PyTorch optimizer**. For the fractional penalties used in the paper, the
scalar global proximal maps are evaluated in closed form and the normal-map
scale is chosen from the insertion warm start. A correction is retained only
when it does not increase the objective.

## Main result: accuracy per neuron

Representative $H^1$-trained Van der Pol runs reported in the paper are:

| activation | penalty | neurons | rel. $H^1$ error |
| --- | --- | ---: | ---: |
| softplus | normalized log penalty | **16** | 0.241 |
| Gaussian | normalized log penalty | 34 | 0.236 |
| tanh | normalized log penalty | 38 | 0.237 |
| ReLU<sup>2</sup> | $|c|^{2/3}$ | 45 | 0.236 |
| ReLU<sup>3</sup> | $|c|^{1/2}$ | 28 | **0.235** |

These are the paper's activation-comparison checkpoints, selected by minimum
training objective. The first three use $\alpha=10^{-4}$, $\gamma=10$,
$p=2.01$; the two fractional-power runs use $\alpha=10^{-5}$.
Reported $H^1$ errors combine errors in the original-scale $V$ and its gradient
with respect to the original state variables, not the normalized training
coordinates. They supersede the earlier approximately 0.10 figures.

The nonconvex models reach a similar error scale with different support sizes.
In the paper's support-budget comparison they have lower error at small support
budgets, while the traditional ReLU+$\ell^1$ network reaches a lower error with
a larger support. This is not a claim of dominance at every support budget.

The two algorithm families also leave different geometric signatures in the
learned parameters. The fractional-power formulation constrains its atoms to
the unit sphere, whereas the normalized log-penalty formulation operates on an
unbounded parameter domain. Their insertion frontiers and weight portraits are
shown in Section 6 of the [paper](paper/paper_0805.pdf).

Run records, derived reports, and figures remain local. The tracked manuscript
and compiled PDF are the publication record.

## Probing the limit: value functions with nonsmooth gradients

The Van der Pol value function is smooth. HJB value functions can instead have
**gradient jumps across switching sets**, where the optimal strategy changes
branch. The pendulum swing-up benchmark targets this regime deliberately: its
switching curve separates braking to the upright at $\theta=0$ from swinging
over the top to $\theta=2\pi$, and the training data contains samples on both
sides of the jump.

The findings are sharp:

- **The switching region is harder for four of the five reported models.**
  ReLU<sup>3</sup> is the exception, with similarly high error in both regions.
  The Gaussian has the smallest regional errors, 0.304 near the switching set
  and 0.191 elsewhere. Adding samples near the switching set gives no
  systematic or material reduction at the tested widths.
- **No model reproduces the full gradient jump.** ReLU<sup>2</sup> develops
  the sharpest fitted change of slope, while the smooth activations interpolate
  through the discontinuity.
- **Regional error and feedback quality rank the models differently.**
  Softplus and ReLU<sup>3</sup> reach an upright neighbourhood from both tested
  starts. From the harder start their costs are 69.8 and 110.9, respectively,
  against the reference cost 26.2. Gaussian, tanh, and ReLU<sup>2</sup> succeed
  only from the easier start; there ReLU<sup>2</sup> reaches cost 10.5 against
  the reference 10.2.

A parallel local theory program studies why activation regularity matters for
such targets. Its notes and claims registry under `docs/research/` are not
distributed with a clean checkout.

## Reproduce it

```bash
uv sync --extra dev          # install (Python ≥ 3.12)
uv run pytest                # test suite
make help                    # list experiment targets
```

The experiment definitions are tracked Hydra configs. For example:

```bash
uv run python scripts/train.py -m +experiment=log_penalty +data=vdp
uv run python scripts/train.py -m +experiment=frac_exp_penalty +data=pendulum
```

Their run records and derived artifacts are written only to local ignored
paths.

`make sweep EXPERIMENT=<name>` runs training and saves run records; it does not
run analysis scripts or generate reports. Pendulum runs include switching-region
and rest-region evaluation metrics. Tables and figures are generated separately.

### Under the hood

- **`src/SSN/` — a semismooth Newton optimizer in PyTorch**: a
  `torch.optim.Optimizer` subclass with matrix-free CG for the Newton system
  and proximal handling of the nonconvex penalties. Algorithm 2 uses the
  closed-form global scalar proximal maps for $q\in\{1/2,2/3,1\}$
  ([ADR-0009](docs/adr/0009-use-verified-closed-form-global-proximal-maps.md)),
  while the outer acceptance guard prevents a local coefficient correction
  from increasing the objective
  ([ADR-0004](docs/adr/0004-model-trainer-eval-separation.md)).
- **Golden-output tests** guard the PDAP solver: refactors of the numerical
  core are checked against stored reference solutions, not just unit
  assertions (`tests/`).
- **Runs are records**: each training run writes a JSON record under
  `rawdata/logs/multirun/`; those records are the source of truth, and
  `scripts/upload_run_records_to_mlflow.py` projects them into the MLflow
  dashboard defined in [`deploy/`](deploy). See
  [docs/adr/mlflow.md](docs/adr/mlflow.md).
- **Paper inputs stay local**: current experiment run records, generated paper
  reports, figures, and paper-support scripts are intentionally not tracked.
  Historical curated summaries remain tracked and are marked as superseded.
  The manuscript source and compiled PDF are the publication record.
- CI runs the test suite and `ruff` on every push.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Library code: signed shallow networks, `PDAP/`, `SSN/`, data/evaluation/plotting |
| `conf/` | Hydra configs: data, model, evaluation, experiment sweeps |
| `scripts/` | Training entry point (`train.py`), dataset generators, MLflow importer |
| `experiments/` | Experiment definitions and legacy curated studies; current paper outputs stay local |
| `tests/` | pytest suite, including golden-output solver tests |
| `docs/` | Research program, claims registry, ADRs, and MLflow guide |
| `deploy/` | Containerized MLflow tracking server |
| `vault/` | Deeper implementation notes |
