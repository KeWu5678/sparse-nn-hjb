# SparseNNforHJB

[![CI](https://github.com/kewu5678/nnforhjb/actions/workflows/ci.yml/badge.svg)](https://github.com/kewu5678/nnforhjb/actions/workflows/ci.yml)

Sparse shallow neural networks for Hamilton–Jacobi–Bellman (HJB) value
functions, trained from value/gradient samples with a primal–dual proximal
algorithm (PDAP) under non-convex sparsity penalties and a semismooth Newton
(SSN) outer solve.

**The problem in three sentences.** Optimal-control value functions V solve an
HJB equation and are typically only semiconcave: their gradient jumps across
switching sets. We fit V (and ∇V) with shallow networks ∑ uₖ σ(aₖ·x + bₖ)
regularized by non-convex penalties (log penalty φ_γ, fractional powers |c|^q)
that drive most outer weights to zero, so accuracy is bought with few neurons.
The experiments compare activations and penalties on a smooth benchmark
(Van der Pol) and a switching-set benchmark (pendulum swing-up); a parallel
theory program studies why semiconcavity-adapted atoms need fewer neurons.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Library code: `models/` (signed/semiconcave nets), `PDAP/`, `SSN/`, data/eval/plots |
| `scripts/` | Hydra training entrypoint (`train.py`), MLflow backfill, one-off diagnostics |
| `conf/` | Hydra configs: data, model, eval, experiment sweeps |
| `experiments/` | Curated studies with `README` (scope), `results.md` (findings), `analysis.py`, `figures/` — ordered `00_openloop` (data) → `01_vdp` (smooth) → `02_pendulum` (switching set) → `03_region_split_pendulum` |
| `tests/` | pytest suite incl. golden-output equivalence tests for the PDAP solver |
| `docs/` | ADRs, research program (`docs/research/`, claims registry), [MLflow guide](docs/mlflow.md) |
| `papar/` | Working paper (LaTeX); `make paper-figures` syncs its figures from `experiments/` |
| `deploy/` | Terraform for the MLflow tracking server on EC2 |
| `vault/` | Deeper implementation notes (algorithm map, model details, benchmarks) |
| `rawdata/`, `outdated/` | Generated run records/datasets and archived legacy material (gitignored) |

## Quickstart

```bash
uv sync --extra dev          # install (Python ≥ 3.12)
uv run pytest                # test suite
uv run ruff check .          # lint
make help                    # list experiment targets
```

Run a curated experiment sweep (Hydra multirun + analysis → `results.md`):

```bash
make region_split_pendulum
make penaltypowers DATA=pendulum
```

Or a single Hydra-composed training run:

```bash
uv run python scripts/train.py +experiment=activationsearch data=vdp
```

Each run writes a JSON Run Record under `rawdata/logs/multirun/`; with
`MLFLOW_TRACKING_URI` set, records are also published to MLflow — see
[docs/mlflow.md](docs/mlflow.md) for the EC2/Terraform tracking-server
workflow and backfill.

## Results & research program

- Experiment findings live next to their code: start at
  [`experiments/01_vdp`](experiments/01_vdp) and
  [`experiments/02_pendulum`](experiments/02_pendulum) (`results.md` per study),
  with provenance in [docs/research-directions.md](docs/research-directions.md).
- The theory program (semiconcave vs signed representation capacity) is in
  [docs/research/OVERVIEW.md](docs/research/OVERVIEW.md) with a
  proved/refuted/open claims registry in
  [docs/research/CLAIMS.md](docs/research/CLAIMS.md).
- Architecture decisions are recorded in [docs/adr/](docs/adr).
