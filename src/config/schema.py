"""Typed configuration schema for a PDAP run (Hydra structured configs).

Four sections compose into :class:`ExperimentConfig`:

  * ``model``    — a registered model: structure + insertion rule + hyperparameters.
  * ``training`` — how the model is fit: outer PDAP loop + SSN solver + insertion
    numeric constants.
  * ``data``     — the data source (a key-based ``.npy``/``.npz`` path with
    arrays ``x``, ``v``, and ``dv``).
  * ``env``      — runtime: seed + logging.

Every default equals the value currently in force for the VDP signed-profile
baseline and the hardcoded library literals, so the default
``ExperimentConfig`` reproduces today's behavior.

The config is **domain-agnostic** — it describes the PDAP model and how it is
trained, not any specific control problem. The only problem-specific input is
``data.path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """A registered model = structure + insertion rule + hyperparameters.

    ``kind`` and ``insertion`` form the model identity (the ``conf/model/*.yaml``
    config group: signed/profile, semiconcave/profile, signed/finite_step).
    ``activation`` is a registry name resolved to a callable at build time; its
    sphere geometry is bundled with the activation in the registry (see
    ``src.config.activations``), not configured here.
    """

    # identity
    kind: str = "signed"          # "signed" | "semiconcave"
    insertion: str = "profile"    # "profile" | "finite_step"
    # structure
    activation: str = "relu"      # name resolved via src.config.activations
    power: float = 1.0
    # (w1, w2) = (value loss weight, gradient loss weight); l2 = (1, 0), h1 = (1, 1)
    loss_weights: Tuple[float, float] = (1.0, 1.0)
    c_init: float = 1.0           # semiconcave only
    # Regularization.  The penalty on the atom weights is  alpha * sum_i phi(|c_i|^q),
    # with q = 2/(power+1) (power is the activation exponent set above).  The two
    # penalties this project uses are selected by how you set power and gamma:
    #   * power penalty   alpha * sum |c|^q   -- set gamma = 0 (phi becomes the
    #     identity) and power > 1 (so q < 1 is genuinely non-convex).
    #   * log penalty     alpha * sum phi(|c|) -- set power = 1 (so q = 1) and
    #     gamma > 0; th interpolates L1 (th=1) <-> non-convex log (th=0).
    alpha: float = 1e-5
    gamma: float = 0.0   # 0 => log term off (power penalty); > 0 => log penalty
    th: float = 0.5      # L1 (th=1) <-> non-convex log (th=0); only acts when gamma > 0


@dataclass
class TrainingConfig:
    """How the model is fit: outer PDAP loop + SSN solver + insertion constants."""

    # outer PDAP loop
    num_iterations: int = 10
    num_insertion: int = 50
    max_insert: int = 15
    prune_amp_tol: float = 1e-8
    # SSN solver (src/SSN/optimizer.py defaults + the hardcoded iterations=20)
    lr: float = 1.0
    method: str = "levenberg_marquardt"   # "levenberg_marquardt" | "steihaug_cg"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0
    fit_outer_iterations: int = 20
    display_every: int = 2
    # insertion numeric constants (src/PDAP/insertion.py)
    ins_merge_tol: float = 1e-2    # cosine-similarity threshold for merging near-duplicate candidates (both methods)
    lbfgs_lr: float = 1e-2        # L-BFGS step size for dual-profile maximisation inside candidate search (both methods)
    lbfgs_steps: int = 200        # max L-BFGS iterations per candidate direction (both methods)
    newton_tol: float = 1e-12     # relative residual tolerance for the Newton solve in finite_step (finite_step only)
    newton_max_iter: int = 50     # max Newton iterations for the finite-step insertion weight solve (finite_step only)


@dataclass
class DataConfig:
    """The data source: a key-based ``.npy`` or ``.npz`` with ``x``, ``v``, ``dv``.

    ``path`` is a bare filename under ``DATA_DIR`` (see ``src.paths``); absolute
    paths are allowed. Resolution happens in ``src.data.load_value_samples``.
    The default points at the existing legacy VDP ``.npy``; new OpenLoop
    generators save ``.npz`` files with the same keys.
    ``train_fraction`` is the train/validation split applied in
    ``src.data.split_value_samples`` (first fraction trains, rest validates).
    ``normalize`` applies max-abs scaling (with chain-rule gradient transform);
    data loading / normalization / splitting happen in the run script (see
    ``scripts/train.py``), not the trainer.
    """

    path: str = "VDP_beta_0.1_grid_30x30.npy"
    train_fraction: float = 0.9
    normalize: bool = True


@dataclass
class EvalConfig:
    """Post-fit evaluation: which metrics to compute on the fitted model.

    ``kind="global"`` (the default) reproduces today's behavior — only the
    global ``summary_metrics``.  ``kind="region_split"`` additionally reports
    errors split into the **switching tube** — the fixed-radius tubular
    neighborhood {distance to the ±2π-tiled switching curve ≤ ``tube_radius``}
    — and the **rest**, scored on the dense **region-eval pool** (``eval_pool``:
    the certified two-sided point set with the training rows excluded, built by
    ``scripts/investigation/build_region_eval_pool.py``). A fixed radius on an
    out-of-sample pool replaces the earlier percentile band over the emitted
    samples: the percentile region was endogenous to the sampling design (adding
    switching-band samples shrank it) and was evaluated mostly on seen data.

    ``distance_cache`` (per-sample distance aligned to the *dataset*, from
    ``scripts/investigation/precompute_region_distances.py``) remains for the
    distance-binned error profile diagnostic.
    """

    kind: str = "global"                  # "global" | "region_split"
    tube_radius: float = 0.3              # switching tube = distance <= tube_radius
    eval_pool: Optional[str] = None       # npz with x/v/dv/distance (out-of-sample pool)
    distance_cache: Optional[str] = None  # npz with per-sample distance (binned profile)


@dataclass
class EnvConfig:
    """Runtime: random seed + logging.

    Fixed (not configured): device is CPU-only (no GPU path exists) and dtype is
    float64 (hardcoded across PDAP/models/SSN). Surfacing those is future work.
    """

    seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    name: str = "run"


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EvalConfig",
    "EnvConfig",
    "ExperimentConfig",
]
