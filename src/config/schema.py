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

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Tuple

from torch import Tensor

from .activations import ACTIVATIONS, get_activation, get_growth, get_use_sphere


@dataclass
class ModelConfig:
    """A registered model = structure + insertion rule + hyperparameters.

    ``kind`` is retained as the run-record model-family label and must be
    ``"signed"``. ``insertion`` selects profile or finite-step insertion.
    ``activation`` is a registry name resolved to a callable at build time; its
    sphere geometry is bundled with the activation in the registry (see
    ``src.config.activations``), not configured here.
    """

    # identity
    kind: str = "signed"          # the only active model family
    insertion: str = "profile"    # "profile" | "finite_step"
    # structure
    activation: str = "relu"      # name resolved via src.config.activations
    # Closed-form coefficient correction is implemented for powers 1, 2, and 3.
    power: float = 1.0
    # (w1, w2) = (value loss weight, gradient loss weight); l2 = (1, 0), h1 = (1, 1)
    loss_weights: Tuple[float, float] = (1.0, 1.0)
    # Regularization uses two separate scalar penalties:
    #   * power penalty  alpha * sum |c|^q, q = 2/(power+1): power > 1, gamma = 0.
    #   * log penalty    alpha * sum phi(|u|): power = 1, gamma >= 0, with the
    #     Algorithm 1 coefficient u = w_p*c and a fixed half L1 / half log mixture.
    # With a ReLU activation, power=1 and gamma=0 give the ReLU--L1 baseline.
    alpha: float = 1e-5
    gamma: float = 0.0   # 0 => log term off (power penalty); > 0 => log penalty
    # Algorithm 1 evaluates the penalty on the normalized measure
    # mu_p = w_p*mu, with w_p(omega) = 1 + |omega|^p.  For a signed profile model
    # with a nonhomogeneous activation this is the only supported objective;
    # SparseModel applies it automatically.  Algorithm 2 is sphere-normalized and does not use
    # this parameter.
    moment_order: Optional[float] = None  # Explicit for nonhomogeneous models; unused on the sphere.


@dataclass
class TrainingConfig:
    """How the model is fit: outer PDAP loop + SSN solver + insertion constants."""

    # outer PDAP loop
    num_iterations: int = 10      # T_out
    num_of_initial_pts: int = 50  # N_trial, initial points of the multistart search per iteration
    max_insert: int = 15          # N_ins, cap on atoms inserted per iteration (batch mode)
    prune_amp_tol: float = 1e-8   # eps_prune: drop atoms with |c_n| <= this
    # --- paper-conformance axes (paper/paper_0805.tex, Section 5) --------------
    # Defaults retain the general-purpose training behavior; the Algorithm 1
    # experiment preset selects the paper-specific values explicitly.
    #
    # insert_init -- initial outer weight of a freshly inserted atom.
    #   "warm_start"  -- the coordinate-descent batch warm start: one combined
    #                    descent direction, one scalar prox step, so the best atom
    #                    gets a real coefficient and the rest ~sqrt(eps).
    #   "guaranteed"  -- the theorem's per-atom coefficient
    #                    c(omega) = -Delta(mu,omega)/(w_p(omega)*||K_p(omega)||^2)
    #                              * sign(P_p(omega)),
    #                    which decreases the objective by at least
    #                    Delta^2/(2||K_p||^2).  Ignored by finite_step insertion,
    #                    whose c* is already the paper's initialization.
    insert_init: str = "warm_start"
    # insert_mode -- how many atoms enter per outer iteration.
    #   "batch"       -- up to max_insert candidates at once, against one frozen
    #                    residual.  Algorithm 1/2 as printed; the paper states that
    #                    the per-step guarantees do not survive the cross terms.
    #   "sequential"  -- exactly one selected atom per iteration. This removes the
    #                    batch cross terms; the rate bound additionally requires an
    #                    exact global maximizer. Raise num_iterations to match width.
    insert_mode: str = "batch"
    # correction_guard -- accept the SSN correction only if it did not increase the
    # objective, otherwise keep the post-insertion coefficients.  SSN is a local
    # method on a nonconvex penalty with no descent guarantee (and for q < 1 the
    # penalty is not locally Lipschitz at 0), so this is what makes the outer loop
    # monotone.
    correction_guard: bool = False
    # loop_order -- where the correction sits relative to the insertion.
    #   "correction_first"  -- insert once up front, then per iteration
    #                          correct -> prune -> record -> insert.  Leaves the
    #                          final batch uncorrected and counted in final_neurons.
    #   "insertion_first"   -- per iteration insert -> correct -> prune -> record,
    #                          the order of Algorithms 1 and 2.
    loop_order: str = "correction_first"
    # radial_cap -- upper bound on the radial candidate search (non-sphere only).
    #   "fixed"     -- the exp(5) clamp (see docs/adr/0006).
    #   "theorem"   -- min(R(mu_t), exp(5)) with R from the quantitative insertion
    #                  theorem; requires growth data for the activation, and falls
    #                  back to the clamp when the activation declares none.
    radial_cap: str = "fixed"
    # SSN solver (src/SSN/optimizer.py defaults + the hardcoded iterations=20)
    method: str = "levenberg_marquardt"   # "levenberg_marquardt" | "steihaug_cg"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0
    fit_outer_iterations: int = 20
    display_every: int = 2  # Log checkpoint indices 0, N, 2N; <=0 disables periodic progress.
    # insertion numeric constants (src/PDAP/insertion.py)
    # Candidate-to-candidate deduplication: Algorithm 1 uses absolute Euclidean
    # distance in omega=(a,b); sphere searches use cosine-similarity gap.  The
    # much tighter Algorithm 2 check against the existing support is internal.
    ins_merge_tol: float = 1e-2
    lbfgs_lr: float = 1e-2        # L-BFGS step size for dual-profile maximisation inside candidate search (both methods)
    lbfgs_steps: int = 200        # max L-BFGS iterations per candidate direction (both methods)


@dataclass
class DataConfig:
    """The data source: a key-based ``.npy`` or ``.npz`` with ``x``, ``v``, ``dv``.

    ``path`` is a bare filename under ``DATA_DIR`` (see ``src.data``); absolute
    paths are allowed. Resolution happens in ``src.data.load_value_samples``.
    The default points at the existing legacy VDP ``.npy``; new OpenLoop
    generators save ``.npz`` files with the same keys.
    ``train_fraction`` is the train/validation split applied in
    ``src.data.split_value_samples`` (first fraction trains, rest validates).
    ``normalize`` applies max-abs scaling (with chain-rule gradient transform);
    data loading / normalization / splitting happen in the run script (see
    ``scripts/train.py``), not the trainer.
    """

    path: str = "VDP_20260924_62305c7ac2534952a7c85d6c4656a491/VDP_pmp_grid_30x30_20260924.npz"
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
    # Abort at startup when MLflow is not reachable, instead of training for hours
    # and discovering at run.finish() that publish_record_to_mlflow silently
    # returned False because MLFLOW_TRACKING_URI was unset. Sweeps set it.
    require_mlflow: bool = False
    verbose: bool = True  # Readable stderr; JSONL file output is always enabled.
    log_level: str = "INFO"  # Shared threshold for both outputs.
    log_file: Optional[str] = None  # Defaults to run.jsonl in Hydra's output directory.


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    name: str = "run"


def resolve_model(config, input_dim: int) -> dict:
    """Prepare model configuration for construction without changing the raw input.

    Hydra and ordinary mappings use the existing ModelConfig defaults. The
    returned runtime mapping contains the activation callable and its geometry;
    raw named configuration remains suitable for run records. Prediction accepts
    any callable and positive finite power, including historical settings.
    Training setup must additionally call validate_model_config before construction.
    """
    config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    if config.kind != "signed":
        raise ValueError(
            f"the active implementation only supports kind='signed'; got {config.kind!r}"
        )
    activation: str | Callable[[Tensor], Tensor] = config.activation
    if isinstance(activation, str):
        activation_name = activation
        activation = get_activation(activation_name)
    elif callable(activation):
        activation_name = next(
            (name for name, (fn, _) in ACTIVATIONS.items() if fn is activation), None,
        )
    else:
        raise TypeError("activation must be a registry name or callable")
    use_sphere = get_use_sphere(activation_name) if activation_name is not None else False
    if not math.isfinite(config.power) or config.power <= 0.0:
        raise ValueError("power must be positive and finite")
    if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    return {
        "input_dim": input_dim,
        "activation": activation,
        "activation_name": activation_name,
        "use_sphere": use_sphere,
        "power": config.power,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "loss_weights": tuple(config.loss_weights),
        # Old records and mixed sweeps may supply a numeric sphere placeholder.
        "moment_order": None if use_sphere else config.moment_order,
        "insertion": config.insertion,
    }


def validate_model_config(config) -> None:
    """Validate resolved model settings before constructing a training problem.

    The insertion rule in this mapping must also be the one supplied to PDAP.
    Prediction-only reconstruction does not call this training configuration gate.
    """
    activation_name, power = config["activation_name"], config["power"]
    gamma, use_sphere = config["gamma"], config["use_sphere"]
    moment_order, insertion = config["moment_order"], config["insertion"]
    if activation_name is None:
        raise ValueError("coefficient correction requires a registered activation")
    if power not in (1.0, 2.0, 3.0):
        raise ValueError(
            "the coefficient correction supports activation powers 1, 2, 3; "
            f"got power={power}"
        )
    if not use_sphere and power != 1.0:
        raise ValueError(
            "nonhomogeneous signed profile insertion is Algorithm 1 and requires power == 1"
        )
    if power != 1.0 and gamma != 0.0:
        raise ValueError(f"the power penalty requires gamma == 0; got power={power}, gamma={gamma}")
    if not use_sphere:
        if moment_order is None:
            raise ValueError("nonhomogeneous training requires an explicit moment_order")
        if not math.isfinite(moment_order) or moment_order <= 0.0:
            raise ValueError("moment_order must be positive and finite")
        growth = get_growth(activation_name)
        if growth is not None and moment_order <= growth.s1:
            raise ValueError(
                f"moment_order must exceed {growth.s1:g} for activation {activation_name!r}"
            )
        # An undeclared growth bound permits numerical training, but cannot
        # certify H1 admissibility or supply a theorem search radius.
    if insertion not in ("profile", "finite_step"):
        raise ValueError(f"insertion must be 'profile' or 'finite_step', got {insertion!r}")
    if insertion == "finite_step":
        if not use_sphere:
            raise ValueError("finite_step insertion is Algorithm 2 and requires a sphere activation")
        if gamma != 0.0:
            raise ValueError("finite_step insertion minimizes the power penalty and requires gamma == 0")


def validate_optimizer_config(training: TrainingConfig) -> None:
    """Validate the training settings before constructing PDAP.

    Each setting with a fixed set of values is checked here rather than at use,
    so a typo fails at setup instead of midway through a sweep.
    """
    for name, value, allowed in (
        ("training.insert_init", training.insert_init, ("warm_start", "guaranteed")),
        ("training.insert_mode", training.insert_mode, ("batch", "sequential")),
        ("training.loop_order", training.loop_order, ("correction_first", "insertion_first")),
        ("training.radial_cap", training.radial_cap, ("fixed", "theorem")),
        ("training.method", training.method, ("levenberg_marquardt", "steihaug_cg")),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    iterations = training.fit_outer_iterations
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("training.fit_outer_iterations must be a nonnegative integer")


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EvalConfig",
    "EnvConfig",
    "ExperimentConfig",
    "resolve_model",
    "validate_model_config",
    "validate_optimizer_config",
]
