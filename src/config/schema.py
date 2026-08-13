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
    # Parameter-moment axis (paper/paper_0805.tex, Section 3): adds, on top
    # of alpha*Phi_1, the weighted-TV term  beta * sum_j (1 + |omega_j|^p) |c_j|
    # with omega_j = (a_j, b_j).  It prices distant neurons out and supplies the
    # tightness Phi_1 lacks (existence by narrow compactness; confined support).
    # moment_beta = 0 turns it off (default => today's behavior).  When on it is
    # confined to its meaningful regime -- a non-sphere activation (|omega| is a
    # free scale, not gauge-fixed to the sphere), the q=1 log family (power == 1),
    # signed kind, and profile insertion -- and PDAP raises otherwise.
    moment_beta: float = 0.0    # 0 => moment axis off; > 0 => on
    moment_order: float = 2.0   # p in the weight w_p(omega) = 1 + |omega|^p
    # How the moment weight w_p enters the objective (paper/paper_0805.tex).
    #   "additive_moment"   -- the preserved comparator
    #                          J = l^M + alpha*sum phi(|c_n|) + beta*sum w_p|c_n|,
    #                          i.e. the moment norm as a separate weighted-L1 term.
    #   "normalized_moment" -- the revised paper objective (Section 3),
    #                          J = l^M + alpha*sum phi(w_p(omega_n)*|c_n|),
    #                          the penalty of the *normalized* measure mu_p = w_p*mu.
    #                          moment_beta plays no role and must be 0; moment_order
    #                          p is active here even at beta = 0.
    # Implemented by the substitution u = w_p*c, which turns the normalized problem
    # into the ordinary one with the dictionary K replaced by K_p = K/w_p.
    objective: str = "additive_moment"


@dataclass
class TrainingConfig:
    """How the model is fit: outer PDAP loop + SSN solver + insertion constants."""

    # outer PDAP loop
    num_iterations: int = 10      # T_out
    num_insertion: int = 50       # N_trial, candidates sampled per iteration
    max_insert: int = 15          # N_ins, cap on atoms inserted per iteration (batch mode)
    prune_amp_tol: float = 1e-8   # eps_prune: drop atoms with |c_n| <= this
    # --- paper-conformance axes (paper/paper_0805.tex, Section 5) --------------
    # Each defaults to the behavior in force before the revised paper, so the
    # default config reproduces the preserved comparator end to end.
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
    lr: float = 1.0
    method: str = "levenberg_marquardt"   # "levenberg_marquardt" | "steihaug_cg"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0
    fit_outer_iterations: int = 20
    display_every: int = 2
    # insertion numeric constants (src/PDAP/insertion.py)
    # Nonhomogeneous Algorithm 1: absolute Euclidean distance in omega=(a,b).
    # Sphere searches: cosine-similarity gap.
    ins_merge_tol: float = 1e-2
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
