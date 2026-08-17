"""The unified PDAP outer loop.

``PDAP`` is a pure trainer: it holds only configuration (the objective, the SSN
solver settings, and the insertion constants).  The model and the data are
arguments to :meth:`PDAP.fit`, which mutates the model's support in place and
returns a :class:`History` of per-iteration metrics — the runtime state lives
with the caller, not the trainer.

The loop is model-agnostic: it drives the model through its contract
(:class:`src.models.base.PDAPModel`) and runs the SSN outer solve, warm start,
and insertion as trainer steps (:mod:`ssn_solve`, :mod:`warmstart`,
:mod:`insertion`); evaluation is recorded into the :class:`History`.

Two loop orders, selected by ``training.loop_order``:

  correction_first (preserved)
    init:  insert -> warm-start -> set_atoms
    loop:  ssn_solve -> prune -> record -> insert -> warm-start -> set_atoms
    The last inserted batch is never corrected or pruned, and is counted in
    ``final_neurons``; the recorded metrics are unaffected, being snapshots taken
    before each insertion.

  insertion_first (the paper's Algorithms 1 and 2)
    loop:  insert -> correct -> prune -> record, stopping as soon as no candidate
    clears the insertion threshold.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from ..config.activations import get_growth, get_use_sphere
from ..models.signed import SignedModel
from .history import History, objective_value
from .insertion import finite_step, profile_threshold
from .radius import certificate_radius, sample_extent
from .ssn_solve import (
    ALGORITHM2_COEFFICIENT_SOLVER,
    ALGORITHM2_PROX_RHO,
    Objective,
    SolverConfig,
    ssn_solve,
)
from .warmstart import warm_start

logger = logging.getLogger(__name__)


class PDAP:
    def __init__(self, cfg) -> None:
        """Configure the trainer from a composed config (``model`` / ``training``).

        Only configuration is read here; the model is built by
        :func:`src.models.build_model` and passed to :meth:`fit` along with the
        data — both owned by the caller.
        """
        m, t = cfg.model, cfg.training
        if m.insertion not in ("profile", "finite_step"):
            raise ValueError(f"model.insertion must be 'profile' or 'finite_step', got {m.insertion!r}")

        self.insertion_kind = m.insertion
        self._use_sphere = get_use_sphere(m.activation)
        self._growth = get_growth(m.activation)

        # Paper-conformance axes.  Each is validated here rather than at use, so a
        # typo fails at construction instead of midway through a sweep.
        for field, value, allowed in (
            ("model.objective", m.objective, ("additive_moment", "normalized_moment")),
            ("training.insert_init", t.insert_init, ("warm_start", "guaranteed")),
            ("training.insert_mode", t.insert_mode, ("batch", "sequential")),
            ("training.loop_order", t.loop_order, ("correction_first", "insertion_first")),
            ("training.radial_cap", t.radial_cap, ("fixed", "theorem")),
        ):
            if value not in allowed:
                raise ValueError(f"{field} must be one of {allowed}, got {value!r}")

        self.objective_kind = m.objective
        self.insert_init = t.insert_init
        self.insert_mode = t.insert_mode
        self.correction_guard = bool(t.correction_guard)
        self.loop_order = t.loop_order
        self.radial_cap = t.radial_cap
        self.coefficient_solver_provenance: dict[str, str | float] = {}

        if (
            m.insertion == "finite_step"
            and m.objective != "normalized_moment"
            and m.moment_beta == 0.0
        ):
            if m.kind != "signed":
                raise ValueError(
                    "finite_step insertion is Algorithm 2 and requires a signed model; "
                    f"got kind={m.kind!r}"
                )
            if not self._use_sphere:
                raise ValueError(
                    "finite_step insertion is Algorithm 2 and requires a sphere activation; "
                    f"got activation={m.activation!r}"
                )
            if m.power not in (1.0, 2.0, 3.0):
                raise ValueError(
                    "finite_step insertion supports activation powers 1, 2, or 3; "
                    f"got power={m.power}"
                )
            if m.gamma != 0.0:
                raise ValueError(
                    "finite_step insertion minimizes the power penalty and requires "
                    f"gamma == 0; got gamma={m.gamma}"
                )
            if m.power == 1.0:
                self.coefficient_solver_provenance = {
                    "coefficient_solver": "soft_threshold",
                }
            else:
                self.coefficient_solver_provenance = {
                    "coefficient_solver": ALGORITHM2_COEFFICIENT_SOLVER,
                    "rho": ALGORITHM2_PROX_RHO,
                }

        # The revised objective replaces the additive moment term by the moment
        # weight inside phi; carrying both would price the same distance twice and
        # make the comparator meaningless.
        if m.objective == "normalized_moment":
            if m.moment_beta > 0.0:
                raise ValueError(
                    "model.objective='normalized_moment' is the revised paper objective "
                    "J = l^M + alpha*sum phi(w_p(omega)|c|), in which the moment norm is "
                    f"not an additive term; it requires moment_beta == 0, got {m.moment_beta}"
                )
            # The u = w_p*c substitution needs theta to be exactly the per-atom outer
            # weight, and Algorithm 1's penalty is the q=1 log family phi_gamma.
            if m.kind != "signed":
                raise ValueError(
                    f"model.objective='normalized_moment' requires a signed model; got kind={m.kind!r}"
                )
            if m.insertion != "profile":
                raise ValueError(
                    "model.objective='normalized_moment' is Algorithm 1 (profile insertion); "
                    f"got insertion={m.insertion!r}.  The k-homogeneous formulation behind "
                    "finite_step normalizes to the sphere instead and carries no w_p."
                )
            if m.power != 1.0:
                raise ValueError(
                    "model.objective='normalized_moment' requires power == 1 (the q=1 log "
                    f"penalty family of Algorithm 1); got power={m.power}"
                )
        if t.insert_init == "guaranteed" and m.kind != "signed":
            raise ValueError(
                "training.insert_init='guaranteed' is the signed theorem step; "
                f"got kind={m.kind!r}"
            )

        # Parameter-moment axis validation: confine it to the regime where it is
        # meaningful (|omega| is a free scale, not sphere-gauge-fixed) and where it
        # is implemented (signed kind, q=1 log family, profile insertion).  When
        # off (moment_beta == 0) nothing is checked -- today's behavior.
        if not m.moment_beta >= 0.0:
            raise ValueError(
                f"moment_beta must be nonnegative; got moment_beta={m.moment_beta}"
            )
        if not m.moment_order > 0.0:
            raise ValueError(
                f"moment_order must be positive; got moment_order={m.moment_order}"
            )
        if m.moment_beta > 0.0:
            if m.kind != "signed":
                raise ValueError(
                    f"moment_beta > 0 requires a signed model (kind='signed'); got kind={m.kind!r}"
                )
            if self._use_sphere:
                raise ValueError(
                    f"moment_beta > 0 requires a non-sphere activation (|omega| a free scale); "
                    f"activation {m.activation!r} is sphere-gauge-fixed (|omega|=1), so the "
                    f"moment term is a degenerate constant"
                )
            if m.power != 1.0:
                raise ValueError(
                    f"moment_beta > 0 requires power == 1 (the q=1 log penalty family); "
                    f"got power={m.power}"
                )
            if m.insertion != "profile":
                raise ValueError(
                    f"moment_beta > 0 requires profile insertion; got insertion={m.insertion!r}"
                )

        # The objective (what is minimized) and the SSN solver settings.
        self.objective = Objective(
            alpha=m.alpha, gamma=m.gamma, th=m.th, loss_weights=tuple(m.loss_weights),
            moment_beta=m.moment_beta, moment_order=m.moment_order, kind=m.objective,
        )
        self.solver = SolverConfig(
            lr=t.lr, method=t.method, max_ls_iter=t.max_ls_iter,
            tolerance_ls=t.tolerance_ls, tolerance_grad=t.tolerance_grad, sigmamax=t.sigmamax,
        )
        # outer-loop + insertion settings
        self.fit_outer_iterations = t.fit_outer_iterations
        self.ins_merge_tol = t.ins_merge_tol
        self.lbfgs_lr = t.lbfgs_lr
        self.lbfgs_steps = t.lbfgs_steps

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _sphere(self, d: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample N candidate neurons uniformly on S^d in R^{d+1}."""
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return v[:, :d].contiguous(), v[:, d].contiguous()

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor, biases: torch.Tensor, outer_weights: torch.Tensor,
        amp_tol: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Defensive gate: drop atoms whose outer weight is negligible.

        An atom with ``|c| <= amp_tol`` is effectively the zero measure (the
        regularizer ``alpha * ||mu||_M`` charges per ``|c|``), so it is removed.
        No clustering/merging is performed; redundant near-duplicate atoms are
        harmless and handled by the next solver iteration.

        Returns ``(W (n,d), b (n,), c (n,), pruned)`` with ``pruned`` the number
        of atoms dropped.
        """
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        keep = ow.abs() > amp_tol
        w_out, b_out, ow_out = w[keep], b[keep], ow[keep]
        pruned = n - int(keep.sum().item())
        return w_out, b_out, ow_out, pruned

    # ------------------------------------------------------------------ #
    # Per-step helpers (model + data are arguments, not state).
    # ------------------------------------------------------------------ #
    def _residual(self, model, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction - target)`` for value and gradient.

        Both models predict with no atoms — the semiconcave envelope, or the
        signed zero network (V = 0) — so the first insertion sees ``-target``.
        """
        X, V, dV = data
        Vp, dVp = model.predict_tensors(X)
        return (Vp - V).detach(), (dVp - dV).detach()

    def _search_radius(self, model, data_train):
        """Radial search cap for this iterate; ``None`` keeps the fixed comparison bound.

        The theorem radius depends on the iterate only through ``||r_mu||``, so it
        is recomputed each insertion; everything else is fixed by the activation
        and the samples.
        """
        if self.radial_cap != "theorem" or self._use_sphere:
            return None
        res_v, res_dv = self._residual(model, data_train)
        w1, w2 = self.objective.loss_weights
        M = int(data_train[0].shape[0])
        # ||r_mu|| in the same empirical norm the fidelity uses: l^M = ||r||^2 / 2.
        residual_norm = float(
            torch.sqrt(
                (w1 * res_v.pow(2).sum() + w2 * res_dv.pow(2).sum()) / max(M, 1)
            )
        )
        return certificate_radius(
            self._growth,
            extent=sample_extent(data_train[0]),
            residual_norm=residual_norm,
            alpha=self.objective.alpha,
            moment_order=self.objective.moment_order,
        )

    def _warm_start(self, model, data_train, W, b, verbose: bool) -> torch.Tensor:
        """Coordinate-descent initial outer weights for new atoms (W, b)."""
        o = self.objective
        return warm_start(
            W, b, self._residual(model, data_train), data_train[0],
            activation=model.activation, power=model.power,
            loss_weights=o.loss_weights, alpha=o.alpha, th=o.th, gamma=o.gamma,
            use_sphere=self._use_sphere, nonneg=not isinstance(model, SignedModel),
            moment_beta=o.moment_beta, moment_order=o.moment_order,
            normalized=o.normalized, verbose=verbose,
        )

    def _initial_outer_weights(self, model, data_train, W, b, c, verbose: bool) -> torch.Tensor:
        """Outer weights for freshly inserted atoms: warm-start (profile, c is
        None) or the finite-step's own coefficients."""
        if c is None:
            return self._warm_start(model, data_train, W, b, verbose)
        return torch.as_tensor(c, dtype=torch.float64).reshape(-1)

    # ------------------------------------------------------------------ #
    # Insertion dispatch.  The insertion-candidate merge tolerance
    # (self.ins_merge_tol, default 1e-2) is independent of the prune amplitude
    # gate (fit's amp_tol, used only in prune_small_weights).
    # ------------------------------------------------------------------ #
    def _insert(self, model, data_train, num_insertion: int, max_insert: int, verbose: bool):
        """Return (W, b, c) where c is None for the profile strategy (needs warm-start)."""
        # Sequential insertion admits the highest-ranked candidate returned by the
        # multistart search, one atom per outer iteration. The theorem's rate bound
        # additionally requires that this candidate be an exact global maximizer; a
        # batch also introduces cross terms through its shared frozen residual.
        if self.insert_mode == "sequential":
            max_insert = 1
        X = data_train[0]
        res_v, res_dv = self._residual(model, data_train)
        existing = None
        if model.n_neurons > 0:
            Wc, bc, _ = model.get_atoms()
            existing = (Wc, bc) if Wc.shape[0] > 0 else None

        d = int(model.input_dim)
        two_sided = isinstance(model, SignedModel)
        common = dict(
            activation=model.activation, power=model.power,
            loss_weights=self.objective.loss_weights, alpha=self.objective.alpha,
            sample_sphere=lambda N: self._sphere(d, N), N=num_insertion,
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            use_sphere=self._use_sphere, existing_atoms=existing, verbose=verbose,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        radius = self._search_radius(model, data_train)
        if self.insertion_kind == "profile":
            return profile_threshold(
                X, res_v, res_dv, two_sided=two_sided,
                moment_beta=self.objective.moment_beta, moment_order=self.objective.moment_order,
                normalized=self.objective.normalized, insert_init=self.insert_init,
                radius=radius, **common,
            )
        W, b, c = finite_step(
            X, res_v, res_dv, radius=radius, **common,
        )
        return W, b, c

    def _correct(self, model, data_train, verbose: bool) -> None:
        """Run the SSN coefficient solve, optionally rejecting a worsening step.

        SSN is a local method on a nonconvex penalty and carries no descent
        guarantee -- for ``q < 1`` the penalty is not even locally Lipschitz at the
        origin.  With ``correction_guard`` on, a correction that raises the
        objective is discarded and the post-insertion coefficients are kept, which
        is what the paper's algorithms prescribe and what makes the outer loop
        monotone.
        """
        if not self.correction_guard:
            ssn_solve(
                model, data_train, self.objective, self.solver,
                iterations=self.fit_outer_iterations, verbose=verbose,
            )
            return

        before = objective_value(model, self.objective, data_train)
        snapshot = {name: t.detach().clone() for name, t in model.state_dict().items()}
        ssn_solve(
            model, data_train, self.objective, self.solver,
            iterations=self.fit_outer_iterations, verbose=verbose,
        )
        after = objective_value(model, self.objective, data_train)
        if after > before:
            model.load_state_dict(snapshot)
            if verbose:
                logger.info(
                    "Correction guard  objective=%.6e->%.6e  accepted=false",
                    before,
                    after,
                )
        elif verbose:
            logger.info(
                "Correction guard  objective=%.6e->%.6e  accepted=true",
                before,
                after,
            )

    def _prune(self, model, amp_tol: float) -> int:
        """Drop atoms with ``|c_n| <= amp_tol``; return how many were removed."""
        W, b, c = model.get_atoms()
        W, b, c, pruned = self.prune_small_weights(W, b, c, amp_tol=amp_tol)
        model.set_atoms(W, b, c)
        return pruned

    def _insert_atoms(
        self, model, data_train, num_insertion: int, max_insert: int, verbose: bool
    ) -> int:
        """Insert accepted atoms into ``model``; return how many were added."""
        W_np, b_np, c_new = self._insert(model, data_train, num_insertion, max_insert, verbose)
        W_new = torch.as_tensor(W_np, dtype=torch.float64)
        b_new = torch.as_tensor(b_np, dtype=torch.float64)
        if W_new.shape[0] == 0:
            return 0
        c_new = self._initial_outer_weights(model, data_train, W_new, b_new, c_new, verbose)
        if model.n_neurons > 0:
            W, b, c = model.get_atoms()
            W_new = torch.cat([W, W_new], dim=0)
            b_new = torch.cat([b, b_new], dim=0)
            c_new = torch.cat([c, c_new], dim=0)
        model.set_atoms(W_new, b_new, c_new)
        return int(W_np.shape[0])

    # ------------------------------------------------------------------ #
    # The PDAP outer loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        model,
        data_train,
        data_valid,
        *,
        num_iterations: int,
        num_insertion: int,
        max_insert: int = 15,
        amp_tol: float = 1e-8,
        verbose: bool = True,
    ) -> History:
        """Train ``model`` in place on ``data_train``; return the :class:`History`."""
        history = History()
        o = self.objective
        if verbose:
            logger.info("PDAP run")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24s |", "model", type(model).__name__)
            logger.info("  | %-16s | %-24s |", "insertion rule", self.insertion_kind.replace("_", " "))
            logger.info("  | %-16s | %-24s |", "samples",
                        f"{int(data_train[0].shape[0])} train, {int(data_valid[0].shape[0])} validation")
            logger.info("  | %-16s | %-24d |", "input dimension", int(model.input_dim))
            logger.info("  | %-16s | %-24.2e |", "alpha", o.alpha)
            logger.info("  | %-16s | %-24.2e |", "gamma", o.gamma)
            logger.info("  | %-16s | %-24.3g |", "activation power", model.power)
            logger.info("  +------------------+--------------------------+")

        if self.loop_order == "insertion_first":
            return self._fit_insertion_first(
                model, data_train, data_valid, history,
                num_iterations=num_iterations, num_insertion=num_insertion,
                max_insert=max_insert, amp_tol=amp_tol, verbose=verbose,
            )

        # --- initialization: insert + warm-start ---
        W_np, b_np, c = self._insert(model, data_train, num_insertion, max_insert, verbose)
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            # No candidate clears the insertion threshold.  Under strong
            # regularization the zero measure is a valid terminal PDAP result,
            # not a training failure; record it once and skip the SSN loop,
            # which requires a nonempty outer-parameter vector.
            history.record(model, self.objective, data_train, data_valid)
            history.final_neurons = 0
            if verbose:
                logger.info("Initial insertion accepted no atoms; returning the zero measure")
            return history
        c = self._initial_outer_weights(model, data_train, W, b, c, verbose)
        model.set_atoms(W, b, c)
        if verbose:
            max_weight = float(c.abs().max().item()) if c.numel() else 0.0
            logger.debug("Initial support  neurons=%d  max |output|=%.2e", int(W.shape[0]), max_weight)
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            supp_before = model.n_neurons

            # 1. SSN on outer weights (inner weights frozen)
            self._correct(model, data_train, verbose)

            # 2. prune: defensive gate — drop negligible atoms
            pruned = self._prune(model, amp_tol)

            # 3. record (evaluation lives in History.record, not the loop)
            history.record(model, self.objective, data_train, data_valid)

            if verbose:
                if supp_before != model.n_neurons:
                    logger.debug(
                        "Support changed during pruning at iteration %d: %d -> %d neurons",
                        i + 1, supp_before, model.n_neurons,
                    )
                logger.info(
                    "  | %-7s | %7d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", model.n_neurons, pruned,
                    history.train_loss[-1], history.val_loss[-1],
                    history.err_l2_val[-1], history.err_h1_val[-1],
                )

            # 4. insert new neurons + warm-start
            self._insert_atoms(model, data_train, num_insertion, max_insert, verbose)

        history.final_neurons = int(model.n_neurons)
        if verbose:
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")
            logger.info("Result")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24d |", "best iteration", history.best_iteration + 1)
            logger.info("  | %-16s | %-24.3e |", "best train loss", history.best_train_loss)
            logger.info("  | %-16s | %-24d |", "best neurons", history.best_neurons)
            logger.info("  +------------------+--------------------------+")
        return history

    def _fit_insertion_first(
        self, model, data_train, data_valid, history: History, *,
        num_iterations: int, num_insertion: int, max_insert: int,
        amp_tol: float, verbose: bool,
    ) -> History:
        """The paper's loop: per iteration insert -> correct -> prune -> record.

        Every inserted batch is corrected before the loop ends, so ``final_neurons``
        describes the same network as the last recorded metrics -- unlike the
        preserved order, which ends on an uncorrected insertion.  The loop also
        terminates when no candidate clears the insertion threshold, as the paper's
        algorithms specify; the preserved order keeps iterating instead.
        """
        if verbose:
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "added", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            # 1. insert; no accepted candidate is the algorithms' stopping rule
            added = self._insert_atoms(model, data_train, num_insertion, max_insert, verbose)
            if added == 0:
                if verbose:
                    logger.info(
                        "No candidate clears the insertion threshold at iteration %d; stopping",
                        i + 1,
                    )
                break

            # 2. correct (rejected if it raises the objective, when guarded)
            self._correct(model, data_train, verbose)

            # 3. prune
            pruned = self._prune(model, amp_tol)

            # 4. record
            history.record(model, self.objective, data_train, data_valid)
            if verbose:
                logger.info(
                    "  | %-7s | %7d | %6d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", model.n_neurons, added, pruned,
                    history.train_loss[-1], history.val_loss[-1],
                    history.err_l2_val[-1], history.err_h1_val[-1],
                )

        if not history.train_loss:
            # Nothing was ever inserted: the zero measure is the terminal result.
            history.record(model, self.objective, data_train, data_valid)
        history.final_neurons = int(model.n_neurons)
        if verbose:
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("Result")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24d |", "best iteration", history.best_iteration + 1)
            logger.info("  | %-16s | %-24.3e |", "best train loss", history.best_train_loss)
            logger.info("  | %-16s | %-24d |", "best neurons", history.best_neurons)
            logger.info("  +------------------+--------------------------+")
        return history
