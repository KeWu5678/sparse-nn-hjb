"""The unified PDAP outer loop.

``PDAP`` holds iteration and search settings. ``fit`` receives SparseProblem
and external loaders, mutates the network support, and returns a History.
SparseProblem owns the network and mathematical operations; SSN performs each
complete coefficient correction. PDAP owns insertion and pruning.

Two loop orders, selected by ``training.loop_order``:

  correction_first (preserved)
    init:  insert -> warm-start -> set_atoms
    loop:  correct -> prune -> record -> insert -> warm-start -> set_atoms
    The last inserted batch is never corrected or pruned, and is counted in
    ``final_neurons``; the recorded metrics are unaffected, being snapshots taken
    before each insertion.

  insertion_first (the paper's Algorithms 1 and 2)
    loop:  insert -> correct -> prune -> record, stopping as soon as no candidate
    is accepted.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sized
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from ..config.activations import get_growth
from ..data import ValueSampleNormalizer, full_batch
from ..problems import SparseProblem
from ..SSN import SSN
from .history import History
from .insertion import FIXED_LOG_CLAMP, finite_step, profile_threshold
from .radius import certificate_radius, sample_extent
from .warmstart import warm_start

logger = logging.getLogger(__name__)


class PDAP:
    def __init__(self, training, *, insertion: str = "profile") -> None:
        """Configure iteration and search settings; fit receives the problem."""
        t = training
        if insertion not in ("profile", "finite_step"):
            raise ValueError(f"insertion must be 'profile' or 'finite_step', got {insertion!r}")
        self.insertion_kind = insertion

        # Paper-conformance axes.  Each is validated here rather than at use, so a
        # typo fails at construction instead of midway through a sweep.
        for field, value, allowed in (
            ("training.insert_init", t.insert_init, ("warm_start", "guaranteed")),
            ("training.insert_mode", t.insert_mode, ("batch", "sequential")),
            ("training.loop_order", t.loop_order, ("correction_first", "insertion_first")),
            ("training.radial_cap", t.radial_cap, ("fixed", "theorem")),
        ):
            if value not in allowed:
                raise ValueError(f"{field} must be one of {allowed}, got {value!r}")

        self.insert_init = t.insert_init
        self.insert_mode = t.insert_mode
        self.correction_guard = bool(t.correction_guard)
        self.loop_order = t.loop_order
        self.radial_cap = t.radial_cap
        # The cap the last insertion search used, and whether the theorem
        # supplied it. Recorded per iteration so a `radial_cap=theorem` run
        # whose hypothesis p > s1 fails stays distinguishable from `fixed`:
        # both search inside exp(5), so the metrics are otherwise identical.
        self._last_search_radius: float | None = None
        self._last_theorem_applied: float | None = None

        self.solver: dict[str, Any] = dict(
            method=t.method, max_ls_iter=t.max_ls_iter,
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
    # Per-step helpers (problem is an argument, not trainer state).
    # ------------------------------------------------------------------ #
    def _search_radius(self, problem, residual):
        """Bound candidate search using the theorem, numerical cap and fallback."""
        cap = math.exp(FIXED_LOG_CLAMP)
        if self.radial_cap != "theorem" or problem.use_sphere:
            self._last_search_radius = 1.0 if problem.use_sphere else cap
            self._last_theorem_applied = -1.0
            return None
        X, res_v, res_dv = residual
        zeros = torch.zeros_like(res_v), torch.zeros_like(res_dv)
        residual_norm = float(torch.sqrt(2 * problem.obj((res_v, res_dv), zeros)))
        radius = certificate_radius(
            get_growth(problem.activation_name) if problem.activation_name is not None else None,
            extent=sample_extent(X), residual_norm=residual_norm,
            alpha=problem.alpha, moment_order=problem.moment_order,
        )
        self._last_search_radius = cap if radius is None else min(float(radius), cap)
        self._last_theorem_applied = 0.0 if radius is None else 1.0
        return None if radius is None else self._last_search_radius

    # ------------------------------------------------------------------ #
    # Insertion dispatch.  The insertion-candidate merge tolerance
    # (self.ins_merge_tol, default 1e-2) is independent of the prune amplitude
    # gate (fit's amp_tol, used only in prune_small_weights).
    # ------------------------------------------------------------------ #
    def _insert(
        self, problem, train_loader, num_insertion: int, max_insert: int, verbose: bool
    ):
        """Return accepted atoms and their initialized outer weights."""
        # Sequential insertion admits the highest-ranked candidate returned by the
        # multistart search, one atom per outer iteration. The theorem's rate bound
        # additionally requires that this candidate be an exact global maximizer; a
        # batch also introduces cross terms through its shared frozen residual.
        if self.insert_mode == "sequential":
            max_insert = 1
        X, V, dV = full_batch(train_loader)
        Vp, dVp = problem.forward_with_gradient(X)
        residual = X.detach(), (Vp - V).detach(), (dVp - dV).detach()
        existing = (problem.inner_weight.detach(), problem.inner_bias.detach()) if problem.n_neurons else None

        d = int(problem.input_dim)
        common: dict[str, Any] = dict(
            sample_sphere=lambda N: self._sphere(d, N), N=num_insertion,
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            existing_atoms=existing, verbose=verbose,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        radius = self._search_radius(problem, residual)
        if self.insertion_kind == "profile":
            W, b, c = profile_threshold(
                problem, residual, two_sided=True, insert_init=self.insert_init,
                radius=radius, **common,
            )
        else:
            W, b, c = finite_step(
                problem, residual, radius=radius, **common,
            )
        if c is None and W.shape[0] > 0:
            c = warm_start(
                torch.as_tensor(W, dtype=torch.float64),
                torch.as_tensor(b, dtype=torch.float64),
                residual, problem, verbose=verbose,
            )
        return W, b, c

    def _correct(self, problem, train_loader, verbose: bool) -> None:
        """Prepare a correction, run SSN, and apply its accepted coefficients."""
        prepared = problem.prepare_coefficients(train_loader)
        theta = torch.nn.Parameter(
            problem.coefficient_coordinates(problem.c.detach(), prepared)
        )
        optimizer = SSN(
            [theta], problem, prepared, max_iter=self.fit_outer_iterations,
            correction_guard=self.correction_guard, verbose=verbose, **self.solver,
        )
        optimizer.step()
        with torch.no_grad():
            problem.c.copy_(problem.model_coefficients(theta.detach(), prepared))

    def _prune(self, problem, amp_tol: float) -> int:
        """Drop atoms with ``|c_n| <= amp_tol``; return how many were removed."""
        W, b, c, pruned = self.prune_small_weights(
            problem.inner_weight, problem.inner_bias, problem.c, amp_tol=amp_tol,
        )
        if pruned:
            problem.set_atoms(W, b, c)
        return pruned

    def _insert_atoms(
        self, problem, train_loader, num_insertion: int, max_insert: int, verbose: bool
    ) -> int:
        """Insert accepted atoms into ``problem``; return how many were added."""
        W_np, b_np, c_new = self._insert(
            problem, train_loader, num_insertion, max_insert, verbose,
        )
        W_new = torch.as_tensor(W_np, dtype=torch.float64)
        b_new = torch.as_tensor(b_np, dtype=torch.float64)
        if W_new.shape[0] == 0:
            return 0
        c_new = torch.as_tensor(c_new, dtype=torch.float64).reshape(-1)
        if problem.n_neurons > 0:
            W_new = torch.cat([problem.inner_weight.detach(), W_new], dim=0)
            b_new = torch.cat([problem.inner_bias.detach(), b_new], dim=0)
            c_new = torch.cat([problem.c.detach(), c_new], dim=0)
        problem.set_atoms(W_new, b_new, c_new)
        return int(W_np.shape[0])

    # ------------------------------------------------------------------ #
    # The PDAP outer loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        problem: SparseProblem,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        *,
        num_iterations: int,
        num_insertion: int,
        max_insert: int = 15,
        amp_tol: float = 1e-8,
        verbose: bool = True,
        reporting_normalizer: ValueSampleNormalizer | None = None,
    ) -> History:
        """Train in place; optionally undo data scaling for reporting errors only.

        ``reporting_normalizer`` must be the transform already applied to the
        supplied samples. It never changes the objective or checkpoint selection.
        With no transform, errors use the supplied samples' units as before.

        Loaders supply the complete dataset for each mathematical evaluation;
        batches do not introduce additional optimizer steps. Omitted validation
        uses the training loader.
        """
        if valid_loader is None:
            valid_loader = train_loader
        problem.validate_training(insertion=self.insertion_kind)
        # A trainer can be reused; a zero-iteration fit has no search to report.
        self._last_search_radius = None
        self._last_theorem_applied = None
        history = History(reporting_normalizer=reporting_normalizer)
        if verbose:
            logger.info("PDAP run")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24s |", "model", type(problem).__name__)
            logger.info("  | %-16s | %-24s |", "insertion rule", self.insertion_kind.replace("_", " "))
            logger.info("  | %-16s | %-24d |", "input dimension", int(problem.input_dim))
            if isinstance(train_loader.dataset, Sized) and isinstance(valid_loader.dataset, Sized):
                logger.info(
                    "  | %-16s | %-24s |", "samples",
                    f"{len(train_loader.dataset)} train, {len(valid_loader.dataset)} validation",
                )
            logger.info("  +------------------+--------------------------+")

        if self.loop_order == "insertion_first":
            return self._fit_insertion_first(
                problem, train_loader, valid_loader, history,
                num_iterations=num_iterations, num_insertion=num_insertion,
                max_insert=max_insert, amp_tol=amp_tol, verbose=verbose,
            )

        # --- initialization: insert + warm-start ---
        W_np, b_np, c = self._insert(
            problem, train_loader, num_insertion, max_insert, verbose,
        )
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            # No candidate clears the insertion threshold.  Under strong
            # regularization the zero measure is a valid terminal PDAP result,
            # not a training failure; record it once and skip the SSN loop,
            # which requires a nonempty outer-parameter vector.
            history.record(
                problem, train_loader, valid_loader,
                search_radius=self._last_search_radius,
                theorem_applied=self._last_theorem_applied,
            )
            history.final_neurons = 0
            if verbose:
                logger.info("Initial insertion accepted no atoms; returning the zero measure")
            return history
        c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
        problem.set_atoms(W, b, c)
        if verbose:
            max_weight = float(c.abs().max().item()) if c.numel() else 0.0
            logger.debug("Initial support  neurons=%d  max |output|=%.2e", int(W.shape[0]), max_weight)
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            supp_before = problem.n_neurons

            # 1. SSN on outer weights (inner weights frozen)
            self._correct(problem, train_loader, verbose)

            # 2. prune: defensive gate — drop negligible atoms
            pruned = self._prune(problem, amp_tol)

            # 3. record (evaluation lives in History.record, not the loop)
            history.record(
                problem, train_loader, valid_loader,
                search_radius=self._last_search_radius,
                theorem_applied=self._last_theorem_applied,
            )

            if verbose:
                if supp_before != problem.n_neurons:
                    logger.debug(
                        "Support changed during pruning at iteration %d: %d -> %d neurons",
                        i + 1, supp_before, problem.n_neurons,
                    )
                logger.info(
                    "  | %-7s | %7d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", problem.n_neurons, pruned,
                    history.train_loss[-1], history.val_loss[-1],
                    history.err_l2_val[-1], history.err_h1_val[-1],
                )

            # 4. insert new neurons + warm-start
            self._insert_atoms(problem, train_loader, num_insertion, max_insert, verbose)

        history.final_neurons = int(problem.n_neurons)
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
        self, problem, train_loader, valid_loader, history: History, *,
        num_iterations: int, num_insertion: int, max_insert: int,
        amp_tol: float, verbose: bool,
    ) -> History:
        """The paper's loop: per iteration insert -> correct -> prune -> record.

        Every inserted batch is corrected before the loop ends, so ``final_neurons``
        describes the same network as the last recorded metrics -- unlike the
        preserved order, which ends on an uncorrected insertion.  The loop also
        terminates when no candidate is accepted; the preserved order keeps
        iterating instead.
        """
        if verbose:
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "added", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            # 1. insert; no accepted candidate is the algorithms' stopping rule
            added = self._insert_atoms(
                problem, train_loader, num_insertion, max_insert, verbose,
            )
            if added == 0:
                if verbose:
                    logger.info(
                        "Insertion accepted no candidate at iteration %d; stopping",
                        i + 1,
                    )
                break

            # 2. correct (rejected if it raises the objective, when guarded)
            self._correct(problem, train_loader, verbose)

            # 3. prune
            pruned = self._prune(problem, amp_tol)

            # 4. record
            history.record(
                problem, train_loader, valid_loader,
                search_radius=self._last_search_radius,
                theorem_applied=self._last_theorem_applied,
            )
            if verbose:
                logger.info(
                    "  | %-7s | %7d | %6d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", problem.n_neurons, added, pruned,
                    history.train_loss[-1], history.val_loss[-1],
                    history.err_l2_val[-1], history.err_h1_val[-1],
                )

        if not history.train_loss:
            # Nothing was ever inserted: the zero measure is the terminal result.
            history.record(
                problem, train_loader, valid_loader,
                search_radius=self._last_search_radius,
                theorem_applied=self._last_theorem_applied,
            )
        history.final_neurons = int(problem.n_neurons)
        if verbose:
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("Result")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24d |", "best iteration", history.best_iteration + 1)
            logger.info("  | %-16s | %-24.3e |", "best train loss", history.best_train_loss)
            logger.info("  | %-16s | %-24d |", "best neurons", history.best_neurons)
            logger.info("  +------------------+--------------------------+")
        return history
