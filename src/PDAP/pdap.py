"""PDAP support updates and complete outer steps.

SparseModel owns the network and its fitting; Penalty owns the penalty term
and its proximal maps. PDAP owns candidate search, insertion, coefficient
correction, pruning and the no-candidate stopping rule.
The external training loop owns repetition, measurement and history.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader

from ..config.activations import get_growth
from ..config.schema import TrainingConfig
from ..data import full_batch
from ..models import Penalty, SparseModel
from ..SSN import SSN
from .insertion import FIXED_LOG_CLAMP, finite_step, profile_threshold
from .radius import certificate_radius, sample_extent
from .ssn_refine import ssn_refine_outerweights
from .warmstart import warm_start


class PDAP:
    def __init__(
        self, training: TrainingConfig, penalty: Penalty, *, insertion: str = "profile",
    ) -> None:
        """Read the settings for one outer step; run_training owns its repetition.

        penalty is shared by insertion, correction and evaluation. Training setup
        validates training with validate_optimizer_config and the insertion rule
        with validate_model_config before construction.
        """
        self.penalty = penalty
        self.insertion_kind = insertion
        self.insert_init = training.insert_init
        self.insert_mode = training.insert_mode
        self.correction_guard = bool(training.correction_guard)
        self.loop_order = training.loop_order
        self.radial_cap = training.radial_cap
        # The cap the last insertion search used, and whether the theorem
        # supplied it. Recorded per iteration so a `radial_cap=theorem` run
        # whose hypothesis p > s1 fails stays distinguishable from `fixed`:
        # both search inside exp(5), so the metrics are otherwise identical.
        self._last_search_radius: float | None = None
        self._last_theorem_applied: float | None = None

        self.solver: dict[str, Any] = dict(
            method=training.method, max_ls_iter=training.max_ls_iter,
            tolerance_ls=training.tolerance_ls, tolerance_grad=training.tolerance_grad,
            sigmamax=training.sigmamax,
        )
        # outer-loop + insertion settings
        self.fit_outer_iterations = training.fit_outer_iterations
        self.ins_merge_tol = training.ins_merge_tol
        self.lbfgs_lr = training.lbfgs_lr
        self.lbfgs_steps = training.lbfgs_steps
        self.num_insertion = training.num_insertion
        self.max_insert = training.max_insert
        self.prune_amp_tol = training.prune_amp_tol

    def step(
        self, problem: SparseModel, train_loader: DataLoader, *, verbose: bool = False,
    ) -> dict:
        """Insert, correct and prune one complete step of the paper algorithms.

        Mutate the supplied network and return execution facts. No admissible
        insertion leaves it unchanged and reports ``stop_reason="no_candidate"``;
        this is the search stopping rule, not a convergence certificate.
        The historical correction-first schedule uses the explicit operations
        below so its measurement can still precede the trailing insertion.
        """
        diagnostics = self.insert(problem, train_loader, verbose=verbose)
        diagnostics["stop_reason"] = "no_candidate" if diagnostics["inserted"] == 0 else None
        if diagnostics["stop_reason"] is None:
            diagnostics.update(self.train_ssn_outerweights_and_prune(problem, train_loader, verbose=verbose))
        return diagnostics

    def initialize(
        self, problem: SparseModel, train_loader: DataLoader, *, verbose: bool = False,
    ) -> dict:
        """Perform the historical correction-first initial support replacement."""
        W, b, c = self._insert(
            problem, train_loader, self.num_insertion, self.max_insert, verbose,
        )
        count = int(W.shape[0])
        if count:
            problem.set_atoms(
                torch.as_tensor(W, dtype=torch.float64),
                torch.as_tensor(b, dtype=torch.float64),
                torch.as_tensor(c, dtype=torch.float64).reshape(-1),
            )
        diagnostics = self._search_diagnostics(count)
        diagnostics["stop_reason"] = "no_candidate" if count == 0 else None
        return diagnostics

    def insert(
        self, problem: SparseModel, train_loader: DataLoader, *, verbose: bool = False,
    ) -> dict:
        """Extend the current support with accepted, initialized candidates."""
        W_np, b_np, c_new = self._insert(
            problem, train_loader, self.num_insertion, self.max_insert, verbose,
        )
        W_new = torch.as_tensor(W_np, dtype=torch.float64)
        b_new = torch.as_tensor(b_np, dtype=torch.float64)
        if W_new.shape[0]:
            c_new = torch.as_tensor(c_new, dtype=torch.float64).reshape(-1)
            if problem.n_neurons > 0:
                W_new = torch.cat([problem.inner_weight.detach(), W_new], dim=0)
                b_new = torch.cat([problem.inner_bias.detach(), b_new], dim=0)
                c_new = torch.cat([problem.c.detach(), c_new], dim=0)
            problem.set_atoms(W_new, b_new, c_new)
        return self._search_diagnostics(int(W_np.shape[0]))

    def train_ssn_outerweights_and_prune(
        self, problem: SparseModel, train_loader: DataLoader, *, verbose: bool = False,
    ) -> dict:
        """Apply the accepted SSN correction, then remove negligible atoms."""
        diagnostics = self.train_ssn_outerweights(problem, train_loader, verbose=verbose)
        diagnostics["corrections"] = 1
        diagnostics["pruned"] = self._prune(problem, self.prune_amp_tol)
        return diagnostics

    def train_ssn_outerweights(
        self, problem: SparseModel, train_loader: DataLoader, *, verbose: bool = False,
    ) -> dict:
        """Prepare a correction, run SSN, and apply its accepted coefficients."""
        fitting, hess_obj, moment_weight = self.coefficient_fitting(problem, train_loader)
        # The normalized coefficients u = w_p*c; the physical ones are c = u/w_p.
        u = torch.nn.Parameter(problem.c.detach() * moment_weight)
        optimizer = SSN([u], self.penalty, hess_obj, **self.solver)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            value = fitting(u)
            value.backward()
            return value

        diagnostics = ssn_refine_outerweights(
            optimizer, u, closure, iterations=self.fit_outer_iterations,
            correction_guard=self.correction_guard, verbose=verbose,
        )
        with torch.no_grad():
            problem.c.copy_(u.detach() / moment_weight)
        return diagnostics

    def coefficient_fitting(
        self, problem: SparseModel, train_loader: DataLoader,
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]:
        """The fitting term of the normalized coefficients u = w_p*c, its Hessian, and w_p.

        SSN adds the penalty to this fitting term.
        """
        X, V, dV = full_batch(train_loader)
        value_jacobian, gradient_jacobian = problem.kernel(X)
        moment_weight = problem.moment_weight_of_atoms()
        value_jacobian = value_jacobian / moment_weight
        gradient_jacobian = gradient_jacobian / moment_weight
        V, dV = (target.detach().reshape(-1).clone() for target in (V, dV))

        def fitting(u: torch.Tensor) -> torch.Tensor:
            return problem.obj((value_jacobian @ u, gradient_jacobian @ u), (V, dV))

        return fitting, problem.hess_obj(value_jacobian, gradient_jacobian), moment_weight

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
        with torch.no_grad():
            Vp, dVp = problem(X)
        residual = X.detach(), (Vp - V).detach(), (dVp - dV).detach()
        existing = (problem.inner_weight.detach(), problem.inner_bias.detach()) if problem.n_neurons else None

        d = int(problem.input_dim)
        radius = self._search_radius(problem, residual)
        search_settings: dict[str, Any] = dict(
            sample_sphere=lambda N: self._sphere(d, N), N=num_insertion,
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            existing_atoms=existing, radius=radius, verbose=verbose,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        if self.insertion_kind == "profile":
            W, b, c = profile_threshold(
                problem, self.penalty, residual, two_sided=True, insert_init=self.insert_init,
                **search_settings,
            )
        else:
            W, b, c = finite_step(problem, self.penalty, residual, **search_settings)
        if c is None and W.shape[0] > 0:
            c = warm_start(
                torch.as_tensor(W, dtype=torch.float64),
                torch.as_tensor(b, dtype=torch.float64),
                residual, problem, self.penalty, verbose=verbose,
            )
        return W, b, c

    def _prune(self, problem, amp_tol: float) -> int:
        """Drop atoms with ``|c_n| <= amp_tol``; return how many were removed."""
        W, b, c, pruned = self.prune_small_weights(
            problem.inner_weight, problem.inner_bias, problem.c, amp_tol=amp_tol,
        )
        if pruned:
            problem.set_atoms(W, b, c)
        return pruned

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

    def _sphere(self, d: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample N candidate neurons uniformly on S^d in R^{d+1}."""
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return v[:, :d].contiguous(), v[:, d].contiguous()

    def _search_radius(self, problem, residual):
        """Bound candidate search using the theorem, numerical cap and fallback."""
        cap = math.exp(FIXED_LOG_CLAMP)
        if self.radial_cap != "theorem" or problem.use_sphere:
            self._last_search_radius = 1.0 if problem.use_sphere else cap
            self._last_theorem_applied = -1.0
            return None
        X, res_v, res_dv = residual
        assert problem.moment_order is not None  # Nonhomogeneous training was validated before search.
        zeros = torch.zeros_like(res_v), torch.zeros_like(res_dv)
        residual_norm = float(torch.sqrt(2 * problem.obj((res_v, res_dv), zeros)))
        radius = certificate_radius(
            get_growth(problem.activation_name) if problem.activation_name is not None else None,
            extent=sample_extent(X), residual_norm=residual_norm,
            alpha=self.penalty.alpha, moment_order=problem.moment_order,
        )
        self._last_search_radius = cap if radius is None else min(float(radius), cap)
        self._last_theorem_applied = 0.0 if radius is None else 1.0
        return None if radius is None else self._last_search_radius

    def _search_diagnostics(self, inserted: int) -> dict:
        return {
            "searches": 1,
            "inserted": inserted,
            "search_radius": self._last_search_radius,
            "radius_theorem_applied": self._last_theorem_applied,
        }
