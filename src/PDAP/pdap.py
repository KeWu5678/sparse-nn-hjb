"""PDAP support updates and complete outer steps.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader

from ..config.activations import get_growth
from ..config.schema import TrainingConfig
from ..data import full_batch
from ..models import Penalty, SparseNetwork
from ..SSN import SSN
from .insertion import FIXED_LOG_CLAMP, finite_step, profile_threshold
from .radius import certificate_radius, sample_extent
from .ssn_refine import ssn_refine_outerweights
from .warmstart import warm_start


class PDAP:
    def __init__(
        self, training: TrainingConfig, penalty: Penalty, network: SparseNetwork,
        train_loader: DataLoader, *, insertion: str = "profile",
    ) -> None:
        """Read the settings for one outer step; run_training owns its repetition.

        penalty is shared by insertion, correction and evaluation. Training setup
        validates training with validate_optimizer_config and the insertion rule
        with validate_model_config before construction.
        """
        self.penalty = penalty
        self.network = network
        self.train_loader = train_loader
        self.insertion_kind = insertion
        self.insert_init = training.insert_init
        self.insert_mode = training.insert_mode
        self.correction_guard = bool(training.correction_guard)
        self.loop_order = training.loop_order
        self.radial_cap = training.radial_cap
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
        self.num_of_initial_pts = training.num_of_initial_pts
        self.max_insert = training.max_insert
        self.prune_amp_tol = training.prune_amp_tol

    def initialize(self) -> dict:
        """Perform the historical correction-first initial support replacement."""
        W, b, c = self._insert(
            self.num_of_initial_pts, self.max_insert,
        )
        count = self._add_atoms(W, b, c, replace=True)
        diagnostics = self._search_diagnostics(count)
        diagnostics["stop_reason"] = "no_candidate" if count == 0 else None
        return diagnostics


    def step(self) -> dict:
        """Insert, correct and prune one complete step of the paper algorithms.
        """
        diagnostics = self.insert()
        diagnostics["stop_reason"] = "no_candidate" if diagnostics["inserted"] == 0 else None
        if diagnostics["stop_reason"] is None:
            diagnostics.update(self.train_ssn_outerweights_and_prune())
        return diagnostics


    def insert(self) -> dict:
        """Extend the current support with accepted, initialized candidates."""
        W, b, c = self._insert(
            self.num_of_initial_pts, self.max_insert,
        )
        return self._search_diagnostics(self._add_atoms(W, b, c))

    def _insert(
        self, num_of_initial_pts: int, max_insert: int
    ):
        """Return accepted atoms and their outer weights, or None for weights to warm-start."""
        # Sequential insertion admits the highest-ranked candidate returned by the
        # multistart search, one atom per outer iteration. The theorem's rate bound
        # additionally requires that this candidate be an exact global maximizer; a
        # batch also introduces cross terms through its shared frozen residual.
        if self.insert_mode == "sequential":
            max_insert = 1
        X, V, dV = full_batch(self.train_loader)
        with torch.no_grad():
            Vp, dVp = self.network(X)
        residual = X.detach(), (Vp - V).detach(), (dVp - dV).detach()
        existing_atoms = (self.network.inner_weight.detach(), self.network.inner_bias.detach()) if self.network.n_neurons else None

        d = int(self.network.input_dim)
        radius = self._search_radius()
        search_settings: dict[str, Any] = dict(
            sample_sphere=lambda num_of_initial_pts: self._sphere(d, num_of_initial_pts),
            num_of_initial_pts=num_of_initial_pts,
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            existing_atoms=existing_atoms, radius=radius,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        if self.insertion_kind == "profile":
            W, b, c = profile_threshold(
                self.network, self.penalty, residual, insert_init=self.insert_init,
                **search_settings,
            )
        else:
            W, b, c = finite_step(self.network, self.penalty, residual, **search_settings)
        return W, b, c


    def train_ssn_outerweights_and_prune(self) -> dict:
        """Apply the accepted SSN correction, then remove negligible atoms."""
        diagnostics = self.train_ssn_outerweights()
        diagnostics["pruned"] = self._prune(self.prune_amp_tol)
        return diagnostics


    def train_ssn_outerweights(self) -> dict:
        """Prepare a correction, run SSN, and apply its accepted coefficients."""
        fitting, hess_obj, moment_weight = self.fit_obj_with_outerweights()
        # The normalized coefficients u = w_p*c; the physical ones are c = u/w_p.
        u = torch.nn.Parameter(self.network.c.detach() * moment_weight)
        optimizer = SSN([u], self.penalty, hess_obj, **self.solver)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            value = fitting(u)
            value.backward()
            return value

        diagnostics = ssn_refine_outerweights(
            optimizer, u, closure, iterations=self.fit_outer_iterations,
            correction_guard=self.correction_guard,
        )
        with torch.no_grad():
            self.network.c.copy_(u.detach() / moment_weight)
        return diagnostics

    def fit_obj_with_outerweights(self) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]:
        """The fitting term of the normalized coefficients u = w_p*c, its Hessian, and w_p.

        SSN adds the penalty to this fitting term.
        """
        X, V, dV = full_batch(self.train_loader)
        moment_weight = self.network.moment_weight_of_atoms()
        target = V.detach().clone(), dV.detach().clone()

        def fitting(u: torch.Tensor) -> torch.Tensor:
            return self.network.obj(X, target, u / moment_weight)

        hess_obj = self.network.hess_obj(X) / torch.outer(moment_weight, moment_weight)
        return fitting, hess_obj, moment_weight


    def _add_atoms(self, W, b, c, *, replace: bool = False) -> int:
        """Append or, if replace, substitute the atoms, warm-starting weights given as None; return their count."""
        count = int(W.shape[0])
        if count:
            W, b = torch.as_tensor(W, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
            c_new = torch.zeros(count, dtype=torch.float64) if c is None else torch.as_tensor(
                c, dtype=torch.float64,
            ).reshape(-1)
            if not replace and self.network.n_neurons > 0:
                W = torch.cat([self.network.inner_weight.detach(), W], dim=0)
                b = torch.cat([self.network.inner_bias.detach(), b], dim=0)
                c_new = torch.cat([self.network.c.detach(), c_new], dim=0)
            self.network.set_atoms(W, b, c_new)
            if c is None:
                X, V, dV = full_batch(self.train_loader)
                warm_start(self.network, self.penalty, X, (V, dV), count)
        return count

    def _prune(self, amp_tol: float) -> int:
        """Drop atoms with ``|c_n| <= amp_tol``; return how many were removed."""
        W, b, c, pruned = self.prune_small_weights(
            self.network.inner_weight, self.network.inner_bias, self.network.c, amp_tol=amp_tol,
        )
        if pruned:
            self.network.set_atoms(W, b, c)
        return pruned

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor, biases: torch.Tensor, outer_weights: torch.Tensor,
        amp_tol: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Defensive gate: drop atoms whose outer weight is negligible.
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

    def _sphere(self, d: int, num_of_initial_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample num_of_initial_pts points uniformly on S^d in R^{d+1}."""
        v = torch.randn(num_of_initial_pts, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return v[:, :d].contiguous(), v[:, d].contiguous()

    def _search_radius(self):
        """Bound candidate search using the theorem, numerical cap and fallback."""
        cap = math.exp(FIXED_LOG_CLAMP)
        if self.radial_cap != "theorem" or self.network.use_sphere:
            self._last_search_radius = 1.0 if self.network.use_sphere else cap
            self._last_theorem_applied = -1.0
            return None
        assert self.network.moment_order is not None  # Nonhomogeneous training was validated before search.
        X, V, dV = full_batch(self.train_loader)
        residual_norm = float(torch.sqrt(2 * self.network.obj(X, (V, dV)).detach()))
        radius = certificate_radius(
            get_growth(self.network.activation_name) if self.network.activation_name is not None else None,
            extent=sample_extent(X), residual_norm=residual_norm,
            alpha=self.penalty.alpha, moment_order=self.network.moment_order,
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
