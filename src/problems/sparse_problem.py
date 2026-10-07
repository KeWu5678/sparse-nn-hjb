"""Neuron and objective mathematics evaluated on externally supplied data."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.activations import ACTIVATIONS, get_activation, get_growth, get_use_sphere
from ..data import full_batch
from ..eval import relative_errors
from .moment import amplitude_mass_radius, moment_weight
from .penalty import (
    _noncvx_correction,
    _noncvx_correction_dd,
    _noncvx_penalty,
    _phi,
    jac_noncvx_penalty,
    phi_prox,
)
from .prox import _power_prox, _power_prox_derivative
from .radius import certificate_radius, sample_extent


@dataclass(frozen=True)
class PreparedObjective:
    """Fitting calculations and scaling fixed for one coefficient correction."""

    obj: Callable[[Tensor], Tensor]
    hess_obj: Tensor
    coefficient_scale: Tensor | None = None


class SparseProblem(nn.Module):
    """Own the sparse network and regularized objective without owning samples.

    DataLoaders deliver (X, V, dV), shaped (n, d), (n, 1), and (n, d).
    The data module assembles full samples for this full-objective algorithm.
    prepare_coefficients returns an independent objective for one SSN solve.

    The prepared obj is data fitting. obj_augmented adds the nonconvex
    normal-map correction; loss adds the full penalty.
    For the log family loss = obj_augmented + alpha*||u||_1. For a fractional
    penalty obj_augmented = obj and the proximal map handles the full penalty.
    Optimizers own parameter mutation, differentiation of loss, and iterations.
    Prediction accepts positive finite powers and named or callable activations.
    Coefficient correction additionally requires the supported penalty/geometry.
    """

    W: Tensor
    b: Tensor

    def __init__(
        self,
        input_dim: int,
        *,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        power: float = 1.0,
        alpha: float = 1e-5,
        gamma: float = 0.0,
        loss_weights: tuple[float, float] = (1.0, 1.0),
        moment_order: float = 2.0,
        penalized_mask: Tensor | None = None,
        nonneg_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self._activation_name = (
            activation if isinstance(activation, str) else
            next((name for name, (fn, _) in ACTIVATIONS.items() if fn is activation), None)
        )
        self._activation = get_activation(activation) if isinstance(activation, str) else activation
        self._use_sphere = (
            get_use_sphere(self._activation_name) if self._activation_name is not None else False
        )
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError("power must be positive and finite")
        if not moment_order > 0.0:
            raise ValueError("moment_order must be positive")
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self._power = power
        self._alpha, self._gamma = alpha, gamma
        self._loss_weights = loss_weights
        self._moment_order = moment_order
        self._penalized_mask = penalized_mask
        self._nonneg_mask = nonneg_mask
        self.register_buffer("W", torch.empty((0, input_dim), dtype=torch.float64))
        self.register_buffer("b", torch.empty(0, dtype=torch.float64))
        self.c = nn.Parameter(torch.empty(0, dtype=torch.float64))

    @classmethod
    def from_config(cls, config, input_dim: int):
        """Read problem specifications from the existing model configuration."""
        if config.kind != "signed":
            raise ValueError(
                f"the active implementation only supports kind='signed'; got {config.kind!r}"
            )
        problem = cls(
            input_dim,
            activation=config.activation,
            power=config.power,
            alpha=config.alpha,
            gamma=config.gamma,
            loss_weights=tuple(config.loss_weights),
            moment_order=config.moment_order,
        )
        problem.validate_correction()
        return problem

    @property
    def input_dim(self) -> int:
        return self.W.shape[1]

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def n_neurons(self) -> int:
        return self.W.shape[0]

    @property
    def use_sphere(self) -> bool:
        return self._use_sphere

    @property
    def normalized(self) -> bool:
        return not self._use_sphere

    @property
    def penalty_exponent(self) -> float:
        return 2.0 / (self._power + 1.0)

    @property
    def loss_weights(self) -> tuple[float, float]:
        return self._loss_weights

    def description(self) -> dict[str, object]:
        """Problem metadata for a run log, independent of solver settings."""
        return {
            "alpha": self._alpha,
            "gamma": self._gamma,
            "activation power": self._power,
        }

    def set_atoms(self, W: Tensor, b: Tensor, c: Tensor) -> None:
        """Replace support with exact atoms, preserving the module's dtype/device.

        W is (n, input_dim); b and c contain one entry per atom. Replacing
        support replaces c's Parameter, so subsequent corrections use the new c.
        """
        W, b, c = (
            torch.as_tensor(tensor, dtype=self.W.dtype, device=self.W.device).detach().clone()
            for tensor in (W, b, c)
        )
        b, c = b.reshape(-1), c.reshape(-1)
        if W.ndim != 2 or W.shape[1] != self.input_dim:
            raise ValueError("W must have shape (n, input_dim)")
        if b.numel() != W.shape[0] or c.numel() != W.shape[0]:
            raise ValueError("b and c must contain one entry per atom")
        self.W, self.b, self.c = W, b, nn.Parameter(c)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the network; ordinary autograd remains available to callers."""
        features = self._activation(F.linear(x, self.W, self.b)) ** self._power
        return F.linear(features, self.c.unsqueeze(0))

    def predict_tensors(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return detached value (N,1) and input gradient (N,d), including empty support."""
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = self(x)
            dV = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
        return V.detach(), dV.detach()

    def jacobians(
        self, x: Tensor, W: Tensor | None = None, b: Tensor | None = None,
        *, warm_start: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Coefficient features (N,n), (N*d,n), for current or supplied atoms.

        Supplied atom parameters retain their gradients for candidate search.
        warm_start preserves the historical sphere-comparator derivative:
        its activation derivative is the positive-preactivation indicator,
        including for leaky_relu and relu2. It does not select an initial step.
        """
        W = self.W if W is None else W
        b = self.b if b is None else b
        differentiable = W.requires_grad or b.requires_grad
        with torch.enable_grad():
            z = F.linear(x.detach(), W, b).requires_grad_(True)
            act = self._activation(z)
            Phi_v = act ** self._power
            if warm_start and self.use_sphere:
                act_deriv = (z > 0).to(z.dtype)
                dS_dz = (
                    act_deriv if self._power == 1.0
                    else self._power * act ** (self._power - 1) * act_deriv
                )
            else:
                dS_dz = torch.autograd.grad(
                    Phi_v, z, grad_outputs=torch.ones_like(Phi_v),
                    create_graph=differentiable,
                )[0]
            dS_dx = dS_dz.unsqueeze(2) * W.unsqueeze(0)
            Phi_g = dS_dx.permute(0, 2, 1).reshape(x.shape[0] * self.input_dim, W.shape[0])
        return (Phi_v, Phi_g) if differentiable else (Phi_v.detach(), Phi_g.detach())

    def coefficient_scale(self, W: Tensor, b: Tensor) -> Tensor:
        """Moment divisor for normalized atoms, or one on the sphere."""
        return moment_weight(W, b, self._moment_order) if self.normalized else torch.ones_like(b)

    def validate_correction(self) -> None:
        """Validate training restrictions without excluding historical prediction."""
        if self._activation_name is None:
            raise ValueError("coefficient correction requires a registered activation")
        if self._power not in (1.0, 2.0, 3.0):
            raise ValueError(
                "the coefficient correction supports activation powers 1, 2, 3; "
                f"got power={self._power}"
            )
        if not self._use_sphere and self._power != 1.0:
            raise ValueError(
                "nonhomogeneous signed profile insertion is Algorithm 1 and requires power == 1"
            )
        if self._power != 1.0 and self._gamma != 0.0:
            raise ValueError(
                f"the power penalty requires gamma == 0; got power={self._power}, gamma={self._gamma}"
            )

    def validate_insertion(self, insertion: str) -> None:
        self.validate_correction()
        if insertion == "finite_step":
            if not self._use_sphere:
                raise ValueError(
                    "finite_step insertion is Algorithm 2 and requires a sphere activation"
                )
            if self._gamma != 0.0:
                raise ValueError(
                    "finite_step insertion minimizes the power penalty and requires gamma == 0"
                )

    def prepare_coefficients(self, loader) -> PreparedObjective:
        """Return independent fixed-neuron calculations for one correction."""
        self.validate_correction()
        X, V, dV = full_batch(loader)
        Phi_v, Phi_g = self.jacobians(X)
        coefficient_scale = None
        if self.normalized:
            coefficient_scale = self.coefficient_scale(self.W, self.b)
            Phi_v = Phi_v / coefficient_scale
            Phi_g = Phi_g / coefficient_scale
        V, dV = (target.detach().reshape(-1).clone() for target in (V, dV))
        n = X.shape[0]
        w1, w2 = self._loss_weights

        def obj(u):
            rv = Phi_v @ u - V
            rg = Phi_g @ u - dV
            return (w1 / (2 * n)) * (rv @ rv) + (w2 / (2 * n)) * (rg @ rg)

        hess_obj = (w1 / n) * (Phi_v.T @ Phi_v) + (w2 / n) * (Phi_g.T @ Phi_g)
        return PreparedObjective(obj, hess_obj, coefficient_scale)

    def coefficient_coordinates(self, coefficients: Tensor, prepared: PreparedObjective) -> Tensor:
        """Physical coefficients to the coordinates used by the proximal map."""
        return (
            coefficients
            if prepared.coefficient_scale is None
            else coefficients * prepared.coefficient_scale
        )

    def model_coefficients(self, u: Tensor, prepared: PreparedObjective) -> Tensor:
        """Return physical model coefficients from a solved proximal iterate."""
        return u if prepared.coefficient_scale is None else u / prepared.coefficient_scale

    def obj_augmented(self, u: Tensor, prepared: PreparedObjective) -> Tensor:
        """Objective plus the nonconvex correction in the proximal splitting."""
        if self.penalty_exponent != 1.0:
            return prepared.obj(u)
        penalized, nonneg = self._masks(u)
        magnitudes = torch.where(nonneg[penalized], u[penalized].clamp_min(0), u[penalized].abs())
        return prepared.obj(u) + self._alpha * (_phi(magnitudes, self._gamma) - magnitudes).sum()

    def loss(self, u: Tensor, prepared: PreparedObjective) -> Tensor:
        """Evaluate the complete objective at u, without modifying u."""
        penalized, nonneg = self._masks(u)
        return prepared.obj(u) + _noncvx_penalty(
            u,
            penalized,
            nonneg,
            alpha=self._alpha,
            gamma=self._gamma,
            penalty_exponent=self.penalty_exponent,
        )

    def normal_map(
        self,
        q: Tensor,
        u: Tensor,
        jac_obj_augmented: Tensor,
        inverse_step: float,
    ) -> Tensor:
        """Evaluate G(q), using u = P(q) and the smooth gradient at u."""
        return inverse_step * (q - u) + jac_obj_augmented

    def jac_normal(self, u: Tensor, jac_prox: Tensor, inverse_step: float, prepared: PreparedObjective) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return inverse_step * (I - jac_prox) + self.hess_obj_augmented(u, prepared) @ jac_prox

    def jac_obj_augmented(self, u: Tensor, jac_loss: Tensor) -> Tensor:
        """Derivative of obj_augmented from the supplied derivative of loss(u).

        Retain the original full-gradient subtraction and correction arithmetic,
        including its derivative selection at masked nonsmooth coordinates.
        The caller computes jac_loss and controls gradient tracking.
        """
        alpha_vec = self._alpha_vec(u)
        abs_u, sign_u = torch.abs(u), torch.sign(u)
        reg_grad = jac_noncvx_penalty(
            abs_u, sign_u, alpha_vec, self._gamma, penalty_exponent=self.penalty_exponent
        )
        jac_obj = jac_loss - reg_grad
        correction = _noncvx_correction(
            abs_u, sign_u, self._gamma, penalty_exponent=self.penalty_exponent
        )
        return jac_obj + alpha_vec * correction

    def hess_obj_augmented(self, u: Tensor, prepared: PreparedObjective) -> Tensor:
        """Hessian of obj plus the diagonal Hessian of the nonconvex correction."""
        correction = _noncvx_correction_dd(
            torch.abs(u), self._gamma, penalty_exponent=self.penalty_exponent
        )
        return torch.diag(self._alpha_vec(u) * correction) + prepared.hess_obj

    def prox(self, q: Tensor, inverse_step: float) -> Tensor:
        """Evaluate the selected proximal map, including nonnegative coordinates."""
        out = _power_prox(q, self._alpha_vec(q) / inverse_step, penalty_exponent=self.penalty_exponent)
        kill = self._masks(q)[1] & (q < 0)
        return torch.where(kill, torch.zeros_like(out), out)

    def jac_prox(self, q: Tensor, inverse_step: float, *, prox_result: Tensor) -> Tensor:
        """Selected proximal Jacobian at q, using its already computed image."""
        jac = _power_prox_derivative(
            q,
            self._alpha_vec(q) / inverse_step,
            penalty_exponent=self.penalty_exponent,
            prox_result=prox_result,
        )
        kill = self._masks(q)[1] & (q < 0)
        diagonal = torch.where(kill, torch.zeros_like(prox_result), torch.diagonal(jac))
        return torch.diag(diagonal)

    def prox_preimage(self, u: Tensor, jac_obj_augmented: Tensor, inverse_step: float) -> Tensor:
        """Reconstruct q from a prox-consistent iterate, following MATLAB SSN.m.

        Nonzero penalized entries use the penalty's stationarity equation;
        zero entries use the smooth gradient. The proximal map is the identity
        on unpenalized coordinates, whose preimage remains u.
        """
        penalized, _ = self._masks(u)
        alpha_vec = self._alpha_vec(u)
        abs_u, sign_u = torch.abs(u), torch.sign(u)
        active = (abs_u > 0) & penalized
        if self.penalty_exponent == 1.0:
            grad = torch.where(active, -alpha_vec * sign_u, jac_obj_augmented)
        else:
            s = abs_u.clamp(min=1e-30)
            base = alpha_vec * self.penalty_exponent * s ** (self.penalty_exponent - 1)
            grad = torch.where(active, -base * sign_u, jac_obj_augmented)
        q = u - (1.0 / inverse_step) * grad
        return torch.where(penalized, q, u)

    def inverse_step(self, prox_scale: float | None) -> float:
        """Translate the solver's proximal scale to the normal-map inverse step.

        The log family retains the reference scaling 1 + alpha*gamma. The
        fractional family requires the scale chosen from its warm start.
        This calculation stores no numerical setting on the problem.
        """
        if self.penalty_exponent < 1.0 and prox_scale is None:
            raise ValueError("prox_scale is required when penalty_exponent < 1")
        if prox_scale is not None and (not math.isfinite(prox_scale) or prox_scale <= 0.0):
            raise ValueError(f"prox_scale must be positive and finite, got {prox_scale}")
        if self.penalty_exponent < 1.0:
            assert prox_scale is not None
            return self._alpha / prox_scale
        return 1.0 + self._alpha * self._gamma

    def warmstart_prox_scale(self, coefficients: Tensor, rho: float) -> float | None:
        """Fractional proximal scale satisfying the warm-start upper bound."""
        if self.penalty_exponent == 1.0:
            return None
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
        if self._alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self._alpha}")
        nonzero = self.active_magnitudes(coefficients)
        if nonzero.numel() == 0:
            raise ValueError("warm-start proximal scaling requires a nonzero coefficient")
        minimum = float(nonzero.min())
        exponent = self.penalty_exponent
        bound = rho * minimum ** (2.0 - exponent) / (2.0 * (1.0 - exponent))
        return min(self._alpha / (1.0 + self._alpha * self._gamma), bound)

    def active_magnitudes(self, coefficients: Tensor) -> Tensor:
        """Nonzero penalized magnitudes, for proximal scaling and diagnostics."""
        penalized, _ = self._masks(coefficients)
        magnitudes = coefficients[penalized].abs()
        return magnitudes[magnitudes > 0]

    def residual(self, loader) -> tuple[Tensor, Tensor, Tensor]:
        """Pair inputs with detached residuals for one candidate search."""
        X, V, dV = full_batch(loader)
        Vp, dVp = self.predict_tensors(X)
        return X.detach(), (Vp - V).detach(), (dVp - dV).detach()

    def _regularizer_terms(self) -> dict[str, float]:
        """Raw functionals and weighted regularizer terms on the current support."""
        W, b, c = self.W, self.b, self.c.detach()
        if c.numel():
            theta = c
            penalized, nonneg = self._masks(theta)
            # Normalized objective: the penalty is evaluated at the normalized measure,
            # phi(w_p(omega_n)|c_n|) = phi(|u_n|), so score it on u = w_p * c -- the same
            # quantity SSN minimizes, so the recorded objective cannot drift from it.
            if self.normalized:
                scale = self.coefficient_scale(W, b)
                theta = theta * scale
            phi = _noncvx_penalty(
                theta,
                penalized,
                nonneg,
                alpha=1.0,
                gamma=self._gamma,
                penalty_exponent=self.penalty_exponent,
            )
        else:
            phi = c.new_zeros(())
        weights = moment_weight(W, b, self._moment_order)
        psi = torch.sum(weights.reshape(-1) * c.reshape(-1).abs())
        total_variation = c.reshape(-1).abs().sum()
        if c.numel():
            radius = torch.sqrt((W * W).sum(dim=-1) + b * b)
            radius_max = radius.max()
        else:
            radius_max = c.new_zeros(())

        return {
            "sparsity_functional": float(phi.detach()),
            "psi_p": float(psi.detach()),
            "alpha_phi": float((self._alpha * phi).detach()),
            "total_variation": float(total_variation.detach()),
            "radius_r95": float(amplitude_mass_radius(W, b, c).detach()),
            "radius_max": float(radius_max.detach()),
        }

    def metrics(self, train_loader, valid_loader=None, reporting_normalizer=None) -> dict[str, float]:
        """Evaluate the objective and errors, keeping persisted metric names."""
        regularizer = self._regularizer_terms()
        metrics = dict(regularizer)
        if valid_loader is None:
            valid_loader = train_loader
        for split, loader in (("train", train_loader), ("val", valid_loader)):
            data = full_batch(loader)
            prediction = self.predict_tensors(data[0])
            target = data[1:]
            obj, value_loss, gradient_loss = self.obj_terms(prediction, target)
            metrics[f"{split}_loss"] = float(obj.detach()) + regularizer["alpha_phi"]
            metrics[f"data_loss_{split}"] = float(obj.detach())
            metrics[f"value_loss_{split}"] = float(value_loss.detach())
            metrics[f"gradient_loss_{split}"] = float(gradient_loss.detach())
            if reporting_normalizer is not None:
                prediction = reporting_normalizer.denormalize_tensors(*prediction)
                target = reporting_normalizer.denormalize_tensors(*target)
            for name, error in zip(("l2", "grad", "h1"), relative_errors(*prediction, *target)):
                metrics[f"err_{name}_{split}"] = error
        return metrics

    def obj_terms(self, prediction, target) -> tuple[Tensor, Tensor, Tensor]:
        """Objective and its unweighted value/gradient contributions."""
        Vp, dVp = prediction
        V, dV = target
        n = V.shape[0]
        value_loss = torch.sum((Vp - V) ** 2) / (2 * n)
        gradient_loss = torch.sum((dVp - dV) ** 2) / (2 * n)
        w1, w2 = self._loss_weights
        return w1 * value_loss + w2 * gradient_loss, value_loss, gradient_loss

    def profile(self, a: Tensor, b: Tensor, residual) -> Tensor:
        """Signed fidelity derivative in a supplied candidate-neuron direction."""
        Kv, Kg = self.jacobians(residual[0], a.reshape(1, -1), b.reshape(1))
        res_v, res_dv = (r.detach().reshape(-1, 1) for r in residual[1:])
        return self._fidelity_pairing(Kv, Kg, res_v, res_dv, reduction="sum").squeeze(0)

    def directional_obj(self, Kv: Tensor, Kg: Tensor, residual) -> tuple[Tensor, Tensor]:
        """Fidelity slope and curvature for supplied neuron columns or a direction.

        A matrix has one column per neuron; a vector represents one direction.
        Residuals are fixed during insertion search. Candidate differentiation
        passes through the supplied features; callers choose when to detach.
        """
        vector = Kv.ndim == 1
        Kv, Kg = Kv.reshape(residual[0].shape[0], -1), Kg.reshape(residual[2].numel(), -1)
        res_v, res_dv = (r.detach().reshape(-1, 1) for r in residual[1:])
        slope = self._fidelity_pairing(Kv, Kg, res_v, res_dv)
        curvature = self._fidelity_pairing(Kv, Kg, Kv, Kg)
        return (slope.squeeze(0), curvature.squeeze(0)) if vector else (slope, curvature)

    def _fidelity_pairing(
        self, Kv: Tensor, Kg: Tensor, V: Tensor, dV: Tensor, *, reduction: str = "dot",
    ) -> Tensor:
        """Weighted empirical pairing with the two historical reduction orders.

        Search used product-sum then division; insertion used dot products with
        pre-scaled weights. Reassociating either changes L-BFGS/PDAP trajectories.
        Keep that numerical distinction here, shared by all fidelity derivatives.
        """
        n = Kv.shape[0]
        parts = []
        for left, right in ((Kv, V), (Kg, dV)):
            if reduction == "sum":
                part = (left * right).sum(0) / n
            elif left.shape[1] == 1:
                part = left[:, 0].dot(right[:, 0]).reshape(1)
            elif right.shape[1] == 1:
                part = left.T @ right[:, 0]
            else:
                part = torch.stack([a.dot(b) for a, b in zip(left.T, right.T)]) if left.shape[1] else left.new_zeros(0)
            parts.append(part)
        w1, w2 = self._loss_weights
        if reduction == "dot":
            w1, w2 = w1 / n, w2 / n
        return w1 * parts[0] + w2 * parts[1]

    def penalty_prox(self, value: float, inverse_step: float) -> float:
        """Selected nonnegative proximal value for the full scalar penalty.

        The log family's closed-form root requires value >= alpha/inverse_step
        to guarantee a minimizer; PDAP owns that eligibility check. SSN's prox
        instead uses the split penalty required by its normal map.
        """
        if self.penalty_exponent == 1.0:
            return phi_prox(value, self._alpha / inverse_step, gamma=self._gamma)
        return max(self.power_prox(value, inverse_step), 0.0)

    def power_prox(self, value: float, inverse_step: float) -> float:
        """Scalar power-penalty proximal value, including its q=1 safeguard.

        Retain scalar scale arithmetic; SSN's masked vector prox has a different
        floating-point evaluation order and must keep its own numerical contract.
        """
        v = torch.tensor([value], dtype=torch.float64)
        return float(_power_prox(
            v, self._alpha / inverse_step, penalty_exponent=self.penalty_exponent,
        )[0])

    def insertion_increment(self, coefficient: float, slope: float, curvature: float) -> float:
        """Exact power-penalty increment for a distinct atom on the sphere.

        Unlike full-loss differentiation, this scalar evaluation must not floor
        tiny magnitudes: the floor can reverse the insertion-decrease test.
        Algorithm 2 validates gamma == 0 before using this increment.
        """
        return (
            slope * coefficient + 0.5 * curvature * coefficient**2
            + self._alpha * abs(coefficient)**self.penalty_exponent
        )

    def certificate_radius(self, residual) -> float | None:
        """The theorem's radius before applying PDAP's numerical search cap."""
        X, res_v, res_dv = residual
        w1, w2 = self._loss_weights
        n = X.shape[0]
        residual_norm = float(
            torch.sqrt((w1 * res_v.pow(2).sum() + w2 * res_dv.pow(2).sum()) / max(n, 1))
        )
        return certificate_radius(
            get_growth(self._activation_name) if self._activation_name is not None else None,
            extent=sample_extent(X),
            residual_norm=residual_norm,
            alpha=self._alpha,
            moment_order=self._moment_order,
        )

    def _alpha_vec(self, u: Tensor) -> Tensor:
        penalized, _ = self._masks(u)
        return torch.where(
            penalized, torch.full_like(u, float(self._alpha)), torch.zeros_like(u),
        )

    def _masks(self, u: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._mask(self._penalized_mask, u, True, "penalized_mask"),
            self._mask(self._nonneg_mask, u, False, "nonneg_mask"),
        )

    @staticmethod
    def _mask(mask: Tensor | None, coordinates: Tensor, default: bool, name: str) -> Tensor:
        if mask is None:
            return torch.full_like(coordinates, default, dtype=torch.bool)
        if mask.shape != coordinates.shape:
            raise ValueError(f"{name} length must match the problem dimension")
        return mask.to(dtype=torch.bool, device=coordinates.device)
