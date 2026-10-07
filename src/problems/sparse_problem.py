"""The sampled learning problem shared by PDAP and semismooth Newton."""

import math

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector

from ..config.activations import get_activation, get_growth, get_use_sphere
from ..eval import relative_errors
from .moment import amplitude_mass_radius, atom_normalizer, moment_weight
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


class SparseProblem:
    """Own samples, the neuron definition, and the regularized objective.

    Each sample split is (X, V, dV), with shapes (n, d), (n, 1), and (n, d).
    Validation defaults to the training samples when no second split is given.
    Models and candidate parameters are supplied to evaluations, never stored.
    prepare_coefficients(model) refreshes the fixed-neuron matrices for one SSN
    solve; candidate and full-model evaluations are independent of that cache.

    obj is the data-fitting objective. obj_augmented adds the nonconvex
    correction used by the normal-map formulation; loss adds the full penalty.
    For the log family loss = obj_augmented + alpha*||u||_1. For a fractional
    penalty obj_augmented = obj and the proximal map handles the full penalty.
    Optimizers own parameter mutation, differentiation of loss, and iterations.
    Construction validates power and its compatibility with activation and gamma;
    internal proximal helpers receive the derived exact penalty exponent.
    """

    def __init__(
        self,
        data_train,
        data_valid=None,
        *,
        activation: str = "relu",
        power: float = 1.0,
        alpha: float = 1e-5,
        gamma: float = 0.0,
        loss_weights: tuple[float, float] = (1.0, 1.0),
        moment_order: float = 2.0,
        penalized_mask: Tensor | None = None,
        nonneg_mask: Tensor | None = None,
    ) -> None:
        self._activation_name = activation
        self._activation = get_activation(activation)
        self._use_sphere = get_use_sphere(activation)
        self._growth = get_growth(activation)
        if power not in (1.0, 2.0, 3.0):
            raise ValueError(
                f"the coefficient correction supports activation powers 1, 2, 3; got power={power}"
            )
        if not self._use_sphere and power != 1.0:
            raise ValueError(
                "nonhomogeneous signed profile insertion is Algorithm 1 and requires power == 1"
            )
        if power != 1.0 and gamma != 0.0:
            raise ValueError(
                f"the power penalty requires gamma == 0; got power={power}, gamma={gamma}"
            )
        if not moment_order > 0.0:
            raise ValueError("moment_order must be positive")
        self._data_train = data_train
        self._data_valid = data_train if data_valid is None else data_valid
        self._power = power
        self._penalty_exponent = 2.0 / (power + 1.0)
        self._alpha, self._gamma = alpha, gamma
        self._loss_weights = loss_weights
        self._moment_order = moment_order
        self._normalized = not self._use_sphere
        self._penalized_mask = penalized_mask
        self._nonneg_mask = nonneg_mask
        self._hess_obj = None
        self._coefficient_scale = None

    @classmethod
    def from_config(cls, config, data_train, data_valid=None):
        """Read problem specifications from the existing model configuration."""
        if config.kind != "signed":
            raise ValueError(
                f"the active implementation only supports kind='signed'; got {config.kind!r}"
            )
        return cls(
            data_train,
            data_valid,
            activation=config.activation,
            power=config.power,
            alpha=config.alpha,
            gamma=config.gamma,
            loss_weights=tuple(config.loss_weights),
            moment_order=config.moment_order,
        )

    @property
    def input_dim(self) -> int:
        return self._data_train[0].shape[1]

    @property
    def use_sphere(self) -> bool:
        return self._use_sphere

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def penalty_exponent(self) -> float:
        return self._penalty_exponent

    @property
    def loss_weights(self) -> tuple[float, float]:
        return self._loss_weights

    def description(self) -> dict[str, object]:
        """Problem metadata for a run log, independent of solver settings."""
        return {
            "samples": f"{len(self._data_train[0])} train, {len(self._data_valid[0])} validation",
            "alpha": self._alpha,
            "gamma": self._gamma,
            "activation power": self._power,
        }

    def validate_model(self, model) -> None:
        """Check the supplied parametrization against this neuron definition."""
        if model.activation is not self._activation:
            raise ValueError(
                f"model activation must match configured activation={self._activation_name!r}"
            )
        if model.power != self._power:
            raise ValueError(
                f"model power={model.power} does not match configured power={self._power}"
            )

    def validate_insertion(self, insertion: str) -> None:
        if insertion == "finite_step":
            if not self._use_sphere:
                raise ValueError(
                    "finite_step insertion is Algorithm 2 and requires a sphere activation"
                )
            if self._gamma != 0.0:
                raise ValueError(
                    "finite_step insertion minimizes the power penalty and requires gamma == 0"
                )

    def prepare_coefficients(self, model) -> None:
        """Build the fixed-neuron objective without retaining or changing model."""
        self.validate_model(model)
        Phi_v, Phi_g = model.jacobians(self._data_train[0])
        Phi_v, Phi_g = Phi_v.detach(), Phi_g.detach()
        self._coefficient_scale = None
        if self._normalized:
            W, b, _ = model.get_atoms()
            self._coefficient_scale = atom_normalizer(W, b, p=self._moment_order)
            if self._coefficient_scale.numel() != Phi_v.shape[1]:
                raise RuntimeError(
                    "normalized Algorithm 1 expects one trainable coordinate per atom"
                )
            Phi_v = Phi_v / self._coefficient_scale
            Phi_g = Phi_g / self._coefficient_scale
        self._Phi_v, self._Phi_g = Phi_v, Phi_g
        n = self._data_train[0].shape[0]
        w1, w2 = self._loss_weights
        self._hess_obj = (w1 / n) * (Phi_v.T @ Phi_v) + (w2 / n) * (Phi_g.T @ Phi_g)
        coordinates = torch.diagonal(self._hess_obj)
        penalized, nonneg = self._masks(coordinates)
        self._alpha_vec = torch.where(
            penalized,
            torch.full_like(coordinates, float(self._alpha)),
            torch.zeros_like(coordinates),
        )
        self._prox_penalized, self._prox_nonneg = penalized, nonneg

    def coefficient_coordinates(self, coefficients: Tensor) -> Tensor:
        """Physical coefficients to the coordinates used by the proximal map."""
        return (
            coefficients
            if self._coefficient_scale is None
            else coefficients * self._coefficient_scale
        )

    def model_coefficients(self, u: Tensor) -> Tensor:
        """Return physical model coefficients from a solved proximal iterate."""
        return u if self._coefficient_scale is None else u / self._coefficient_scale

    def obj(self, u: Tensor) -> Tensor:
        """Quadratic value/gradient fitting objective on the prepared neurons."""
        if self._hess_obj is None:
            raise RuntimeError(
                "prepare_coefficients(model) is required before coefficient evaluation"
            )
        X, V, dV = self._data_train
        rv = self._Phi_v @ u - V.reshape(-1).detach()
        rg = self._Phi_g @ u - dV.reshape(-1).detach()
        w1, w2 = self._loss_weights
        return (w1 / (2 * X.shape[0])) * (rv @ rv) + (w2 / (2 * X.shape[0])) * (rg @ rg)

    def hess_obj(self) -> Tensor:
        """Constant coefficient Hessian on the most recently prepared support."""
        if self._hess_obj is None:
            raise RuntimeError(
                "prepare_coefficients(model) is required before coefficient evaluation"
            )
        return self._hess_obj

    def obj_augmented(self, u: Tensor) -> Tensor:
        """Objective plus the nonconvex correction in the proximal splitting."""
        if self._penalty_exponent != 1.0:
            return self.obj(u)
        penalized, nonneg = self._masks(u)
        magnitudes = torch.where(nonneg[penalized], u[penalized].clamp_min(0), u[penalized].abs())
        return self.obj(u) + self._alpha * (_phi(magnitudes, self._gamma) - magnitudes).sum()

    def loss(self, u: Tensor) -> Tensor:
        """Evaluate the complete objective at u, without modifying u."""
        penalized, nonneg = self._masks(u)
        return self.obj(u) + _noncvx_penalty(
            u,
            penalized,
            nonneg,
            alpha=self._alpha,
            gamma=self._gamma,
            penalty_exponent=self._penalty_exponent,
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

    def jac_normal(self, u: Tensor, jac_prox: Tensor, inverse_step: float) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return inverse_step * (I - jac_prox) + self.hess_obj_augmented(u) @ jac_prox

    def jac_obj_augmented(self, u: Tensor, jac_loss: Tensor) -> Tensor:
        """Derivative of obj_augmented from the supplied derivative of loss(u).

        Retain the original full-gradient subtraction and correction arithmetic,
        including its derivative selection at masked nonsmooth coordinates.
        The caller computes jac_loss and controls gradient tracking.
        """
        abs_u, sign_u = torch.abs(u), torch.sign(u)
        reg_grad = jac_noncvx_penalty(
            abs_u, sign_u, self._alpha_vec, self._gamma, penalty_exponent=self._penalty_exponent
        )
        jac_obj = jac_loss - reg_grad
        correction = _noncvx_correction(
            abs_u, sign_u, self._gamma, penalty_exponent=self._penalty_exponent
        )
        return jac_obj + self._alpha_vec * correction

    def hess_obj_augmented(self, u: Tensor) -> Tensor:
        """Hessian of obj plus the diagonal Hessian of the nonconvex correction."""
        correction = _noncvx_correction_dd(
            torch.abs(u), self._gamma, penalty_exponent=self._penalty_exponent
        )
        return torch.diag(self._alpha_vec * correction) + self.hess_obj()

    def prox(self, q: Tensor, inverse_step: float) -> Tensor:
        """Evaluate the selected proximal map, including nonnegative coordinates."""
        out = _power_prox(q, self._alpha_vec / inverse_step, penalty_exponent=self._penalty_exponent)
        kill = self._prox_nonneg & (q < 0)
        return torch.where(kill, torch.zeros_like(out), out)

    def jac_prox(self, q: Tensor, inverse_step: float, *, prox_result: Tensor) -> Tensor:
        """Selected proximal Jacobian at q, using its already computed image."""
        jac = _power_prox_derivative(
            q,
            self._alpha_vec / inverse_step,
            penalty_exponent=self._penalty_exponent,
            prox_result=prox_result,
        )
        kill = self._prox_nonneg & (q < 0)
        diagonal = torch.where(kill, torch.zeros_like(prox_result), torch.diagonal(jac))
        return torch.diag(diagonal)

    def prox_preimage(self, u: Tensor, jac_obj_augmented: Tensor, inverse_step: float) -> Tensor:
        """Reconstruct q from a prox-consistent iterate, following MATLAB SSN.m.

        Nonzero penalized entries use the penalty's stationarity equation;
        zero entries use the smooth gradient. The proximal map is the identity
        on unpenalized coordinates, whose preimage remains u.
        """
        abs_u, sign_u = torch.abs(u), torch.sign(u)
        active = (abs_u > 0) & self._prox_penalized
        if self._penalty_exponent == 1.0:
            grad = torch.where(active, -self._alpha_vec * sign_u, jac_obj_augmented)
        else:
            s = abs_u.clamp(min=1e-30)
            base = self._alpha_vec * self._penalty_exponent * s ** (self._penalty_exponent - 1)
            grad = torch.where(active, -base * sign_u, jac_obj_augmented)
        q = u - (1.0 / inverse_step) * grad
        return torch.where(self._prox_penalized, q, u)

    def inverse_step(self, prox_scale: float | None) -> float:
        """Translate the solver's proximal scale to the normal-map inverse step.

        The log family retains the reference scaling 1 + alpha*gamma. The
        fractional family requires the scale chosen from its warm start.
        This calculation stores no numerical setting on the problem.
        """
        if self._penalty_exponent < 1.0 and prox_scale is None:
            raise ValueError("prox_scale is required when penalty_exponent < 1")
        if prox_scale is not None and (not math.isfinite(prox_scale) or prox_scale <= 0.0):
            raise ValueError(f"prox_scale must be positive and finite, got {prox_scale}")
        if self._penalty_exponent < 1.0:
            assert prox_scale is not None
            return self._alpha / prox_scale
        return 1.0 + self._alpha * self._gamma

    def warmstart_prox_scale(self, coefficients: Tensor, rho: float) -> float | None:
        """Fractional proximal scale satisfying the warm-start upper bound."""
        if self._penalty_exponent == 1.0:
            return None
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
        if self._alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self._alpha}")
        nonzero = self.active_magnitudes(coefficients)
        if nonzero.numel() == 0:
            raise ValueError("warm-start proximal scaling requires a nonzero coefficient")
        minimum = float(nonzero.min())
        exponent = self._penalty_exponent
        bound = rho * minimum ** (2.0 - exponent) / (2.0 * (1.0 - exponent))
        return min(self._alpha / (1.0 + self._alpha * self._gamma), bound)

    def active_magnitudes(self, coefficients: Tensor) -> Tensor:
        """Nonzero penalized magnitudes, for proximal scaling and diagnostics."""
        penalized, _ = self._masks(coefficients)
        magnitudes = coefficients[penalized].abs()
        return magnitudes[magnitudes > 0]

    def residual(self, model) -> tuple[Tensor, Tensor]:
        X, V, dV = self._data_train
        Vp, dVp = model.predict_tensors(X)
        return (Vp - V).detach(), (dVp - dV).detach()

    def obj_terms(self, prediction, target) -> tuple[Tensor, Tensor, Tensor]:
        """Objective and its unweighted value/gradient contributions."""
        Vp, dVp = prediction
        V, dV = target
        n = V.shape[0]
        value_loss = torch.sum((Vp - V) ** 2) / (2 * n)
        gradient_loss = torch.sum((dVp - dV) ** 2) / (2 * n)
        w1, w2 = self._loss_weights
        return w1 * value_loss + w2 * gradient_loss, value_loss, gradient_loss

    def model_loss(self, model) -> float:
        """Full training loss on any support, including the zero model."""
        X, V, dV = self._data_train
        obj = self.obj_terms(model.predict_tensors(X), (V, dV))[0]
        return float(obj.detach()) + self._regularizer_terms(model)["alpha_phi"]

    def _regularizer_terms(self, model) -> dict[str, float]:
        """Raw functionals and weighted regularizer terms on the current support."""
        W, b, c = model.get_atoms()
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if params:
            theta = parameters_to_vector(params).detach()
            penalized, nonneg = self._masks(theta)
            # Normalized objective: the penalty is evaluated at the normalized measure,
            # phi(w_p(omega_n)|c_n|) = phi(|u_n|), so score it on u = w_p * c -- the same
            # quantity ssn_solve minimizes, so the recorded objective cannot drift from it.
            if self._normalized:
                scale = atom_normalizer(W, b, p=self._moment_order)
                if scale.numel() != theta.numel():
                    raise RuntimeError(
                        "normalized Algorithm 1 expects one trainable coordinate per atom "
                        f"(got {theta.numel()} for {scale.numel()} atoms)"
                    )
                theta = theta * scale
            phi = _noncvx_penalty(
                theta,
                penalized,
                nonneg,
                alpha=1.0,
                gamma=self._gamma,
                penalty_exponent=self._penalty_exponent,
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

    def metrics(self, model, reporting_normalizer=None) -> dict[str, float]:
        """Evaluate the objective and errors, keeping persisted metric names."""
        regularizer = self._regularizer_terms(model)
        metrics = dict(regularizer)
        for split, data in (("train", self._data_train), ("val", self._data_valid)):
            prediction = model.predict_tensors(data[0])
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

    def profile(self, a: Tensor, b: Tensor, residual, *, two_sided: bool = True) -> Tensor:
        """Differentiable fidelity derivative in the candidate-neuron direction."""
        neuron_v, neuron_dv = self._neuron_value_grad(a, b)
        n = self._data_train[0].shape[0]
        w1, w2 = self._loss_weights
        val_part = (neuron_v * residual[0]).sum() / n
        grad_part = (neuron_dv * residual[1]).sum() / n
        signed = w1 * val_part + w2 * grad_part
        return torch.abs(signed) if two_sided else signed

    def search_score(self, a: Tensor, b: Tensor, residual, *, two_sided: bool = True) -> Tensor:
        score = self.profile(a, b, residual, two_sided=two_sided)
        if self._normalized:
            score = score / moment_weight(a, b, self._moment_order)
        return score

    def profile_insertion(
        self, a: Tensor, b: Tensor, residual, *, two_sided: bool = True, guaranteed: bool = False
    ) -> tuple[float, float | None] | None:
        """Return the threshold margin and optional descent coefficient, or None."""
        neuron_v, neuron_dv = self._neuron_value_grad(a, b)
        p_signed, curvature = self._directional_obj(neuron_v, neuron_dv, residual)
        v = abs(p_signed) if two_sided else p_signed
        w_p = float(moment_weight(a, b, self._moment_order))
        threshold = self._alpha * w_p if self._normalized else self._alpha
        score = v / w_p - self._alpha if self._normalized else v - threshold
        # Keep this comparison in physical-profile units, including rounding.
        if v <= threshold or (guaranteed and curvature < 1e-30):
            return None
        coefficient = None
        if guaranteed:
            magnitude = (v - threshold) / curvature
            coefficient = -magnitude if p_signed > 0 else magnitude
        return score, coefficient

    def finite_insertion(self, a: Tensor, b: Tensor, residual) -> tuple[float, float] | None:
        """Return (coefficient, loss increment), or None without a decreasing step."""
        neuron_v, neuron_dv = self._neuron_value_grad(a, b)
        return self.solve_insertion_weight(*self._directional_obj(neuron_v, neuron_dv, residual))

    def solve_insertion_weight(self, p_omega: float, S_sq: float) -> tuple[float, float] | None:
        """Minimize c*P + ||K||²*c²/2 + alpha*|c|^penalty_exponent.

        Return (coefficient, loss increment), or None without a decreasing step.
        """
        if S_sq < 1e-30 or abs(p_omega) < 1e-30:
            return None
        abs_p = abs(p_omega)
        prox_input = torch.tensor([abs_p / S_sq], dtype=torch.float64)
        magnitude = float(
            _power_prox(prox_input, self._alpha / S_sq, penalty_exponent=self._penalty_exponent)[0]
        )
        if magnitude <= 0.0:
            return None
        dJ = (
            -abs_p * magnitude
            + 0.5 * S_sq * magnitude**2
            + self._alpha * magnitude**self._penalty_exponent
        )
        if dJ >= 0.0:
            return None
        sign = 1.0 if p_omega < 0 else -1.0
        return sign * magnitude, dJ

    def warmstart_profiles(self, W: Tensor, b: Tensor, residual) -> Tensor:
        S_val, S_grad, _ = self._warmstart_matrices(W, b)
        n = self._data_train[0].shape[0]
        w1, w2 = self._loss_weights
        res_v, res_dv = (r.detach().reshape(-1) for r in residual)
        return (w1 / n) * (S_val.T @ res_v) + (w2 / n) * (S_grad.T @ res_dv)

    def warmstart_coefficients(self, W: Tensor, b: Tensor, direction: Tensor, residual) -> Tensor:
        """The one-dimensional proximal step along PDAP's chosen direction."""
        S_val, S_grad, scale = self._warmstart_matrices(W, b)
        slope, curvature = self._directional_obj(S_val @ direction, S_grad @ direction, residual)
        if -slope >= self._alpha and curvature > 1e-30:
            if self._power == 1.0:
                tau = phi_prox(-slope / curvature, self._alpha / curvature, gamma=self._gamma)
            else:
                v = torch.tensor([-slope / curvature], dtype=torch.float64)
                tau = max(
                    float(
                        _power_prox(
                            v, self._alpha / curvature, penalty_exponent=self._penalty_exponent
                        )[0].item()
                    ),
                    0.0,
                )
        else:
            tau = 0.0
        out = tau * direction
        return out if scale is None else out / scale

    def certificate_radius(self, residual) -> float | None:
        """The theorem's radius before applying PDAP's numerical search cap."""
        res_v, res_dv = residual
        w1, w2 = self._loss_weights
        n = self._data_train[0].shape[0]
        residual_norm = float(
            torch.sqrt((w1 * res_v.pow(2).sum() + w2 * res_dv.pow(2).sum()) / max(n, 1))
        )
        return certificate_radius(
            self._growth,
            extent=sample_extent(self._data_train[0]),
            residual_norm=residual_norm,
            alpha=self._alpha,
            moment_order=self._moment_order,
        )

    def _neuron_value_grad(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        X = self._data_train[0].detach().clone().requires_grad_(True)
        pre = X @ a.reshape(-1) + b.reshape(())
        act = self._activation(pre).reshape(-1, 1)
        neuron_v = act**self._power
        neuron_dv = torch.autograd.grad(neuron_v.sum(), X, create_graph=True, retain_graph=True)[0]
        return neuron_v, neuron_dv

    def _directional_obj(self, Kv: Tensor, Kg: Tensor, residual) -> tuple[float, float]:
        Kv, Kg = Kv.detach().reshape(-1), Kg.detach().reshape(-1)
        res_v, res_dv = (r.detach().reshape(-1) for r in residual)
        n = self._data_train[0].shape[0]
        w1, w2 = self._loss_weights
        slope = float((w1 / n) * Kv.dot(res_v) + (w2 / n) * Kg.dot(res_dv))
        curvature = float((w1 / n) * Kv.dot(Kv) + (w2 / n) * Kg.dot(Kg))
        return slope, curvature

    @torch.no_grad()
    def _warmstart_matrices(self, W: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        pre = self._data_train[0].detach() @ W.T + b
        act = self._activation(pre)
        S_val = act**self._power
        if self._use_sphere:
            act_deriv = (pre > 0).double()
        else:
            pre_tmp = pre.detach().requires_grad_(True)
            with torch.enable_grad():
                act_tmp = self._activation(pre_tmp)
                act_deriv = torch.autograd.grad(act_tmp.sum(), pre_tmp, create_graph=False)[
                    0
                ].detach()
        dS_dz = (
            act_deriv if self._power == 1.0 else self._power * act ** (self._power - 1) * act_deriv
        )
        S_grad = (dS_dz.unsqueeze(2) * W.unsqueeze(0)).permute(0, 2, 1).reshape(-1, W.shape[0])
        scale = None
        if self._normalized:
            scale = moment_weight(W, b, self._moment_order).reshape(-1)
            S_val = S_val / scale
            S_grad = S_grad / scale
        return S_val, S_grad, scale

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
