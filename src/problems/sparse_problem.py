"""Neuron and objective mathematics evaluated on externally supplied data."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .moment import moment_weight
from .penalty import (
    _noncvx_correction,
    _noncvx_correction_dd,
    _noncvx_penalty,
    _phi,
    jac_noncvx_penalty,
    phi_prox,
)
from .prox import _power_prox, _power_prox_derivative


class SparseProblem(nn.Module):
    """Own the sparse network and regularized objective without owning samples.

    obj evaluates data fitting from predictions and targets. obj_augmented adds the nonconvex
    normal-map correction; callers form the objective obj + alpha*sparsity_functional.
    For the log family that objective is obj_augmented + alpha*||u||_1. For a fractional
    penalty obj_augmented = obj and the proximal map handles the full penalty.
    Optimizers own parameter mutation, differentiation of the objective, and iterations.
    Construction consumes an already resolved configuration. Training setup must
    validate its penalty, geometry and insertion before constructing the problem.
    """

    inner_weight: Tensor
    inner_bias: Tensor

    def __init__(
        self,
        config,
        *,
        penalized_mask: Tensor | None = None,
        nonneg_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self._activation_name = config["activation_name"]
        self._activation = config["activation"]
        self._use_sphere = config["use_sphere"]
        self._moment_order = config["moment_order"]
        self._power = config["power"]
        self._alpha, self._gamma = config["alpha"], config["gamma"]
        self._loss_weights = config["loss_weights"]
        self._penalized_mask = penalized_mask
        self._nonneg_mask = nonneg_mask
        self.register_buffer("inner_weight", torch.empty((0, config["input_dim"]), dtype=torch.float64))
        self.register_buffer("inner_bias", torch.empty(0, dtype=torch.float64))
        self.c = nn.Parameter(torch.empty(0, dtype=torch.float64))

    @property
    def input_dim(self) -> int:
        return self.inner_weight.shape[1]

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def activation(self):
        return self._activation

    @property
    def activation_name(self) -> str | None:
        return self._activation_name

    @property
    def moment_order(self) -> float | None:
        return self._moment_order

    @property
    def n_neurons(self) -> int:
        return self.inner_weight.shape[0]

    @property
    def use_sphere(self) -> bool:
        return self._use_sphere

    @property
    def normalized(self) -> bool:
        return not self._use_sphere

    @property
    def power(self) -> float:
        return self._power

    @property
    def penalty_exponent(self) -> float:
        return 2.0 / (self._power + 1.0)

    @property
    def loss_weights(self) -> tuple[float, float]:
        return self._loss_weights

    def set_atoms(self, inner_weight: Tensor, inner_bias: Tensor, c: Tensor) -> None:
        """Replace support with independent copies of the supplied atom tensors.

        Inputs must be detached tensors shaped (n, input_dim), (n,), and (n,),
        respectively, already matching the module's dtype and device.
        Replacing support replaces c's Parameter, so subsequent corrections use the new c.
        """
        self.inner_weight = inner_weight.clone()
        self.inner_bias = inner_bias.clone()
        self.c = nn.Parameter(c.clone())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return value (N,1) and input gradient (N,d), including empty support.

        The input gradient is a torch.func vector-Jacobian product, so both
        outputs follow the caller's grad mode, including the caller's input
        graph. Samples are independent, so a unit cotangent gives each
        sample's gradient.
        """
        def value(x: Tensor) -> Tensor:
            return F.linear(self._rho(F.linear(x, self.inner_weight, self.inner_bias)), self.c.unsqueeze(0))

        V, value_vjp = torch.func.vjp(value, x)
        dV, = value_vjp(torch.ones_like(V))
        return V, dV

    def kernel(
        self, x: Tensor, inner_weight: Tensor | None = None, inner_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Kernel matrices of the atoms K(omega) sampled at x.

        Columns are the current or supplied atoms; the network is the kernel
        applied to c. Return the value kernel (N,n) and input-gradient kernel
        (N*d,n), whose rows are sample-major. Both follow the caller's grad
        mode, so supplied atoms remain differentiable during candidate search.
        """
        inner_weight = self.inner_weight if inner_weight is None else inner_weight
        inner_bias = self.inner_bias if inner_bias is None else inner_bias
        z = F.linear(x.detach(), inner_weight, inner_bias)
        value_kernel, rho_vjp = torch.func.vjp(self._rho, z)
        rho_derivative, = rho_vjp(torch.ones_like(value_kernel))
        gradient_kernel = (rho_derivative.unsqueeze(1) * inner_weight.T.unsqueeze(0)).reshape(
            x.shape[0] * self.input_dim, inner_weight.shape[0],
        )
        return value_kernel, gradient_kernel

    def _rho(self, z: Tensor) -> Tensor:
        """The paper's activation rho = activation**power, applied elementwise.

        Being elementwise, its Jacobian is diagonal: a unit cotangent gives rho'(z).
        """
        return self._activation(z) ** self._power

    def moment_weight_of_atoms(
        self, inner_weight: Tensor | None = None, inner_bias: Tensor | None = None,
    ) -> Tensor:
        """Moment weights w_p(omega) = 1 + |omega|^p of the current or supplied atoms.

        Return one weight per atom, or ones on the sphere; the atoms are not
        modified. The weights remain differentiable during candidate search.
        """
        inner_weight = self.inner_weight if inner_weight is None else inner_weight
        inner_bias = self.inner_bias if inner_bias is None else inner_bias
        return (
            moment_weight(inner_weight, inner_bias, self._moment_order)
            if self.normalized else torch.ones_like(inner_bias)
        )

    def obj(self, prediction: tuple[Tensor, Tensor], target: tuple[Tensor, Tensor]) -> Tensor:
        """Weighted empirical H1 quadratic, excluding the sparsity penalty.

        The value channel has one entry per sample. Both channels remain
        differentiable; the gradient channel is flattened sample-major.
        """
        rv, rg = ((value - reference).reshape(-1) for value, reference in zip(prediction, target))
        n = rv.numel()
        w1, w2 = self._loss_weights
        return 0.5 * ((w1 / n) * rv.dot(rv) + (w2 / n) * rg.dot(rg))

    def jac_obj(self, Kv: Tensor, Kg: Tensor, residual) -> Tensor:
        """Coefficient gradient of obj from value/input-gradient Jacobians.

        Kv is (N,n), Kg is (N*d,n); residual contains (X, value error,
        input-gradient error). Residuals are fixed during candidate search;
        supplied Jacobians retain their gradients with respect to the neurons.
        """
        rv, rg = (r.detach().reshape(-1, 1) for r in residual[1:])
        n = Kv.shape[0]
        w1, w2 = self._loss_weights
        return w1 * ((Kv * rv).sum(0) / n) + w2 * ((Kg * rg).sum(0) / n)

    def hess_obj(self, Kv: Tensor, Kg: Tensor) -> Tensor:
        """Full coefficient Hessian of obj, including cross terms between neurons."""
        n = Kv.shape[0]
        w1, w2 = self._loss_weights
        return (w1 / n) * (Kv.T @ Kv) + (w2 / n) * (Kg.T @ Kg)

    def sparsity_functional(self, u: Tensor) -> Tensor:
        """Unweighted log/power functional on the supplied penalty coordinates."""
        penalized, nonneg = self._masks(u)
        return _noncvx_penalty(
            u,
            penalized,
            nonneg,
            alpha=1.0,
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

    def jac_normal(self, u: Tensor, jac_prox: Tensor, inverse_step: float, hess_obj: Tensor) -> Tensor:
        """Selected G'(q), using u = P(q) and jac_prox = P'(q)."""
        I = torch.eye(u.numel(), device=u.device, dtype=u.dtype)
        return inverse_step * (I - jac_prox) + self.hess_obj_augmented(u, hess_obj) @ jac_prox

    def obj_augmented(self, u: Tensor, fitting: Tensor) -> Tensor:
        """Add the nonconvex splitting correction to the scalar fitting value.

        fitting is obj evaluated at the same u; the caller preserves its
        autograd graph when derivatives of the augmented objective are needed.
        """
        if self.penalty_exponent != 1.0:
            return fitting
        penalized, nonneg = self._masks(u)
        magnitudes = torch.where(nonneg[penalized], u[penalized].clamp_min(0), u[penalized].abs())
        return fitting + self._alpha * (_phi(magnitudes, self._gamma) - magnitudes).sum()

    def jac_obj_augmented(self, u: Tensor, jac_loss: Tensor) -> Tensor:
        """Derivative of obj_augmented from the supplied derivative of the objective at u.

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

    def hess_obj_augmented(self, u: Tensor, hess_obj: Tensor) -> Tensor:
        """Hessian of obj plus the diagonal Hessian of the nonconvex correction."""
        correction = _noncvx_correction_dd(
            torch.abs(u), self._gamma, penalty_exponent=self.penalty_exponent
        )
        return torch.diag(self._alpha_vec(u) * correction) + hess_obj

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

    def prox_scale_for_fractional_penalty(self, coefficients: Tensor, rho: float) -> float | None:
        """Fractional proximal scale satisfying the warm-start upper bound."""
        if self.penalty_exponent == 1.0:
            return None
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
        if self._alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self._alpha}")
        penalized, _ = self._masks(coefficients)
        magnitudes = coefficients[penalized].abs()
        nonzero = magnitudes[magnitudes > 0]
        if nonzero.numel() == 0:
            raise ValueError("warm-start proximal scaling requires a nonzero coefficient")
        minimum = float(nonzero.min())
        exponent = self.penalty_exponent
        bound = rho * minimum ** (2.0 - exponent) / (2.0 * (1.0 - exponent))
        return min(self._alpha / (1.0 + self._alpha * self._gamma), bound)

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
