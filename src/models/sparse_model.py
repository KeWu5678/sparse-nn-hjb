"""Neuron and fitting mathematics evaluated on externally supplied data."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseModel(nn.Module):
    """Own the sparse network and its data fitting without owning samples.

    obj evaluates data fitting from predictions and targets; Proximal owns the
    penalty term, and callers form the objective obj + Proximal.penalty. The
    problem supplies the penalty exponent q = 2/(power+1) its activation implies.
    Optimizers own parameter mutation, differentiation of the objective, and iterations.
    Construction consumes an already resolved configuration. Training setup must
    validate its penalty, geometry and insertion before constructing the problem.
    """

    inner_weight: Tensor
    inner_bias: Tensor

    def __init__(self, config) -> None:
        super().__init__()
        self._activation_name = config["activation_name"]
        self._activation = config["activation"]
        self._use_sphere = config["use_sphere"]
        self._moment_order = config["moment_order"]
        self._power = config["power"]
        self._loss_weights = config["loss_weights"]
        self.register_buffer("inner_weight", torch.empty((0, config["input_dim"]), dtype=torch.float64))
        self.register_buffer("inner_bias", torch.empty(0, dtype=torch.float64))
        self.c = nn.Parameter(torch.empty(0, dtype=torch.float64))

    @property
    def input_dim(self) -> int:
        return self.inner_weight.shape[1]

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
        if not self.normalized:
            return torch.ones_like(inner_bias)
        squared_norm = (inner_weight * inner_weight).sum(dim=-1) + inner_bias * inner_bias
        return 1.0 + squared_norm.clamp_min(0.0).pow(self._moment_order / 2.0)

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
