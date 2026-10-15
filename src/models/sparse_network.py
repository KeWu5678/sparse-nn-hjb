"""Neuron and fitting mathematics evaluated on externally supplied data."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseNetwork(nn.Module):
    """Own the sparse network and its data fitting without owning samples."""

    inner_weight: Tensor
    inner_bias: Tensor

    def __init__(self, config) -> None:
        super().__init__()
        self.activation_name = config["activation_name"]
        self.activation = config["activation"]
        self.use_sphere = config["use_sphere"]
        self.moment_order = config["moment_order"]
        self.power = config["power"]
        self.penalty_exponent = config["penalty_exponent"]
        self.loss_weights = config["loss_weights"]
        self.register_buffer("inner_weight", torch.empty((0, config["input_dim"]), dtype=torch.float64))
        self.register_buffer("inner_bias", torch.empty(0, dtype=torch.float64))
        self.c = nn.Parameter(torch.empty(0, dtype=torch.float64))

    @property
    def input_dim(self) -> int:
        return self.inner_weight.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.inner_weight.shape[0]

    @property
    def normalized(self) -> bool:
        return not self.use_sphere

    def set_atoms(self, inner_weight: Tensor, inner_bias: Tensor, c: Tensor) -> None:
        """Replace support with independent copies of the supplied atom tensors.
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
        return self.activation(z) ** self.power

    def moment_weight_of_atoms(
        self, inner_weight: Tensor | None = None, inner_bias: Tensor | None = None,
    ) -> Tensor:
        """Moment weights w_p(omega) = 1 + |omega|^p of the current or supplied atoms.
        """
        inner_weight = self.inner_weight if inner_weight is None else inner_weight
        inner_bias = self.inner_bias if inner_bias is None else inner_bias
        if not self.normalized:
            return torch.ones_like(inner_bias)
        squared_norm = (inner_weight * inner_weight).sum(dim=-1) + inner_bias * inner_bias
        return 1.0 + squared_norm.clamp_min(0.0).pow(self.moment_order / 2.0)

    def obj(self, x: Tensor, target: tuple[Tensor, Tensor], c: Tensor | None = None) -> Tensor:
        """Weighted empirical H1 quadratic of the network at x, excluding the sparsity penalty.
        """
        prediction = self(x) if c is None else torch.func.functional_call(self, {"c": c}, (x,))
        rv, rg = ((value - reference).reshape(-1) for value, reference in zip(prediction, target))
        n = rv.numel()
        w1, w2 = self.loss_weights
        return 0.5 * ((w1 / n) * rv.dot(rv) + (w2 / n) * rg.dot(rg))

    def jac_obj(
        self, residual, inner_weight: Tensor | None = None, inner_bias: Tensor | None = None,
    ) -> Tensor:
        """Gradient of obj in the outer weights of the current or supplied atoms.

        residual is (x, value error, input-gradient error) and is held fixed. A
        supplied atom's entry is the slope at weight zero and stays
        differentiable with respect to the atom during candidate search.
        """
        Kv, Kg = self.kernel(residual[0], inner_weight, inner_bias)
        rv, rg = (r.detach().reshape(-1, 1) for r in residual[1:])
        n = Kv.shape[0]
        w1, w2 = self.loss_weights
        return w1 * ((Kv * rv).sum(0) / n) + w2 * ((Kg * rg).sum(0) / n)

    def hess_obj(
        self, x: Tensor, inner_weight: Tensor | None = None, inner_bias: Tensor | None = None,
    ) -> Tensor:
        """Full Hessian of obj in the outer weights of the current or supplied atoms."""
        Kv, Kg = self.kernel(x, inner_weight, inner_bias)
        n = Kv.shape[0]
        w1, w2 = self.loss_weights
        return (w1 / n) * (Kv.T @ Kv) + (w2 / n) * (Kg.T @ Kg)
