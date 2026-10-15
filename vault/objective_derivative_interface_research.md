# Objective derivatives and temporary preparation

Research checked on 2026-10-08 against baseline `69a8fa9` and primary
PyTorch, SciPy, and Firedrake documentation/source. The user accepted removing
`PreparedObjective`. After discussing the combined
`forward`, coefficient derivatives and autograd behavior, the user authorized
implementation with "understood, go". The interfaces below and removal of the
separate pairing helper are now accepted. Implementation status lives in
[REFACTOR_HANDOFF.md](../REFACTOR_HANDOFF.md); this note records the research.

## One forward operation

The accepted `forward(x)` returns `(value, input_gradient)`. It replaces both
the value-only `forward` contract and `forward_with_gradient`;
`jac_forward_with_gradient` becomes `jac_forward`. `obj` evaluates the H1
quadratic from this pair and the target pair. This is compatible with
`nn.Module`: PyTorch's own
[`LSTM`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
returns an output and a tuple of states, and its
[`forward` source](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/rnn.py#L1114)
returns that structure directly. A tensor is not the only legitimate result.

Callers should use `problem(x)`, unpacking the pair. PyTorch's
[`Module.forward` documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.forward)
explains that calling the module runs registered hooks, whereas calling
`problem.forward(x)` directly bypasses them. This requires updating consumers,
not introducing another value-only wrapper or output-selection flag.

The pre-change `forward_with_gradient` could not simply be renamed unchanged: it
detached both outputs and computed the input derivative with
`create_graph=False`. H1 training must retain how both outputs depend on the
outer coefficients. If the input derivative is obtained by differentiating the
network value with `autograd.grad`, differentiable use needs
`create_graph=True`; an analytic derivative can instead preserve that
coefficient dependence directly. The caller's input graph should also remain
connected when the caller requests input differentiation.
([Implementation](../src/problems/sparse_problem.py),
[`autograd.grad`](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html))

Before cleanup, some reporting paths used `no_grad`; others relied on
the helper's detached outputs and now request `no_grad` explicitly. An
implementation using autograd to obtain input gradients needs a local
`enable_grad` region for that calculation. PyTorch supports this nesting.
Preserve the caller's
gradient mode for returned training versus reporting tensors; do not use
`model.eval()` as a substitute for gradient-mode handling. None of this requires
a new public option.
([`no_grad`](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html),
[`enable_grad`](https://docs.pytorch.org/docs/2.14/generated/torch.enable_grad.html),
[`Module.eval`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval))

## Output derivatives and objective derivatives

| Operation before cleanup, at `69a8fa9` | Quantity and dependency |
|---|---|
| `jac_forward_with_gradient` | Change in every predicted value and input gradient when an outer coefficient changes. Returns feature matrices; does not use targets or residuals. |
| `profile` | First derivative of `obj` in one proposed neuron's coefficient direction, using those features and the current residual. Candidate-search differentiation continues through the proposed inner parameters. |
| `directional_obj` | First and second derivatives of `obj` along supplied feature columns or a combined direction. The second result is per-direction curvature, not a full Hessian with cross terms. |

This distinction follows the chain rule, not a required class layout. PyTorch's
[`jacrev`](https://docs.pytorch.org/docs/2.14/generated/torch.func.jacrev.html)
returns a function's Jacobian with respect to a specified argument;
[`vjp`](https://docs.pytorch.org/docs/2.14/generated/torch.func.vjp.html)
contracts output derivatives with supplied output weights. SciPy's
[`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
likewise distinguishes the residual Jacobian from the scalar cost gradient.

At `69a8fa9`, `directional_obj` consumed the result of
`jac_forward_with_gradient`; it did not reimplement feature differentiation.
Coefficient preparation also used the forward Jacobian independently. The
warm-start caller evaluated individual columns and their combined direction.
([Problem](../src/problems/sparse_problem.py),
[warm start](../src/PDAP/warmstart.py), [insertion](../src/PDAP/insertion.py))

The accepted interface makes the distinction explicit: `jac_forward` supplies
output derivatives with respect to outer coefficients; `jac_obj` and `hess_obj`
supply the first and second objective derivatives with respect to those
coefficients, replacing `directional_obj`. For one candidate coefficient they
give its slope and curvature. For several coefficients the Hessian includes
cross terms; the former per-column curvature vector contained only its diagonal.
This supports the user's naming decision. The implementation replaces the
separate `profile` and `directional_obj` interfaces with these two operations.

## Does an inner product need its own method?

There is no universal extraction rule in the inspected implementations.
PyTorch's [`LBFGS.step`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/lbfgs.py#L404)
computes the directional derivative directly as `flat_grad.dot(d)`. SciPy
extracts recognizable operations such as
[`compute_grad`, `build_quadratic_1d`, and `evaluate_quadratic`](https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/_lsq/common.py),
while leaving their ordinary dot products inline. PyTorch's
[`MSELoss.forward`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/loss.py#L571)
delegates to the public functional loss, not to a generic pairing layer.

The recognizable mathematical name here is a **weighted empirical H1 inner
product**: multiply corresponding values and gradients, sum them with the two
loss weights, and divide by the sample count. Firedrake's official
[`norm` implementation](https://www.firedrakeproject.org/_modules/firedrake/norms.html)
uses the value inner product plus gradient inner product for H1. Our sampled,
weighted version is an inference from that definition and the local formula;
it is not a continuous-domain integral. A zero channel weight gives a
semidefinite form on the full value/gradient space.

Earlier this note conditionally suggested `_h1_inner_product` as a clearer
name. That recommendation is superseded by the user's requested consolidation:
remove `_fidelity_pairing` without a renamed replacement. Keep the short
weighted contractions directly in `obj`, `jac_obj`, and `hess_obj`, sharing the
existing loss weights and feature calculations. The objective, its first
derivative, and its second derivative are distinct formulas; writing their
direct contractions does not duplicate competing objective definitions. No new
class, generic contraction API, or mode flags are justified here.

`torch.inner` itself is only a tensor contraction; it does not supply the H1
channel weights or sample normalization. Default `MSELoss` averages over all
elements, which would incorrectly divide the gradient channel by the input
dimension as well. These primitives require the repository's existing scaling.
([`torch.inner`](https://docs.pytorch.org/docs/2.14/generated/torch.inner.html),
[`MSELoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html))

## Does preparation require a class or network copy?

At `69a8fa9`, `PreparedObjective` was a plain frozen dataclass, not a
`SparseProblem` subclass. Its three fields were an objective closure, a fixed
Hessian, and normalization weights. The closure retained feature matrices and
target tensors for one solve; it contained no copied network or inner-parameter
backup. SSN separately copies the starting outer coefficients for rollback.
([Preparation](../src/problems/sparse_problem.py), [SSN](../src/SSN/optimizer.py))

The implementation removes this record. Reusing fixed features/Hessian
avoids recomputing them at every trial. That is an efficiency argument for local
tensors, not a necessity argument for this
dataclass. PyTorch's `vjp` returns an output and a reusable closure as a tuple;
[`LBFGS.step`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.step)
accepts a callable. Neither requires or justifies our three-field type. Returning
ordinary local values is a viable alternative; replacing the dataclass with a
new wrapper class or hidden model cache would not simplify the boundary.

The original research did not test a numerical migration. The old helper
preserved different reduction orders for search and correction. Subsequent
implementation uses one product-sum coefficient gradient for every caller;
both PDAP reference summaries and all four SSN trajectories pass at their
unchanged tolerances. The initial matrix-product alternative changed search
trajectories and was rejected. Historical arithmetic does not prove that a
helper or class is necessary. The repository's [refactoring rules](../refactor.md)
require both meaningful operations and preservation of numerical behavior;
final verification is recorded in the handoff.
