# SSN problem and optimizer responsibilities

Research checked on 2026-10-06. The clarified question is whether `optimizer.py`
can own PyTorch optimization mechanics while calling a problem object for the
SSN-specific mathematical operations and nonconvex penalty. Constructor inputs
are a separate, smaller question. This note records evidence and assessment;
it adopts no new design decision or class hierarchy.

## Three responsibilities to distinguish

Our proposed vocabulary for evaluating this design:

| Responsibility | Examples | Practical ownership for the proposed split |
|---|---|---|
| PyTorch integration | Parameter groups, optimizer state, flat vectors, graph/gradient execution, committing parameters | `SSN` and its local evaluation helpers |
| Numerical procedure | Newton iterations, linear/CG solves, globalization, acceptance, stopping, radius/damping updates | SSN's solver code, including strategy and MPCG helpers |
| Problem mathematics | Objective and penalty, smooth splitting, selected proximal map/Jacobian, normal-map evaluations | Problem operations evaluated on explicit vectors |

A mathematical operation can use Torch tensor arithmetic while living on the
problem. Conversely, Newton/CG, acceptance, and stopping remain numerical
procedures even though they contain mathematical formulas. These distinctions
concern responsibilities; they do not require one new class per row.

## What the primary implementations actually do

**PyTorch permits different derivative-execution arrangements.** Its optimizer
guide shows `loss.backward()` before an ordinary `step()`, and also a closure
that computes gradients when an optimizer requests reevaluation. The base
optimizer owns parameter groups and optimization state. Thus the documented
interfaces do not require the `backward` call to be written inside the optimizer.
([PyTorch 2.10 optimizer guide](https://docs.pytorch.org/docs/2.10/optim.html#taking-an-optimization-step))

**LBFGS owns integration and iteration, with closure-supplied gradients.**
`LBFGS.step` enables gradients for the closure, gathers parameter `.grad`
values, maintains history, selects a direction, invokes its line search, and
updates parameters. Its directional evaluator installs a trial, calls the
closure, gathers gradients, and restores the parameter snapshot on normal
return. That restoration has no `try/finally`; it is not a model of exception
rollback.
([PyTorch 2.10 LBFGS source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/lbfgs.py#L325))

**torchmin gives a direct precedent for optimizer-owned autodiff.** Its
`Minimizer`, a PyTorch `Optimizer` subclass, gathers and writes flat parameters.
Its evaluation method calls `backward` and gathers derivatives. Its `step`
documentation explicitly requires a closure without a `backward` call, then
its implementation delegates the numerical solve to `minimize` and installs
the final result.
([Minimizer source, commit 7fed93e](https://github.com/rfeinman/pytorch-minimize/blob/7fed93e99ae7d1af6576737165b8068ea65c2a29/torchmin/optim/minimizer.py#L109))

For an ordinary explicit-vector callable, torchmin instead uses
`ScalarFunction.closure(x)`, which prepares a differentiable tensor and calls
`autograd.grad`. Its constructor returns an existing `Minimizer` unchanged;
the trust-region driver can therefore consume the evaluation interface from
either route. The integration adapter and numerical procedure are separable.
([ScalarFunction source](https://github.com/rfeinman/pytorch-minimize/blob/7fed93e99ae7d1af6576737165b8068ea65c2a29/torchmin/function.py#L81),
[Trust-region driver](https://github.com/rfeinman/pytorch-minimize/blob/7fed93e99ae7d1af6576737165b8068ea65c2a29/torchmin/trustregion/base.py#L163))

**DeepInv demonstrates mathematical objects called by an iteration.**
`Potential`, `DataFidelity`, and `Prior` supply operations such as gradients
and proximal maps. Its PGD example calls these operations to construct the
next iterate; predefined GD, PGD, and ADMM solvers provide the iteration
machinery. This supports keeping Torch-based mathematical evaluations in
problem objects. It does not prescribe our SSN module layout.
([DeepInv optimization guide](https://deepinv.org/user_guide/reconstruction/optimization.html))

Our inference: the proposed optimizer-owned autodiff pipeline is practical and
has direct precedent. Derivative execution inside a problem/evaluation object
also has precedent. The useful comparison is which interface makes graph
lifetime, mathematical inputs, and iteration state easiest to understand.

## Current code and the concrete point to revisit

`SSN.step()` already owns parameter extraction, initial graph creation,
normal-map system assembly, stopping, strategy dispatch, and accepted parameter
updates. Trial evaluation calls the problem on explicit vectors.
([Current SSN](../src/SSN/optimizer.py#L69))

`SparseProblem` owns the objective, penalty splitting, proximal operations,
and analytic smooth Hessian. Its `smooth_grad` currently also owns
`detach`/`requires_grad`, `enable_grad`, and `autograd.grad`; it optionally
accepts an already-built loss graph. Autograd mechanics are therefore currently
shared between optimizer and problem.
([Current problem](../src/SSN/problem.py#L65))

**Practical assessment, not an accepted change:** if optimizer-owned autodiff
is the intended boundary, the optimizer can prepare the graph and compute the
full objective gradient, while a problem operation applies the existing
penalty subtraction and smooth correction. This preserves the current
mathematical sequence and can remove the problem's dependence on a supplied
loss graph. The numerical procedure still determines when evaluations occur,
which Newton system to solve, and whether to accept the result.
([Existing gradient arithmetic](../src/SSN/problem.py#L73))

Normal-map evaluation and its selected Jacobian can also be exposed as
SSN-aware problem operations if that makes the optimizer clearer. Their inputs
would include the iterate and relevant scale. Deciding when to evaluate them,
solving the resulting Newton equation, and managing globalization would remain
solver responsibilities. Keeping these evaluations in an SSN helper is another
workable boundary; their current presence in `optimizer.py` is not itself a
correctness defect.

Moving derivative execution should preserve gradient reuse, operation order,
derivative selections, and graph lifetime. The interface discussion alone
does not justify changing those numerical behaviors.
([Refactoring principles](../refactor.md))

## Constructor injection remains a separate choice

The current trainer supplies `data_loss(u)` and its constant Hessian to
`SparseProblem`; the problem adds the penalty, and SSN calls the problem.
([Trainer construction](../src/PDAP/ssn_solve.py#L140))

Pymanopt similarly accepts a decorated cost in a plain `Problem`, then runs
an optimizer on that object. Its PyTorch backend supports autodiff, although
its 2.2.1 implementation crosses a NumPy/Torch boundary. This is interface
precedent rather than a proposed solver replacement.
([Problem API](https://pymanopt.org/docs/stable/problem.html),
[Optimizer workflow](https://pymanopt.org/docs/stable/quickstart.html),
[PyTorch backend](https://github.com/pymanopt/pymanopt/blob/2.2.1/src/pymanopt/autodiff/backends/_pytorch.py#L18))

Our assessment: retaining a callable leaves data-formula/Hessian consistency
with the caller; a concrete data-owning constructor can derive both from shared
feature/target inputs. Either can support the proposed optimization pipeline.
Changing this constructor is not necessary merely to move PyTorch mechanics
into the optimizer.
