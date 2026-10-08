---
status: accepted
---

# SparseProblem owns the network and mathematics; PDAP and SSN own optimization

## Current decision: one SparseProblem built on PyTorch (updated 2026-10-08)

The agreed refactor replaces the separate model/problem representation with
one `SparseProblem(torch.nn.Module)` that owns the current network and
problem-specific evaluations. `SignedModel`, `ShallowNetwork`, `PDAPModel`, and
their factory are removed, and callers use the single module. The implementation uses
buffers named `inner_weight` and `inner_bias`, and an `nn.Parameter` for outer
coefficients `c`. These registration details implement the agreement; they are
not additional user requirements. Direct registration avoids randomly
initialized layers that would immediately be overwritten with exact atoms.

The single `forward(x)` returns the network value and its gradient with respect
to x; callers use `problem(x)` and unpack the pair. `jac_forward` returns the
Jacobians of both outputs with respect to outer coefficients. Ordinary gradient
mode preserves differentiation through both outputs; reporting requests
`no_grad`. Loss weights belong to the objective, not the prediction outputs.
This replaces the separate value-only and value-and-gradient prediction methods.

Data stays outside the module, supplied through standard Dataset/DataLoader
delivery. Data utilities own reading, validation, physical normalization,
splitting and assembly; the problem owns mathematical evaluations of delivered
samples. The agreed preparation lifetime is one correction, outside persistent
problem state. The user rejected the `PreparedObjective` carrier class: useful
reuse does not require a named type. Preparation returns an objective callable,
fitting Hessian and normalization tensor as ordinary values. PDAP keeps the
normalization tensor local for coefficient conversion; SSN receives the callable
and Hessian. Features, targets and fitting Hessian are reused while SSN changes
outer coefficients. These calculations provide no network backup. The
normalization tensor contains ones for sphere atoms; otherwise its values depend
on the neurons even when the configured moment order stays fixed.
`SparseProblem.obj(prediction, target)` is the canonical scalar H1 fitting
objective. Prepared coefficient fitting calls it using the fixed features and
targets. This closure can reference the same problem during its correction;
avoiding a second network does not require forbidding that reference.
`jac_obj` and `hess_obj` provide the coefficient gradient and full Hessian of
that objective, including cross terms. PDAP uses those derivatives for individual
neuron or combined-direction slopes and curvatures. The former `profile` and
`directional_obj` methods and `_fidelity_pairing` helper are removed, without a
renamed helper. The short objective, gradient and Hessian contractions represent
distinct mathematical operations. The existing warm-start sphere derivative
convention remains a separately documented historical numerical choice.

PDAP owns candidate search/ranking, insertion acceptance, the warm-start
procedure and initial-coefficient selection, support insertion/pruning,
correction scheduling and outer stopping. SparseProblem evaluates supplied candidates/directions:
features, objective derivatives, and penalty/proximal operations.
PDAP owns its insertion-increment calculation and certificate radius, including
the theorem formula and numerical cap/fallback. Mathematical evaluation does not transfer search, initialization,
or acceptance decisions to the problem. The mixed insertion/warm-start methods
have been removed; their procedures now live in PDAP's insertion and warm-start
modules. The assistant's former proposal to move complete warm-start selection
into SparseProblem was withdrawn; it was not an agreed ownership change.

The problem may evaluate a threshold or proximal minimizer, including branches
defined by the mathematical operator. PDAP decides how to use those evaluations
for search, initialization and insertion. PDAP's proposed atoms and SSN's
temporary working coefficients do not create a second network. PDAP decides
when accepted changes are applied to the persistent network.

SSN remains a `torch.optim.Optimizer`; its `step()` owns the complete configured
iteration budget and the final objective acceptance/fallback on temporary
coefficients. PDAP applies the result. The old `ssn_solve` layer is removed.
Historical explicit unguarded configurations remain comparators, with their
setting passed to SSN. This and the exact iteration defaults are compatibility
choices; an iteration budget does not establish convergence. Hydra stays at
repository level. One configuration function owns training compatibility rules
and takes explicit settings. `SparseProblem.validate_training` supplies its own
settings to that function; construction/preparation and optimizers use this
single entry operation. Configuration and optimizers do not inspect private
model fields, and no second settings object or getters are needed. The ordinary
constructor still supports historical prediction settings.

`moment_order` defaults to `None`; sphere models have no moment order. Legacy
numeric placeholders and mixed-preset values are ignored for that geometry.
Nonhomogeneous training requires an explicit positive finite order, checked in
configuration against the activation's declared H1 growth (`p > s1`). An
activation's growth constrains the choice without uniquely determining it.
Registered activations without growth metadata retain explicit-order numerical
training and the existing radius fallback; no theoretical bound is invented.
Prediction alone needs no moment order. Evaluation reports `psi_p=None` for
sphere models while retaining their total variation.

The user subsequently requested separation of metrics from mathematics.
Evaluation owns split iteration, reporting denormalization, support diagnostics
and metric dictionaries; it calls the problem's objective and raw sparsity
functional. History invokes evaluation and owns one detached checkpoint with
existing artifact views and a single restoration. Broader logging/backend
redesign remains deferred. Read-only activation identity and moment order expose
the existing mathematical facts needed by PDAP and evaluation without copying
configuration or inspecting private fields.

Implementation and verification status live in
[REFACTOR_HANDOFF.md](../../REFACTOR_HANDOFF.md), including the user's statements,
verification, deferred work, and the cause/prevention record. That evidence
takes precedence over assistant-authored architectural summaries. The remainder
of this ADR records earlier designs; its separate-model, data-owning problem,
and cached-objective descriptions are superseded by this decision.

## Context

`src/models/signed.py` and `src/models/semiconcave.py` had each grown to carry
five concerns at once: the network parametrization (forward, weights), the SSN
optimization (both models `from ..SSN import SSN`), the warm-start coordinate
descent, the loss + nonconvex regularizer, and the relative-error evaluation.
The two models duplicated the warm-start (~75 vs ~54 lines, differing only in a
sign convention and a nonnegativity clamp) and the relative-error computation
(identical arithmetic), and they each embedded an SSN solve.

Reading the two SSN solves side by side shows the models are the *same kind of
object*. Both are **linear in their outer parameters θ** for the outer solve:
the data Hessian is `(w1/Nx)·Φvᵀ Φv + (w2/Nx)·Φgᵀ Φg` in both cases — the signed
model's `Φv` is its network matrix and `Φg` its gradient kernel; the semiconcave
model appends structural columns for the curvature `C` and affine `a, b0`. The
differences reduce to (1) which feature columns exist, (2) the penalized/nonneg
masks, (3) sign convention — and (3) is just the nonneg mask. The duplication
existed because nobody factored out "the linear-in-θ SSN problem."

`V = ½C‖x‖² − g(x)` for the semiconcave model is a composition of submodules
(quadratic head + shallow network + affine), each linear in its own parameters,
so it needs no bespoke analytic feature code: predictions and the constant
Jacobians `(Φv, Φg)` can come from autodiffing `forward`. SSN is retained for
*both* models — it is the prox-Newton solver for the nonconvex sparse
regularizer that defines this project; a smooth optimizer (Adam/SGD) cannot
deliver the sparsity or handle the nonsmooth penalty, and the signed model needs
SSN for the identical reason.

## Decision

Separate the three concerns along these layers:

- **`src/models/`** — parametrizations only: `forward` / predictions, named
  parameters, and a tag for which parameters are penalized / nonnegative. No
  import of `..SSN`. The semiconcave model becomes a composed network; its
  Jacobians come from autodiff (constant per support, recomputed only when the
  support changes at insertion), retiring the hand-built feature maps.
- **`src/PDAP/`** — the trainer and the loop-in-loop
  (`insert → warm-start → SSN → prune → insert`). One generic SSN outer solve and
  one generic warm-start, driven by the model's Jacobians and masks.
- **`src/eval.py`** — performance evaluation as pure functions of
  `(prediction, target)` tensors: relative L2/gradient/H1 errors and the
  value/gradient data-loss split. No model state. The nonconvex penalty is *not*
  here — it depends on which parameters a model penalizes, so it stays the
  model's responsibility and is added to the data loss to form the objective.
- **`src/data.py`** — value-sample loading, normalization, and the train/valid
  split (`split_value_samples`), surfaced as `DataConfig.train_fraction`.

This revises ADR-0003's note that "model-level relative errors may remain in the
model/PDAP training path until there is a concrete extraction need": that need
has arrived (verbatim duplication across two models). Evaluation moves to a
dedicated `src/eval.py` rather than into `src/metric.py`, which ADR-0003 reserves
as a reporting/table utility — `metric.py` remains a presentation layer and does
not own performance evaluation.

## Consequences

The change lands as small, golden-backed steps (`tests/test_pdap_equivalence.py`
pins exact summaries), regenerating the golden only when a step deliberately
changes numerics:

1. Extract evaluation to `src/eval.py`; models and PDAP call it; delete the
   duplicated `_compute_relative_errors`. Behavior-preserving (golden unchanged).
2. Delete the dead non-SSN path in `signed.py` (`train`, the generic-optimizer
   branch, SGD fallback, `optimizer_type`); PDAP always uses SSN.
3. Move warm-start into `src/PDAP/` as one generic function.
4. Move the SSN outer solve into `src/PDAP/`, driven by model Jacobians.
5. Rewrite the semiconcave model as a composed network with autodiff Jacobians.
6. Define the thin model contract (`src/models/base.py:PDAPModel`); `SignedModel`
   becomes an adapter over `ShallowNetwork`.

All six steps are implemented. In the event, only step 1's split changed
numerics (a real train/validation split, golden regenerated once); steps 2-6 all
landed bit-exact, including the semiconcave autodiff Jacobians (autograd runs the
same operations the analytic feature maps did). The result: the duplication is
gone, the layering inversion is fixed (neither model imports the SSN optimizer),
evaluation has a single home, and the trainer depends on one explicit model
contract.

## Follow-up: models as nn.Modules; objective/solver config out of the model

Pushing the same principle (model = parametrization, use the framework) further,
three more steps, all bit-exact:

7. **Both models are `nn.Module`s.** `SignedModel` *subclasses* `ShallowNetwork`
   (it is the shallow net, not a wrapper around `self.net`); `SemiconcaveModel`
   is an `nn.Module` with `c, C, a, b0` as parameters registered in θ-order and
   `W, b` as buffers. `theta` (the SSN working vector) is just the trainable
   parameters, read/written with torch's `parameters_to_vector` /
   `vector_to_parameters` — so `get_theta` / `set_theta` leave the contract.
8. **The objective leaves the model.** `compute_loss` is removed; the trainer
   computes the objective (`PDAP._record_loss` for the data term via
   `src.eval.data_loss_terms`, plus `src.SSN.penalty._noncvx_penalty` for the
   regularizer — the same penalty the SSN closure uses, so it exists once).
9. **Objective and solver hyperparameters leave the model.** `alpha, gamma, th,
   loss_weights` and `lr, method, max_ls_iter, tolerance_*, sigmamax` move into
   trainer dataclasses (`Objective`, `SolverConfig` in `ssn_solve`), built by PDAP
   from config and threaded to the solve / warm-start / recording. Model
   constructors take only forward-defining parameters (`activation`, `power`,
   plus `c_init`/`dtype` for semiconcave); `power` stays because it defines
   `σ^p` and induces `q = 2/(power+1)`.

The `PDAPModel` contract is now minimal: `power, q, input_dim, last_fit_summary`,
`parameters()`, and the methods `set_atoms / get_atoms / predict / predict_tensors
/ jacobians / penalty_masks`. `jacobians` and `penalty_masks` are the only
model-specific surface the trainer can't obtain from `nn.Module` built-ins —
`jacobians` is the feature-map computation, `penalty_masks` the irreducible
penalty/nonneg structure (the nonneg part *is* semiconcavity, so it cannot be
flattened away). Bit-exactness held because building `nn.Parameter`s consumes no
RNG (unlike `nn.Linear`, whose draw `set_atoms` reuses via `ShallowNetwork`'s
constructor) and `parameters()` yields θ in the same `[c | C | a | b0]` order the
old hand-packing used.

## Penalty selection stays config-driven

The penalty is `alpha * Σ phi(|c|^q)` with two independent knobs — the power-q
exponent (`q = 2/(power+1)`) and the log-nonconvex `phi` (`gamma`, `th`). We
considered a named `penalty: l1 | power_q | log | composed` selector (which would
have decoupled the penalty exponent from the activation power so `log`/`l1` force
`q = 1`), but rejected it: it adds an enum plus validation and a second notion of
`q` for a choice the existing parameters already express. Instead the two
penalties this project uses are selected by parameter values, documented inline in
`ModelConfig`:

- **power penalty** `alpha * Σ |c|^q` — `gamma = 0` (phi is the identity),
  `power > 1` (so `q < 1` is non-convex).
- **log penalty** `alpha * Σ phi(|c|)` — `power = 1` (so `q = 1`), `gamma > 0`.

So the penalty is configurable but not a named axis; `power` and `gamma` remain
its controls, and the penalty exponent stays coupled to the activation power.

## Model compatibility and numerical precision

`PDAP.fit` checks that the supplied model uses the configured activation
callable and power before reading data or changing model state. Callers use
`build_model` with the same configuration as the trainer. This prevents the
trainer's geometry, growth data, and normalized objective from disagreeing with
the actual network. `PDAPModel` also declares activation and state save/restore,
which insertion, history, and the correction guard already require.

Floating-point data, model layers, and numerical work arrays use float64.
Layers are created in float64 before assigning supplied coefficients, so
support replacement cannot round an SSN solution through float32. This changes
the layer-initialization random stream and supersedes the historical bit-exact
behavior above; the PDAP reference summaries are refreshed for this deliberate
numerical change.

## Follow-up: one shared problem for PDAP and SSN

`src/problems/sparse_problem.py` now defines `SparseProblem`. It owns the sample
splits `(X, V, dV)`, neuron definition, objective weights, penalty, normalization,
proximal and normal maps, insertion formulas, and mathematical radius bound.
The penalty, proximal, moment, and radius helpers live inside `src/problems/`.
The historical trainer-owned `Objective` and callable/Hessian constructor are
replaced by this data-based interface:

```python
problem = SparseProblem.from_config(cfg.model, train_data, valid_data)
model = build_model(cfg, problem.input_dim)
trainer = PDAP(cfg.training, insertion=cfg.model.insertion)
history = trainer.fit(model, problem, num_iterations=10, num_insertion=50)
```

The model owns the persistent parameters and predictions. It is passed to
problem operations explicitly and is never retained by the problem. PDAP owns
candidate search, insertion/pruning, warm-start direction selection, correction
scheduling and acceptance guards. SSN owns Torch parameter integration,
autograd of the coefficient loss, iteration, globalization and accepted updates.
History stores metrics supplied by the problem and model snapshots; generic
relative-error computations remain in `src/eval.py`. Persisted metric names are
unchanged.

Before each coefficient solve, `problem.prepare_coefficients(model)` refreshes
the fixed-neuron prediction matrices, coefficient normalization, and objective
Hessian. `ssn_solve` then creates a fresh SSN optimizer because support replacement
replaces the model's Parameter objects. The problem is reused for the whole fit;
full-model loss, residual, and insertion evaluations work independently of this
cache, including on an empty support. A problem has one prepared coefficient
system at a time, so simultaneous coefficient solves need separate instances.

The mathematical API uses `obj` for the fitting objective, `obj_augmented` for
that objective plus the nonconvex correction, and `loss` for the fitting
objective plus the full penalty. Derivatives use `jac_` and `hess_`, including
`jac_obj_augmented`, `hess_obj_augmented`, `jac_prox`, and `jac_normal`.
`penalty_exponent` names the scalar exponent; `q` remains the proximal preimage.
For the log family, `loss = obj_augmented + alpha*||u||_1`; for the fractional
family, `obj_augmented = obj` and the prox handles the complete penalty.

The mathematical radius is returned before applying a numerical cap; PDAP owns
the existing `exp(5)` search limit and fallback. Numerical regression fixtures
are retained unchanged by this ownership refactor.
