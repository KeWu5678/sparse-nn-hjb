# Metrics and measuring: ownership and framework options

> **Superseded (2026-10-15).** Metric names, History storage and the run
> artifacts now follow [ADR 0017](../docs/adr/0017-metric-catalog.md) and
> [ADR 0016](../docs/adr/0016-mlflow-is-the-system-of-record.md). The JSON run
> records, `src/experiment_logging.py` and `src/problems/` described below no
> longer exist. The ownership decisions (native PyTorch, full-split evaluation,
> no metric framework) still hold.

Investigated 2026-10-08. **Design accepted; implementation verified.** The user has
confirmed prediction quality, solver diagnostics, and runtime costs are in scope,
with an explicit constraint to keep the codebase small.
This note supplements the read-only [earlier investigation](data_monitoring_research.md).
The investigation initially changed only this note. The user has now confirmed
the complete plan, including both loop orders and the limited measurement scope.
The accepted ownership is recorded in [ADR 0004](../docs/adr/0004-model-trainer-eval-separation.md);
the public training flow is described in [the algorithm guide](algorithm.md).

**Native PyTorch and the current full-split evaluation are retained.** The user
rejected introducing streaming: the production path evaluates the whole split,
and DataLoader delivery does not imply minibatch training. Continue manual MLflow
publication downstream of local records. The user explicitly permits redesigning
the existing interfaces from scratch and asks for concrete PyTorch implementation
patterns. The user accepted one complete `PDAP.step` plus a small external
training loop that evaluates and records results. The user also accepted keeping
PDAP plain and composing the existing SSN optimizer rather than inheriting
`torch.optim.Optimizer` at the outer level.
Requiring evaluation to move out of History was too strong as a universal rule;
its placement should follow the chosen integration. TorchMetrics is not selected.
Avoid a class per scalar, event bus, or backend hierarchy for these requirements.

## Boundary and baseline observations

The separation is sound when measurement means observing/reporting a calculation.
Objectives, derivatives, search criteria, and correction guards remain necessary
mathematics even when no output is requested. At the investigation baseline
`e8d02e3`, History invoked evaluation. That was a valid arrangement: a recorder
can coordinate evaluation while formulas stay in evaluation functions. The user
subsequently selected the external-loop design recorded in
[ADR 0004](../docs/adr/0004-model-trainer-eval-separation.md).
History now receives supplied metrics. This follows our chosen integration,
not a universal rule requiring all History implementations to be passive.

| Concern | Accepted responsibility split |
|---|---|
| Samples, preprocessing, splits, delivery | Data utilities and Dataset/DataLoader. They do not define metric reductions. |
| Prediction, input derivatives, fitting objective, penalty | SparseProblem and mathematical helpers. Evaluation reuses the canonical operations. |
| Insertion, correction, acceptance, stopping, required counters | PDAP/SSN. Report existing outcomes without recalculating them for a logger. |
| Dataset evaluation, norm reductions, physical-coordinate reporting, support diagnostics | Evaluation/measurement functions with temporary state. Return scalar values for consumers. |
| Observation positions and cadence | Training orchestration, configured before execution. Preserve quantities needed for selection/control independently of optional output. |
| Metric records and detached checkpoints | History stores supplied measurements and preserves minimum-training-objective selection; the external loop initiates evaluation. |
| Run identity, persistence, tracking transport | Runner/ExperimentRun. MLflow consumes the resulting record. |
| Presentation and profiling | Logging/plots and optional timing/profiler instrumentation consume or surround these operations. |

The following observations refer to the pre-refactor baseline `e8d02e3`.
File links identify the modules; their current contents include the refactor.

- [History](../src/PDAP/history.py) took loaders, invoked metrics, cloned state
  and selected the minimum training loss. Keeping this coordination would minimize migration, but migration
  size is not the user's design criterion. A new callback hierarchy or record
  class is unnecessary in either a fresh design or an incremental change.
- [src/eval.py:41](../src/eval.py) calls `full_batch` for each split. The runner
  uses the loader's full-split default ([src/data.py:79](../src/data.py)). Keep
  that evaluation and its arithmetic; no streaming accumulator is proposed.
- [src/SSN/optimizer.py:102](../src/SSN/optimizer.py) already exposes iteration,
  failed-step, and correction-acceptance state. Capture it at the owning operation;
  these are events/counters, not prediction-error metrics. A fixed SSN iteration
  budget does not establish convergence (`step` documentation at line 111).
  [PDAP._correct](../src/PDAP/pdap.py) discarded that optimizer after applying its
  coefficients, without returning diagnostics. The refactor returns those facts
  for the runner to count and History to retain.

**Cadence is not merely cosmetic:** reducing training-objective observation points
can change the selected checkpoint. Decouple optional diagnostics/display cadence
from required selection/control quantities before changing evaluation frequency.

## Concrete PyTorch integration

PyTorch's ordinary control flow is an external loop that calls an optimizer's
`step()` and separate evaluation code. Its [optimizer documentation](https://docs.pytorch.org/docs/2.10/optim.html#taking-an-optimization-step)
defines parameter updates through `step`, and its
[optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation)
demonstrates separate training and evaluation functions. Applying that control
pattern to this project is a design inference, not a built-in PDAP interface.

The implemented interfaces used by the external loop are:

```python
diagnostics = pdap.step(problem, train_loader)
quality = metrics(
    problem, train_loader, valid_loader, reporting_normalizer=normalizer,
)
history.record(problem, quality, diagnostics)
```

| Operation | Inputs | Returned value and effect |
|---|---|---|
| `PDAP.step` | Current network, full training population; settings supplied at setup | Mutates the network through search/insertion, SSN correction and pruning; returns execution facts and an algorithmic stop reason. |
| `metrics` | Current network, full evaluation populations, reporting normalization | Returns scalar quality/support metrics without updating the network. |
| `History.record` | Current model, calculated values and solver facts | Appends a record and detached checkpoint; its list position supplies the checkpoint index. No solver decisions. |

The runner owns repetition within the configured budget, timing, evaluation and
recording. PDAP still determines whether an admissible candidate exists and
whether to stop; the runner obeys that result. Ordinary no-candidate termination
is distinct from an error or a convergence proof. Each loop iteration is a PDAP
iteration using the complete training split, not a minibatch update.

Both paper algorithms have a natural complete step:
search/select/initialize/insert, then SSN correction and acceptance, then pruning
([Algorithms 1 and 2](../paper/paper_0805.tex), lines 2902–2921 and 3015–3034).
PDAP need not inherit `torch.optim.Optimizer` solely to use this arrangement;
SSN already supplies that optimizer interface for its fixed-support solve.

Initial-state evaluation and terminal failed-search diagnostics need deliberate
handling. Recording the initial model permits zero-iteration and no-insertion
runs. A failed terminal search has execution cost even if it produces no new
model-quality checkpoint. Returning stop reasons and actual counters avoids
inferring solver work from history length.

The historical correction-first algorithm observes between pruning and a trailing
insertion. An explicit runner branch preserves this order through PDAP's
`initialize`, `train_ssn_outerweights_and_prune` and `insert` operations, including the final
uncorrected insertion. No historical algorithm has been removed.

## Should PDAP inherit torch.optim.Optimizer?

**Recommendation: use a plain PDAP class with `step`; compose SSN as a native
optimizer for coefficient correction.** This is a suitability judgment for this
algorithm and the accepted external loop, not a PyTorch restriction.

The base class supplies parameter groups, parameter-associated state, gradient
clearing, serialization and step hooks. PDAP's outer operation also searches for,
inserts and removes neurons; it requires the model's structural operations and
training objective/data in addition to the coefficient tensors. That remains true
when designing the representation from scratch. A parameter registry adds value
only if the outer algorithm actually uses it or an integration requires it.

Dynamic membership is possible: [`add_param_group`](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.add_param_group.html)
adds parameters during training. The [base implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/optimizer.py)
has no corresponding removal/remapping helper for pruning a packed coefficient
vector. A subclass can implement the required lifecycle. In the current model,
[`set_atoms`](../src/problems/sparse_problem.py) replaces `c` with a new Parameter;
optimizer groups would need rebinding and any associated state would need deliberate
discarding or migration. This implementation detail illustrates the cost; preserving
it is not a requirement. Zero-neuron support is not a blocker: the model contains
one zero-element Parameter, which the installed optimizer constructor accepts.

Native [`state_dict`](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.state_dict.html)
saves optimizer state and group metadata, not model parameter values or buffers.
Loading matches saved state to current parameters in group order; it does not
reconstruct the support or remap per-atom state after pruning. Exact run resume
would also require model/configuration and RNG/data state, irrespective of inheritance.
That capability is not introduced by this investigation.

The documented [`step`](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.step.html)
accepts an optional objective closure and returns a loss or None. A custom
`step(problem, loader) -> diagnostics` is possible in Python but does not become
interchangeable with generic optimizer consumers merely by inheriting the class.
PDAP can use tensors, autograd and profiling without that inheritance. Conversely,
gradient-free logic or internal repeated objective evaluations do not rule out
inheritance: native LBFGS already uses a closure for repeated evaluations.

SSN corrects a fixed temporary normalized coefficient vector and already uses
native per-parameter trust-region state. The outer insertion-first PDAP currently
has no essential persistent per-parameter optimizer state across steps; model
support carries the iterate. Keeping the outer class plain avoids maintaining an
additional parameter registry without a demonstrated consumer. Reconsider if a
specific optimizer-consuming integration or standard outer step hooks provide
concrete simplification.

## Earlier callback alternative (not implemented)

An injected callable is simply a Python function passed as an argument. A
proposed callback contract is `observe(iteration, problem, diagnostics) -> None`.
Its inputs are an iteration number, the live corrected/pruned model, and solver
facts. The return value is `None`; the function's effect is storing measurements.
There is no stdin/stdout transport. Run-local datasets, normalization and history
can be captured by the surrounding function:

```python
def observe(iteration, problem, diagnostics):
    quality = metrics(
        problem, train_loader, valid_loader, reporting_normalizer=normalizer,
    )
    history.record(problem, quality, diagnostics)

pdap.fit(problem, train_loader, observe=observe)
```

This was proposed project syntax, not an existing PyTorch or PDAP method. PDAP
would call `observe(iteration, problem, diagnostics)` synchronously. Retaining the
live model reference does not retain a checkpoint; saved state needs detached
copies. The external-step design above makes these calls directly and therefore
does not need an observation callback.

An optimization loop chooses when its state is meaningful to observe. A call
from that loop to an evaluation or recording function does not transfer ownership
of the called function's calculations to the optimizer. Required loss evaluations,
acceptance decisions and execution counters remain with PDAP/SSN; optional quality
evaluation and presentation can run synchronously at the chosen observation point.
The observer can inspect the current model before optimization continues; it does
not need to wait for `fit()` to return.

Established libraries use several arrangements:

- [PyTorch's optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation)
  calls training and evaluation functions directly from an outer loop.
- [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
  accepts an iteration callback that observes intermediate solver results.
- [Ignite's example](https://github.com/pytorch/ignite#simplified-training-and-validation-loop)
  uses an epoch-completion event handler to run evaluation and display results.
- [Keras History](https://github.com/keras-team/keras/blob/master/keras/src/callbacks/history.py)
  appends supplied metric values. Its [PyTorch trainer](https://github.com/keras-team/keras/blob/master/keras/src/backend/torch/trainer.py)
  performs validation before invoking epoch-end callbacks. Keras also explicitly
  supports callbacks that [inspect model predictions](https://keras.io/guides/writing_your_own_callbacks/#usage-of-selfmodel-attribute).

These support separating calculation from retention, but not a universal rule
that a recorder may never initiate evaluation. A single reporting operation may
coordinate both. Independent evaluation functions are useful for saved-model
scoring without appending a training-history row.

| Integration | Assessment for this repository |
|---|---|
| `PDAP.step` with an external training loop | Preferred starting point for the fresh design: explicit inputs/outputs, ordinary PyTorch control flow, no callback contract. The runner respects PDAP's stop result. |
| Keep `History.record` delegating to evaluation | Valid existing coordination, but preserving it is not a requirement. |
| Pass one synchronous observer callable into PDAP | Alternative when PDAP should retain the whole repetition loop. Run setup binds samples, normalizer and History; PDAP passes the current model and solver facts. No new evaluator class is required. |
| Compute metrics in PDAP, then record | Valid, but merely moves coordination into both outer loops. It is not inherently a better seam. |
| Expose a generator or adopt an engine framework | Changes execution/lifecycle contracts beyond the current need. No such change is proposed. |

The current [PDAP observation](../src/PDAP/pdap.py) follows correction and pruning.
In contrast, a native [PyTorch post-step hook](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.register_step_post_hook.html)
fires after the whole SSN correction but before PDAP copies its temporary
coefficients into `problem.c` ([PDAP correction](../src/PDAP/pdap.py)). Evaluating
the persistent model there would observe stale state. That hook can inspect
correction counters; inner Newton traces require an inner observation point.

Recorded checkpoints are also distinct from executed operations: correction-first
has a final insertion after its last record; insertion-first can reject a terminal
search before recording; an empty terminal result can have one record and no SSN
correction. Actual work totals must come from executed operations, including final
work outside checkpoint rows. The accepted SSN objective precedes pruning and
cannot blindly replace the recorded post-pruning training objective.

## Compatible choices

[pyproject.toml:5](../pyproject.toml) declares Python >=3.12, torch >=2.6,
hydra-core >=1.3, and MLflow >=3.15. Read-only `.venv` metadata inspection found
Python 3.14.5, torch 2.10.0, Hydra 1.3.2, and MLflow 3.15.1. TorchMetrics,
TorchEval, TensorBoard, Ignite, and Lightning are not installed. The current
[runtime contract](../src/config/schema.py) is CPU/float64 (`EnvConfig`, line 183).

| Choice | Assessment for this project |
|---|---|
| Native torch + small project functions | Preferred starting point if current reductions are the whole requirement. Exact H1, sparsity, coordinate transforms, and solver diagnostics already belong to this project; retain their explicit meaning. |
| Standalone TorchMetrics | Provides `update`/`compute`/`reset` and distributed state reduction without Lightning. Useful when a common accumulator lifecycle would replace repeated code. H1 and domain diagnostics still require project definitions. [v1.9.0 overview](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/docs/source/pages/overview.rst) |
| Existing MLflow | Tracking destination, not the metric calculator. Its manual API accepts scalar dictionaries and integer steps, including optional asynchronous publication. [API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_metrics) |
| TensorBoard | Optional visualization sink: PyTorch's SummaryWriter writes event files asynchronously, and TensorBoard is installed separately. Add only for a requested visualization capability beyond the existing dashboard. [PyTorch 2.10](https://docs.pytorch.org/docs/2.10/tensorboard.html) |
| TorchEval | Similar streaming metric API; distributed `sync_and_compute` is explicit. PyPI currently offers 0.0.7 (2023-08-24) and describes the library as alpha. Its older release provides less assurance for this Python 3.14 runtime; age alone does not prove incompatibility or abandonment. [Maintainer package page](https://pypi.org/project/torcheval/) |
| Ignite / Lightning | Broader loop frameworks: Ignite uses an Engine and events; Lightning manual optimization remains within Lightning's lifecycle. These could fit a future training-framework decision, but metric recording alone does not justify migrating PDAP/SSN into them. [Ignite Engine](https://docs.pytorch.org/ignite/generated/ignite.engine.engine.Engine.html), [Lightning manual optimization](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.0/docs/source-pytorch/model/manual_optimization.rst) |

TorchMetrics 1.9.0 is the latest published release shown by
[PyPI](https://pypi.org/project/torchmetrics/), dated 2026-03-09, requiring Python
>=3.10. Its base requirements include torch >=2.0, NumPy, packaging, and
lightning-utilities. The source's torch <2.9 bound is marked for CI stability;
normal packaging removes non-strict upper bounds. This is not an installation
prohibition on torch 2.10 and is not evidence that our exact environment works.
No resolver, import, or integration check with TorchMetrics was run.
([Requirements](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/requirements/base.txt),
[packaging code](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/setup.py))

Hydra need only configure the chosen measurement policy at application setup;
no special Hydra–metric adapter is needed for ordinary Python objects/functions.
Hydra already configures Python logging through `dictConfig`, supporting separate
job and Hydra logging settings. It does not determine metric mathematics.
([Hydra 1.3 logging](https://hydra.cc/docs/1.3/configure_hydra/logging/))

## Numerical and state contracts, regardless of library

The following follows directly from [src/eval.py:60](../src/eval.py) and
[SparseProblem.obj:205](../src/problems/sparse_problem.py):

1. Reuse `problem.obj(prediction, target)` and add the penalty once. Fitting loss
   divides by `2N`; gradient coordinates are summed, not averaged over `N*d`.
2. Preserve four full-split sums: value error squared `E_v`, gradient error squared `E_g`,
   target value squared `T_v`, and target gradient squared `T_g`. Final relative
   errors are `sqrt(E_v/clamp(T_v))`, `sqrt(E_g/clamp(T_g))`, and
   `sqrt((E_v+E_g)/clamp(T_v+T_g))`, with the existing `1e-30` clamp.
   Averaging batch-relative errors is different mathematics. For example,
   two one-sample targets `[1, 10]` and predictions `[2, 10]` give a mean batch
   relative error of `0.5`, but the global relative L2 error is `1/sqrt(101)`.
3. Keep loss weights/training coordinates separate from reporting transforms.
   Current fitting loss uses training coordinates; relative errors can use physical
   coordinates. Preserve the unweighted reported H1 convention, metric names,
   and empty-region NaNs.
4. Region sums need region counts and their specified global reference sums;
   distance-bin edges must be common to the complete evaluation, not recomputed
   independently per batch. Support radii and total variation are observations
   of one model support, not numbers to average over the sample loader.

Preserve `no_grad` around reporting, not a blanket `inference_mode` conversion.
[SparseProblem.forward:137](../src/problems/sparse_problem.py) temporarily enables
autograd to obtain input derivatives, then detaches outputs when reporting.
Inference mode imposes additional restrictions on autograd and tensors created
inside it. Our evaluator needs input gradients even without parameter-gradient
recording; this is a project-specific incompatibility with a routine inference-only
recipe. ([PyTorch inference mode](https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html))

## Records and measuring performance

[ADR 0002:9](../docs/adr/0002-mlflow-as-run-record-backend.md) makes local JSON and
artifacts authoritative, MLflow optional, and per-iteration curves local. Failed
training runs currently retain logs without a Run Record. [ADR 0014](../docs/adr/0014-retire-the-ec2-mlflow-target.md)
places the projection in a local container. [ExperimentRun:40](../src/experiment_logging.py)
collects events in memory and writes JSON before publishing at completion.
Live dashboard curves or crash-surviving partial metrics would require an explicit
lifecycle/durability decision; neither follows from choosing TorchMetrics.
Current `elapsed_s` starts before data loading and ends before JSON publication;
it is excluded from MLflow summary extraction. A solver-only duration therefore
needs its own meaning and value. ([Runner:214](../scripts/train.py),
[record timing:82 and projection:178](../src/experiment_logging.py))

Manual MLflow tracking is the direct fit. Full PyTorch autologging targets
Lightning; vanilla `nn.Module` autologging forwards TensorBoard `add_scalar` and
`add_hparams` calls. It does not discover PDAP iterations or derive H1 errors.
([Official API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html#mlflow.pytorch.autolog))

Use separate measures for elapsed run time, phase cost, and diagnostic profiling.
For CPU phase durations, `time.perf_counter()` is a monotonic high-resolution
timer; name whether a duration includes data loading, evaluation, and persistence.
([Python time](https://docs.python.org/3/library/time.html#time.perf_counter))
For repeatable operation comparisons, `torch.utils.benchmark.Timer` includes
warmup, controlled thread counts, replicates, and accelerator synchronization.
Future manual CUDA wall timing needs synchronization at the timed boundaries.
([Benchmark API](https://docs.pytorch.org/docs/2.10/benchmark_utils.html),
[CUDA synchronization](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html))
Use `torch.profiler` with a limited schedule for operator/memory investigations;
shape/stack recording adds overhead, so profiled time should be identified
separately from ordinary run time. Tensor allocation profiling is not a whole-process
peak-RSS measurement. No profiling or performance comparison was performed here.
([Profiler](https://docs.pytorch.org/docs/2.10/profiler.html))

## Interview status

- **Q1 accepted:** include prediction quality, solver diagnostics, and runtime
  costs, with a small codebase.
- **Q2 accepted:** keep native PyTorch for now; add no metric-framework dependency.
  TorchMetrics remains a future option if state-management needs justify it.
- **Q3 resolved:** preserve current full-split evaluation and arithmetic. The
  user rejected adding streaming and emphasized avoiding overengineering.
- **Q4 accepted:** one complete `PDAP.step` plus a small external training loop
  owning repetition, timing, evaluation and recording. PDAP owns algorithmic stop
  decisions. The user permits redesign from scratch and prioritizes PyTorch
  practice, explicit inputs/outputs and low complexity. An observer callback is
  unnecessary for this design; the earlier universal requirement for passive
  History is withdrawn. Implementation is authorized; historical algorithm removal
  is not part of the task. The correction-first schedule remains an explicit
  compatibility branch in the external runner using PDAP-owned operations.
- **Native optimizer inheritance resolved:** the user accepted a plain PDAP class
  composing SSN and instructed implementation to proceed. Inheritance is feasible
  but provides no established simplification for the outer structural algorithm;
  see the contract assessment above.
- **Q5 accepted:** preserve the historical correction-first order alongside the
  complete insertion-first paper step.
- **Q6 accepted:** report actual SSN iterations, failed trials and correction
  acceptance, solver/evaluation wall time, and process lifetime peak resident
  memory. Detailed inner profiling remains separate.
- The user confirmed both remaining choices after reviewing the consolidated
  plan. The grilling session is complete. Keep implementation concise and report
  uncertainty, especially when additions exceed removals.

## Implementation and verification

The implementation adds `src/training.py` and removes the training/reporting
loops from PDAP. History records supplied values; existing full-split metric
formulas and numerical fixtures are retained. A zero-budget historical run now
has a usable checkpoint, and a failed historical initialization reports an
existing model's actual neuron count. Ordinary checkpoint positions and selection
are preserved.

The metrics-only production change removes **41 lines overall**. This count
excludes concurrent logging and coefficient-preparation changes. Thirteen focused
training tests cover the new integration, including a real SSN correction with a
rejected trial. Line count is a size check; the design benefit is explicit
ownership with ordinary functions and no additional metric framework or classes.

Independent standards and specification reviews passed. Before/after comparisons
matched all recorded metrics, checkpoints, final model tensors, selected indices
and RNG state for both insertion methods and both loop orders. Subsequent
concurrent solver/logging changes were checked for compatibility and preserved.

Final current-workspace verification: **515 passed, one preexisting stale
controller-selection artifact failure**, 19 warnings, in 43.49 seconds. All 115
source/configuration/fixture/dependency files hashed before this run were unchanged
at completion. Ruff and whitespace checks pass. Evidence is in
`/tmp/sparse-metrics-current-pytest.log`,
`/tmp/sparse-metrics-current-before-hashes.json`,
`/tmp/sparse-metrics-spec-comparison.json` and
`/tmp/sparse-metrics-spec-ssn.json`. The deleted handoff file was not restored.
