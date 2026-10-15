# Data, measurement and logging

Research checked on 2026-10-08 against `491af6a`, installed packages, and the
primary sources linked below. This records investigation and the current grilling
discussion. The user's existing `refactor.md` changes are untouched.

**Accepted dependency:** structlog integrated with Python standard logging,
with Hydra managing configuration. The user confirmed this choice and limited
this session to logging. Work from the desired behavior in small implementation
increments; the current implementation should not dictate the new design.
Data loading, metric computation, History/checkpoint redesign, experiment
tracking, and backend deployment are outside this session's scope.

**Accepted first increment:** one shared logging setup and a standalone example
that emits events through both structlog and standard logging. The user approved
readable stderr output and JSON lines in an explicitly supplied file, both at
`INFO` and above by default. Expose `configure_logging(log_file=..., level="INFO")`
and use structlog directly at call sites. Verify levels, structured fields,
exceptions, and exactly one emission per destination. Hydra integration and
automatic run context are completed in the later increments below.

**Accepted second increment:** migrate training to the shared setup, preserve
`env.verbose`, `env.log_level`, and `env.log_file`, and retire
`src/logging_config.py`. The default file becomes `run.jsonl` in Hydra's per-run
output directory. Hydra supplies settings; `hydra/job_logging: none` leaves
application handler configuration to `src/logging.py`. Run context and the
runner/training event conversion followed in the third increment.

**Latest scope clarification:** implement local logging only when the design is
agreed: one shared configuration, concise console output, and structured per-run
files under the existing run directories. Future backend compatibility means
retaining structured output and configuring destinations at startup. Prometheus,
Grafana, Loki, Alloy, exporters, and backend abstractions are outside the current
scope. The deployment discussion below is reference material for later work.

## First logging increment

Implemented against `e8d02e3`, with structlog `26.1.0` added to the project and
lockfile. [The setup](../src/logging.py) exposes one function and uses structlog's
built-in formatters; there is no custom logger class or backend interface.

```python
import structlog
from src.logging import configure_logging

configure_logging(log_file="logs/demo.jsonl")
structlog.get_logger(__name__).info("experiment_started", run_id="demo")
```

Each JSON line contains `timestamp` (UTC), `level`, `logger`, `event`, and supplied
fields. Ordinary standard-library logging uses the same outputs and preserves
`extra` fields. Exceptions retain tracebacks in both outputs. Reconfiguration
closes and replaces root handlers; files use append mode.

Run [the standalone demo](../scripts/logging_demo.py) with
`uv run --frozen python -m scripts.logging_demo --log-file /tmp/demo.jsonl`.
The demo includes an intentional caught exception to show traceback formatting;
add `--level DEBUG` to see the otherwise filtered debug event.

Verification: **23 tests passed** across `tests/test_logging.py` and the existing
`tests/test_experiment_logging.py`; Ruff and whitespace checks passed. An
independent review found no material findings and reran the three new tests.
Separate INFO/DEBUG demo checks confirmed event counts, fields, and exceptions.
Tests use subprocesses to isolate global logging configuration.

This established the standalone foundation, subsequently integrated below.

## Hydra integration and old setup retirement

[Training](../scripts/train.py) now calls the shared setup using Hydra's composed
settings and per-run output directory. The obsolete `src/logging_config.py` and
its `get_logger` wrapper have been removed. Existing standard-library log calls
flow through the new formatters without a call-site migration.

| Setting | Behavior |
|---|---|
| `env.verbose=true` (default) | Readable application logs on stderr |
| `env.verbose=false` | Suppress application console output; retain JSONL file logs |
| `env.log_level=INFO` (default) | Shared minimum level for console and file |
| `env.log_file=null` (default) | Write `run.jsonl` in Hydra's output directory |
| `env.log_file=/path/events.jsonl` | Use that file instead of the default |

[The top-level config](../conf/config.yaml) selects `hydra/job_logging: none`,
so Hydra does not install a second set of job handlers or create `train.log`.
Hydra's own launcher messages remain framework-controlled. The shared Python
setup accepts `console=False` for direct callers as well.

Verification: **69 tests passed** across `tests/test_logging.py`,
`tests/test_experiment_logging.py`, and `tests/test_config.py`; Ruff and whitespace
checks passed. Three real Hydra CLI cases cover default output, a quiet console
with a custom file, and level filtering. A separate two-worker joblib sweep
confirmed isolated logs for both jobs, one copy of each application event, and
no redundant job log files. Independent review found no material findings and
reran the 26 logging/experiment tests successfully. In-process runner tests
restore pytest's logging handlers and verify capture still works afterward.

## Run context, lifecycle, and progress cadence

The third increment adds built-in structlog context variables to both structlog
and standard-library events. The runner generates one `run_id` for both logs
and the Run Record, and scopes `run_id`, `experiment`, and `seed` to the job.
Caller context is restored on both success and failure, including when a Hydra
worker executes another job afterward.

Application events now have stable names and separate fields: `run_started`,
`data_loaded`, `training_progress`, `training_finished`, `evaluation_summary`,
optional `region_summary`, and `run_completed` with the Run Record path.
An execution or tracking-preflight exception produces `run_failed` with its
traceback and is re-raised. This adds no failed Run Record. Hydra still reports
the original exception; with console enabled, its traceback may appear alongside
the application traceback. Empty-region H1 values use JSON `null` in logs while
persisted metric values retain their existing representation.

`training.display_every` now controls progress output at recorded checkpoint
indices `0, N, 2N, ...`; its default is `2`. Values at or below zero suppress
periodic progress, while `training_finished` still reports termination. Every
checkpoint is still evaluated and saved. SSN correction details now use `DEBUG`, alongside
existing candidate and backtracking diagnostics.

For Hydra runs, `training.display_every=10` reduces progress frequency;
`training.display_every=0` keeps lifecycle and summary messages. Use
`env.log_level=DEBUG` to include solver details, or `env.verbose=false` to keep
application logs in the JSONL file only.

Verification: **171 distinct tests passed** across the regression and targeted
runs (169 initially, then both new empty-region cases and all affected cases in
a 31-test rerun). The regression selection included logging, configuration,
training, History, PDAP, and SSN. Ruff and whitespace checks passed. Independent
review found no material findings and verified 37 focused tests plus the two
region regressions. A four-job/two-worker Hydra sweep verified distinct run
contexts, quiet console output, and preserved terminal summaries/checkpoints
with periodic output disabled. Compared with the working snapshot before this
increment, production code added 21 net lines; the numerical operations and
recording order were preserved.

## Follow-up 2026-10-13: logging contract and primary references

The recorded choice was **structlog integrated with Python logging**; Loguru
was considered as an alternative. The implementation in [src/logging.py](../src/logging.py)
uses `ProcessorFormatter` with `JSONRenderer` for the file and `ConsoleRenderer`
for stderr. [The runner](../scripts/train.py) initializes these handlers for each
job and binds `run_id`, `experiment`, and `seed` around the run. This confirms
structured logging was an accepted goal; it does not establish that Hydra is
incapable of providing it.

Hydra 1.3 configures ordinary Python logging and its default output is plain
text. Its documentation explicitly supports opting out with `none` and replacing
the job logging configuration with a custom `dictConfig` configuration. Therefore,
`hydra/job_logging: none` delegates application handler setup to our Python entry
point; it does not itself create structured output. A custom Hydra configuration
using a JSON formatter is also possible. Our setup replaces root handlers, so
re-enabling Hydra's default would create an extra job log file before the shared
setup replaces those handlers.
([Hydra logging](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/logging/),
[customization](https://hydra.cc/docs/1.3/configure_hydra/logging/),
[current config](../conf/config.yaml))

Primary guidance and its application to this repository:

- structlog documents `ProcessorFormatter` for formatting both standard logging
  and structlog events, including different renderers for different destinations.
  This directly supports our shared formatting approach.
  ([Standard library integration](https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging))
- structlog recommends structured output for aggregation and readable output
  during development. Its deployment guidance favors stdout with external
  collection. Keeping per-run JSONL files for this local batch workflow is our
  recommendation; a future container deployment could instead emit JSON to
  stdout. The current console emits readable text, so collecting it alone would
  not collect the JSON representation.
  ([Logging best practices](https://www.structlog.org/en/stable/logging-best-practices.html))
- Execution context can carry shared identifiers into each event. Our scoped
  run context follows that pattern.
  ([Context variables](https://www.structlog.org/en/stable/contextvars.html))
- Python supports multiple handlers and formats, but its ordinary file handlers
  do not coordinate writes across processes. Keep separate files for concurrent
  sweep jobs; a shared `env.log_file` would need a different writing arrangement.
  ([Logging cookbook](https://docs.python.org/3.14/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes))

Assessment: retain the existing setup for this repository. These sources support
its design principles; they do not prescribe this exact configuration as a
universal best practice. The older deployment discussion below is historical;
the implemented state is described above. Backend-specific findings are in the
dated follow-up at the end of this note.

## Observed baseline

| Component | Project requirement | Lock and installed version |
|---|---|---|
| Python | `>=3.12` | Runtime: `3.14.5` |
| PyTorch | `>=2.6` | `2.10.0` |
| Hydra | `>=1.3` | `1.3.2` |
| OmegaConf | Transitive | `2.3.0` |
| MLflow | `>=3.15` | `3.15.1` |
| TorchMetrics, TensorBoard, Lightning, Ignite | Absent | Not locked or installed |

Versions came from [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock),
and `.venv` package metadata at the initial investigation, before the accepted
logging implementation. The active numerical contract remains CPU/float64.

[ADR 0002](../docs/adr/0002-mlflow-as-run-record-backend.md) already chooses
MLflow as an optional comparison dashboard, with local JSON Run Records and
Run Artifacts authoritative. It explicitly keeps per-iteration curves local
and omits failed training runs from Run Records.
[ADR 0014](../docs/adr/0014-retire-the-ec2-mlflow-target.md) moves this projection
to a local container. Choosing live tracking, failed-run records, or MLflow as
authoritative storage would reopen those decisions.

Current [evaluation](../src/eval.py) owns reported errors and diagnostics and
calls `SparseProblem.obj` and `sparsity_functional`. However,
[History.record](../src/PDAP/history.py) also initiates evaluation, copies the
checkpoint, appends values, and selects the minimum training loss. Each split
is materialized through [full_batch](../src/data.py); using DataLoader has not
made evaluation streaming. [The runner](../scripts/train.py) submits only final
summary metrics to [ExperimentRun](../src/experiment_logging.py), which writes
JSON before publishing to MLflow. `env.require_mlflow` optionally requires a
reachable server before training; the default is false.

## Ownership assessment

The existing separation is sound if “measurement” means reporting evaluation,
not every mathematical quantity that can be printed. The current decision in
[ADR 0004](../docs/adr/0004-model-trainer-eval-separation.md) supports this split:

| Responsibility | Proposed owner |
|---|---|
| Sample validation, preprocessing, splits and delivery | Data utilities and PyTorch Dataset/DataLoader |
| Prediction, Empirical Fidelity, derivatives, sparsity functional | SparseProblem and mathematical helpers |
| Candidate decisions, correction acceptance, stopping and iteration positions | PDAP/SSN |
| Split traversal, physical-coordinate reporting, error reductions, support diagnostics | Evaluation |
| Measurement timing and optional reporting cadence | Training orchestration, configured by Hydra |
| Recorded values, detached snapshots, existing best-model policy | History |
| Run identity, persistence, backend publication | ExperimentRun / runner |
| Human-readable progress, warnings and debugging | Python logging, configured once at the entry point |

Theorem-Derived Search Radius inputs, candidate derivatives, and objectives used
by the Correction Acceptance Guard are required algorithm calculations. Turning
off monitoring must not disable them or alter optimization. Logging consumes
their reported results. Conversely, timing and experiment transport do not
belong in SparseProblem or DataLoader.

Making evaluation explicit before `History.record`, then sharing its values
with history, progress and tracking, would clarify the remaining boundary. This
is a proposal, not a requirement for a new callback framework or record class.
History already clones a checkpoint once and exposes legacy artifact views;
the old duplicated-checkpoint complaint is not a current defect.

## Framework recommendation

**Provisional recommendation: retain PyTorch calculations, Python logging with
Hydra configuration, and manual MLflow tracking through the existing run-record
boundary. Add a metric library only for a concrete aggregation requirement.**

| Candidate | Fit and decision |
|---|---|
| Python logging + Hydra | Already available. Hydra configures standard logging with `dictConfig`, including separate job/Hydra configurations. Prefer one configuration owner for Hydra jobs; direct scripts/notebooks can retain a separate entry-point setup. [Hydra 1.3 logging](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/logging/), [customization](https://hydra.cc/docs/1.3/configure_hydra/logging/) |
| MLflow manual tracking | Already installed and used. `log_metrics(values, step=...)` accepts explicitly computed scalars without adopting a training framework. Installed 3.15.1 signatures also expose asynchronous logging and flushing. [Tracking API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_metrics) |
| TorchMetrics standalone | Optional accumulator/state-management library, independent of Lightning. Useful for reusable metric collections or distributed reduction; does not define this project's H1 mathematics. [Official overview source, v1.9.0](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/docs/source/pages/overview.rst), [custom metrics](https://lightning.ai/docs/torchmetrics/stable/user-guide/implement) |
| TensorBoard via PyTorch | Optional alternative visualization sink for live scalar curves, figures and histograms. `SummaryWriter` writes event files asynchronously and needs a separate TensorBoard installation. It does not calculate metrics. Add only for a UI capability the user actually wants. [PyTorch 2.10 integration](https://docs.pytorch.org/docs/2.10/tensorboard.html) |
| Ignite / Lightning | Both can accommodate custom work, but introduce loop/lifecycle machinery. Ignite's Engine normally processes data batches and emits events; Lightning manual optimization still integrates with its Trainer. PDAP outer iterations and complete SSN corrections are not minibatch optimizer steps. Adopting either solely for logging is disproportionate. [Ignite Engine](https://docs.pytorch.org/ignite/generated/ignite.engine.engine.Engine.html), [Lightning manual optimization source](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.0/docs/source-pytorch/model/manual_optimization.rst) |

The single-owner logging recommendation addressed an initial conflict:
the original `src/logging_config.py` cleared root handlers after Hydra had
configured them. Hydra supports either custom `hydra/job_logging` or opting out
of job-handler configuration. The accepted second increment uses the latter;
[train.py](../scripts/train.py) passes composed settings to the shared setup.

MLflow PyTorch autologging is not a substitute for measurement. Full autologging
targets Lightning; vanilla PyTorch support forwards TensorBoard scalar and
hyperparameter calls. It cannot discover our H1 errors or PDAP diagnostics.
Manual tracking is the direct fit. This is stated by the
[official API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html#mlflow.pytorch.autolog)
and confirmed in the installed 3.15.1 source.

TorchMetrics 1.9.0 declares Python `>=3.10` and a PyTorch minimum of `2.0`.
Its requirements file's `torch<2.9` bound is documented as CI stability only;
packaging removes non-strict upper bounds. Thus it is not a declared prohibition
on torch 2.10. It is also not proof that Python 3.14.5 / torch 2.10.0 is tested
upstream or works here. An adoption would require a focused compatibility check.
([Requirements](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/requirements/base.txt),
[packaging](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/setup.py))

## Correctness constraints independent of library

For batches of sizes `n_b`, use `sum(n_b * problem.obj(batch)) / sum(n_b)`
and add regularization once. For relative H1 error, accumulate value/gradient
squared-error sums and reference sums, then take the final ratio and square
root. Never average batch-relative errors or equally weight uneven batch means.
Retain the existing denominator clamp, physical-coordinate transformation,
metric names, and division by samples rather than gradient components.
These follow from [the local formulas](../src/eval.py) and
[Empirical Fidelity](../CONTEXT.md); standard
[MSELoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.MSELoss.html)
instead averages all elements by default.

TorchMetrics can represent the requisite sums with `add_state(...,
dist_reduce_fx="sum")`; `MeanMetric` accepts weights, but does not turn a mean
of ratios into a global norm ratio. Keep train/validation accumulators separate,
reset between evaluations, and explicitly retain float64 state. In v1.9.0,
`set_dtype(torch.float64)` is the supported conversion; `double()` is overridden
to do nothing. These are library mechanics, not reasons to move reporting state
onto SparseProblem. ([Custom metrics](https://lightning.ai/docs/torchmetrics/stable/user-guide/implement),
[MeanMetric source](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/src/torchmetrics/aggregation.py),
[Metric source](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/src/torchmetrics/metric.py))

Distributed metric reduction would need summed numerators, denominators and
counts across processes. Correct reduction alone does not correct duplicate
samples: PyTorch's DistributedSampler pads when `drop_last=False` and the split
is not evenly divisible. Exact evaluation needs appropriate sample coverage.
This is a future consideration; current execution is CPU, and DataLoader
workers are not distributed model training.
([PyTorch 2.10 sampler source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/data/distributed.py))

H1 evaluation still needs input derivatives. Current
[forward](../src/problems/sparse_problem.py) obtains the input gradient with
`torch.func.vjp`, which evaluates regardless of the outer grad mode, so
reporting `no_grad` returns untracked outputs while training differentiates
through both outputs. Preserve this behavior. The earlier implementation
nested `enable_grad` and `autograd.grad`, which failed inside `inference_mode`;
a 2026-10-08 probe of the current forward evaluated inside `inference_mode`.
([torch.func.vjp](https://docs.pytorch.org/docs/2.10/generated/torch.func.vjp.html),
[inference mode](https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html))

## User steering and concrete proposals

The user requested live information without excessive output or codebase
complexity, and confirmed that local records remain authoritative. Live
information may be terminal progress; whether it includes live dashboard curves
is still open. The following are proposals, not accepted implementation choices.

**A. Concise progress logs — implemented above.**
The shared setup configures Python logging from Hydra settings. The runner and
external training loop use structured events, while existing lower-level modules
continue using standard-library logging. The third increment wires
`training.display_every` to presentation only, retains its default value 2, and
always reports normal termination even between scheduled progress messages.
Removing remaining per-function `verbose` parameters is a separate possible
cleanup, not necessary for the current logging behavior.

Use INFO for run lifecycle and progress; DEBUG for SSN correction details,
candidate statistics and backtracking; WARNING for unexpected recoverable
conditions. Ordinary rejected trials, exhausted configured iteration budgets,
and stopping because no candidate was accepted are not inherently warnings.
Keep the per-run log file at INFO by default, with targeted DEBUG available
when needed. Severity selects detail; cadence controls volume. This follows
the [Python logging level guidance](https://docs.python.org/3.14/howto/logging.html).

Make measurement explicit at the current observation points: evaluate once,
pass the resulting dictionary to `History.record(problem, values, ...)`, then
use those same values for progress and any selected tracking. History can
continue to snapshot once and apply the existing best-checkpoint rule. Keep
normalization metadata needed by saved-run readers. Record existing SSN
counters/outcomes before the temporary optimizer is discarded; do not repeat
objective evaluations just to log them. This uses ordinary functions and
dictionaries without adding an event bus, callback hierarchy, or logger class.

**B. One updating terminal status line.**
If scrolling progress lines are still too much, use `tqdm` for terminal progress
over the same measured values, retaining standard logging for durable events.
Its `mininterval` controls refresh rate, `set_postfix(..., refresh=False)` updates
displayed measurements without forcing a redraw, and `disable=None` disables
the bar on non-interactive output. Its documented logging integration preserves
file logging while routing console messages around the bar. Disable individual
worker bars during parallel Hydra sweeps as well. This is a small explicit new
dependency: neither tqdm nor Rich is installed or locked in the observed repo.
([tqdm reference](https://tqdm.github.io/docs/tqdm/),
[logging integration](https://tqdm.github.io/docs/contrib.logging/))

[Rich Progress](https://rich.readthedocs.io/en/stable/progress.html) is an
alternative when multiple tasks or custom display columns are needed. A single
PDAP status line does not yet justify that additional presentation scope.

Both proposals keep mathematical evaluation, checkpoint selection, terminal
presentation, and optional backend publication distinct. Standard log levels
do not control whether research measurements are computed or saved.

## Remaining design decisions

1. **Live destination:** select concise terminal lines or an updating status
   line; separately decide whether MLflow curves must update during training.
   Current MLflow publication happens at finish. Live tracking needs a run
   lifecycle and explicit step/phase semantics; it changes ADR 0002.
2. **Failure durability, if live tracking is chosen:** local authority is now
   confirmed. If live curves must survive interruption, an in-memory event list
   followed by end-of-run JSON is insufficient. Failed-run records and
   publication-failure behavior need an explicit decision. Existing artifact
   paths are tags, not uploaded backups.
3. **Cadence:** separate expensive evaluation from publication frequency.
   Throttling checkpoint-selection evaluation can change the returned model:
   current best selection uses minimum training loss, `best_iteration` is a
   history-row index, and `iterations` counts rows. Preserve that policy unless
   deliberately changed; store actual outer-step identity if records become
   sparse. SSN trial values must not be confused with accepted model snapshots.

If live publication is selected, batch scalar metrics per event; do not send
network calls from every objective evaluation or DataLoader worker. Keep one
writer per run, close runs on success/error, and drain any async work at the
lifecycle boundary. MLflow's fluent API is not thread-safe; explicit run IDs or
its client API are available when concurrency requires them. TensorBoard likewise
needs an explicit run directory and writer closing. ([MLflow API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html),
[SummaryWriter](https://docs.pytorch.org/docs/2.10/tensorboard.html))

## Follow-up: centralized logs and deployment

**Confirmed requirements:** centrally configured logging across experiments and
a possible future service; concise application code and easy management; future
backend compatibility. The user clarified that no Prometheus/Grafana or other
backend infrastructure is needed now. Recommendations below remain pending
acceptance; local records and files are sufficient for this increment.

**Recommendation:** add structlog over the existing standard logging/Hydra
configuration, with one initialization point. `ProcessorFormatter` can format
application and third-party standard logs consistently as JSON. Application
calls remain `log.info("correction_completed", iteration=i, duration_s=dt)`.
The initial configuration is more involved than Loguru's; the benefit here is
retaining the existing logging ecosystem with one event schema.
([structlog integration](https://www.structlog.org/en/stable/standard-library.html))

Loguru is a credible simpler-setup alternative: `serialize=True` provides JSON,
and file rotation, retention and queued sinks are built in. Capturing Hydra and
third-party standard logging requires its documented interception bridge.
Both libraries support bound fields and context-local metadata; neither is
required by Grafana. Plain standard logging with a JSON formatter also works,
but requires more explicit structured-context plumbing at call sites.
([Loguru overview](https://loguru.readthedocs.io/en/stable/overview.html),
[contextualize](https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.contextualize),
[structlog context](https://www.structlog.org/en/stable/contextvars.html))

If deployment is requested later, these two independent paths are available;
neither is part of the current implementation proposal:

- Logs: per-run JSONL locally, or container stdout when deployed → Grafana Alloy
  → Loki → Grafana. Alloy can read files or container logs; collection does not
  require application code to call Loki. Use Alloy for new work: Promtail reached
  EOL on March 2, 2026.
  ([File collection](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.source.file/),
  [Docker collection](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.source.docker/),
  [Promtail status](https://grafana.com/docs/loki/latest/send-data/promtail/))
- Operational numbers: `prometheus_client` counters/gauges/histograms → HTTP
  `/metrics` → Prometheus scraping → Grafana. Logging an error count as JSON does
  not itself expose a Prometheus metric. Keep exact per-run research curves in
  the existing history/Run Records/MLflow boundary.
  ([Python client](https://prometheus.github.io/client_python/exporting/http/),
  [Prometheus architecture](https://prometheus.io/docs/introduction/overview/))

Bind `run_id`/experiment once per run; reset context between runs and initialize
each worker deliberately. Use bounded deployment labels such as service and
environment. Keep unbounded run IDs in log fields or Loki structured metadata,
not indexed Loki labels or Prometheus labels.
([Loki cardinality](https://grafana.com/docs/loki/latest/get-started/labels/cardinality/),
[Prometheus labels](https://prometheus.io/docs/practices/naming/))

Local Run Records and artifacts remain authoritative under the existing ADR;
centralized diagnostic logs do not replace them. `prometheus-client==0.24.1` is
already installed transitively; direct application use should declare it.
structlog and Loguru are absent. No collector/backend deployment or new package
integration has been tested.

This investigation changed no implementation, tests, dependencies, ADRs, or
handoff. The structlog dependency choice is accepted; implementation details
remain under discussion. Earlier framework comparisons are research history.

## 2026-10-13 — Prometheus and Loki/Alloy backend verification

**Source facts (official documentation):**

- **Collection:** Alloy `loki.source.file` tails newline-terminated entries,
  discovers files, and persists read offsets. `loki.process` extracts JSON
  fields and timestamps before forwarding to `loki.write`; extraction and
  indexed-label promotion are separate. File collection adds the full path as
  a `filename` label.
  ([Files](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.source.file/),
  [Processing](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.process/))
- **Search:** LogQL parses JSON at query time, enabling searches without
  indexing `run_id`. Illustrative query, assuming these labels/fields:
  `{service_name="sparse-nn-hjb"} | json | __error__="" | run_id="example-run"`.
  ([JSON queries](https://grafana.com/docs/loki/latest/query/log_queries/))
- **Cardinality:** Loki recommends stable origin labels; distinct combinations
  create streams. Prometheus similarly creates series per label combination
  and discourages unbounded values. Loki structured metadata supports searching
  unindexed fields without JSON parsing; it requires schema ≥13, TSDB, and
  `allow_structured_metadata: true`.
  ([Loki labels](https://grafana.com/docs/loki/latest/get-started/labels/cardinality/),
  [Prometheus labels](https://prometheus.io/docs/practices/naming/#labels),
  [Metadata](https://grafana.com/docs/loki/latest/get-started/labels/structured-metadata/))
- **Metrics:** Prometheus scrapes exposed numerical metrics; JSONL alone does
  not expose them. Loki can derive rates/aggregations through LogQL. Explicit
  Alloy `stage.metrics` configuration can alternatively expose derived metrics
  at Alloy's `/metrics` endpoint.
  ([Prometheus](https://prometheus.io/docs/introduction/overview/),
  [LogQL metrics](https://grafana.com/docs/loki/latest/query/metric_queries/),
  [Alloy metrics](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.process/#stagemetrics))
- **Batch jobs:** Prometheus recommends last-success/completion times and
  durations; jobs lasting several minutes can also be scraped while running.
  Its narrower Pushgateway guidance limits the usual appropriate case to
  service-level batch outcomes independent of particular machines/instances.
  Machine maintenance jobs fit Node Exporter's textfile collector. Pushed
  series require explicit deletion, and gateway health does not establish job
  health. Pushgateway explicitly excludes event-store use and automatic TTL.
  ([Batch instrumentation](https://prometheus.io/docs/practices/instrumentation/#batch-jobs),
  [Pushgateway guidance](https://prometheus.io/docs/practices/pushing/),
  [Project README](https://github.com/prometheus/pushgateway#readme))

**Our recommendation:** retain local JSONL now; consider Alloy → Loki → Grafana
for future diagnostic search. Use bounded `service_name`/`environment` labels;
keep `run_id` and iteration in JSON or optional structured metadata. Audit/remove
per-run `filename` labels too. Reserve Prometheus for operational monitoring;
consider Pushgateway only for a defined recurring service outcome with cleanup
ownership, never as the default for per-experiment curves. Preserve
[ADR 0002](../docs/adr/0002-mlflow-as-run-record-backend.md): local Run Records and
artifacts are authoritative; History curves stay local; MLflow receives
completed-run summaries.

**Caveats:** documentation research only; no deployment or query execution.
Recheck `latest` documentation against selected versions. Promtail reached EOL
on 2026-03-02; choose Alloy for new collection.
([Promtail status](https://grafana.com/docs/loki/latest/send-data/promtail/))
