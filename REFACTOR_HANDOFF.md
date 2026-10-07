# Refactor plan and handoff

Updated: 2026-10-08. Branch: `work/problem-solver-refactor`.
Implementation baseline: `edb166a666bf2dc123198c373d9abc90f52f5830`.
Full-review baseline: `49eeb0349a0a8ee489c075400d23cb1cb31e1208`.

## Current status and working rules

**Inline review implemented and independently reviewed, baseline `32b0878`.** The user's pasted cache
recovers six comments lost during simultaneous editing. It predates the accepted
`inner_weight` / `inner_bias` rename; apply the comments to the current file,
not by restoring the cached source wholesale. Root plans/documents; subagents
implement and independently review.

| Review comment | Resolution / work |
| --- | --- |
| `coefficient_scale` is confusing; suggested `_scaling_innner_weights`. | Rename to `normalization_weight`: it evaluates a factor from the inner parameters, without scaling or mutating those parameters. PDAP also uses this mathematical evaluation, so it remains public. |
| Move `validate_correction` to configuration. | One `src/config.validate_training` function owns the compatibility rules and takes explicit settings. One model entry operation supplies its own settings; construction/preparation and direct training retain their checks before sample access. |
| Move `validate_insertion` to configuration. | Share the configuration rules with insertion compatibility, without duplicating model settings or adding getters. |
| Why a prepared class / is it an inner-weight backup? | The existing three-field record stores the fitting objective, its Hessian and normalization weights for one correction. It contains no network backup or optimization procedure. Retain this plain record, not a second model or behavioral class. Its exact representation is an implementation choice, not a user requirement. |
| Why can the factor be `None`? | Remove this identity sentinel. Every preparation carries a tensor: moment weights for normalized neurons, ones otherwise. Coordinate conversion becomes multiplication/division. |
| Configuration is fixed; why a class or whole-network copy for conversion? | Moment order is configured once; factor values depend on the neurons and change when support changes. The prepared factors match that solve's features. Proposals remain PDAP-owned; SSN's outer-coefficient backup handles correction rollback. No new copy of inner weights is required. |

`state_cleanup` implemented the changes; independent Standards and Spec reviewers
identified private-setting reads in the first config validator as an unresolved
encapsulation violation. Root's assumption that every model validation method
must disappear was not a user requirement and caused that coupling. The correction
keeps one `SparseProblem.validate_training` entry operation, replacing the two
former methods. It supplies its own settings to the pure configuration function;
all compatibility rules live in configuration. PDAP/SSN request this behavior
without reading model internals. No settings copy, getter, flag, adapter class or
new module is needed. Do not optimize method counts at the expense of ownership,
or treat a documented tradeoff as user approval for a standards violation.
Construction, preparation, PDAP and direct SSN retain their checks; ordinary
prediction construction remains broader than training.

Coordinate conversion now always creates independent working coefficients, so
the immediate duplicate clone in PDAP was removed; SSN's rollback copy remains.
Numerical formulas, solver acceptance, masks, derivative conventions and fixtures
remain unchanged. Combined focused checks: **148 passed**; after the validation
review fix, **76 impacted tests passed**. Ruff passes all 15 changed Python files,
and config/core/state/evaluation Pyright reports zero errors. Fixture hashes and
whitespace checks pass. Independent Standards (`inline_review_standards`) and
Spec (`inline_review_spec`) reviews verified the actual correction, closed the
encapsulation finding and found no remaining issue within this diff. The final
full suite reports **428 passed, one preexisting stale controller-selection
artifact failure, 18 warnings** in 28.11 seconds. The ignored artifact and its
test were left unchanged. No new class or module was added; SparseProblem is
21 lines shorter. Logs: `/tmp/sparse-preparation-validation-reviewed-pytest.log`
and `/tmp/sparse-preparation-validation-{focused,ruff,pyright}.log`.
This narrow configuration validation move is requested by the new comments;
broader configuration and logging redesign remains deferred.

**Inner-parameter rename implemented and reviewed, baseline `edc1720`.** The
user accepted the [naming research](vault/parameter_naming.md): `W` becomes
`inner_weight` and `b` becomes `inner_bias` on SparseProblem and its public
arguments. Callers and checkpoint capture/restoration use the new names while
preserving existing saved histories. The optional `c` rename was a separate
proposal and is not part of this change. `state_cleanup` implemented and verified;
root updated documentation. Independent Standards (`inner_names_standards`) and
Spec (`spec_audit`) reviews found no issues in this diff.
No additional interface, numerical, or ownership change is included.

Verification: **425 passed, 1 preexisting stale controller-selection artifact
failure, 18 warnings** in 29.29 seconds; 92 focused tests passed. The three extra
cases exercise the existing restoration test across new names, prior `W/b/c`,
older layer names, and legacy atom fields. Checkpoint independence remains tested.
Ruff passes all seven changed Python files; core/state/evaluation Pyright reports
zero errors; all three numerical fixture hashes are unchanged. Logs:
`/tmp/sparse-inner-names-{focused,final-pytest,ruff,pyright}.log`.

**Naming follow-up implemented and reviewed, baseline `b9fba49`.** During the user's method-by-
method review, they directed removal of `description()`, retention of `forward`,
and the name `forward_with_gradient` for predictions plus input gradients. They
then corrected the proposed `jac_forward` name to `jac_forward_with_gradient`.
The latter takes inputs and current/supplied inner weights and biases; its two
outputs are derivatives of predicted values and input gradients with respect
to outer coefficients. Input gradients are computed internally, not supplied.
`state_cleanup` implemented these narrow changes and all callers; root documented
them. Independent Standards (`naming_standards`) and Spec (`spec_audit`) reviews
found no issues within this diff. `description` and its banner loop are removed;
there are no compatibility aliases or replacement getters.

Verification: 89 focused tests passed; full suite **422 passed, 1 preexisting
stale controller-selection artifact failure, 18 warnings** in 28.37 seconds.
Ruff passes all 15 changed Python files; core/state/evaluation Pyright reports
zero errors; fixture hashes and numerical behavior are preserved. Git whitespace
checks pass. Logs: `/tmp/sparse-api-names-final-pytest.log`,
`/tmp/sparse-api-names-focused.log`, `/tmp/sparse-api-names-ruff.log`, and
`/tmp/sparse-api-names-pyright.log`.

The review also exposed remaining duplicated feature construction in `forward`
and the coefficient-feature calculation, and repeated quadratic fitting logic
in prepared `obj` and `obj_terms`. This naming follow-up does not consolidate
those calculations or change derivative conventions. The earlier review's
completion report did not establish that all duplication or naming issues were
resolved. Continue the user's method-by-method review without expanding scope.

**Ownership cleanup at `b9fba49` implemented and independently reviewed.** The
instruction was "refactor (or clean up) based on the boundary and ownership we
set up first". PDAP now owns insertion and warm-start decisions; SparseProblem
supplies their mathematical evaluations. Final verification: 422 passed and
one preexisting stale-artifact failure; Ruff and core Pyright pass. Both final
reviews report no unresolved material finding within this cleanup. Mask
retirement and reporting/configuration redesign remain explicitly deferred.

The reviewed scope is the diff from `49eeb03`: DataLoader integration, the
module/SSN migration, and this ownership cleanup, including the correction test
and independently sourced trajectory fixture. It excludes the user's
pre-existing edits preserved in 49eeb03. The implementation baseline edb166a
remains the reference for the migration, not the finished revision.

The exact working source/tests before this cleanup are preserved at
`/tmp/sparse-boundary-before-y_e9gbjp`, with a binary working-tree patch. Use this
snapshot to distinguish current cleanup from the earlier uncommitted migration,
compare numerical behavior, and review actual deletions. Do not regenerate
fixtures from the modified implementation.

SparseProblem is currently the nn.Module; src/models and ssn_solve are deleted;
SSN owns the complete coefficient correction. Configuration/logging redesign
and separate known numerical bugs remain deferred. The earlier DataLoader-only
restriction was the assistant's scope mistake, not a user decision.

**The root agent plans, documents, delegates, and coordinates only.** The user
explicitly assigned implementation and review to subagents. Root does not edit
production code or tests. Implementation agents own separate file sets; review
assignments are independent of the implementation under review. Keep this plan
current as work, verification, and review finish. Do not call the overall task
complete merely because one part or its tests pass.

Follow `refactor.md` and the established vocabulary in `CONTEXT.md`. Do not
rewrite either to justify a new design. Use plain text, not LaTeX, in messages.
Do not reopen settled decisions or ask routine implementation questions.

## Agreed boundaries and current implementation

User statements below are evidence for ownership. Assistant-authored plans and
historical ADR descriptions cannot establish additional user approval.

| Owner | Agreement from the discussion | Current implementation / remaining discrepancy |
| --- | --- | --- |
| SparseProblem | One `torch.nn.Module` represents the current network and owns problem-specific evaluations. The user reiterated "sparseproblem should use the pytorch nn.model and delete the models" and "all the problem specific evaluation is owned by sparseproblem class". | The module owns persistent neuron state, prediction, features/input derivatives, residuals, objectives, penalties, proximal/normal-map evaluations, supplied-candidate profiles, and mathematical radius. Mixed insertion/warm-start procedures and `src/models` are deleted. |
| PDAP | Own candidate search/ranking, insertion acceptance, warm-start direction and initial-coefficient selection, support insertion/pruning, correction scheduling and outer stopping. The user said "the pdap owns the neuron insertion", described pruning and retaining proposals until accepted, and directed scope review against the paper. | These procedures live in PDAP, insertion.py and warmstart.py. They call problem evaluations for the mathematics. Historical stopping/order variants remain distinct from the paper algorithms. |
| SSN | Retain `torch.optim.Optimizer`; optimize supplied outer coefficients with inner weights fixed, using SparseProblem's mathematics. Own correction iterations, working coefficients, and the starting-coefficient backup discussed by the user. | Complete correction, trial acceptance, final comparison/fallback and rollback are here; PDAP applies the result without a second correction acceptance loop. The legacy `ssn_solve` layer is deleted. Guard options and iteration defaults are compatibility/implementation choices, not new user requirements. |
| Temporary preparation | The user selected "Have SparseProblem return a temporary prepared objective that SSN uses for that solve" over persistent cached attributes. | Features/targets/Hessian preparation is correction-local and reused for the solve. It is neither a second network nor a separate owner of mathematical formulas. No persistent loader or feature cache lives on SparseProblem. |
| Data module and Dataset/DataLoader | The user reiterated "the data lives outside of the problem" and requested the DataLoader dependency. Data preparation and delivery leave PDAP/SSN. | `src/data.py` owns reading, validation, physical normalization, splitting and full-data assembly; Dataset/DataLoader retains and delivers aligned samples. Mathematical residuals/features/reductions remain evaluations, not DataLoader responsibilities. Native BatchSampler delivery now works while dropped samples remain rejected. |
| Repository configuration | The user said configuration "is a repo level function and we are using hydra to manage it". | Hydra/runner constructs configured objects. Modules consume the settings relevant to their responsibilities; DataLoader does not own repository configuration. Redesign remains deferred. |
| Measurement, logging, monitoring and history | The user identified these as separate concerns, then asked to "save the configuration and logging etc., part for later". | Existing metric/history and saved-run behavior is preserved. History owns one detached checkpoint; existing artifact views share it and restoration happens once. History still invokes problem metrics with supplied loaders; its final internal organization remains deferred. |
| Work allocation | The user assigned root "only in charge of the planning and orchestrating" and asked for subagents to implement and review. | Root edits plans/docs only. The latest user instruction authorizes source cleanup through implementation agents, followed by independent review. |

SparseProblem defines what a mathematical operator evaluates, including its
parameter-dependent branches. PDAP decides which candidates/directions to use,
when to initialize or insert, and whether to accept them. SSN makes decisions
inside the coefficient correction. A mathematical threshold or proximal
minimizer can be evaluated by the problem without giving it insertion control.

Paper cross-check: the authoritative `paper/paper_0805.tex` specifies this outer
procedure in Algorithm 1 (`alg:log`, lines 2898–2924) and Algorithm 2 (`alg:q`,
lines 3011–3037). Both search/filter candidates, test and rank them, initialize
and insert selected neurons, request a fixed-inner-weight SSN correction,
prune, and stop on no accepted candidate or the outer iteration budget. Both
require correction acceptance against the post-insertion objective. Assigning
that correction check to SSN is our software organization of the paper's
requirement; the manuscript does not prescribe Python classes. The algorithm
map in `vault/algorithm.md` records the different insertion tests and the
separate baseline warm-start, so they are not conflated.

The persistent network, PDAP's proposed atoms, and SSN's working coefficients
have different lifetimes. A proposal can be evaluated without first inserting
it into the live network. PDAP decides when accepted support changes are
applied; a module operation may enforce tensor/registration invariants. SSN
retains the starting coefficients needed for its own rollback. None of this
requires another network instance or copies of problem formulas.

Established mathematical names remain `obj`, `obj_augmented`, `loss`, `jac_*`,
`hess_*`, `penalty_exponent`; `q` means the SSN proximal preimage. Physical data
normalization and coefficient/moment normalization remain distinct.

The training entry point is `trainer.fit(problem, train_loader,
valid_loader, ...)`. There is no separately constructed model passed beside
problem. Prediction and construction require no dataset. SSN sees supplied
coefficient vectors and problem mathematics, not loaders, layers, or insertion.
Synthetic coefficient objectives continue to work without dummy networks/data.

## Work assignments

The naming follow-up above takes precedence over these completed cleanup
assignments and the earlier migration table:

| Task | Owner and files | Status / required result |
| --- | --- | --- |
| Problem evaluation / PDAP procedures | `ownership_cleanup`: SparseProblem, insertion.py, warmstart.py, their direct problem/model/normalization/paper/config/SSN-problem/Algorithm-2-insertion tests | Complete. Procedures moved to PDAP; feature/residual calculations shared; warm-start matrices built once; unused interfaces and four derived stored attributes removed. |
| State ownership | `state_cleanup`: pdap.py, history.py, history/results/PDAP-equivalence/end-to-end tests; results.py only if needed | Complete. Native tensors for inspection, one owned checkpoint, one restoration, and no parameter replacement when pruning removes nothing. Reporting behavior preserved. |
| Data/solver follow-up | `state_cleanup`: data.py, dataloader tests, SSN optimizer documentation and correction tests | Complete. Native BatchSampler works; tolerance behavior documented accurately; multi-iteration trust-region acceptance/rejection/recovery covered. No new convergence policy. |
| Plan, integration and evidence | Root; independent `standards_audit` and `spec_audit` | Implementation and final reviews complete. Root coordinates documentation and commit; root did not edit source or tests. |

Tests owned by `ownership_cleanup`: test_sparse_problem, test_model_contract,
test_paper_conformance, test_normalized_measure, test_ssn_problem,
test_ssn_equivalence, test_algorithm2_correction, test_algorithm2_insertion,
test_config. DataLoader tests belong to `state_cleanup`; their callers were
migrated when get_atoms/model_loss were removed. No other agent edits shared files
without an explicit handoff. Existing `spec_audit` and `standards_audit` remain
independent of these implementation changes.

Keep the temporary prepared objective limited to fixed-support correction reuse.
Do not resolve warm-start duplication by expanding it. Preserve the known
historical sphere warm-start derivative convention explicitly. Mask retirement
is deferred: existing masked goldens, finite differences and proximal tests
exercise a real mathematical contract. Removing it needs a separate decision;
this cleanup neither deletes that coverage nor creates a second solver/problem.

Implemented interface (a means to the agreed ownership, not new
user requirements): removed `search_score`, `profile_insertion`,
`finite_insertion`, `solve_insertion_weight`, `warmstart_profiles`, and
`warmstart_coefficients`. A signed differentiable `profile` shares the residual
pairing used by directional slope/curvature. `jac_forward_with_gradient` accepts
supplied atoms; its warm-start option isolates the existing historical derivative
convention. PDAP reads one penalty-strength value (`alpha`) and owns the tests
and strategy choices. One coefficient-scale evaluation is reused. Proximal
evaluations reuse existing kernels; the insertion objective increment preserves
its exact scalar arithmetic. These mathematical operations do not reject or
select candidates.
There is no new class, settings bundle, prepared-object protocol, or persistent
feature cache. Agent verification must check candidate derivatives and trajectory
rounding against the pre-cleanup snapshot, not only algebraic equivalence.

Validation and review of ownership cleanup at `b9fba49`:

- Full suite on frozen source: **422 passed, 1 preexisting failure, 18 warnings**
  in 30.00 seconds. The sole failure is the stale ignored local controller
  selection described below. Ruff passes all 29 changed Python files; Pyright
  reports zero errors/warnings in src/problems, src/PDAP, src/SSN, src/data.py
  and src/results.py. Git whitespace checks pass.
- Both PDAP reference summaries match all eight recorded values exactly; SSN
  references pass. All three fixture hashes match the pre-cleanup snapshot.
  No fixture, tolerance, or numerical stopping policy was changed by this cleanup.
- Standards and Spec reviewers independently checked the final interfaces,
  procedure ownership, duplicate calculations/state, and numerical compatibility.
  No unresolved material finding remains within this cleanup. Mask generality
  remains deferred; reporting redesign and full-data memory costs are not solved
  or benchmarked by these changes.
- SparseProblem decreased from **53 to 46 methods**, eight to seven properties,
  **17 to 13 stored attributes**, and 658 to 607 lines. The four core production
  files have 48 fewer lines. These measure actual cleanup, not design acceptance
  by method count.
- Logs: `/tmp/sparse-boundary-final-pytest.log`,
  `/tmp/sparse-boundary-final-ruff.log`, `/tmp/sparse-boundary-final-pyright.log`;
  focused numerical checks: `/tmp/sparse-boundary-prox-final.log`.

Numerical compatibility decisions: the shared private residual pairing retains
historical sum/dot operation order. Forcing one order changed a seeded support
from 51 to 54 neurons. Scalar `power_prox` preserves finite-insertion arithmetic
and is reused by fractional `penalty_prox`; SSN's vector proximal calculation is
unchanged. The scalar insertion increment keeps its exact power evaluation,
without importing the full-loss magnitude floor. Independent review caught both
tiny-coefficient regressions, now covered through actual insertion calls. These
are numerical contracts under refactor.md rule 9, not new procedures, classes,
or public arithmetic-selection protocols.

Earlier migration/review assignments (historical status before this cleanup):

| Task | Owner | Files / responsibility | Status |
| --- | --- | --- | --- |
| Network/problem consolidation | `objective_design` | SparseProblem; delete src/models; model-contract, sparse-problem, and DataLoader/problem tests | Implemented; ownership and duplication findings remain open |
| Complete correction and outer loop | `training_integration` | src/SSN; src/PDAP/pdap.py; delete ssn_solve; SSN and Algorithm 2 correction tests | Implemented; stopping description and focused coverage concerns remain open |
| Caller and artifact migration | `data_pipeline` | History, results, training runner, plotting references, remaining integration tests | Implemented; snapshot duplication remains open |
| Plan and decision record | Root | This handoff and existing architecture documentation; coordinate interfaces and evidence | Ownership/cause record corrected; no cleanup implementation assigned |
| Full Standards review | `standards_audit` | All changes since 49eeb03; documented rules, duplication, excessive interfaces/state | Complete: four material design findings |
| Full specification/design review | `spec_audit` | Actual simplification, mathematical/data/optimizer ownership, compatibility and tests | Complete: broad findings plus boundary check against user statements |
| Method/attribute usage audit | `interface_audit` | SparseProblem and callers; keep/delete/combine inventory and smaller interface | Complete: inventory, concrete reductions and current ownership reviewed |

Agents must report changed files, validation, remaining failures, and numerical
or compatibility differences. Agents do not commit until root coordinates the
completed review and checks. Root does not substitute its own implementation
for delegated work.

After the server restart, the three prior implementation agents were no longer
live. Their code, test logs, and reports were retained; three new independent
read-only reviewers were assigned, rather than repeating implementation or tests.
The pre-cleanup audit found 53 methods (eight properties), 17 stored
attributes, test-only interfaces, and mask generality with no production caller.
Do not delete useful mathematical operations merely because they have one
caller, or move the same complexity into a new class. The review must identify
real deletions/combination opportunities and their behavior consequences.

## Original full-review findings and cleanup disposition

The findings and line references below describe the pre-cleanup snapshot, not
the finished source. Feature duplication, obsolete interfaces, duplicate state,
native batching, mixed procedure ownership and missing focused coverage are
resolved. Mask generality remains deferred for the reasons recorded above.
The cause/prevention record remains applicable to future work.

Scope: all 40 tracked changed paths since 49eeb03, the new correction test and
trajectory fixture, and caller searches including ignored Python under scripts
and experiments. The reviewers did not rerun the full suite or modify source,
tests, or fixtures. They used small read-only probes for batching, stopping, and
derivative behavior. Performance/copy observations were not benchmarked.

### Standards review — four material findings

1. **P2: unused mask generality** — `sparse_problem.py:65`, `:91`, `:640`.
   Two constructor options, two attributes, three helpers, and branches through
   penalty/proximal calculations have no production caller. Every active network
   is signed, with all coefficients penalized and unconstrained. This is retained
   speculative generality, judged against refactor.md rules 7 and 10–12. Fixed
   masks also do not naturally follow changing support size. Removal must account
   for genuine existing masked numerical tests; do not silently discard them.
2. **P2: repeated feature calculations** — `sparse_problem.py:195`, `:598`,
   `:616`, and `:553`/`:560`. Correction, candidate, and warm-start calculations
   separately construct activated features and input derivatives. The two
   warm-start methods rebuild identical matrices consecutively. Refactor.md rule
   7 requires removing shared knowledge. Consolidation must preserve candidate
   differentiation and explicitly handle the historical derivative discrepancy.
3. **P2: obsolete model convenience interfaces** — `sparse_problem.py:120`,
   `:124`, `:190`, `:411`. The activation/power getters, NumPy `predict`, and
   `model_loss` survive only because tests call them. Delete these unnecessary
   interfaces and test actual prediction/metrics behavior. This is a Middle Man /
   Speculative Generality judgment under the deletion test and rule 10. In
   contrast, `obj_augmented` has a real mathematical meaning and derivative-test
   purpose; test-only use alone is not sufficient reason to delete it.
4. **P2: duplicate snapshots and defensive copies** — `history.py:93`–`:98`,
   `:134`–`:136`; `pdap.py:192`, `:211`. History separately clones atom tuples and
   the same native state, then restores both. PDAP clones for inspection and
   again during support replacement. Use one owned checkpoint and clone at
   actual ownership transfers, preserving existing artifact fields. This is
   duplicated state under refactor.md rules 7 and 12, not a request to redesign
   history/logging.

### Specification/design review — four material findings

1. **P2: standard DataLoader batching rejected** — `data.py:39` treats
   `batch_size=None` as unbatched. A native DataLoader with an explicit sequential
   BatchSampler also has `batch_size=None` while yielding correctly aligned
   batches and retaining every sample. The reviewer reproduced this rejection.
   Correct the batching validation while retaining the complete-data/no-drop
   contract; do not introduce another loader abstraction.
2. **P2: unsupported production generality retained** — the mask subsystem above
   remains despite the signed-only active network. This predates the current
   merge, but the refactor promised to remove unnecessary complexity. Masked
   synthetic/golden cases are real mathematical coverage, so the supported
   coefficient contract must be decided before removing the subsystem.
3. **P2: old model interface transplanted** — the four unused convenience
   interfaces above were carried into SparseProblem. `model_loss` is a second
   full-evaluation path after its production guard caller moved to SSN. Remove
   dead compatibility surface without discarding meaningful mathematical oracles.
4. **P2: copying still follows the removed representation** — read-only inspection,
   filtering/concatenation, support replacement, and history each copy the same
   state. Read registered tensors for inspection; snapshot at an actual ownership
   transfer. Preserve independence from the live network and between checkpoints.

Additional stopping-documentation concern: `optimizer.py:118`/`:156` does not
exit a complete correction when positive `tolerance_grad` is satisfied. It skips
Newton work in that iteration, then repeats evaluations for the remaining budget
and counts them. A stationary four-iteration probe performed four evaluations
and reported four iterations. This preserves the former wrapper's behavior;
calling it whole-correction early stopping is misleading. Correct the description
without silently changing the accepted numerical/stopping policy. The known
false-zero residual is a separate problem.

The Spec reviewer also identified missing focused coverage for native
BatchSampler delivery and complete multi-iteration trust-region fallback.
Existing numerical, state-lifetime, artifact, and fixture-provenance checks remain
useful; their passing result does not erase these design findings.

### Ownership correction after the user's follow-up

**Cause and accountability:** the first error was misunderstanding
"problem-specific evaluation" as ownership of optimization procedures that
contain mathematics. The repeated departures after the user's clarifications
were failures to follow and verify the agreed plan. Calling the entire sequence
a misunderstanding understates that failure. Root's account is that it did not
deliberately choose to override the plan; the observable record establishes the
incorrect interpretation and repeated noncompliance, not a separate approved
design change.

The concrete failure chain was:

1. The assistant broadened mathematical evaluation into insertion/warm-start
   procedure ownership and retained mixed legacy methods.
2. It recorded its own interpretation ("insertion formulas") as accepted
   architecture. Implementation details and user requirements were also mixed.
3. Implementation/review did not catch departures from the original discussion,
   including the previously omitted single-nn.Module consolidation. Passing
   tests and later consolidation did not establish correct ownership.
4. A subsequent cleanup proposal prioritized removing methods/repeated matrices
   and proposed moving PDAP's largest-profile direction selection into the
   problem, repeating the ownership error.

The user did not create this ambiguity by changing the requirement. Root owns
the planning, delegation and review failure. Independent ownership reviews
confirmed these pre-cleanup code discrepancies, now corrected:

- `profile_insertion` rejects candidates and chooses an initial coefficient;
  `finite_insertion`/`solve_insertion_weight` include insertion optimization and
  acceptance; `warmstart_coefficients` includes eligibility and step selection.
  These methods mix evaluations with decisions that belong to PDAP.
- Evaluating features, signed profiles, directional slope/curvature, objective
  changes, and penalty/proximal operations remains problem mathematics. PDAP
  owns how to search, rank, initialize, accept, and apply those evaluations.
- `src/PDAP/warmstart.py` already selects the largest-profile direction. It is
  not a pass-through. The earlier proposal to move that procedure into
  SparseProblem and delete this module is withdrawn.

The initial review itself changed no source. The latest user instruction
authorized the completed cleanup assignments above. Review must compare ownership against the user's
agreement, not treat assistant-authored descriptions or a lower method count as
acceptance.

**Required workflow for subsequent implementation:**

1. Attach the applicable ownership row and user evidence to each agent task.
   Label agreed requirements, implementation choices, and proposals separately.
   A proposal or agent suggestion never becomes accepted merely by being copied
   into this handoff or an ADR.
2. Trace the affected caller and state lifetime. Classify each operation as
   problem evaluation, PDAP decision/support update, SSN correction, data
   preparation/delivery, or existing reporting. Check candidate rejection,
   warm-start selection and correction rollback explicitly. Do not classify by
   method name, the presence of formulas, or every conditional in isolation.
3. Name the duplicated responsibility that disappears and the existing owner
   that retains it. Do not justify a new class, copied formula, or responsibility
   transfer by a shorter method count. Preserve the vocabulary in `CONTEXT.md`
   and rules in `refactor.md`; do not rewrite them to fit the implementation.
4. Have an independent reviewer compare the resulting code with the user
   evidence and ownership row as well as numerical tests. Report compliance and
   remaining discrepancies separately from passing tests. Root coordinates
   corrections; root does not replace the implementation agent.

This review reaffirmed all rows above. It also identified overclaims in the old
"accepted architecture" table: exact tensor registration, prepared-record fields,
full-data materialization, default iteration count and historical guard options
are implementation/compatibility choices. Reporting placement is still deferred.

### Authorized cleanup priorities and outcome

Restore ownership and consolidate duplicated evaluation first (item 5), in
parallel with independent state cleanup; then finish the data/solver follow-up
and broader verification. The numbered findings retain their previous order
for traceability.

1. Correct the DataLoader batching check and cover the reproduced native
   BatchSampler case. Retain ordinary/uneven batching, no-drop behavior, and RNG
   isolation. This is a small defect fix within the accepted DataLoader contract.
2. Delete `predict`, the `activation`/`power` getters, and `model_loss`. Migrate
   tests to `forward`, `predict_tensors`, `metrics`, and independent objective
   expectations instead of preserving unused methods for tests.
3. Derive `_input_dim` from `W.shape[1]`, `_normalized` from `not use_sphere`, and
   `_penalty_exponent` from power. Obtain growth data from the registry when
   computing the radius. Keep the existing useful property names. This removes
   four stored attributes (17 to 13) without adding a settings class or getters.
4. Make History own one detached snapshot and populate its existing artifact
   views from it. Use native tensors in PDAP for read-only inspection; keep
   `set_atoms` as the validated support replacement. Remove `get_atoms` once its
   copying protocol is gone. Preserve old artifact readers and checkpoint
   isolation; avoid two consecutive state restorations.
5. Keep the warm-start procedure in the existing `src/PDAP/warmstart.py` and
   insertion procedures in `src/PDAP/insertion.py`. Move the acceptance and
   initialization decisions currently hidden in SparseProblem into those
   procedures. Keep problem-specific evaluations with SparseProblem and its
   existing mathematical helpers. Prepare shared warm-start calculations once
   without moving direction selection into the problem, duplicating objective
   formulas in PDAP, or introducing another class or collection of getters.
6. Keep necessary mathematical calculations together. `model_loss` is removed;
   `obj_terms` remains the mathematical value/gradient fitting decomposition used
   by metrics. Its single caller is not itself a reason to remove the operation.
   Do not move the entire `solve_insertion_weight` procedure into `prox.py`:
   it includes insertion decisions as well as a proximal evaluation. Existing
   mathematical helpers supply the calculation; PDAP owns the insertion solve
   and its acceptance.
   Neither a single-caller method nor a long class is by itself a defect.
7. Treat mask retirement and cross-path derivative unification as separate
   explicit decisions. No production caller needs masks, but masked goldens,
   normal-map finite differences, nonnegative prox tests, and fractional-scale
   tests currently exercise them. Do not drop those assertions or add a second
   mathematical implementation merely to make the class look smaller.
8. Recheck caller simplicity and meaningful tests after cleanup, including the
   complete correction with both SSN strategies. Correct stopping documentation;
   do not silently add a new convergence policy.

The first three simplifications of the class (unused interfaces, derived state,
and snapshot ownership) remove five public operations and four stored attributes.
This is a consequence of removing unnecessary obligations, not a method-count
target. The SSN loss, smooth-gradient transformation, prox, preimage, normal map,
and their Jacobians remain distinct mathematical operations. Combining them into
a large tuple-returning method would hide the same complexity and disrupt the
current lazy Jacobian/tolerance ordering.

Numerical constraint on feature unification: for inputs `[-1, 2]`, a unit inner
weight, zero bias, and power 1, correction versus warm-start derivatives are
`[0.01, 1]` versus `[0, 1]` for leaky ReLU, and `[0, 4]` versus `[0, 1]` for squared
ReLU. Ordinary ReLU agrees. Remove repeated work within a warm start first;
blindly replacing every derivative path with autodiff changes behavior.

Items 1–6 and 8 are complete, with targeted verification, full-suite comparison
and independent boundary review recorded above. Item 7 retains masked numerical
coverage and the explicitly tested historical derivative convention; changing
either contract is deferred.

## Implementation decisions and compatibility checks

These implement the ownership requirements above; their exact fields, defaults
and storage mechanics are not separate user-approved requirements.

- Register `inner_weight` and `inner_bias` as buffers and `c` as an `nn.Parameter`, with coherent
  empty support. Direct registration avoids randomly initialized layers that
  would immediately be overwritten with exact atoms.
- `prepare_coefficients(loader)` currently returns the three-field
  `PreparedObjective` record (`obj`, `hess_obj`, `coefficient_scale`). Features
  and targets are captured by its temporary closure; the record adds no
  mathematical implementation or second network.
- Full-data evaluation currently materializes a complete loader pass in the
  data module. This is a memory tradeoff in the implementation, not a promise
  of out-of-core training or a requirement to store data on the problem.
- Use `SSN.step()` for one complete correction, with an internal iteration
  budget (`max_iter`, default 20). Existing tests of one numerical iteration
  explicitly request one iteration. No mandatory new convergence criterion.
- Keep a starting coefficient copy inside SSN. With the guard enabled, accept
  only a finite final full objective no greater than the initial objective;
  otherwise restore the starting coefficients. Exceptions must not mutate the
  network. Report iterations and correction acceptance, not convergence.
- Preserve existing explicit `correction_guard=False` configurations as the
  historical unguarded comparator; the guard is implemented only in SSN. Do
  not silently change Hydra defaults while configuration redesign is deferred.
- Preserve the existing optional `tolerance_grad` setting for compatibility;
  do not present it as a newly validated convergence test. Its known false-zero
  residual requires separate mathematical investigation.
- Use the existing feature-residual loss arithmetic and cached fitting Hessian.
  Compact quadratic storage, new penalty/activation semantics, and numerical
  bug fixes are separate work.
- Direct atom registration removes random draws from discarded nn.Linear
  initialization. Check seeded PDAP trajectories explicitly. Do not add dummy
  random draws to production, overwrite historical fixtures, loosen tolerances, or hide
  failures to claim equivalence. Establish whether observed differences are
  this RNG change or a mathematical regression, and record the evidence.
- Preserve reading historical atom snapshots and physical-coordinate reporting,
  including zero support and evaluating another checkpoint without modifying
  the current one. Historical layer-key state dictionaries need treatment at
  the reader, not a retained model hierarchy.
- Saved prediction may use historical activation powers unsupported by today's
  SSN correction. Plain construction permits positive finite powers and an
  activation name or callable. `validate_correction()` retains the existing
  supported-power/gamma/geometry restrictions, called by training `from_config`,
  insertion validation, and coefficient preparation. This preserves historical
  prediction without a second class or a prediction-only mode flag.
- Keep state_dict limited to network state, with no samples, loader, feature
  cache, prepared objective, or optimizer state stored on SparseProblem.

## Verification and completion gate

Before declaring completion, require all of the following:

1. Inspect the actual diff for the accepted architecture: SparseProblem is an
   nn.Module, obsolete model files and ssn_solve are deleted, and callers use
   one problem. No replacement hierarchy or duplicate formula owner.
2. Check native module behavior, prediction/input derivatives, exact coefficient
   storage, support changes, empty support, checkpoint restoration, and no
   incidental RNG changes from assigning exact atoms.
3. Check prepared objective value/gradient/Hessian and lifetime; full versus
   uneven DataLoader batches; candidate/warm-start/radius mathematics.
4. Check complete SSN iterations, both strategies, final acceptance/fallback,
   normalized coefficients, historical comparator behavior, and exception
   isolation from persistent network state.
5. Check PDAP integration, history, saved-run readers, runner, and logging
   contracts. Preserve numerical fixtures; explain any trajectory differences.
6. Run appropriate focused tests, type checks/lint, then the full suite once
   after integration. Investigate new failures rather than classifying them as
   pre-existing. Independent standards and specification reviews must both
   resolve material findings.
7. Update this record with evidence and outstanding limitations, then have an
   implementation subagent commit the reviewed changes on the current branch.
   Do not push or publish without a separate user instruction.

Baseline at `edb166a`: **398 passed, 2 failed, 20 warnings**. Existing failures:

- `test_pendulum_selection_prioritizes_stabilization_then_global_error`: stale
  ignored local `controller_selection.json`; do not regenerate unrelated data.
- `test_pdap_summary_matches_golden[signed_profile]`: prior platform/reference
  mismatch. Pre-change L2 = 0.020708285737220288, H1 =
  0.05464074122149227, loss = 0.0028238830405932294. The old fixture is unchanged.

Earlier prerequisite work was preserved in `49eeb03`; DataLoader integration
was committed in `edb166a`. The working tree was clean when this correction
started. `refactor.md`, `CONTEXT.md`, and numerical fixtures were unchanged.

Focused verification reported by the implementation agents:

- 43 module/problem/DataLoader tests pass; src/models is deleted.
- 45 SSN tests pass, including complete iteration budgets, worsening/nonfinite
  final fallback, the explicit unguarded comparator, and exception rollback.
- 10 Algorithm 2 correction tests pass after caller/validation migration.
- 104 History/results/runner/config/normalization/radius/Algorithm 2/paper
  conformance checks pass.
- Both trajectory-reference tests pass with unchanged strict tolerances.

These selections overlap; they are not an aggregate suite count.

Integrated verification before the reopened design review, reported by `data_pipeline`:

- `.venv/bin/python -m pytest -q`: **410 passed, 1 failed, 18 warnings** in
  30.07 seconds. The sole failure is the unchanged stale ignored local
  controller-selection artifact listed above. No new failures remain.
- Ruff on all changed production/test files and Git whitespace checks pass.
- Pyright has no errors in SparseProblem, SSN, PDAP, History, or results.
  train.py/plots.py report 26 diagnostics (4/22); independent checks of their
  `edb166a` versions produced exactly the same diagnostic identities. No new
  typing errors were introduced.
- Earlier standards/specification reviews reported no unresolved findings.
  The user subsequently rejected the class/interface design; the full review
  above supersedes those reports as evidence of design acceptance.
- Production code changes total 307 added / 708 removed lines (401 fewer).
  This is separate from tests and documentation. refactor.md, CONTEXT.md, and
  both historical numerical fixtures are unchanged.

Earlier migration logs: `/tmp/sparse-final-pytest.log`,
`/tmp/sparse-final-pyright.log`. Final cleanup results and review disposition are
at the top of this document. No push is authorized.

The caller agent independently reran `edb166a` in a temporary snapshot and then
changed only its discarded layer-initialization RNG consumption in a diagnostic
process. Neither production code nor test fixtures were patched for this probe.
This provides a reference for attributing the observed trajectory differences:

| Reference | Final neurons | L2 | H1 | Last loss |
| --- | ---: | ---: | ---: | ---: |
| Original signed_profile | 54 | 0.020708285737220288 | 0.05464074122149227 | 0.0028238830405932294 |
| Original signed_finite_step | 57 | 0.02066964472206974 | 0.05551263296586934 | 0.0029939213075414516 |
| Old math without layer RNG, signed_profile | 57 | 0.02125353101279779 | 0.048004871851952166 | 0.0022523097271912127 |
| Old math without layer RNG, signed_finite_step | 51 | 0.026508712474840275 | 0.05603594165442737 | 0.0028927412770338714 |

The current implementation matches all eight reported values from the old-code
diagnostic without layer RNG **exactly**, including final support widths. The
coordinated regression treatment keeps `pdap_golden.npz` unchanged and adds
`pdap_direct_registration_golden.npz`, populated from that independent old-code
diagnostic, never from the new implementation's results. Existing strict
tolerances remain unchanged. This records the accepted representation's RNG
consequence without manufacturing random draws, skipping tests, or silently
replacing the historical reference. Both current trajectory tests and the full
suite's numerical checks pass against this independently sourced reference.

Review follow-ups addressed before final verification: the existing fractional
power/log-penalty rejection test now exercises `validate_correction()` (plain
construction also serves historical prediction), and the trajectory test no
longer silently regenerates an independent reference from current code. The
SSN owner reports all 10 Algorithm 2 correction tests passing after the former
migration. These change test setup, not the mathematical rejection or tolerances.

Evidence lives at
`/var/folders/11/x3msc_sj5236k25kdjys_9_r0000gn/T/sparse-pdap-baseline-zuhk5wgd`.

## Historical discussion and recovery notes

The following records preserve the original decisions, research, and recovery
context. Statements about what is not yet authorized, separate models, mutable
prepared caches, or a future model merge describe earlier stages; the current
plan above supersedes them. The configuration/logging deferral remains current.

## Accepted decisions

### External datasets and data loading

Training and validation datasets remain outside SparseProblem. The user has now explicitly chosen standard PyTorch DataLoader integration. Use Dataset/DataLoader for sample access and batch delivery; keep Hydra configuration at repository level and measurement/monitoring/logging separate. Passing batches to mathematical evaluations does not imply storing them permanently on the model.

A DataLoader does not itself make already-resident tensors out-of-core; the Dataset/loading implementation determines that. Current PDAP/SSN solve the full empirical objective. Chunking must aggregate the full contributions; stochastic minibatch updates would change the algorithm.

Feature-based fitting evaluations and the cached fitting Hessian in the temporary prepared objective are agreed. The current implementation scope is recorded above; the older broader plan below also contains deferred model, configuration, and logging changes.

### Temporary prepared coefficient objective

The purpose is limited to reusing fixed calculations within one SSN correction.
With inner weights, biases, samples and objective settings fixed, neuron values,
input-derivative features, coefficient normalization and the data-fitting Hessian
do not change when SSN changes the outer coefficients. SparseProblem calculates
them once; SSN reuses them for iterations, trial evaluations and the final check.
Only the data-fitting Hessian is fixed: penalty corrections and normal-map
Jacobians can still depend on the current coefficients.

Preparation is not mathematically necessary. Recomputing these quantities on
every evaluation is a valid alternative, at the cost of repeated neuron/input
derivative evaluation and feature-matrix products. Keeping this reuse local to
one solve implements the user's choice to avoid persistent prepared attributes
on SparseProblem. It is not an additional architectural layer.

The current method-free record contains only `obj`, `hess_obj`, and optional
`coefficient_scale`. The fitting callable captures fixed features and target
snapshots; the formulas are defined by SparseProblem. PDAP passes this return
value to SSN and releases it after correction. Changing support, data or relevant
settings requires fresh preparation. No separate setup/reset/invalidation
protocol or persistent loader is needed.

This record is neither a proposed network nor a coefficient backup. SSN keeps
its own starting-coefficient copy. Candidate-search and warm-start calculations
are separate local evaluations; their duplicated work does not justify expanding
this record into a general cache or creating more prepared-object classes.
It must not own insertion decisions, solver state, loaders or logging. Preserve
the current feature-residual arithmetic during structural cleanup; expanded
quadratic storage or other numerical changes remain separate decisions.

### One network/problem representation directly on PyTorch

The target SparseProblem subclasses torch.nn.Module and owns the network representation and problem-specific mathematical operations.

The custom SignedModel / ShallowNetwork hierarchy and redundant model contract should be removed as callers migrate. PDAP and SSN remain separate algorithms. Centralization does not require moving all mathematical helper functions into one enormous class/file.

The accepted parameter representation is:

| Quantity | PyTorch representation | Meaning |
| --- | --- | --- |
| W | Registered buffer | Inner weights |
| b | Registered buffer | Inner biases |
| c | nn.Parameter | Outer coefficients |

Direct registration was chosen over two nn.Linear submodules. PDAP creates exact atoms and changes support size; constructing layers and overwriting their initialization adds little value.

Illustrative forward calculation only, not an agreed final signature:

~~~python
z = torch.nn.functional.linear(x, self.W, self.b)
neurons = self.activation(z) ** self.power
return neurons @ self.c[:, None]
~~~

An nn.Module submodule is a contained nn.Module object, not a method. Directly registered buffers/parameters are native PyTorch too.

### SSN remains independent of neural-network structure

SSN works with a coefficient vector and mathematical operations. It should not need knowledge of layers, prediction, datasets, or neuron insertion.

Generic mathematical solver cases, including synthetic objectives in tests, should remain possible without manufacturing a dummy network.

### Complete SSN solve and non-increase guarantee

SSN owns numerical state, iterations, stopping/convergence, internal trial acceptance, and the final acceptance check for the complete solve. PDAP receives coefficients and convergence/outcome information from that solve.

SSN compares the full objective at its final coefficients with the objective at its starting coefficients. If the result worsens the objective, it returns the starting coefficients and reports that outcome. The comparison is against the coefficients supplied to this correction, which are the post-insertion coefficients in the paper's insertion-first loop. It does not undo the insertion.

The user explicitly accepted running a configured number of SSN iterations, with no required convergence test. On completing the iteration budget, return the final coefficients if they pass the final objective check; otherwise return the starting coefficients. Report iteration count and correction acceptance. Completing the budget is ordinary completion, not an exception or evidence of nonconvergence. Optional early stopping/convergence reporting can be added later with a validated criterion. Evaluation/linear-algebra exceptions are a separate issue.

PDAP owns insertion, pruning, outer stopping, and applying the coefficients returned by SSN to SparseProblem. It does not repeat SSN's final acceptance check or keep a network snapshot for that check. This supersedes the earlier design that assigned final correction acceptance to PDAP.

The accepted recommendation uses a temporary coefficient working vector during SSN. The model's c remains the accepted physical coefficient vector. Where required, SSN works in normalized coordinates u; the candidate is converted back for application. This requires coefficient storage, not another network or a dataset copy.

The target should **remove the legacy ssn_solve wrapper**. It has no independent responsibility under the agreed ownership: SparseProblem provides the prepared objective and problem-specific coefficient transformations/calculations; SSN owns the coefficient working vector, complete numerical solve, and final check; PDAP initiates the correction and applies its result. Retaining Optimizer inheritance does not justify retaining this wrapper.

A caller search found PDAP._correct as its production caller and direct imports in correction/integration tests. Migrate those callers and tests to the new boundary; their current imports are not a reason to preserve the wrapper. The module also contains SolverConfig and ALGORITHM2_PROX_RHO, whose settings and behavior must be accounted for during migration without retaining a mixed-responsibility solve layer. No requirement for a compatibility wrapper has been identified. This remains a design decision, not authorization to start implementation before the remaining design discussion is finished.

The user explicitly accepted retaining torch.optim.Optimizer inheritance. Use its parameter handling and numerical-state facilities directly. This is compatible with every accepted ownership decision; the exact optimizer-facing API remains to be specified.

### Existing mathematical naming

Inherited from the recovered ownership-refactor session:

- obj: data-fitting objective.
- obj_augmented: fitting objective plus the nonconvex correction used in the normal-map formulation.
- loss: fitting objective plus the full penalty.
- Derivatives use jac_ and hess_.
- penalty_exponent is the scalar exponent; q is the SSN proximal preimage.

For the log family, loss = obj_augmented + alpha * ||u||_1. For the fractional family, obj_augmented = obj and the prox handles the full penalty.

## Remaining discussion agenda

This is an inventory, not accepted defaults. Follow dependencies and discuss one question at a time.

1. **SSN implementation details.** Optimizer inheritance and removal of ssn_solve are settled. Specify the complete-correction entry point and supply the prepared mathematical objective directly using PyTorch's existing facilities. The proposed entry point is step(). Resolve routine coding details without turning each into a new architecture question.
2. **SSN numerical contract.** Fixed-iteration execution without a required convergence criterion is accepted. Resolve minimal iteration/acceptance reporting, exceptions, numerical tolerance for the agreed final objective comparison, and treatment of existing tolerance_grad/correction_guard settings. Investigate the false-zero residual independently because it also feeds the Newton equation. Optional convergence reporting must use a validated criterion. Keep problem-specific mathematics separate from numerical iteration logic.
3. **DataLoader integration (current plan).** DataLoader adoption and external data ownership are now accepted. Use the integration plan below for sample preparation, loader construction, batch-aware problem calculations, insertion residual lifetime, correction preparation, and caller migration. Hydra owns configuration; measurement, monitoring, history, and persistence have separate boundaries. Preserve complete coverage, sample/residual alignment, and full-objective normalization. Retaining all prepared features still consumes memory proportional to sample count.
4. **Construction and prediction without samples.** Explicit input dimension, zero support, restoration from weights, and separating training-valid settings from prediction-valid settings.
5. **Authoritative neuron evaluation.** Centralize values and input derivatives across prediction, candidate search, warm starts, and coefficient preparation. Decide when gradients remain differentiable versus detached. Candidate search needs differentiation of input gradients with respect to candidate parameters.
6. **Activation/power semantics.** Configured power is currently an extra activation exponent, not necessarily the activation's homogeneity degree. Registry relu2 is already squared; some sphere flags cover nonhomogeneous activations. Preserve or separately correct these semantics.
7. **Support replacement invariants.** Shapes/device/dtype, empty support/input batches, replacing c when width changes, and stale prepared/optimizer state. Define one coherent empty network state.
8. **Configuration ownership and compatibility.** cfg.model mixes network/objective settings with insertion selection; cfg.training mixes PDAP and SSN settings. Internal ownership and external CLI/artifact-key migration are separate decisions.
9. **Artifacts and history.** Checkpoint contents and key migration, old run/config/History compatibility, old Python imports, and avoiding dataset/cache persistence or unnecessary duplicate snapshots.
10. **State left by fit.** Final iterate versus best recorded iterate; best currently means smallest training objective, not validation error. Define the returned history/result and preserve nonmutating alternate-checkpoint evaluation.
11. **Coordinate conventions.** Physical versus training input coordinates, normalizer/gradient-chain-rule ownership, and distinction from coefficient/moment normalization.
12. **Numerical preservation and known fixes.** Mathematical equivalence versus tolerance-based equivalence versus seeded trajectories. Decide RNG changes, objective evaluation order, and derivative fixes explicitly. This question was raised earlier but the user redirected to architecture before answering.
13. **Migration and validation.** After design agreement: incremental changes, caller migration, meaningful behavioral tests, historical readers, and documentation. This list is not authorization to start implementing the refactor.

## Inspection findings before the DataLoader increment

### Current correction path

[SparseProblem](src/problems/sparse_problem.py) currently owns samples and a mutable prepared coefficient cache. prepare_coefficients builds value/gradient feature matrices, coefficient normalization, Hessian, and masks. obj evaluates squared residuals using those features and original targets.

[ssn_solve](src/PDAP/ssn_solve.py) prepares the problem, creates a temporary nn.Parameter, constructs SSN, loops over step(), and writes converted coefficients back to the model.

[PDAP._correct](src/PDAP/pdap.py) currently optionally snapshots the entire model state, calls ssn_solve, checks the objective, and restores the snapshot on worsening. In the agreed target, SSN performs the final comparison before returning coefficients for PDAP to apply.

[SSN](src/SSN/optimizer.py) currently subclasses Optimizer, evaluates explicit trial vectors through problem operations, and commits accepted steps to supplied parameters. Both damped-Newton and trust-region strategies exist.

Its current step() takes no closure and instead obtains mathematical operations from a problem object annotated as SparseProblem. That is a specialized interface; compatibility with a generic step(closure) caller should not be assumed. The iteration loop lives in ssn_solve, while SSN.step performs one iteration. Neither that split nor the concrete problem annotation is imposed by Optimizer inheritance. Refactoring these boundaries does not establish a need to remove the base class or to add a closure adapter without a concrete consumer.

Prepared coefficient state is consumed during correction. Insertion and warm-start routines evaluate candidate atoms independently against samples/residuals; they do not need this fixed-support cache.

For nonhomogeneous normalization, u = scale * c. Physical input/data normalization is a separate operation.

### Convergence check discrepancy

SSN currently reconstructs q using problem.prox_preimage(u, grad, inverse_step), then calls normal_map(q, u, grad, inverse_step). The mathematical normal-map definition assumes u = prox(q). That consistency is not checked before the current stopping test. On zero penalized coordinates, the reconstruction uses q = -grad / inverse_step, so inverse_step * (q - u) + grad can vanish even when prox(q) differs from u.

A read-only Python reproduction used the existing SyntheticProblem (which inherits SparseProblem's proximal/normal-map operations) for loss(u) = 0.5 * (u - 2)^2 + 0.1 * |u|, with u = 0 and tolerance_grad = 1e-8. The current residual was exactly 0 and SSN.step left u at 0 with loss 2. The scaled proximal fixed-point residual at u was 1.9; the actual minimizer u = 1.9 has loss 0.195. This verifies a concrete false convergence case, not just a hypothetical numerical concern.

The default tolerance_grad = 0 disables the early-exit check, but the reconstruction also feeds the Newton equation. The wider effect on iterations needs investigation before a correction is implemented. Do not silently alter the numerical algorithm or regenerate goldens as part of moving methods. The proposed convergence criterion is not yet accepted, and no source implementation was changed by this diagnostic.

### Duplicated evaluation and a derivative discrepancy

[ShallowNetwork](src/models/net.py), [SignedModel](src/models/signed.py), candidate evaluation, and warm-start matrix construction duplicate neuron calculations.

The warm-start derivative branch uses the ReLU indicator for all sphere-marked activations. A read-only check with x = [-1, 1], W = 1, b = 0, power = 1 found:

| Activation | Model/candidate input derivative | Warm-start derivative |
| --- | --- | --- |
| leaky_relu | [0.01, 1] | [0, 1] |
| relu2 | [0, 2] | [0, 1] |

Correcting this is a behavioral change to validate explicitly, not silently hide in the class merge.

### Prediction, history, and compatibility

- [model_from_history](src/results.py) reconstructs predictions from activation, power, and W/b/c without samples. However, load_run currently opens/hashes the dataset when data_sha256 provenance is present. Portable prediction and provenance-checked loading are distinct.
- [History](src/PDAP/history.py) stores atom copies and every state_dict tensor at each iteration. Do not accidentally register datasets/features as persistent buffers copied into each checkpoint.
- Alternate-checkpoint evaluation currently reconstructs a separate predictor and leaves the saved model unchanged; tests require this.
- The runner leaves the final model in place and separately reconstructs the best recorded checkpoint for regional evaluation. Correction-first ordering can end with an insertion not subsequently corrected or recorded.
- Empty models can currently be layerless, whereas restored empty support uses zero-width layers.
- Inspection counted 588 current run records, all signed models with powers 1/2/3. Among 6,336 archived records there are also powers 2.01/4/5 and combinations rejected by the current training constructor. This does not establish that today's loader reads every archive schema.
- Active ignored paper scripts consume src.results and existing config keys. Gitignored does not mean obsolete. No active notebooks were found outside archived/generated directories.
- MLflow projects config, metrics, and artifact paths; it does not serialize the network class itself.

### Numerical behavior and scope

- SignedModel.set_atoms reconstructs nn.Linear layers then overwrites initialized values. This consumes random numbers. Direct tensors can change later insertion searches and seeded trajectories despite equivalent forward evaluation.
- A compact quadratic objective is mathematically equivalent to squared-residual evaluation, but cancellation and arithmetic order can differ. Representation choice has numerical consequences.
- PDAP golden tests check exact neuron counts and tolerant summaries, with some platform-sensitive checks skipped on CI. SSN goldens check per-step behavior. Do not automatically regenerate fixtures.
- CPU/float64 is the production contract; generic solver tests also cover float32. GPU expansion is not requested.
- Loop orders, insertion modes, correction guards, radius policies, deduplication, and empty-support behavior need deliberate treatment, not silent changes.
- Semiconcave models were retired: [ADR 0012](docs/adr/0012-retire-the-semiconcave-model.md). Unrelated open-loop physical-system “problem” classes are outside this merge.

## Research already completed

The user requested established practice for nested optimizers. Both direct parameter mutation and candidate-returning inner solves are established patterns; the outer algorithm's acceptance/state lifecycle determines suitability.

- [PyTorch optimizers](https://docs.pytorch.org/docs/2.14/optim.html): accept supplied parameters, including standalone tensors. Temporary nn.Parameters do not require another model.
- [PyTorch L-BFGS](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/lbfgs.py): strong-Wolfe evaluation saves parameters, applies/evaluates trial updates, restores values on normal return, then later installs a selected step. This is not a guarantee of exception rollback or restoration of arbitrary closure side effects.
- [PyTorch 2.10 L-BFGS implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/lbfgs.py), also checked in the restored local environment: one parameter group, global state registered under the first parameter, and multiple iterations inside step(). These are valid precedents for retaining SSN's Optimizer inheritance while moving the complete correction into SSN. They are not reasons to replace SSN's mathematical algorithm with L-BFGS.
- [SciPy basin-hopping documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) and [source](https://raw.githubusercontent.com/scipy/scipy/main/scipy/optimize/_basinhopping.py): copies current coordinates, calls a local minimizer, tests acceptance, and replaces the accepted state only when appropriate. This is an ownership analogy, not a proposal to use that algorithm.
- [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html): complete solve returning a solution and termination information.
- [PyTorch functional_call](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html): supports evaluation with supplied parameters. Do not introduce it automatically when the objective already takes explicit coefficient vectors.

The temporary-coefficient recommendation follows from correction acceptance and normalized coefficient coordinates; it is not a universal best-practice claim and does not imply removing Optimizer inheritance.

### Built-in optimizer research for objective storage

The user explicitly requested documentation and source-code research on the feature-versus-quadratic storage choice. The inspected version is 2.10, matching the restored environment, with matching local source inspection.

- [Optimizer usage documentation](https://docs.pytorch.org/docs/2.10/optim.html#optimizer-step-closure): ordinary steps consume precomputed gradients; methods needing repeated evaluations use a supplied closure. The closure controls loss evaluation and gradient computation.
- [SGD source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/sgd.py): reads parameter gradients and optionally stores momentum buffers. It does not build or store sample features.
- [Adam source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adam.py): reads parameter gradients and stores step counts and moving gradient statistics (with an additional maximum statistic for AMSGrad). Feature/target storage is outside the optimizer.
- [L-BFGS source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/lbfgs.py) and [documentation](https://docs.pytorch.org/docs/2.10/generated/torch.optim.LBFGS.html): keeps a bounded update/gradient history and other numerical iteration values; invokes the closure repeatedly, including at trial parameters. It does not inspect or transform the loss into an exact quadratic representation.

Inference for this project: objective preparation/caching belongs behind SparseProblem's mathematical interface; SSN stores its numerical iteration information. A supplied evaluator may reuse calculations valid for fixed neurons, but these sources do not recommend either full features or quadratic coefficients for this specific problem. Do not attribute the compact-summary proposal to a PyTorch best-practice rule. Retaining current residual-based fitting calculations for the structural refactor is an implementation recommendation to preserve existing arithmetic, not a requirement imposed by these sources. Selecting a different representation should be justified by performance and numerical measurements.

### Data access, history, and logging research

The user requested this investigation after distinguishing external ownership of samples from DataLoader adoption. The subsequent explicit decision is to adopt DataLoader. The sources below inform the integration plan; the earlier tensor-input recommendation has been superseded.

Primary evidence:

- [PyTorch 2.10 data documentation](https://docs.pytorch.org/docs/2.10/data.html), matching the restored environment, and the [DataLoader implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/data/dataloader.py): DataLoader orchestrates dataset traversal, sampling/order, batching/collation, worker processes/prefetch, and optional pinned host memory. It is already part of the project's PyTorch dependency. Its availability does not require using it for resident tensors.
- [Dataset implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/data/dataset.py) and [fetcher implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/data/_utils/fetch.py), also inspected locally: TensorDataset retains and indexes tensors; the fetcher requests samples from the dataset and collates them. File decoding, normalization, and splits are supplied separately. DataLoader does not define losses, metric reduction, history, or checkpoints.
- [PyTorch Ignite Engine documentation](https://docs.pytorch.org/ignite/generated/ignite.engine.engine.Engine.html#ignite.engine.engine.Engine.run) and [source](https://docs.pytorch.org/ignite/_modules/ignite/engine/engine.html): the training engine accepts a repeatable iterable of batches, including a list or DataLoader. Event handlers attach evaluation/logging/checkpoint behavior outside the model. This demonstrates that a standard training boundary need not demand a concrete DataLoader type.
- [Hugging Face Trainer source, v5.19.0](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/trainer.py), following the documentation's source link: get_train_dataloader creates input infrastructure in the trainer; log appends supplied scalar results to TrainerState.log_history and invokes logging callbacks. The [callback documentation](https://huggingface.co/docs/transformers/main_classes/callback) describes separate logging/evaluation/checkpoint schedules and run state. This supports separating model calculations from result recording.
- [Lightning DataModule documentation source](https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/data/datamodule.rst): preparation, splits/transforms, and loader creation can be organized separately from the model. This is an example of responsibility separation, not a recommendation to add Lightning or a new DataModule hierarchy here.

These frameworks offer different interfaces. The project-specific inference is to separate data preparation, mathematical evaluation, training control, and recording. The user's choice determines the concrete use of DataLoader in the plan below.

## DataLoader integration plan

**Historical broader plan:** the implementation record at the top supersedes
this proposed sequence. The single-module migration is now implemented; new
Hydra settings and changed history/logging ownership remain deferred. DataLoader
adoption and full-data mathematical requirements remain in force. Current
full-data assembly is a documented memory tradeoff, not out-of-core training.

### Responsibility boundary

| Component | Responsibility |
| --- | --- |
| Hydra schema and conf/ | Define and compose data, loader, algorithm, evaluation, and reporting settings. The runner resolves them. |
| src/data.py | Read and validate files; apply the existing normalization and split; provide a small factory that returns standard TensorDataset/DataLoader instances from explicit arguments. No Hydra lookups or logging backend ownership. |
| TensorDataset / DataLoader | Retain external samples and deliver batches, including PyTorch's indexing, collation, and worker machinery. No problem formulas, model state, measurement, or run history. |
| SparseProblem(nn.Module) and mathematical helpers | Compute predictions, derivatives, residuals, candidate profiles, objectives, and metric contributions from supplied batches. Own mathematical full-data reductions and temporary preparation. No permanent datasets/loaders or configuration-framework dependency. |
| PDAP | Receive loaders for one fit, coordinate insertion/correction/pruning and measurement timing, and request problem evaluations. Do not implement a second sampler, batching scheme, or set of mathematical reductions. |
| SSN | Solve against the temporary prepared objective. No dataset traversal or loader dependency. |
| Measurement / monitoring | Mathematical metrics use SparseProblem and existing src/eval.py helpers. Timing/counters and report scheduling are separate from data delivery. Produce scalar records/diagnostics for consumers. |
| History / runner / ExperimentRun | History stores supplied metric records and detached snapshots. Runner/ExperimentRun control output destinations, artifacts, and backend publishing. No hidden data loading or metric reevaluation in History. |

SparseProblem may iterate supplied batches to perform a mathematical reduction. That is consumption of the loader's output; fetching/sampling/collation still belong to DataLoader. Do not move residual or feature formulas into a Dataset, collate_fn, or loader subclass merely because they use samples.

### 1. Standard data objects and configuration

- Keep the current loading, sample validation, chain-rule normalization, and NumPy-seeded split in src/data.py. Preserve preprocessing order and normalization scope during this migration; changing how normalization is fitted is a separate numerical decision. Retain split_value_samples for existing dataset/result utilities.
- Wrap each resulting split in TensorDataset(x, v, dv), then construct ordinary DataLoader objects. Use the built-in collator. Do not introduce a custom DataLoader subclass, a new data-module hierarchy, or a wrapper that forwards the entire PyTorch API.
- Batch contract: three tensors that unpack as x, v, dv, shaped (batch_n, input_dim), (batch_n, 1), and (batch_n, input_dim). PyTorch's default collation currently returns a list for TensorDataset samples; accept the unpackable sequence rather than requiring a tuple. Preserve CPU/float64 for the production path.
- Proposed minimal Hydra additions in src/config/schema.py: data.batch_size, a positive integer or null, and data.num_workers, a nonnegative integer defaulting to 0. In this repository, null means the full split; the factory resolves it to len(dataset) before constructing DataLoader. Passing batch_size=None directly to PyTorch disables automatic batching and is not the intended meaning.
- Start with batch_size=null and num_workers=0. Enforce sequential complete passes with shuffle=False and drop_last=False; keep pin_memory=False for the current CPU contract. The factory consumes resolved values passed by the runner; it does not read Hydra or establish a second configuration system. Unsupported sampling/drop behavior must not silently change the empirical objective.
- Give train and validation loaders separate torch.Generator instances seeded from the run seed. Preserve the existing split generator behavior. Local PyTorch 2.10 inspection and a read-only probe confirmed iterator construction changes the global Torch RNG even with shuffle=False and num_workers=0, unless a separate generator is supplied. Validation must not perturb later neuron proposals.
- Keep input_dim, sample counts, and normalization metadata outside the model's sample storage. Determine input_dim from prepared source metadata before construction, not by consuming a loader just to discover a shape. Count samples from batch sizes during mathematical reductions; len(loader) counts batches.

### 2. Entry points and model lifetime

Target calls (illustrative signatures):

~~~python
history = trainer.fit(problem, train_loader, valid_loader)
prepared = problem.prepare_coefficients(train_loader)
metrics = problem.metrics(train_loader, valid_loader, reporting_normalizer=normalizer)
~~~

The latter calls occur inside PDAP when the corresponding calculation is needed. Mathematical methods may accept an iterable of batches; the training runner supplies actual DataLoaders. Always pass the repeatable loader, not an already-consumed iterator. Constructors and forward(x) need no training data. Preserve the current validation-as-training fallback if no validation split is supplied, making that choice in training orchestration.

Remove _data_train/_data_valid and sample counts from persistent SparseProblem state. Store neither loaders nor datasets on the model, in optimizer state, or in History. Do not bypass the new input boundary through loader.dataset.tensors or concatenate all raw samples merely to call the old data-owning constructor.

Loader construction can land independently. Removal of SparseProblem's sample-owning constructor must land coherently with the already accepted nn.Module/temporary-objective migration; this plan does not prescribe an extra long-lived compatibility wrapper.

### 3. Preserve the full-data mathematics

- **Coefficient correction:** traverse the training loader once to build ordered feature/target blocks. Keep the existing feature-based representation, concatenate the feature and target blocks consistently, and retain the fitting Hessian in the temporary objective. Preserve the existing sample-major order of flattened input derivatives. Reuse that prepared objective through every SSN iteration and final acceptance check; SSN never reloads samples. Rebuild preparation after support changes. Retaining this cache does not provide out-of-core correction.
- **Insertion and warm starts:** traverse the loader to compute the current residual once per insertion phase. Keep each input batch paired with its detached residuals in a temporary collection for that search. Pass these paired batches to profile, directional-objective, warm-start, and radius calculations. Reuse them during candidate optimization, then release them when the phase ends. Do not independently re-iterate a loader and zip it with an old residual vector, and do not fetch through DataLoader inside candidate vmap/autograd transformations.
- **Candidate reductions:** combine signed profile contributions over all batches before taking an absolute value, choosing a direction, applying a threshold, or performing a proximal calculation. Accumulate full slope/curvature and radius ingredients before finalizing them. Candidate input derivatives must remain differentiable with respect to candidate parameters.
- **Objectives and measurement:** sum sample contributions and normalize by the actual total sample count; add regularization once per full objective. Relative errors combine global error/reference sums before taking ratios and roots. Preserve existing zero-denominator conventions. An uneven last batch receives its actual sample weight.
- **Training semantics:** batches control data delivery and evaluation memory, not the number of PDAP or SSN updates. Never run one coefficient solve or insertion decision per batch. New batch sizes may change floating-point reduction order; validate mathematical results with appropriate tolerances rather than promising identical trajectories in threshold-sensitive runs.

### 4. Measurement, history, and logging boundary

The loader supplies samples only. SparseProblem and its mathematical helpers supply objective/error values. PDAP determines when to request them and collects solver diagnostics. Move the current hidden problem.metrics invocation out of History.record; History receives already computed records and detached checkpoint copies. Progress output and ExperimentRun consume the same record rather than triggering another full evaluation.

Hydra remains authoritative for evaluation cadence and output settings. Runtime timing/counters belong to monitoring around training/evaluation operations, not the Dataset/DataLoader or mathematical model. A small reporting hook can connect those records to existing output machinery; a general monitoring framework is outside this integration task.

Preserve physical-coordinate reporting via the external normalizer, existing metric keys, recording positions in both loop orders, and best-by-training-loss checkpoint selection. Keep dataset identity/hash, split information, normalization metadata, resolved loader settings, and code revision in run metadata. Do not serialize live loaders, workers, datasets, or temporary feature/residual caches into checkpoints or history. Checkpoint schema and resumable-training changes remain separate work.

### 5. Migration order and verification

1. Establish the restored checkout's test baseline before implementation. Preserve the six existing uncommitted code edits and numerical fixtures.
2. Add the small data-loader factory, repository Hydra fields, and runner wiring. Preserve current file formats, preprocessing, split indices, dtype, and seed behavior. Validate explicit batch-size/worker arguments at this boundary.
3. Implement batch-aware mathematical reductions and temporary coefficient preparation with the accepted model merge. Migrate insertion.py and warmstart.py to paired residual batches, and the correction path to the prepared objective. Remove implicit dataset reads; retain SSN's generic synthetic-objective tests without manufacturing DataLoaders for them.
4. Migrate PDAP.fit and training/evaluation callers to loaders. Update metric/history plumbing so evaluation is explicit and records remain compatible. Finish the already planned ssn_solve removal as part of the broader correction migration; do not attribute that architectural decision to DataLoader.
5. Verify one-full-batch behavior against the baseline, then compare smaller and uneven batches for losses, input-gradient alignment, coefficient gradients/Hessian, signed candidate profiles and their derivatives, warm starts, radius calculations, and reported metrics. Include an example whose per-batch signed profiles cancel so an incorrect sum of absolute values is detected.
6. Verify repeatable complete coverage, train/validation separation, isolated loader RNG, constructor/prediction without samples, and absence of datasets/caches in model/optimizer/history serialization. Exercise configured worker loading with a small macOS-compatible smoke test if exposing num_workers, and check invalid sizes/drop policies fail clearly.
7. Run the affected mathematical, solver, PDAP-equivalence, result-reader, and logging tests, followed by the full suite. Inspect any changed golden outputs before accepting them. Document the new Hydra settings and API in the existing repository documentation.

This historical broader plan also proposed moving evaluation out of History. That part is deferred: the current increment passes loaders explicitly into History.record while preserving its evaluation and serialization behavior. Current implementation checks and baseline results are recorded above.

## Original repository and transfer state (historical)

At the original transfer handoff, before DataLoader implementation:

- Branch: work/problem-solver-refactor.
- HEAD: aa5ada4bb5a81976c8bcb0e0c7da5055cc83f9dc.
- Commit: Refactor shared sparse problem and solvers; revise manuscript.
- No implementation edits were made during this design discussion. This handoff is the only file created for the transfer request.
- Existing uncommitted work must be preserved. Before adding this document, git diff --stat reported 79 insertions and 92 deletions in six files:

~~~text
scripts/investigation/visualize_proximal_deadzone.py
src/problems/penalty.py
src/problems/prox.py
src/problems/sparse_problem.py
tests/test_power_prox.py
tests/test_sparse_problem.py
~~~

These edits predate the design discussion and concern proximal/penalty validation and related tests/diagnostics. They do not implement the newly agreed architecture. Inspect their actual diff on resumption; do not reset or overwrite them.

The earlier full-suite baseline recorded during this discussion was:

~~~text
uv run --frozen pytest -q
365 passed, 20 warnings, 17.19s
~~~

This was an earlier run, not a fresh validation at that handoff. The new Mac's pre-change baseline is now recorded in the implementation scope above.

**Recovery completed on 2026-10-07:** the working tree, ignored local files, original Git history/stashes, and available linked paper worktree were restored from Time Machine. The active checkout is `/Users/chaoruiz/Documents/Repos/sparse-nn-hjb` on `work/problem-solver-refactor`; the original `SparseNNforHJB` path is a local alias. At recovery time only an environment import check had run. The full pre-change baseline has since been established, as recorded above. The six pre-existing code changes were preserved exactly in prerequisite commit 49eeb0349a0a8ee489c075400d23cb1cb31e1208 before committing the DataLoader work.

## Reading map and recovered session

Start here, then inspect the actual working tree:

- [Refactoring principles](refactor.md), [domain vocabulary](CONTEXT.md), [repository map](CLAUDE.md).
- [Current ownership ADR](docs/adr/0004-model-trainer-eval-separation.md): the current-decision section records the agreed single SparseProblem and complete SSN correction. Earlier sections preserve superseded designs.
- [SparseProblem](src/problems/sparse_problem.py): replaces the former model contract and SignedModel/ShallowNetwork hierarchy, visible in Git at `edb166a`.
- [PDAP](src/PDAP/pdap.py), [warm start](src/PDAP/warmstart.py), [SSN](src/SSN/optimizer.py): the separate correction wrapper is deleted.
- [Configuration](src/config/schema.py), [activations](src/config/activations.py), [training runner](scripts/train.py).
- [History](src/PDAP/history.py), [results](src/results.py), [plots](src/plots.py).
- [Synthetic objectives](tests/ssn_test_problem.py), [SSN equivalence](tests/test_ssn_equivalence.py), [PDAP equivalence](tests/test_pdap_equivalence.py), [results tests](tests/test_results.py), [history tests](tests/test_history.py).

The [algorithm module map](vault/algorithm.md) records the implemented ownership
and manuscript scope; its verification link points back to this plan.

The initially lost session was located:

- ID: 01a0f7fe-3b6f-7b10-bff1-9169fc45f9eb.
- Title: “Explain red line in mpcg.py”; continued on October 6.
- Original-machine rollout: /Users/chaoruiz/.codex/sessions/2026/10/01/rollout-2026-10-01T17-03-49-01a0f7fe-3b6f-7b10-bff1-9169fc45f9eb.jsonl.

That session supplied the earlier implemented ownership split. The current discussion revises its architectural direction. The rollout and locally installed skills may not transfer; this handoff should suffice to resume.

Current resumption message:

> Continue the user's method-by-method review. The latest naming follow-up and
> its verification status are recorded at the top; the broader ownership cleanup
> was committed as b9fba49. Keep forward, use forward_with_gradient and
> jac_forward_with_gradient, and remove description. The Jacobian's inputs are x
> and current/supplied inner_weight,inner_bias; both returned matrices differentiate with respect to c.
> Remaining duplicated feature construction and quadratic fitting logic were
> identified after the earlier review and are not resolved by renaming.
> Read the current status and boundary table before the historical record.
> PDAP owns search/insertion/warm-start procedures; SparseProblem owns the current
> nn.Module and mathematical evaluations; SSN owns complete corrections and
> fallback. Final suite: 422 passed, one existing stale local selection failure;
> Ruff and core Pyright pass. Numerical references and tolerances are preserved.
> Mask retirement, configuration/logging redesign and known numerical bugs remain
> deferred. Root coordinates/docs only; subagents implement and review. The
> pre-cleanup source/tests snapshot is /tmp/sparse-boundary-before-y_e9gbjp.

Historical resumption note (review-only status superseded by the message above):

> Read the current plan at the top of REFACTOR_HANDOFF.md and inspect Git state. Root only plans, documents, delegates, and coordinates; subagents implement and independently review. The requested full review of all changes since 49eeb03 is complete; its findings and proposed cleanup are recorded above. No cleanup is implemented, and the working tree remains uncommitted. The user rejected excessive methods/attributes in SparseProblem; the previous 410 passing tests and earlier narrow reviews do not establish a streamlined design. Preserve the accepted one-module/external-data/temporary-preparation/Optimizer architecture, vocabulary, historical fixtures, and configuration/logging deferral. Use the recorded usage evidence to delegate any subsequently authorized cleanup rather than moving the same complexity into more classes or dropping meaningful mathematical tests. Mask support and derivative conventions require explicit treatment.

> Latest ownership correction: PDAP owns warm-start, candidate search/selection,
> insertion acceptance, and support mutation. SparseProblem supplies the network
> and mathematical evaluations. Withdraw the former proposal to move complete
> warm-start selection into SparseProblem. Mixed insertion/warm-start methods
> remain an implementation defect; the documentation correction does not fix
> the source. Do not infer user approval from assistant-authored historical text.
> The cause/prevention record distinguishes the initial misunderstanding from
> repeated failures to follow the clarified plan. Before delegating cleanup,
> attach the applicable user evidence and ownership row; independently review
> responsibility and state lifetime as well as behavior. Exact storage/default
> choices and deferred reporting placement are not additional user agreements.
