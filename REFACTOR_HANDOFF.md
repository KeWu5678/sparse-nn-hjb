# Refactor discussion handoff

Updated: 2026-10-07. The repository has been recovered from Time Machine on the replacement Mac, and the design discussion has resumed.

## Start here

**Status: DataLoader increment implemented and verified.** The user invoked implement and codebase-design and explicitly authorized implementation and delegation. Configuration and logging redesign are deferred. The remaining model/optimizer refactor decisions below are still targets, not claims about implemented code.

The objective is to clarify ownership among **SparseProblem, SSN, and PDAP**, remove duplicated network/problem representations, and build the centralized problem directly on PyTorch.

The governing principle, explicitly stated by the user:

> The refactoring doesn't need to force itself to the current pattern, best practice and principles are the guidelines.

Existing classes, wrappers, inheritance, signatures, and file boundaries describe current behavior; they do not constrain the target architecture. Each retained layer must justify its responsibility. Structural redesign is not blanket authorization to change the mathematical algorithm or historical results. Numerical and artifact compatibility still need discussion.

**Resume with the implementation scope below.** Do not re-ask accepted decisions. Configuration remains a repository-level Hydra concern; measuring, monitoring, history, and logging are separate responsibilities. The DataLoader increment is complete; use the remaining accepted decisions below for subsequent work. Investigate routine technical details without another architecture questionnaire.

**Communication preference:** the user explicitly requested no LaTeX because it is unreadable in their interface. Use plain language, ordinary text, and small code examples or tables when useful; do not use rendered mathematical notation in replies.

## Current discussion point

### Authorized implementation increment

The latest user instruction is to formulate the plan and implement now, using the implement and codebase-design skills and subagents as useful. This supersedes the earlier request to pause implementation. The immediately preceding scope correction explicitly defers configuration and logging work.

Implement standard TensorDataset/DataLoader construction, explicit loader arguments through current training callers, centralized full-data assembly, and one temporary coefficient objective per correction. Preserve the current separate model representation, optimizer iteration/acceptance behavior, logging outputs, History format, normalization/split policy, and numerical formulas. Do not implement the broader nn.Module merge or remove ssn_solve in this increment. Those accepted later changes are independent of adopting DataLoader; the scoped increment supersedes the earlier plan's proposed simultaneous migration.

The implementation plan for this increment is:

1. Add make_dataloader(tensor_samples, ...) and make_dataloaders(prepared_samples, train_fraction, ...) to src/data.py. Return actual PyTorch loaders, use complete sequential batches, default to a full split, and isolate each loader's Torch generator. Keep existing reading/normalization/splitting functions and NumPy split semantics.
2. Remove sample-owning constructor arguments from SparseProblem; provide input_dim explicitly. Pass loaders to residual, model_loss, metrics, and coefficient preparation. Use data.full_batch to assemble delivered samples once per evaluation. Keep candidate inputs paired with detached residuals as one (X, residual_v, residual_dv) tuple for one insertion phase; retain the existing full-data mathematical formulas.
3. Keep every coefficient, penalty, proximal, and normal-map formula on SparseProblem. SparseProblem.prepare_coefficients(model, train_loader) returns a small frozen PreparedObjective record containing obj, hess_obj, and coefficient_scale; the fitting closure is defined by SparseProblem. SSN receives both problem and prepared data explicitly. There is no second mathematical problem class, inheritance hierarchy, or duplicate problem settings. Synthetic objectives use the same record without a dummy network or dataset.
4. Use PDAP.fit(model, problem, train_loader, valid_loader=None, ...) for this increment. Adapt ssn_solve to consume the returned objective while preserving its existing iteration behavior and PDAP correction guard. History.record receives explicit loaders but retains its current evaluation behavior; redesigning history/logging is deferred.
5. Migrate the runner and existing tests. Test the agreed public seams: delivered batches and random-state isolation; prepared objective value/gradient/Hessian; full-data metrics, candidate and warm-start calculations; PDAP integration and numerical summaries. Preserve old fixtures and distinguish existing failures from regressions.

The user corrected the initial implementation: clarify ownership and remove duplication instead of adding layers, and follow refactor.md terminology. coefficient_objective.py was removed entirely; its mathematical methods remain on SparseProblem. Only a three-field PreparedObjective data record is passed separately to SSN. Established names obj, hess_obj, and coefficient_scale are retained. Repeated per-batch mathematical reductions and PDAP loader validation were also removed. DataLoader delivers samples; data.full_batch assembles the complete tensors required by this algorithm. Smaller delivery batches do not make the solver streaming or reduce its full-data memory requirement.

No Hydra schema additions, reporting callbacks, monitoring framework, artifact-schema changes, or new convergence rules belong to this increment. Loader settings are explicit Python arguments for now. The training runner uses full-split batches and zero workers by default.

Fresh pre-change baseline on this Mac: **363 passed, 2 failed, 20 warnings** in 39.39 seconds. Existing failures are a stale local controller_selection.json artifact and the signed_profile PDAP golden summary. The pre-change signed_profile values were L2=0.020708285737220288, H1=0.05464074122149227, loss=0.0028238830405932294. Do not regenerate fixtures or hide these failures. The pre-existing six-file worktree patch and copies were preserved in a temporary snapshot before implementation.

Final validation after removing the extra objective class: **398 passed, 2 failed, 20 warnings** in 32.35 seconds. Both failures match the pre-change baseline; signed_profile retains exactly the same three values above. Pyright reports zero errors on all changed production modules checked; Ruff and Git whitespace checks pass. Standards review (including refactor.md naming) and Spec review have no outstanding findings. A review-found issue with verbose logging on finite loaders without a sample count was fixed and tested. No fixtures were regenerated. Configuration/logging redesign and the broader model/SSN refactor remain deferred.

The latest accepted design decision is to **adopt standard PyTorch DataLoader**, keeping repository configuration in Hydra and defining measurement/monitoring/logging separately. The earlier recommendation to retain tensor tuples as the public training input has been superseded by this explicit user choice; do not reopen DataLoader adoption.

The existing decision to **retain feature-based fitting evaluations and the cached fitting Hessian inside the temporary prepared objective** also stands. Compact quadratic storage is a separate potential performance optimization, not part of this agreed change.

Fixed-iteration SSN correction without a required convergence criterion is also accepted. Run the configured number of SSN iterations (current default 20), then perform the final objective comparison and return the accepted coefficients or the starting coefficients. Report iterations and correction acceptance; completing the budget is a normal outcome, without a convergence claim. Optional early stopping can be addressed later.

Retaining SSN's torch.optim.Optimizer inheritance and removing the legacy ssn_solve wrapper are also settled. Do not reopen these choices.

The complete-solve guarantee is also already accepted:

**SSN owns the final non-increase check for the complete coefficient correction.** If its final coefficients worsen the full objective relative to its starting coefficients, SSN returns the starting coefficients and reports that outcome. PDAP applies the returned coefficients without repeating the acceptance check. The user explicitly agreed to this, superseding the earlier placement of final acceptance in PDAP.

The temporary prepared objective returned by SparseProblem is also already agreed. Do not reopen its ownership/lifetime question.

The SSN ownership, PyTorch integration, and primary stopping policy are settled. Resolve routine implementation mechanics without another architecture questionnaire. Remaining numerical safeguards/configuration compatibility and the independently discovered residual discrepancy still need deliberate handling; do not use optional convergence criteria to block discussion of the remaining design.

After the later model merge, the target training entry point is trainer.fit(problem, train_loader, valid_loader). This increment still passes model separately, as specified above. The runner constructs standard TensorDataset/DataLoader objects from externally prepared samples. SparseProblem obtains full samples through data.full_batch for mathematical evaluation and preparation; it neither constructs nor retains loaders. SSN receives SparseProblem and its temporary PreparedObjective data; it receives no loader or raw samples. DataLoader owns fetching, ordering, and batching; residuals, features, and objective reductions remain problem mathematics.

Every prepared correction and mathematical evaluation still covers the full required dataset. Centralized assembly preserves data/target/residual alignment, total-sample normalization, and one penalty per complete objective without repeating batch reductions throughout the mathematics. Feature matrices for the complete correction remain in memory as agreed; DataLoader does not make that cache independent of dataset size. Delivery defaults to one full-split batch; smaller batches are assembled into the same full tensors and checked for equivalence. No stochastic per-batch optimizer updates are introduced.

The recommendation to remove Optimizer inheritance has been withdrawn after checking PyTorch's implementation. L-BFGS uses one parameter group, stores global numerical state under the first parameter, flattens parameters for its calculations, and runs multiple iterations inside step(). The SSN bookkeeping patterns previously cited as reasons to remove inheritance are therefore normal optimizer patterns, not evidence of incompatibility.

SSN remains an Optimizer subclass and uses its parameter handling, settings, and numerical state directly. A complete correction operates on a temporary coefficient tensor registered with the optimizer. Recommend exposing the complete correction through step(), with internal iterations as in L-BFGS, rather than introducing another public solve wrapper. The exact signature and result reporting are implementation details still to specify; step() performing the complete correction has not yet been separately agreed. Do not reopen inheritance removal merely because SSN handles one temporary vector or runs a complete solve.

The agreed complete-solve guarantee still needs an explicit numerical tolerance, termination reporting, and compatibility treatment for the existing correction_guard setting. Current SSN strategies already check the full objective at trial points, but allow small numerical increases; those checks alone do not establish a strict non-increase contract for the complete solve.

### Optimizer implementation outline

- Register the temporary coefficient tensor with SSN(Optimizer), use standard param_groups/state, and give SSN the prepared mathematical operations returned by SparseProblem directly. Preserve generic synthetic objectives; do not require a network or a chain of adapters.
- Move complete iteration control, stopping, and solve diagnostics into SSN. Retain the existing Newton/trust-region calculations while changing ownership. Recommend one public step() for the complete correction.
- Keep the starting coefficients available for the agreed final comparison and fallback. SSN's numerical working tensor is separate from the network's c. Coefficient transformations and other problem-specific calculations remain supplied by SparseProblem through its prepared objective.
- PDAP requests preparation, invokes SSN, and applies the resulting coefficients. Remove ssn_solve and migrate its configuration definitions, callers, tests, and relevant logging to their owners.
- Before implementation, establish the restored checkout's test baseline. Validate the prepared objective against existing evaluations, solver behavior for both strategies, the final non-increase guarantee, exceptions without network mutation, and coefficient normalization/support changes. Preserve existing golden fixtures unless an explicitly agreed numerical change explains a difference.

Stopping after a configured number of SSN iterations is sufficient for a budgeted correction followed by the agreed objective comparison. Reaching that budget is ordinary completion, not an error or proof of nonconvergence. Without a validated convergence check, report iterations and correction acceptance rather than asserting that the returned point converged or failed to converge. The current defaults are fit_outer_iterations = 20 and tolerance_grad = 0, which disables residual-based early exit.

A convergence criterion is useful if early stopping or an explicit accuracy report is wanted, but need not be established merely to execute a fixed-budget correction. The existing optional check and its configuration compatibility still need deliberate treatment: inspection reproduced a false-zero residual at inactive coefficients. That mathematical issue also affects the Newton equation, so fixed-budget execution does not resolve it. Investigate it as a numerical-correctness issue separately from deciding whether early stopping is needed.

Candidate for optional convergence reporting, not accepted or required for fixed-budget execution: evaluate the scaled proximal fixed-point residual at the actual working coefficients, r(u) = (u - P_tau(u - tau * grad(f)(u))) / tau, and require it to be finite with ||r(u)||_infinity <= epsilon. Here f is obj_augmented and P_tau is the selected proximal map for the remaining penalty/constraints, with tau = 1 / inverse_step fixed for the solve. Its zeros correspond to the paper's proximal/normal-map equation with consistent coefficient and preimage values. SparseProblem supplies this mathematical evaluation through the prepared objective; SSN applies the numerical stopping rule. The norm scaling and tolerance would need explicit treatment if this feature is adopted. This tests the coefficient equation for fixed neurons and is not a claim of a global minimum.

## Accepted decisions

### External datasets and data loading

Training and validation datasets remain outside SparseProblem. The user has now explicitly chosen standard PyTorch DataLoader integration. Use Dataset/DataLoader for sample access and batch delivery; keep Hydra configuration at repository level and measurement/monitoring/logging separate. Passing batches to mathematical evaluations does not imply storing them permanently on the model.

A DataLoader does not itself make already-resident tensors out-of-core; the Dataset/loading implementation determines that. Current PDAP/SSN solve the full empirical objective. Chunking must aggregate the full contributions; stochastic minibatch updates would change the algorithm.

Feature-based fitting evaluations and the cached fitting Hessian in the temporary prepared objective are agreed. The current implementation scope is recorded above; the older broader plan below also contains deferred model, configuration, and logging changes.

### Temporary prepared coefficient objective

SparseProblem constructs and returns a prepared objective for the current fixed neurons and supplied training data. It does not retain the prepared feature matrices, coefficient Hessian, scaling, and masks as persistent attributes on the network object.

PDAP keeps the returned objective available for one complete coefficient correction and supplies the same instance to SSN. Its calculations are reused across all inner SSN iterations and SSN's final acceptance check. The object can be released when that correction finishes.

This is derived calculation data and mathematical operations, not a proposed network or a backup of its weights. The working coefficient vector used by SSN is separate. All formulas remain implemented on SparseProblem; the temporary record does not reimplement them. Feature matrices used to test candidate insertions are also a separate concern; temporary values inside a SparseProblem evaluation method do not become persistent attributes merely because that method created them.

The prepared calculations remain valid while the neurons and relevant data/settings remain fixed; changing coefficients alone does not invalidate them. The user accepted retaining feature-based fitting evaluations and the already-cached fitting Hessian in this temporary objective. Do not replace them with expanded quadratic loss evaluations during the structural refactor. Closures versus a record versus a class, batching policy, and reuse across correction attempts remain implementation questions. Compact storage can be considered separately based on measured memory, speed, and numerical accuracy.

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

**Historical broader plan:** the authorized increment at the top of this document supersedes this plan where it proposed simultaneous model migration, new Hydra settings, and changed history/logging ownership. Those portions are deferred. DataLoader adoption and full-data mathematical requirements remain in force.

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
- [Current ownership ADR](docs/adr/0004-model-trainer-eval-separation.md): its latest section describes the existing implementation and keeps model state outside SparseProblem. The target direction has changed in this discussion; the ADR has not yet been updated.
- [SparseProblem](src/problems/sparse_problem.py), [model contract](src/models/base.py), [SignedModel](src/models/signed.py), [ShallowNetwork](src/models/net.py).
- [PDAP](src/PDAP/pdap.py), [correction wrapper](src/PDAP/ssn_solve.py), [warm start](src/PDAP/warmstart.py), [SSN](src/SSN/optimizer.py).
- [Configuration](src/config/schema.py), [activations](src/config/activations.py), [training runner](scripts/train.py).
- [History](src/PDAP/history.py), [results](src/results.py), [plots](src/plots.py).
- [Synthetic objectives](tests/ssn_test_problem.py), [SSN equivalence](tests/test_ssn_equivalence.py), [PDAP equivalence](tests/test_pdap_equivalence.py), [results tests](tests/test_results.py), [history tests](tests/test_history.py).

The vault algorithm documentation also has stale references; revise it after the architecture is settled.

The initially lost session was located:

- ID: 01a0f7fe-3b6f-7b10-bff1-9169fc45f9eb.
- Title: “Explain red line in mpcg.py”; continued on October 6.
- Original-machine rollout: /Users/chaoruiz/.codex/sessions/2026/10/01/rollout-2026-10-01T17-03-49-01a0f7fe-3b6f-7b10-bff1-9169fc45f9eb.jsonl.

That session supplied the earlier implemented ownership split. The current discussion revises its architectural direction. The rollout and locally installed skills may not transfer; this handoff should suffice to resume.

Suggested resumption message:

> Read REFACTOR_HANDOFF.md, starting with the current implementation scope and validation results, and inspect Git state. The user authorized the DataLoader increment and deferred configuration/logging redesign. Source implementation now uses external loaders, problem-owned mathematics, and a three-field PreparedObjective record; model merging and the complete SSN solve refactor remain future work. Preserve agreed architecture, use plain text rather than LaTeX, and do not reopen DataLoader adoption. Check the two documented pre-existing test failures separately from regressions. The false-zero normal-map residual still requires a separate numerical investigation. Follow the user's next instruction about the remaining refactor.
