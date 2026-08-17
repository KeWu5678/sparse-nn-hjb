# 10. Use only the normalized-measure objective for Algorithm 1

Date: 2026-08-17

## Status

Accepted

## Context

The implementation exposed two objectives for the same nonhomogeneous signed
profile model:

- an additive weighted-total-variation term controlled by `moment_beta`; and
- the paper's current objective, which evaluates the scalar penalty at
  `w_p(omega) |c|`.

The former branch also covered the unweighted objective
`L + alpha * sum phi_gamma(|c_n|)` when `moment_beta=0`.  That plain comparator
was the default of the old `log_penalty` configurations; it is a special case
of the retired branch, not a third objective retained by the current version.

Supporting both required separate branches in candidate refinement, insertion
acceptance, warm-starting, coefficient correction, objective recording, Hydra
configuration, and experiment tooling. The additive formulation is no longer
part of the current manuscript. Keeping it as a runtime option makes the active
algorithm harder to read and permits configurations that do not represent the
algorithm being studied.

The additive implementation and its experiments remain available in Git history,
including commit `5132d12`.

## Decision

The normalized-measure objective is the only objective for nonhomogeneous
Algorithm 1:

```
J = l^M + alpha * sum_n phi_gamma(w_p(omega_n) |c_n|).
```

Remove `model.objective`, `model.moment_beta`, the additive runtime branches, and
the additive experiment presets and Make targets. PDAP now identifies Algorithm 1
from the model itself: signed profile insertion with a nonhomogeneous activation.
For that family it applies the substitution `u = w_p c` automatically.

Algorithm 2 remains sphere-normalized and uses its fractional-power objective.
The separate ReLU--L1 baseline also remains unchanged.

## Consequences

- A current configuration cannot accidentally select the retired formulation.
- Candidate search, insertion, warm-starting, SSN, and recorded objectives have
  one Algorithm 1 definition.
- `moment_order` remains configurable because it is a parameter of the current
  normalized objective.
- Reproducing or extending the additive study requires checking out its historical
  implementation rather than adding a compatibility branch to current code.
- Historical additive-study artifacts may remain local, but are not tracked by
  the current version.
- Historical unweighted `log_penalty` results likewise describe the retired
  `moment_beta=0` special case and must not be presented as current normalized-
  measure results.

## Alternative considered

**Keep the additive formulation as an ablation option.** Rejected because the
ablation is not part of the current paper and its branches materially complicate
the implementation. Git history is the comparison mechanism.
