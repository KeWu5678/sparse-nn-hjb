# Pendulum stability evaluation

## What is being claimed

The infinite-horizon control problem targets convergence to an upright
equilibrium `(theta, velocity) = (2 k pi, 0)` as time tends to infinity.
Asymptotic stability additionally requires Lyapunov stability: sufficiently
small initial perturbations remain small. Neither property is proved by a
finite simulation or by an H1 approximation error alone.

We report **numerical stabilization over a finite observation window**.
This replaces the previous terminal-only test (`angle error < 0.4 rad` and
`speed < 0.4 rad/s` at 10 seconds). A reported pass is not a certificate of
asymptotic stability or of finite infinite-horizon cost.

## Implemented criterion

`src.OpenLoop.pendulum.evaluation.assess_upright` is the shared implementation
used by the paper generators, horizon study, and manuscript consistency tests.
For a requested horizon `T`, success requires:

- A complete, finite trajectory from time 0 through a sample at `T`.
- At **every recorded sample in [T - 2, T]**, wrapped angle error is strictly
  below **0.1 rad** (about 5.7 degrees) and angular speed is strictly below
  **0.5 rad/s**.

The wrapped angle error is `abs((theta + pi) % (2*pi) - pi)`, so all periodic
upright equilibria are equivalent. A truncated or non-finite trajectory fails;
passing at the last available sample cannot substitute for reaching `T`.
Longer traces are assessed by their prefix up to `T`. No claim is made about
behavior between sampled states. `settling_time` is the start of the final
uninterrupted sequence of passing samples, reported only if the dwell test
passes; it is not an infinite-time settling guarantee.

The angle and speed tolerances match the defaults in the DFKI
[simple-pendulum benchmark](https://github.com/dfki-ric-underactuated-lab/torque_limited_simple_pendulum/blob/master/software/python/simple_pendulum/analysis/benchmark.py).
That implementation's regular-execution success flag latches at first entry.
Our **final two-second dwell requirement is an explicit study choice**, not
a criterion from Han--Yang or a universal benchmark standard. The DFKI
[RealAIGym protocol](https://dfki-ric-underactuated-lab.github.io/real_ai_gym_leaderboard/acrobot_real_system_leaderboard_v1.html)
also distinguishes entering a goal region from staying there.

## Horizon and cost checks

Use the same selected model checkpoints and initial states at **T = 10, 20,
and 40 seconds**, with held-control RK4, **dt = 0.005 seconds**, and the
existing numerical control guard **|u| <= 30**. Simulate each law once to
40 seconds and score prefixes. The manuscript's control/cost curves extend to 40 seconds, with newly solved 40-second
free-terminal-state PMP references initialized only from complete rollouts.
Divergent feedback traces are displayed only up to their actual stopping time.
Table 4 retains a labelled 10-second cost column and reports numerical stabilization
results separately at 10 and 40 seconds; the full study also reports 20 seconds.
Longer rollouts test sensitivity
of the upright outcome and residual behavior. The former 10-second PMP
solution is not extended as if it were a stationary controller.

Report success, completion, terminal error and speed, maximum angle error
and speed over the final two seconds, accumulated running cost, and mean
running cost over the same tail window. Costs use held-control RK4-stage
quadrature, without a learned terminal value. Cost at an unreached horizon
is unavailable, not replaced by partial-trajectory cost.

Check whether upright outcomes persist as `T` increases and whether tail
running costs decrease. A nonzero equilibrium offset may pass the tolerance
test but retain positive running cost; if it persists forever, the
infinite-horizon cost diverges. Failure by `T` means the finite-window
criterion was not met, not that eventual convergence is impossible.

For a theoretical convergence claim, verify an invariant attracting region
for the actual controller, accounting for any saturation and sampled control.
A certified local LQR handoff would be a different controller and must be
reported as such. See [Lyapunov analysis](https://underactuated.mit.edu/lyapunov.html).

## Controller selection

Within each activation family, first retain combinations that pass the
criterion from **both A and B at T = 40**, then choose the one with the
**smallest global physical-coordinate relative H1 validation error**.
Regional errors do not rank controllers. Each run contributes its
minimum-training-objective checkpoint; saved iterates are not additional
candidates. If no combination passes, retain the least-global-error fit
only as an explicitly failed comparator.

The available H1-trained fits share one production dataset: 24 runs per
nonhomogeneous activation across the log-penalty and moment-order sweeps,
four per rectified power, and five ReLU-L1 runs. The selector checks runs
in ascending global H1 order and stops at the first pass in each family.
This finds the same minimum as testing every run; 38 of the 85 available
runs needed rollouts. A failed A rollout already excludes a candidate.

`controller_selection.json` records the chosen fits, tested outcomes, and
SHA-256 hashes of every candidate record and fit file. Figure and stability
generators consume this selection and reject changed inputs until selection
is rerun. A/B now serve as selection states: their outcomes are not an
independent test of control generalization or a population success rate.

## Reproduce

With the current local paper scripts, datasets, and trained fits available:

```sh
uv run python scripts/paper/pendulum_selection.py
uv run python scripts/paper/pendulum_full_scope.py
uv run python scripts/paper/pendulum_stability.py
uv run pytest tests/test_pendulum_stability.py tests/test_paper_feedback_results.py
```

The runner uses the same model selection, feedback laws, and geometric starts
A/B as the paper, plus the ReLU-L1 baseline and sampled-PMP heuristic. It does
not retrain controllers; selection is performed by the first command.

Outputs live under `experiments/02_pendulum/paper_log_penalty/`:
`stability.md`, `stability.json` (configuration, starts, source record hashes,
and numerical metrics), and `stability_rollouts.npz` (full traces and costs).
These paper-specific scripts and outputs remain local/ignored under ADR 0011;
the shared evaluator, tests, this protocol, and the manuscript are tracked.
To check integration sensitivity without overwriting the primary results:

```sh
uv run python scripts/paper/pendulum_stability.py --dt 0.0025 \
  --out experiments/02_pendulum/paper_log_penalty/stability_half_step
```

## Current selected fits (September 27, 2026)

At all three horizons, the sampled-PMP heuristic passes from A and B.
Softplus, tanh, ReLU², and the ReLU-L1 baseline pass from A at all three
horizons. All five selected nonconvex fits pass from B at all three horizons;
the baseline passes from B at 20 and 40 seconds, but fails at 10 seconds.
No available Gaussian or ReLU³ combination passes from both starts at 40
seconds, so their selected fits are failed comparators.
Halving the integration/control-hold step
preserves all 42 classifications (seven laws, two starts, three horizons).

| Family | alpha | gamma | p | Global H1 | Stabilized from both at T = 40 |
|---|---:|---:|---:|---:|---|
| Gaussian | 1e-4 | 10 | 2.01 | 0.2094 | no |
| softplus | 1e-5 | 10 | 2.01 | 0.2743 | yes |
| tanh | 1e-5 | 0 | 2.01 | 0.2322 | yes |
| ReLU² | 1e-5 | — | — | 0.2001 | yes |
| ReLU³ | 1e-6 | — | — | 0.2476 | no |
| ReLU-L1 | 1e-2 | — | — | 0.4212 | yes |

From A, softplus, tanh, and ReLU² retain mean tail running costs about
0.1042, 0.0836, and 0.5671 at 40 seconds. ReLU²'s accumulated cost grows
from 86.93 at 10 seconds to 92.60 at 20 and 103.95 at 40. Passing is
compatible with a persistent nonzero offset, not evidence of finite
infinite-horizon cost.
