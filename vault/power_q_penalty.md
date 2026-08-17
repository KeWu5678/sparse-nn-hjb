# Fractional-power coefficient solver

Algorithm 2 uses `ReLU^k` atoms on the unit sphere and the coefficient penalty
`alpha * sum |c_i|^q`, where `q = 2/(k+1)`. The implemented fractional cases
are:

| `k` | `q` | scalar solver |
|---:|---:|---|
| 2 | 2/3 | closed-form global prox |
| 3 | 1/2 | closed-form global prox |

The `k=1`, `q=1` ReLU--L1 endpoint uses ordinary soft thresholding. Other
powers are rejected at trainer construction.

## Scalar global prox

For input `v` and scale `mu > 0`, the selected proximal point minimizes

```text
0.5 * (t - v)^2 + mu * |t|^q.
```

For `q < 1`, define

```text
t_switch = [2 * mu * (1-q)]^(1/(2-q))
v_switch = (2-q) * t_switch / [2 * (1-q)].
```

The implementation returns zero when `|v| <= v_switch`, including the exact
tie. Above the switch it returns `sign(v) * t`, where `t` is the largest
positive solution of

```text
t + mu * q * t^(q-1) = |v|.
```

The `q=1/2` root is evaluated through a depressed cubic in `sqrt(t)`; the
`q=2/3` root is evaluated through a depressed quartic in `t^(1/3)`. The
implementation nondimensionalizes by `mu^(1/(2-q))` before evaluating either
formula, which preserves scale covariance and avoids overflow at the scales
covered by the tests.

## One-atom insertion

For a candidate node, let `P` be the fidelity profile and `A` its sampled
feature curvature. The actual objective increment is

```text
Delta(c) = c * P + 0.5 * A * c^2 + alpha * |c|^q.
```

Completing the square gives the selected global minimizer

```text
c_star = prox_{(alpha/A)|.|^q}(-P/A).
```

The candidate is accepted only when `c_star` is nonzero and `Delta(c_star) < 0`.
The accepted coefficient is also the warm start for the outer-weight
correction.

## Warm-start-scaled normal map

For `q < 1`, let `m` be the smallest nonzero penalized coefficient at the start
of a correction. The proximal scale is fixed for the whole SSN solve:

```text
prox_scale = min(
    alpha / (1 + alpha * gamma),
    rho * m^(2-q) / [2 * (1-q)],
)
```

Algorithm 2 has `gamma=0` and uses `rho=0.5`, selected by the recorded 24-cell
pilot in `experiments/algorithm2_rho_pilot.md`. The bound places every nonzero
warm-start coefficient strictly above the global-prox output jump. The normal
map uses `inverse_step = alpha / prox_scale`; the scale is not retuned during
the inner SSN iterations.

The global prox is discontinuous at its switching input. Consequently the
semismooth Newton claim is local and conditional: proximal inputs must stay
away from the switch and the selected generalized Jacobian must be nonsingular.
The outer correction guard remains authoritative and rolls back a correction
that increases the post-insertion objective.

## Source and tests

- `src/SSN/prox.py`: closed-form global prox and derivative.
- `src/SSN/optimizer.py`: normal map and generalized Jacobian.
- `src/PDAP/insertion.py`: actual one-atom increment and warm coefficient.
- `src/PDAP/ssn_solve.py`: fixed warm-start scale and correction diagnostics.
- `tests/test_power_prox.py`: switch, roots, derivatives, zero scale, and scaling.
- `tests/test_algorithm2_insertion.py`: exact increment and tie rejection.
- `tests/test_algorithm2_correction.py`: warm scale and SSN integration.
