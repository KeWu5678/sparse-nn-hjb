# Historical VDP cross-experiment summary

This superseded report retains the original run selection and numbers. Its
relative errors were stored in normalized training coordinates; they have not
been rescored as errors of the original-scale value function and gradient.
The objectives and supported powers also predate the current solver. Do not
compare these numbers or rankings with the [current paper](../../../paper/paper_0805.pdf).
All figures described below are archived locally and are not distributed with
this repository.

Historical comparison at the fixed operating point **α = 1e-5**. Here “Algorithm 1” =
profile insertion + log penalty (**γ = 1**), activations tanh/softplus/gaussian
(from `../log_penalty`). Algorithm 2 = finite-step insertion + power penalty
(**γ = 0**), ReLU^2/ReLU^5 (from `../frac_exp_penalty`). These are historical
labels, not an identification with the current paper's algorithms.

Historical champion runs (lowest stored validation error at the fixed point)

| method | historical algorithm | neurons | stored rel H1 (training coordinates) |
| ------ | --------- | ------- | ------ |
| tanh | Algo 1 (profile, γ=1) | 66 | 0.314 |
| softplus | Algo 1 (profile, γ=1) | 27 | 0.292 |
| gaussian | Algo 1 (profile, γ=1) | 113 | 0.099 |
| ReLU^2 | Algo 2 (finite-step, γ=0) | 41 | 0.098 |
| ReLU^5 | Algo 2 (finite-step, γ=0) | 21 | 0.104 |

## Frontier — sparsity at equal accuracy

The archived curves show each champion's insertion trajectory (neurons versus
cumulative-minimum stored relative H1). In that historical metric, ReLU^5 reached
approximately 0.10 with 21 neurons and Gaussian with 113. This is not a
current original-coordinate accuracy-per-neuron comparison.

## Feedback — both algorithms stabilize

Historical closed-loop rollout from y₀=(2, 1) under the synthesized feedback
û(x) = −∂_{x₂}V̂/(2β), beside the reference control.

| controller | neurons | stabilizes? | closed-loop cost |
| ---------- | ------- | ----------- | ---------------- |
| reference | — | yes | 6.48 |
| softplus | 27 | yes | 6.68 |
| gaussian | 113 | yes | 6.51 |
| relu5 | 21 | yes | 6.49 |

These archived runs stabilized the system near the reference rollout cost.
The table is historical rollout evidence, not a claim of certified optimality
or a comparison of the current paper's selected models.

## Weights — archived structural portraits

The learned atoms differ structurally: Algorithm 2 constrains them to the unit
sphere S², Algorithm 1 does not (gaussian spans a huge norm range). This is a
*portrait*, not evidence of the cause of an accuracy/sparsity gap.
In the archived figures, dot color = sign of the outer
weight, size ∝ |outer weight|.

**Variant A — stereographic projection of S²** (atoms radially projected onto the
sphere; green circle = equator):

The three archived panels compare Gaussian, softplus and ReLU^5.

**Variant B — raw (a₁, a₂, b) with unit-sphere wireframe** (ReLU on the sphere,
Algo-1 scattered off it):

The same three models are shown in the archived raw-coordinate panels.
