# Pendulum full experimental scope

## Algorithm 1: positive-moment nonhomogeneous model

Selected positive-moment Algorithm 1 fits

| activation | alpha | beta  | p    | gamma | N  | switching H1 | rest H1 |
| ---------- | ----- | ----- | ---- | ----- | -- | ------------ | ------- |
| gaussian   | 1e-04 | 0e+00 | 2.01 | 10    | 25 | 0.563        | 0.579   |
| softplus   | 1e-04 | 0e+00 | 2.01 | 10    | 15 | 0.807        | 0.968   |
| tanh       | 1e-04 | 0e+00 | 2.01 | 10    | 30 | 0.613        | 0.585   |

### Synthesized feedback law

Algorithm 1 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A | upright A | cost B | upright B |
| -------- | ------ | --------- | ------ | --------- |
| true PMP | 26.2   | yes       | 10.2   | yes       |
| Gaussian | 157.1  | no        | 164.2  | no        |
| softplus | 85.0   | no        | 87.7   | no        |
| tanh     | 264.7  | no        | 258.4  | no        |

Control trace: `figures/feedback_control_b_log_penalty.png`

## Algorithm 2: homogeneous ReLU model

### Synthesized feedback law

Algorithm 2 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A   | upright A | cost B | upright B |
| -------- | -------- | --------- | ------ | --------- |
| true PMP | 26.2     | yes       | 10.2   | yes       |
| ReLU^2   | 55.5     | yes       | 10.1   | yes       |
| ReLU^3   | 140002.5 | no        | 10.2   | yes       |
| ReLU^5   | 218.8    | no        | 216.1  | no        |

Control trace: `figures/feedback_control_b_relu.png`

## Cross-model diagnostics

- insertion frontier: `figures/frontier.png`
- switching/rest comparison: `figures/near_far_dumbbell.png`
- atom portrait: `figures/atom_portrait.png`
- value transect: `figures/transect_value.png`
- gradient transect: `figures/transect_normal_gradient.png`
- true branch value: `figures/transect_true_branches_value.png`
- true branch gradient: `figures/transect_true_branches_gradient.png`
- error/distance value: `figures/error_vs_distance_value.png`
- error/distance gradient: `figures/error_vs_distance_gradient.png`

### Learned surfaces

- gaussian: `figures/surface_gaussian.png`
- softplus: `figures/surface_softplus.png`
- tanh: `figures/surface_tanh.png`
- relu^2: `figures/surface_relu2.png`
- relu^3: `figures/surface_relu3.png`
- relu^5: `figures/surface_relu5.png`

### Feedback phase portraits

- true PMP: `figures/feedback_true_pmp.png`
- gaussian: `figures/feedback_gaussian.png`
- softplus: `figures/feedback_softplus.png`
- tanh: `figures/feedback_tanh.png`
- relu^2: `figures/feedback_relu2.png`
- relu^3: `figures/feedback_relu3.png`
- relu^5: `figures/feedback_relu5.png`
- control_b_log: `figures/feedback_control_b_log_penalty.png`
- control_b_relu: `figures/feedback_control_b_relu.png`

## Oversampling control

Oversampling fits are still running.

Figure: ``
