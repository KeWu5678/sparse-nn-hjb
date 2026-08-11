# Pendulum full experimental scope

## Algorithm 1: positive-moment nonhomogeneous model

Selected positive-moment Algorithm 1 fits

| activation | alpha | beta  | p    | gamma | N  | switching H1 | rest H1 |
| ---------- | ----- | ----- | ---- | ----- | -- | ------------ | ------- |
| gaussian   | 1e-04 | 1e-04 | 2.01 | 0     | 62 | 0.542        | 0.506   |
| tanh       | 1e-04 | 1e-05 | 3    | 1     | 43 | 0.540        | 0.480   |
| softplus   | 1e-04 | 1e-10 | 2.01 | 1     | 81 | 0.522        | 0.441   |

### Synthesized feedback law

Algorithm 1 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A | upright A | cost B | upright B |
| -------- | ------ | --------- | ------ | --------- |
| true PMP | 26.2   | yes       | 10.2   | yes       |
| Gaussian | 186.0  | no        | 15.2   | yes       |
| softplus | 89.2   | yes       | 28.2   | yes       |
| tanh     | 434.5  | no        | 18.9   | yes       |

Control trace: `figures/feedback_control_b_log_penalty.png`

## Algorithm 2: homogeneous ReLU model

### Synthesized feedback law

Algorithm 2 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A  | upright A | cost B | upright B |
| -------- | ------- | --------- | ------ | --------- |
| true PMP | 26.2    | yes       | 10.2   | yes       |
| ReLU^2   | 57.9    | yes       | 10.3   | yes       |
| ReLU^3   | 29261.8 | no        | 10.3   | yes       |
| ReLU^5   | 235.9   | no        | 232.4  | no        |

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

Common-set relative H1 error for the switching-best run (three alpha values per variant; all entries come from one run)

| family   | variant        | runs | switching | rest  | N   |
| -------- | -------------- | ---- | --------- | ----- | --- |
| Gaussian | 6k 23% (base)  | 3    | 0.625     | 0.612 | 69  |
| Gaussian | 6k 40% band    | 3    | 0.605     | 0.602 | 106 |
| Gaussian | 6k 60% band    | 3    | 0.617     | 0.699 | 91  |
| Gaussian | 6k+2k band add | 3    | 0.617     | 0.653 | 88  |
| ReLU^2   | 6k 23% (base)  | 3    | 0.246     | 0.156 | 108 |
| ReLU^2   | 6k 40% band    | 3    | 0.289     | 0.172 | 131 |
| ReLU^2   | 6k 60% band    | 3    | 0.346     | 0.235 | 109 |
| ReLU^2   | 6k+2k band add | 3    | 0.288     | 0.184 | 131 |

Figure: `figures/oversampling_control.png`
