# Pendulum full experimental scope

## Algorithm 1: normalized-measure nonhomogeneous model

Selected Algorithm 1 fits

| activation | alpha | p    | gamma | N  | switching H1 | rest H1 |
| ---------- | ----- | ---- | ----- | -- | ------------ | ------- |
| gaussian   | 1e-04 | 2.01 | 10    | 77 | 0.304        | 0.191   |
| softplus   | 1e-04 | 2.01 | 10    | 30 | 0.501        | 0.417   |
| tanh       | 1e-04 | 2.01 | 10    | 75 | 0.358        | 0.197   |

### Synthesized feedback law

Algorithm 1 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A | upright A | cost B | upright B |
| -------- | ------ | --------- | ------ | --------- |
| true PMP | 26.2   | yes       | 10.2   | yes       |
| Gaussian | 8451.8 | no        | 10.1   | yes       |
| softplus | 69.8   | yes       | 11.2   | yes       |
| tanh     | 7822.2 | no        | 10.2   | yes       |

Control trace: `figures/feedback_control_b_log_penalty.png`

## Algorithm 2: homogeneous ReLU model

### Synthesized feedback law

Algorithm 2 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A   | upright A | cost B | upright B |
| -------- | -------- | --------- | ------ | --------- |
| true PMP | 26.2     | yes       | 10.2   | yes       |
| ReLU^2   | 64.9     | yes       | 10.3   | yes       |
| ReLU^3   | 140002.5 | no        | 10.2   | yes       |
| ReLU^5   | 117.8    | yes       | 149.6  | yes       |

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

| family   | variant        | runs | alpha | switching | rest  | N   |
| -------- | -------------- | ---- | ----- | --------- | ----- | --- |
| Gaussian | 6k 23% (base)  | 3    | 1e-04 | 0.278     | 0.136 | 73  |
| Gaussian | 6k 40% band    | 3    | 1e-05 | 0.276     | 0.158 | 143 |
| Gaussian | 6k 60% band    | 3    | 1e-05 | 0.299     | 0.179 | 145 |
| Gaussian | 6k+2k band add | 3    | 1e-04 | 0.296     | 0.165 | 79  |
| ReLU^2   | 6k 23% (base)  | 3    | 1e-05 | 0.238     | 0.146 | 110 |
| ReLU^2   | 6k 40% band    | 3    | 1e-06 | 0.277     | 0.170 | 133 |
| ReLU^2   | 6k 60% band    | 3    | 1e-05 | 0.310     | 0.189 | 112 |
| ReLU^2   | 6k+2k band add | 3    | 1e-06 | 0.246     | 0.160 | 133 |

Figure: `figures/oversampling_control.png`
