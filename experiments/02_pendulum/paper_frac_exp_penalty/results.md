# Algorithm 2 — pendulum swing-up

Fresh sequential runs using the actual one-atom increment and the global-prox warm-start-scaled correction.

Selected Algorithm 2 fits

| activation | alpha | N   | switching H1 | rest H1 |
| ---------- | ----- | --- | ------------ | ------- |
| relu^2     | 1e-06 | 136 | 0.340        | 0.250   |
| relu^3     | 1e-06 | 103 | 0.409        | 0.294   |

## Synthesized feedback law

Algorithm 2 feedback: closed-loop cost and stabilization from A=(0.71, 0.68) and B=(0.23, 0.53)

| model    | cost A | upright A | cost B | upright B |
| -------- | ------ | --------- | ------ | --------- |
| true PMP | 26.2   | yes       | 10.2   | yes       |
| ReLU^2   | 76.3   | yes       | 10.3   | yes       |
| ReLU^3   | 1609.3 | no        | 10.2   | yes       |

Control trace: `figures/feedback_control_b_relu.png`
