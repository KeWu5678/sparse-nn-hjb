# Pendulum moment control: beta at a fixed operating point

**Status: complete.** Four seed-42 records hold `softplus`, alpha=1e-5, gamma=10, p=2.01, H1 loss `[1,1]` fixed and vary only beta. The beta=0 row reproduces the configuration the earlier log-penalty study selected, so the rows differ in the moment weight alone.

## Fits

| beta  | N  | val H1 | switching H1 | rest H1 | R95 | R max | Phi_1 | Psi_p    |
| ----- | -- | ------ | ------------ | ------- | --- | ----- | ----- | -------- |
| 0e+00 | 77 | 0.5149 | 0.5323       | 0.3865  | 1   | 148   | 674   | 3.43e+04 |
| 1e-10 | 69 | 0.5299 | 0.5409       | 0.4348  | 1   | 148   | 291   | 3.35e+04 |
| 1e-05 | 34 | 0.7106 | 0.6765       | 0.7172  | 1   | 148   | 127   | 5.05e+03 |
| 1e-02 | 7  | 0.8571 | 0.8886       | 0.9056  | 2.2 | 2.2   | 3.43  | 25.9     |

## Synthesized feedback

Closed-loop cost and stabilization from A and B, the two sides of the switching curve, under the full-scope rollout protocol (RK4, T=10, dt=0.005, |u| <= 30).

| law          | cost A  | upright A | cost B  | upright B |
| ------------ | ------- | --------- | ------- | --------- |
| true PMP     | 26.2    | yes       | 10.2    | yes       |
| beta = 0e+00 | 73013.3 | no        | 66996.7 | no        |
| beta = 1e-10 | 168.3   | no        | 170.2   | no        |
| beta = 1e-05 | 87.5    | no        | 92.2    | no        |
| beta = 1e-02 | 82.6    | no        | 86.0    | no        |
