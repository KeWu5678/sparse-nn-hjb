# Algorithm 1 log penalty — pendulum

This study applies Algorithm 1 to the pendulum swing-up value function on the
two-sided switching-band data. `analysis.py` writes the numerical summary to
`results.md`; the smooth counterpart is `../../01_vdp/paper_log_penalty`.

## Method

The objective is the empirical fidelity plus
`alpha * sum phi_gamma(w_p(omega_n) * |c_n|)`, where
`w_p(omega)=1+|omega|^p`.

At every insertion iteration, Algorithm 1:

1. computes `R_search=min(R(mu), exp(5))`, falling back to `exp(5)` when the
   theorem radius is unavailable;
2. draws `N_trial` random starts with a uniform direction and a log-uniform
   radius between `exp(-3)` and `R_search`;
3. runs one unconstrained joint L-BFGS maximization of `|P_p(omega)|` from each
   start;
4. discards final points outside `R_search`, then removes Euclidean
   near-duplicates with tolerance `1e-2`;
5. retains candidates satisfying `|P_p(omega)| > alpha*L_phi`, initializes each
   outer coefficient from its certificate violation, and optimizes all outer
   weights with the guarded semismooth Newton correction.

## Sweep

| axis | values |
|---|---|
| activation | leaky_relu, softplus, tanh, gaussian, gausscent_1, matern52, gelu_squared |
| alpha | 1e-2, 1e-3, 1e-4, 1e-5 |
| gamma | 0, 0.1, 1, 10 |
| loss weights | `[1,0]` and `[1,1]` |
| insertion mode | batch and sequential |

Fixed settings are `power=1`, `p=2.01`, normalized data, and seed 42. Batch
insertion uses at most 15 candidates against one residual and 10 outer
iterations. Sequential insertion retains one candidate per iteration and uses a
150-iteration neuron budget. The dedicated p-study varies
`p in {2.01,2.5,3,4}`.

The pendulum evaluator additionally reports errors on the certified switching
neighbourhood and its complement. The oversampling study reruns the same
sequential method on four sampling densities near the switching set.

## Reproduce

```sh
make paper-sweep EXPERIMENT=pendulum/paper_log_penalty
make paper-p-study EXPERIMENT=pendulum/paper_log_penalty
```
