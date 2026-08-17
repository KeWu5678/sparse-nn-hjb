# Fractional-power penalty — pendulum swing-up

This Hydra study exercises the current Algorithm 2 solver on the two-sided
pendulum data. It uses ReLU powers `k={2,3}`, corresponding to `q={2/3,1/2}`,
four values of `alpha`, and both value-only and value-plus-gradient losses. The
ReLU--L1 endpoint and ReLU-squared oversampling study are run separately by the
paper workflow.

Insertion minimizes the actual one-atom objective increment with the selected
global scalar prox. The coefficient correction uses the global-prox normal map
with the warm-start-derived fixed scale documented in
`vault/power_q_penalty.md`.

Run into an empty output root with:

```sh
make sweep EXPERIMENT=pendulum/frac_exp_penalty
```

The validated manuscript-facing run set and current numerical report live in
`../paper_frac_exp_penalty/`.
