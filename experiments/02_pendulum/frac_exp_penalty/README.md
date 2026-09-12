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

This sweep's config (`conf/experiment/pendulum/frac_exp_penalty.yaml`) was retired on 2026-09-10;
only the `paper_*` experiments are runnable now. The archived records under
`rawdata/logs/multirun/pendulum/frac_exp_penalty/` remain readable by `analysis.py`.

The validated manuscript-facing run set and current numerical report stay local
under `../paper_frac_exp_penalty/`.
