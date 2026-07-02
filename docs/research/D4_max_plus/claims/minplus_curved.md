# minplus_curved

**status:** open

**statement.** For $V$ with a **curved** switching set (where the ridge cone fails,
`correction_cost`/`curved_switching_rnorm`), the **min-of-paraboloids** architecture
$V=\min_{i\le m}\varphi_i$ ($\varphi_i$ quadratic, $D^2\varphi_i\preceq CI$) attains a finite count with a
budget→rate bound: $N_{\min}(V,\varepsilon)<\infty$ and small for HJB value functions, so re-composing
with `signed_lower_bound` gives the divergent separation on curved switching sets.

**status of pieces.**
- **succeeds on $v_d$** (min of $2d$ paraboloids: finite atoms, curved switching) where the
  ridge net diverges — the clean "min beats ridge" instance (`curved_switching_rnorm`).
- **not universal:** rotationally-symmetric $g=(\|x\|-1)_+$ likely needs a *continuum* of
  paraboloids in min too. So min suffices only for **finitely-min-representable** $V$.
- **open:** characterize which HJB $V$ are finitely-min-representable (max-plus rank), and
  prove the budget→rate bound there.

**uses (when closed).** `signed_lower_bound`, max-plus rank theory (McEneaney, Gaubert).

**caveat.** This is a **different architecture** from the repo cone-ridge model — pursue only
for the curved-switching frontier (`separation_general`), not the current main line.

**attempts.** Parked; framed via the min-plus capacity note (`../refs/t4-minplus-capacity.md`). Refs:
`curse-free-max-plus`, `Gaubert–McEneaney–Qu`.
