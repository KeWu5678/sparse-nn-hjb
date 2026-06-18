# D4 — Max-plus / tropical algebra (PARKED direction)

**Machinery.** Tropical / min-plus algebra (McEneaney; Gaubert–McEneaney–Qu;
Zhang–Naitzat–Lim). Min-of-paraboloids representation; max-plus rank; curse-free
numerical methods for HJB. A distinct mathematical tool, not a formulation under D1.

**What it is for.** The min-of-paraboloids architecture (Kunisch–Vásquez-Varas), analyzed
via min-plus algebra. Relevant **only** to the curved-switching-set case where the
**ridge** model provably fails (`correction_cost`: $\gamma^+(g)=\infty$, e.g. $(\|x\|-1)_+$).
Open claim: `claims/minplus_curved.md`.

**Status: parked.** It is the natural escape hatch when ridges fail, but:
- The repo model is a **ridge** cone, not min-of-paraboloids — so this is a *different
  architecture*, off the main line.
- min suffices only for **finitely-min-representable** $V$ (clean case: $v_d$ = min of $2d$
  paraboloids, T2). Not a universal fix — rotationally-symmetric $(\|x\|-1)_+$ likely needs
  a continuum of paraboloids in min too. "Which HJB $V$ are finitely-min-representable" =
  open (max-plus rank).

**Why not current.** The current goal fixes flat switching ($\gamma^+<\infty$, ridges work).
Curved switching is the next frontier; only *then* does this machinery become the main line.

**Detail / log.** `refs/t4-minplus-capacity.md` (T4); external refs `curse-free-max-plus`,
`Gaubert–McEneaney–Qu`, `CDC-max-plus-complexity bounds`.
