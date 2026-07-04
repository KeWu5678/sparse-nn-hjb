# curved_switching_rnorm

**status:** proved ($d=2$; $d\ge3$ modulo gaps)

**statement.** There exists a $C$-semiconcave $V$ whose switching set is a smooth **curved**
hypersurface (the Kunisch–Vásquez-Varas target $v_d=\min_{i\le2d}\varphi_i$, paraboloids) such that
its ridge ($\mathrm{ReLU}$) variation norm is infinite:
$$\|v_d\|_{\mathcal R}=\gamma_d\,\big\|\partial_b^{d+1}\mathcal R\{v_d\}\big\|_1=\infty,$$
while the min-of-paraboloids architecture represents $v_d$ with $2d$ exact atoms. Equivalently
$\gamma^+(g)=\infty$ for the convex correction $g$.

**proof.** Sketch: the curvature of the switching surface makes the Radon-domain derivative
$\partial_b^{d+1}\mathcal R\{v_d\}$ non-integrable (a flat kink gives a finite point mass; a curved
kink smears it with a non-summable density). Full argument $d=2$: `../refs/t2-separation-draft.md`
(Fourier witness $v_d$); $d\ge3$ modulo stated gaps. $\square$

**uses.** — (harmonic-analysis machinery: Radon transform, $\mathcal R$-norm = Ongie et al.)

**consequence.** This is the engine of `correction_cost` (curved case) and the obstruction in
`separation_general`: ridge nets cannot finitely represent a curved switching set, so a non-
ridge (min) architecture is necessary — `../../D4_max_plus/claims/minplus_curved.md`.

**attempts.** Referee-corrected ($\delta/2$ vs $\delta$ misquote, $\sqrt2$ Jacobian, mirror sign);
all fixed (see ref). $d\ge3$ general curved switching is not fully closed.
