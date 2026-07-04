# certificate_dichotomy

**status:** proved ($d=1$, modulo one numerically-airtight inequality); open ($d\ge2$)

**statement.** Penalized objective $\min_\mu\tfrac12\|K\mu-f\|^2+\alpha\|\mu\|_{\mathcal M}$, dual
certificate $q(w,b)=\langle\varphi_{w,b},\eta\rangle$ ($\eta$ = residual). In $d=1$ (power-2 ridges): the
signed model activates atoms at **both** bounds $q=\pm\alpha$; the cone activates only at the
**single** bound $q=+\alpha$. Consequently the continuum (contact-region) atom counts satisfy
$$N^{\mathrm{cont}}_{\text{signed}}/N^{\mathrm{cont}}_{\text{cone}}\to 2.$$

**proof.** ($d=1$.) Lemma: $q_+''=2\eta$ (and mirror $q_-''=-2\eta(-b)$). KKT: cone reduced-cost
is one-sided ($q=-\alpha$ only) — proved. Alternation ($q''=2\eta$ flips sign at each contact)
gives $R_+=R_-$. The factor 2 follows **given** the one inequality $q_{\text{cone}}\le+\alpha$
everywhere (cone certificate never overshoots the opposite bound) — proved by KKT on one side,
numerically airtight on the other ($R\le10^{-10}$ over 387 targets), not yet analytic.
Detail/referee record: `../refs/t5-d1-certificate-dichotomy.md`. $\square$ (conditional)

**open ($d\ge2$).** The certificate identity lifts via Radon $\partial_b^2 q=\mathcal R[\eta]$, but the
contact-region count / alternation argument in $d\ge2$ is unproved.

**uses.** — (D2 machinery: Bredies optimality / dual certificate). Feeds `achievability`.

**attempts.** 1. absolute-count law $n=K+1$ — **false** (atoms cluster $\propto1/\alpha$); restated as
a continuum ratio. 2. three spurious "overshoot" measurement bugs — fixed (convention-free
$R$). 3. $d\ge2$ lift — open.
