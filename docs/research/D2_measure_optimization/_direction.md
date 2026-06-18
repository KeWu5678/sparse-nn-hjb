# D2 — Measure-theoretic optimization (current-but-stalled)

**Machinery.** Tikhonov over Radon measures $\min_\mu \tfrac12\|K\mu-f\|^2+\alpha\|\mu\|_{\mathcal M}$;
Fenchel duality; **dual certificate** $q=K^\ast(f-K\mu^\ast)$ with $\mathrm{supp}\,\mu^\ast\subset\{|q|=\alpha\}$;
source condition + Bregman $O(\delta)$ rate. Refs: Bredies–Pikkarainen, Parhi–Nowak,
Boyd–Schiebinger–Recht (ADCG = PDAP's ancestor), Bach; canonical Candès–Fernández-Granda,
de Castro–Gamboa, Duval–Peyré.

**Role.** The **achievability** half of the separation: does penalized PDAP realize the
cone's small count (and not add spurious atoms)? This is the only half where optimality
conditions enter. Cone = one-sided certificate ($q=+\alpha$); signed = two-sided ($|q|=\pm\alpha$).

**Status: delivered $d=1$ only.** The certificate dichotomy is proved in $d=1$; the
source-condition route is $d=1$/recovery-only (Mairhuber–Curtis). $d\ge2$ open.

## Claims → `claims/`  (registry in `../CLAIMS.md`)
- `certificate_dichotomy.md` — **proved $d=1$** (cone one-sided ⟹ ratio→2, modulo one
  numerically-airtight inequality); **open $d\ge2$**.
- `achievability.md` — **open** — penalized PDAP attains the cone count, $d\ge2$ ($d=1$ via
  certificate dichotomy).
- `source_condition_separation_free.md` — **refuted** as a $d\ge2$ separation ($d=1$/recovery-
  only: Mairhuber–Curtis; recovery ≠ count).

## Refs → `refs/`
`t5-d1-certificate-dichotomy.md` (the proved $d=1$ result),
`semiconcave-source-condition.md` (source-condition detail; contains the original
"separation-free in general $d$" overclaim, corrected in the formulation summary).
