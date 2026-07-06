# Review of `Mthesis.tex` — full five-axis report

Date: 2026-07-05 · Reviewed at commit state of 2026-07-05 (46 pp build, clean log).
Ruler: HU Berlin M.Sc. Mathematik thesis (PO 2014): **max ~50 pages at ~11pt, usual
mathematical typesetting**. Current layout is 12pt / linespread 1.3 / 15mm margins —
a conforming re-layout will reflow everything, so the 50-page cap must be re-checked
after fixes (see TEX-1).

How to use this file: each finding has an ID, severity, location (line numbers in
`Mthesis.tex` as of this review), the issue, and a proposed fix. Tick the box to
approve a fix, strike it to reject. Fix batches will be applied per your approval,
math/content first, mechanics last.

Severity legend: **critical** = wrong or unsubstantiated claim a committee would
catch; **major** = proof gap / stale result / structural absence; **medium** =
imprecision that needs repair but has a clear local fix; **minor** = polish.

---

## 0. Headline finding

- [ ] **HEAD-1 (critical). The claimed removal of compactness is not actually
  established, and it silently breaks four proofs.**
  The introduction claims the thesis "remov[es] the assumptions of bounded
  dictionary" (l. 195) and §3.4 states "Unlike [Pieper–Petrosyan], we don't assume
  $\Omega$ is compact" (l. 717) — with $\Omega = \mathbb{R}^{d+1}$ fixed at l. 243.
  Pieper–Petrosyan assume **compact** $\Omega$ throughout (their §2.1, verified in
  the source PDF), and the thesis's proofs still use consequences of compactness
  that fail on $\mathbb{R}^{d+1}$:
  1. Well-definedness of $\mathcal{N}$ on all of $\mathcal{M}(\Omega)$ (l. 329–342):
     for $\sigma$ with linear growth in $\omega$ (forced by Assumption 3.1), and a
     fortiori for $\mathrm{ReLU}^k$, $\sigma(x,\cdot)$ is unbounded on
     $\mathbb{R}^{d+1}$, so $\int_\Omega \sigma\,d\mu$ need not exist for arbitrary
     finite $\mu$. Also note Assumption 3.1's global Lipschitz bound *excludes*
     $\mathrm{ReLU}^k$, $k\ge 2$, on an unbounded $\Omega$ — the assumption and the
     later activation class are incompatible unless $\Omega$ is bounded.
  2. Existence theorem (Thm 3.5, l. 618–635): weak-* l.s.c. of $L$ is justified by
     "$\mathcal{N}$ weak-*-to-weak continuous", which requires
     $\sigma(x,\cdot) \in C_0(\Omega)$ — false for every activation used
     (ReLU^k grows; softplus/tanh/sigmoid have nonzero limits at infinity). Mass
     escaping to infinity breaks the argument.
  3. Finite-support theorem (Thm 3.7), part 2 (l. 771–783): the
     escape-to-infinity argument "contradicts $\bar p \in C_0(\Omega)$" — but
     $\bar p \in C_0(\Omega)$ is never proved and is false in general (same reason).
  4. Sufficient-condition theorem (Thm 3.8): needs the uniform gap
     $\sup_{\omega \notin \cup B_\epsilon}|\bar p(\omega)| \le (1-\delta)\alpha$,
     which follows from pointwise strict inequality **plus compactness** (that is
     exactly how the source derives it); pointwise $|\bar p| \lneq \alpha$ on a
     non-compact set gives no $\delta$. Also $\|\mathcal{N}\|$ (operator norm
     $\mathcal{M}\to L^2$) is finite only for a bounded dictionary.

  **Status 2026-07-06 — refined after discussion (supersedes the original
  option (a)/(b) recommendation).** Two corrections and a sharper resolution:

  *(i) Corrected scope of the norm blow-up.* For every ridge atom the $H^1$ atom
  norm grows like $\|\nabla_x\sigma(\cdot;(a,b))\|_{L^2(D)} \sim c\,|a|^{1/2}$
  (slab computation), so the dictionary is never norm-bounded on an unbounded
  parameter set. This does **not** force a compact $\Omega$; it only means
  $\mathcal{N}$ acts on the weighted space $M_w$,
  $w(\omega) = \max(1, \|\sigma(\cdot;\omega)\|_{H^1})$, not on all of
  $\mathcal{M}(\mathbb{R}^{d+1})$. Weighting the space and normalizing the
  dictionary are the same operation ($\tilde\mu = w\mu$, $\tilde\sigma = \sigma/w$).
  Norm blow-up and l.s.c. failure are *distinct* failure modes: the $|a|\to\infty$
  concentration channel is harmless for l.s.c. along minimizing sequences
  (concentrating profiles converge weakly to 0); what kills l.s.c. is a
  *non-vanishing strong limit* at infinity.

  *(ii) Approved counterexample — to be added as a remark in §3.* Weak-* l.s.c.
  of $L$ genuinely fails on $\Omega = \mathbb{R}^{d+1}$ for saturating
  activations: take $\sigma = \tanh$, target $V \equiv 1$,
  $\mu_n = \delta_{(a, b_n)}$, $b_n \to \infty$. Then $\mu_n \overset{*}{\rightharpoonup} 0$
  in $\sigma(\mathcal{M}, C_0)$ but $\mathcal{N}\mu_n = \tanh(a\cdot x + b_n) \to 1$
  strongly in $H^1(D)$, so $\liminf L(\mu_n) = 0 < L(0)$. Consequently existence
  of global minimizers is **false** for tanh/softplus-type activations on
  $\mathbb{R}^{d+1}$: the cheapest fit of a constant component drifts to
  $b = \infty$ (coefficient $M/\rho(b)$ strictly decreasing in $b$). Note
  constants *are* exactly representable at finite parameters ($a = 0$ atoms) —
  the escape is a penalty-infimum phenomenon, not a representability one.
  **Softplus is strictly worse** (2026-07-06): $\mu_n = \rho(b_n)^{-1}
  \delta_{(0,b_n)}$ fits $V \equiv 1$ *exactly for every $n$* with penalty
  $\phi(1/\rho(b_n)) \to 0$, so $\inf J = 0$ unattained; since
  $\|\mu_n\|_{\mathcal{M}} \to 0$, even **norm**-l.s.c. of $L$ fails (the
  operator is unbounded: vanishing TV-mass carries unit impact). Moreover
  $\log(1+e^z) \approx z$ for large $z$ makes *every affine function*
  reachable at asymptotically zero penalty ($c_n \to 0$, $a_n = e/c_n$,
  $b_n \to \infty$), so the penalty is blind to the affine part — non-attainment
  is generic. Sharper still (2026-07-06): $(\gamma/n)\,\mathrm{softplus}(n(e\cdot x - t))
  \to \gamma\,(e\cdot x - t)_+$ in $H^1(D)$ at penalty $\phi(\gamma/n) \to 0$, so
  on $\mathbb{R}^{d+1}$ the log-penalized softplus model reaches **every finite
  ReLU network at zero asymptotic penalty** — on finite data the infimum
  collapses toward unregularized interpolation; the regularizer is void in the
  limit. (Tanh's analogous step channel is blocked by the $H^1$ loss:
  $\|\nabla\|\sim n^{1/2}$; softplus leaks because its dilation limit, ReLU, is
  $H^1$-legal.) Practical note: solvers explore a bounded parameter region, so
  experiments are evidence about the compact-$\Omega$ problem — the compact
  assumption is the theory *of* the practice. Rescue for class (c) besides
  truncation: the impact-normalized penalty $\phi(|c_n|\,w(\omega_n))$,
  $w(\omega) = \|\sigma(\cdot;\omega)\|_{H}$, closes all zero-cost channels;
  existence then via the compactified dictionary whose ideal boundary is
  exactly {ReLU ridges + constants} — outlook material. Invariant sorting the
  classes: not "TV-mass at infinity" but whether escaping sequences *retain fit
  impact*; equivalently, condition (D) below (tanh fails by plateau, softplus
  by blow-up, Gaussian passes).

  *(iii) Resolution — taxonomy by activation class* (replaces "compact Ω
  everywhere"):
  - **(a) $k$-homogeneous (ReLU$^k$):** sphere constraint via Prop 4.3 — compact
    without loss of generality; §3 applies verbatim.
  - **(b) Localized (value-channel decay):**
    $\|\sigma(\cdot;\omega)\|_{L^2(D,\nu)} \to 0$ as $|\omega| \to \infty$
    (ridge atoms: $\rho(\pm\infty) = 0$; Gaussian/RBF class). The theory is
    viable **on unbounded $\Omega$** via the distributional-gradient
    formulation (2026-07-06, supersedes the weighted-duality plan): take
    (D) $\rho \in C^1$ Lipschitz, $\rho, \rho' \in L^2(\mathbb{R})$ (forces
    $\rho(\pm\infty)=0$); then $w_0(\omega) = \|\rho(a\cdot x+b)\|_{L^2(D)}
    \in C_0(\Omega)$ (slab estimate), so the value operator is weak-*-to-weak
    continuous against **all** of $L^2$; define $J$ with the *distributional*
    gradient of the value network ($+\infty$ if not $L^2$) — then existence
    needs no weighted duality: TV-coercivity + Alaoglu, value channel converges
    by $w_0 \in C_0$, gradient channel is bounded by $J \le C$ itself and
    identified distributionally. Optimality conditions unchanged (local).
    Finite support: escape leg stated *conditionally* under
    $\limsup_{|\omega|\to\infty}|\bar p(\omega)| < \alpha$ (the exact quantity
    failing in the class-(c) counterexamples), with $V \in H^2$ + a
    $3/2$-moment on $\bar\mu$ as sufficient remark. Thm 3.8 with local
    minimality in $\|\cdot\|_w$ ($\mathcal{N}$ 1-Lipschitz there). Genuine
    extension beyond Pieper–Petrosyan. **Est. +≈1.5 pages** over route (α)
    (shared items — Thm 3.8 rewrite, l.s.c. citation, counterexample remark —
    are owed in both routes).
  - **(c) Saturating (tanh, softplus, sigmoid):** decay fails and existence is
    genuinely false by (ii) — compact $\Omega$ is *necessary*, not cosmetic.
    (Alternative model fix — explicit affine offset term absorbing the
    $b \to \pm\infty$ channel, with the $H^1$ loss killing the
    $|a| \to \infty$ step channel — outlook remark only, unproven.)

  *Heeringa-embedding leverage (2026-07-06):* $B_\sigma \hookrightarrow
  B_{\mathrm{ReLU}^k}$ applies to softplus/tanh/Gaussian ($\partial^2\sigma \in
  L^1$ etc.) and buys two things. (i) **Losslessness of the restriction**: cite
  it where compact $\Omega$ / the sphere is introduced — the theory's function
  class (ReLU$^k$) already contains everything the smooth dictionaries express,
  at controlled Barron norm; upgrades §2.2 to load-bearing. (ii) **Rigorous
  completion story (outlook)**: the push-forward $\sigma(z) = \mathrm{affine} +
  \int \mathrm{ReLU}(z-t)\,\sigma''(t)\,dt$ parametrizes the compactified
  softplus dictionary (boundary atoms = ReLU ridges + affine); with the
  impact-normalized penalty this becomes a well-posed compact-dictionary
  problem ("softplus $\cup$ ReLU"). It does **not** restore existence for the
  raw softplus problem (penalty does not transport: atomic $\mapsto$ diffuse,
  concavity lost, charged at TV rate) and preserves norms, **not neuron
  counts** — no problem-equivalence or sparsity-transfer claims.

  **Decision needed for Batch M:** (α) uniform compact-$\Omega$ everywhere
  (fast, safe, loses the class-(b) novelty) vs. (β) the taxonomy version with
  the new unbounded-$\Omega$ theorems for class (b) (stronger thesis, more
  work). The claims at l. 195 / 717 get rewritten accordingly in either case;
  the counterexample of (ii) goes in as a remark in both.

---

## 1. Mathematical content (statements & proofs)

Verified line-by-line: Lemma 3.1, Prop 3.2, Lemma 3.3, Lemma 3.4, Thm 3.5,
Thm 3.6, Thm 3.7, Thm 3.8, Prop 4.2, Prop 4.3, Cor 4.2.1, Cor 4.2.2, plus all
control-theory formulas in §1 and §6 (HJB sign convention ✓, VDP adjoint system ✓,
pendulum adjoint system ✓ — including the neat identity
$\sin^2\theta+(\cos\theta-1)^2 = 2-2\cos\theta$ matching the code — LQR
linearization ✓, feedback formulas $u^* = -\frac{1}{2\beta}g^Tp$ ✓ and
$u = -\frac{1}{2rml^2}\partial_\omega V$ ✓).

- [ ] **MATH-1 (major). Thm 3.8 (sufficient condition, l. 845–903): proof is a
  corrupted transcription of Pieper–Petrosyan Thm 4 and does not go through as
  written.** Concretely: (i) the $(1-\delta)$ bound at l. 889 appears from nowhere
  (source derives it from compactness — see HEAD-1.4); (ii) the proof only expands
  the *value* fidelity term, so l. 873's identification
  $\langle \mathcal{N}\bar\mu - y, \mathcal{N}\tilde\mu\rangle = \langle \bar p, \tilde\mu\rangle$
  is wrong — $\bar p$ contains the $d$ gradient terms too; the expansion must be
  done for all $d+1$ quadratic terms; (iii) l. 890 drops an $\alpha$:
  $(1-\delta)|\tilde c_n| \to (1-\delta)\alpha|\tilde c_n|$; (iv) l. 894–896: the
  bound $\gamma_1|\tilde c_n|^2/2 \le (\gamma_1\epsilon/2)|\tilde c_n|$ needs the
  $\epsilon$; as written "$(\alpha\delta - \alpha\gamma_1/2)$" can be negative and
  the final combination l. 900–902 has the sign structure garbled — the source's
  choice is $\epsilon \le \alpha\delta/(\|\mathcal{N}\|^2 + \alpha\gamma_1/2)$;
  (v) restricting to $\mu \in M(\Omega_\epsilon)$ (l. 859) loses admissible
  perturbations with small far-away atoms — the source decomposes an *arbitrary*
  $\mu$ with $\|\mu-\bar\mu\| \le \epsilon$; (vi) "Choosing $c_n = \bar c_n$"
  (l. 881) hides the actual use of hypothesis 1 (local minimality of $\bar c$ for
  $J_{\bar\omega}$ gives $J_{\bar\omega}(c) \ge J_{\bar\omega}(\bar c)$ for $c$ near
  $\bar c$). **Fix:** rewrite the proof following the source's Thm 4 verbatim
  structure, adapted to the $H^1$ loss (all $d+1$ terms), on compact $\Omega$.

- [ ] **MATH-2 (major). Lemma 3.4 (weak-* l.s.c., l. 539–616): the proof is not
  rigorous as written.** (i) $A_h := \{\omega : \mu_h(\omega) \ge \epsilon\}$
  should use $|\mu_h|(\{\omega\})$; (ii) "by passing to a subsequence
  $\mu_h|_{A_h} \overset{*}{\rightharpoonup} \mu|_{A_h}$" is ill-formed — the set
  $A_h$ varies with $h$, so the limit statement and every subsequent expression
  $\Phi_1(\mu \mathbf{1}_{A_h})$, $\|\mu\mathbf{1}_{\Omega - A_h}\|$ is
  ill-defined; restriction does not commute with weak-* limits in general;
  (iii) l. 580–581 integrates the *limit's* continuous part $d\mu_{\mathrm{cont}}$ where it
  must be $d|\mu_{h,\mathrm{cont}}|$; (iv) the $(1-\delta)$ factor is applied inconsistently
  between l. 593 and l. 611; (v) missing $|\cdot|$ throughout for signed measures
  (also in the lemma statement: $\int_\Omega d\mu_{\mathrm{cont}}$ should be
  $\int_\Omega d|\mu_{\mathrm{cont}}|$). The *statement* is true — it is essentially
  Bouchitté–Buttazzo Thm 3.3 (their Lemma 3.6 is correctly available for the atomic
  part: the set of measures with $\le k$ atoms is weak-* closed, and the atomic
  functional is l.s.c. on it — verified in the source), with the compatibility
  condition supplied by $\phi'(0)=1$. **Fix (recommended):** either cite
  Bouchitté–Buttazzo Thm 3.3 (and Pieper–Petrosyan's l.s.c. lemma, which does
  exactly this) and delete the flawed proof, or rewrite it properly: fix a
  threshold $\epsilon$, pass to a *fixed* finite atom count $k \le \sup_h
  \|\mu_h\|/\epsilon$, extract a subsequence along which the (at most $k$) heavy
  atoms converge as points, apply BB Lemma 3.6 on that limit configuration, and
  treat the light part via $\phi_1(z) \ge (1-\delta)z$ on $[0,\epsilon]$ against
  the l.s.c. of the total-variation norm; then let $\epsilon \to 0$.

- [ ] **MATH-3 (major). Thm 3.5 (existence, l. 618–635): two defects.**
  (i) "Moreover, $L(\mu)$ is also coercive" (l. 626) is false in general (the
  fidelity term is bounded on TV-bounded sets, and nothing more is needed:
  $J \ge \alpha\Phi_1$ and $\Phi_1$ coercive by Lemma 3.3 suffices) — delete the
  claim; (ii) the weak-* l.s.c. of $L$ needs $\sigma(x,\cdot) \in C_0$/compact
  $\Omega$ (HEAD-1.2) — with fix HEAD-1(a), $\sigma(x,\cdot) \in C(\Omega)$ on
  compact $\Omega$ makes $\mathcal{N}$ weak-*-to-weak continuous, and the proof is
  fine. Also state where Banach–Alaoglu/metrizability is used to extract the
  weak-* convergent subsequence.

- [ ] **MATH-4 (major). Thm 3.7 (finite support, l. 719–836): several repairs.**
  Part 1: (i) $C_\delta := |\bar\mu|(D_\delta)$ but the Taylor identity at
  l. 735–736 needs the replacement atom to carry $\bar\mu(D_\delta)$ (signed), and
  the penalty computation at l. 758 writes $-\bar\mu(D_\delta)$ where the
  construction removes $|\bar\mu|$-mass — the proof implicitly assumes
  $\bar\mu \ge 0$ on $D_\delta$. Fix by working on the positive/negative parts
  separately (by Thm 3.6, $\mathrm{sign}\,\bar\mu_{\mathrm{cont}}$ is locally constant
  where $\bar p = \mp\alpha$, so a small enough ball has one sign), and say so;
  (ii) atoms inside $D_\delta$ are silently merged — either exclude them
  ($D_\delta$ around a point of $\mathrm{cont}\,\bar\mu$ can be chosen with
  $|\bar\mu_{\mathrm{atom}}|(D_\delta)$ arbitrarily small) or handle via subadditivity;
  (iii) l. 768: the choice $\delta^2 \le 2\alpha\gamma_2/((d+1)\Lambda_1^2)$ makes
  the bracket $\alpha\gamma_2 - (d+1)\delta^2\Lambda_1^2 \ge -\alpha\gamma_2$,
  i.e. possibly negative — the factor 2 is on the wrong side; needs
  $\delta^2 < \alpha\gamma_2/((d+1)\Lambda_1^2)$ (part 2, l. 833, has it right);
  (iv) state $C_\delta \le \hat z_2$ for $\delta$ small (used by Assumption
  3.4(2)). Part 2: (v) the escape-to-infinity argument needs $\bar p \in
  C_0(\Omega)$ — unproved and false in general (HEAD-1.3); under fix HEAD-1(a)
  this whole sub-argument is *deleted* (compact $\Omega$ needs no escape case);
  (vi) sign-uniformity "w.l.o.g. $c_n > 0$" should say "after passing to a
  subsequence, infinitely many atoms share a sign"; (vii) $\delta_N :=
  \max_{i\ge N}|\omega_i - \hat\omega|$ should be $\sup$; (viii) l. 810:
  $\Phi(\mu)$ → $\Phi(\bar\mu)$; (ix) if $\hat\omega \in \mathrm{atom}\,\bar\mu$ the penalty
  bookkeeping needs subadditivity (one line).

- [ ] **MATH-5 (medium). Remark after Prop 4.2 (l. 948–954): the constant is off
  by a factor 2.** With $p=q=2$: $s = 4/(k+1)$, so $2/s = (k+1)/2$ and the sphere
  penalty coefficient is $\alpha\,\frac{k+1}{2}\,k^{-k/(k+1)}$, not
  $\alpha\,(k+1)\,k^{-k/(k+1)}$. Consequently $k=1$ gives $\sum|c_n|$ (matching
  Pieper–Petrosyan Appendix F / Prop 3, verified in the source: their coefficient
  is $2\alpha/s = \alpha$ at $p=q=2$, $k=1$), not $2\sum|c_n|$; and $k=2$ gives
  $\tfrac{3}{2}\cdot 2^{-2/3}\sum|c_n|^{2/3}$, not $3\cdot 2^{-2/3}\sum|c_n|^{2/3}$.
  The proposition statement and its proof are internally consistent and correct;
  only the remark's evaluation is wrong.

- [ ] **MATH-6 (medium). §2.2 Barron material (l. 289–306) garbles the source
  definitions.** In Li–Lu–Mathé–Pereverzev the representation is $f(x) = \int_P
  a\,(b\cdot x + c)_+^k \,\rho(da,db,dc)$ with $\rho$ a **probability** measure on
  the parameter space, and the norm is $\|f\|_{B^{k,m}_{p,\rho}} =
  (\sum_{|\alpha|\le m} \|\partial^\alpha f\|^p_{B^{k-|\alpha|}_{p,\rho_\alpha}})^{1/p}$,
  infimum over representing $\rho$ (their Def. 1 and eq. (8); two indices $k,m$).
  The thesis's $\|f\|_{B^k_{p,\mu}} := (\int_{\mathbb{S}^d} d\mu(a,b)^{kp})^{1/p}$
  is not a meaningful expression ("$d\mu^{kp}$"), drops the outer weight $a$,
  conflates the signed-measure-on-sphere picture (Pieper/Ongie) with the
  probability-over-parameters picture (Li et al.), sums over $|\alpha| \le k$
  instead of $\le m$, and flattens $B^{k,m}_p$ to $B^k_p$. Also Prop 2.1 =
  their Thm 7 is correct in substance, but state it with the norm bound
  $\|f\|_{H^m} \le C(\Omega,d,k)\|f\|_{B_1^k}$, $0 \le m \le k$. **Fix:**
  rewrite the subsection following the source's Def. 1 + Thm 7; pick ONE measure
  picture and keep it consistent with §3's $\mathcal{M}(\Omega)$ setting,
  remarking explicitly on the translation between the two.

- [ ] **MATH-7 (medium). Prop 2.2 (l. 319–325) misstates
  Heeringa et al., Thm 1.** Their point 2 (for order $k \ge 1$) requires the
  one-sided **Caputo** derivatives $\partial_+^{k+1}\sigma \in L^1((0,\infty))$,
  $\partial_-^{k+1}\sigma \in L^1((-\infty,0))$ **and $\Omega$ (the state domain)
  bounded**; the classical-integer-derivative version needs (per their remark)
  the distributional derivative to exist as a finite measure. The thesis's
  "$\sigma \in C^k(\mathbb{R})$ with $\partial^{k+1}\sigma \in L^1(\mathbb{R})$"
  drops the bounded-domain hypothesis and the one-sided formulation. Also
  $B_\sigma$ (Barron space for general activation $\sigma$) is used without ever
  being defined. **Fix:** state their Thm 1(1)–(2) precisely, define $B_\sigma$,
  and note $D$ bounded (true in this thesis anyway).

- [ ] **MATH-8 (medium). Eq. (l. 838–841): $J_{\bar\omega}(c) := l(\mathcal{N}_{\bar\omega,c};y)
  + \sum_n \phi(|c_n|)$ is missing the factor $\alpha$** in front of the penalty
  (every later use, e.g. Thm 3.8 and Algorithms 1–2, assumes it). Also decide
  $y$ vs $V$ for the target (both are used; see WORD-4).

- [ ] **MATH-9 (medium). Cor 4.2.2 (l. 1190–1242): hypotheses too weak / notation
  undefined.** (i) "$\sigma$ is continuous" does not support the $\nabla_x\sigma$
  terms inside $J$ nor the dominated-convergence argument — require $\sigma(\cdot)
  \in C^1$ in $x$ (true for ReLU^k, $k\ge 2$); (ii) $\|\cdot\|_H$ and the inner
  product $\langle\cdot,\cdot\rangle_H$ are never defined (they must be the
  $L^2\times (L^2)^d$ product norm on (value, gradient) data — define once, reuse
  in §5's $S_t(\omega)^2 = \|\sigma(\cdot;\omega)\|_H^2$); (iii) step 2's
  stationarity needs the *full* residual including gradient terms — with the
  $H$-inner-product definition it is exactly $\bar p$, say so; (iv) $\phi_k'$
  "strictly decreasing" is not implied by concavity — for $\phi_k = \frac1q z^q$
  it is true, otherwise use the generalized inverse; (v) the index $k$ in
  "$(\omega^k, c^k) \to (\omega,c)$" collides with the power $k$ — rename.

- [ ] **MATH-10 (medium). §2.1 Riesz statement (l. 256–262) is wrong as written:**
  "(C(\Omega,\mathbb{R}^m))^* = \mathcal{M}(\Omega,\mathbb{R}^m)$" must be
  $C_0(\Omega;\mathbb{R}^m)^*$ (for non-compact $\Omega$ the dual of bounded
  continuous functions is strictly larger). Also: "vanishing on the boundary"
  (l. 252) — the displayed definition is *vanishing at infinity*; l. 261 "For each
  $\mu \in \ldots$ and $u \in \ldots$." is a sentence fragment; the dual bracket
  at l. 262 should pair $\mathcal{M}$ with $C_0$. With HEAD-1(a) ($\Omega$
  compact) all of this simplifies to plain $C(\Omega)^* = \mathcal{M}(\Omega)$.

- [ ] **MATH-11 (medium). §3 opening (l. 329–342): well-definedness of
  $\mathcal{N}$ and $\nabla\mathcal{N}$.** "Since for all $x$, $\sigma(\cdot,x)$ is
  bounded by a $\mu$-integrable function" is not an argument (and is false on
  unbounded $\Omega$, HEAD-1.1). On compact $\Omega$ with Assumption 3.1, state:
  $\sigma, \partial_{x_i}\sigma$ are continuous and bounded on $D\times\Omega$
  (if $D$ bounded), hence $\mathcal{N}: \mathcal{M}(\Omega) \to H^1$ is a bounded
  linear operator; differentiation under the integral by dominated convergence.
  Also fix the argument-order drift: $\sigma(\omega,x)$ (l. 331) vs $\sigma(x,\omega)$
  (l. 654, Assumption 3.1) — pick $\sigma(x;\omega)$ everywhere.

- [ ] **MATH-12 (medium). W(D) definition (l. 279–284):** the integral is over
  $\mathbb{S}^d$ but $\mu$ ranges over $\mathcal{M}(\Omega)$ with $\Omega =
  \mathbb{R}^{d+1}$ — make it $\mu \in \mathcal{M}(\mathbb{S}^d)$; the operator
  $\mathcal{N}$ appears in (2.1)/l. 283 before it is defined (§3) — either move the
  definition up or write the constraint out.

- [ ] **MATH-13 (minor). Lemma 3.1 (l. 441–465):** statement says "for all $a,b$"
  — restrict to $a, b \ge 0$; the two bullets vs "1./2." labels mismatch; part 2's
  computation needs $\phi(0)=0$ (cite Assumption 3.3); in the proof $z$, $z_1$,
  $z_2$ drift. In Lemma 3.3's proof (l. 521–533): the subadditivity step is
  applied to a *countable* sum — extend by induction + continuity (one sentence),
  and "$\phi_1$ is sublinear" (l. 532) should be "subadditive with $\phi_1(z)\le z$".
  Also the symbol $\phi_1$ is used in Lemmas 3.3/3.4 while the assumptions are
  stated for $\phi$ — unify (the subscript collides with $\Phi_1$).

- [ ] **MATH-14 (minor). Prop 3.2 (second-order estimate, l. 487–494): no proof
  given.** It is two lines from $\phi(z) = \int_0^z \phi'(\xi)d\xi$ and Assumption
  3.4 — for a thesis, include it (also fixes the wrong-looking "the second order
  estimate" as a name: it is a pair of envelope bounds).

- [ ] **MATH-15 (minor). Thm 3.6 (l. 667–711):** statement (2) has $\omega$ vs $w$
  typo; "defined $\mu$-a.s." → "$|\bar\mu|$-a.e."; l. 708 uses `*` for
  multiplication and integrates against $d|\bar\mu|$ where it should be
  $d|\bar\mu_{\mathrm{cont}}|$; the limit at l. 691 needs $c \ne 0$ (true for atoms of
  $\bar\mu$ — say it); minor: $u \in \mathcal{M}(\Omega)$ fixed then "$\tau$ small
  enough" — quantify as: for each fixed $u$ there is $\tau_0(u)$.

- [ ] **MATH-16 (minor). Prop 4.3 (l. 1039–1076):** (i) "Under Assumption 3.3
  [pen2]" — $\phi_k(z) = \frac1q z^q$ is not differentiable at $0$; state pen2
  with differentiability on $(0,\infty)$ or relax here; (ii) the case
  $(\hat a_n,\hat b_n) = 0$: by $k$-homogeneity $\sigma(x;0)=0$, drop such
  neurons — one sentence; (iii) "equivalent" — state the meaning (equal optimal
  values; minimizers map to each other by the rescaling); (iv) the proof shows
  one direction (project inward-ball solutions to the sphere) — add the trivial
  converse inclusion sentence; (v) $\lneqq/\gneqq$ → $<$ / $>$.

- [ ] **MATH-17 (minor). §4 Remark (l. 916–919):** "For $k > 2$ the activation is
  twice differentiable" — so is $k = 2+\epsilon$; the sentence "The value part of
  the loss is approximated by a high-order ReLU activation function
  $\mathrm{ReLU}^{2+\epsilon}$, and the gradient by $\mathrm{ReLU}^{1+\epsilon}$"
  is garbled (presumably: the *gradient* of a $\mathrm{ReLU}^{2+\epsilon}$ atom is
  a $\mathrm{ReLU}^{1+\epsilon}$ ridge, which is still $C^1$-smooth at the kink) —
  rewrite. Assumption 4.1 fixes $k \ge 2$ but Prop 4.2/4.3 and the Remark use
  $k \ge 1$ — state the assumption for $k \ge 1$ and specialize where needed.

- [ ] **MATH-18 (minor). Editing leftovers with mathematical content:**
  (i) l. 1099–1106: "More generally, we assume the activation function to satisfy
  the following condition:" is followed by *blank lines* and then "…the direct
  method of ." — a missing assumption block and a truncated sentence ("direct
  method of calculus of variations"); (ii) l. 1179: "the merging argument of
  Theorem~5.2" — there is no Theorem 5.2 in this document (it is Thm 3.7; a
  hard-coded stale number — make it a `\ref`); (iii) l. 2090: "…solve the problem
  by over." — sentence cut mid-word (oversampling); (iv) l. 1307: the
  stereographic projection is written $\Psi : \mathbb{S}^d \to \mathbb{R}^d$, but
  the displayed formula maps $z \in \mathbb{R}^d \mapsto \mathbb{S}^d \subset
  \mathbb{R}^{d+1}$ — it is the *inverse* map; fix domain/codomain;
  (v) l. 1311–1314: $\bar f_t \in \arg\min_f \langle f, J'(f_t)\rangle$ has no
  constraint set (unbounded below) — this is the FW step over the TV-ball, write
  it as such; (vi) l. 1264: $U_{ad} := \{u : E(y(u),u) = 0\}$ equals all of $U$
  by construction — the reduced problem is unconstrained; delete $U_{ad}$;
  (vii) l. 1272 "$\mathcal{E} := \mathrm{ran}(E)$" — the Lagrange multiplier lives in the
  dual of the *codomain* space; (viii) l. 1289: $g(y^*)p^*$ → $g(y^*)^T p^*$;
  (ix) l. 1083: Leshno et al. say non-polynomial activations give *density in
  $C(K)$ on compacta*, not "linearly approximate any measurable function".

- [ ] **MATH-19 (minor). Intro precision:** (i) l. 129/135: HJB here is a
  first-order fully nonlinear PDE — "nonlinear hyperbolic" is nonstandard; say
  "first-order nonlinear (Hamilton–Jacobi) equation"; (ii) l. 106 "Hamilton
  Jacobi ODE" → Hamiltonian two-point boundary value problem; (iii) l. 124: the
  robustness claim ("since the existing numerical methods for TPBVP in general
  tend to diverge") conflates open-loop non-robustness to *perturbations* with
  numerical divergence — split the two statements; (iv) l. 177: "$x^m$ is a
  sample of the random variable $x \in L^2(D,\nu)$" — the random variable is
  $D$-valued with law $\nu$; $L^2(D,\nu)$ is where the *loss* lives; (v) l. 190:
  ReLU's *first* derivative exists as an $L^\infty$ function; the obstruction to
  gradient training is the discontinuity of $\nabla_x\mathcal{N}$ /
  non-differentiability needed for the SSN inner solve — state the actual
  obstruction; (vi) l. 278: "the generalization is achieved by limiting the size
  of the network, but rather by controlling the magnitude" — inverted; insert
  "not"; (vii) l. 260 "which also implies the completeness" — completeness of
  $\mathcal{M}$ as a dual space, fine, but say "in particular ... is a Banach
  space".

---

## 2. Numerics vs. experiment ground truth (§6 of the thesis)

Everything below was checked against `experiments/*/results.md`, the experiment
configs (`conf/`, `src/config/schema.py`), the data-generation code
(`src/OpenLoop/pendulum/`), and the actual figure PNGs (all 42 `\includegraphics`
targets exist; all paper figures are byte-identical to their experiment sources).

**Verified correct** (no action): Table `tab:alg1_gradient` H¹ column ✓; both
`tab:alg1_sparsity` H¹ rows ✓; insertion-diagnostics table ✓ (exact match, all 20
rows); `tab:alg1_alpha` (all rows present in results.md ✓, rest plausible from the
same sweep); `tab:alg2_gradient` both panels ✓ (all 10 rows); VDP feedback costs
6.68/6.51/6.49 vs 6.48 ✓; VDP 30×30 grid, β=0.1, T=3 ✓; pendulum ε=2e-4 ✓,
A=(0.71,0.68), B=(0.23,0.53), T=10 ✓; 3,900 = 3,000 + 900 ✓; oversampling table
✓ (all 8 rows); ≈9.4×10⁵ common eval set ✓; tube d ≤ 0.3 ✓.

- [ ] **NUM-1 (major). §6.2.2 insertion-frontier prose and caption are stale
  relative to the current figure/run** (l. 2155–2174 vs
  `plot/pendulum_insertion_frontier.png` = `region_split/figures/frontier.png`):
  the figure shows ReLU² reaching ≈3.1×10⁻¹ at ~131 neurons (≈3.15×10⁻¹ at ~116),
  not "≈3.6×10⁻¹ at 113 neurons"; the figure contains a **leaky ReLU** curve
  (runner-up, ending ≈4.4×10⁻¹) that the text never mentions; softplus ends
  ≈5.2×10⁻¹ (text: 5.4×10⁻¹); Gaussian/ReLU⁵ end ≈6.1×10⁻¹ (text: 5.5–5.9×10⁻¹);
  the caption's "plateau at 1.5–1.6× its error" is actually ≈1.7–2.0×. **Fix:**
  re-read the numbers off the current run records and rewrite the paragraph +
  caption, including leaky ReLU.

- [ ] **NUM-2 (major). Fig. 1 caption (l. 213) misdescribes the frontier
  series.** Per `experiments/01_vdp/baseline/results.md` and
  `02_pendulum/baseline/results.md`, the ReLU^k series in both frontier panels is
  **k = 3** (finite-step, γ=0), not "we choose k = 2"; and the fractional penalty
  is $\psi_k(t) = |t|^q$ with $q = 2/(k+1)$ — the caption's "$\psi(t) = t^{k}$
  with $q = \tfrac{2}{k+1} < 1$" mixes the two symbols. Also both log-penalty
  series use γ=10 ✓ (caption right about that) and α=1e-5 ✓. Also: the four
  curves in each panel include a "ReLU + L1" baseline the caption doesn't
  mention — describe all series.

- [ ] **NUM-3 (medium). Experimental setup misstatements (l. 1455–1456):** the
  runs use up to **15** insertions per outer iteration (`max_insert = 15`,
  `src/config/schema.py:66`; no experiment overrides it), not "up to 50"; pruning
  drops atoms with $|c| \le 10^{-8}$ (`amp_tol = 1e-8`, `src/PDAP/pdap.py`), not
  "$|c| < 10^{-5}$". "10 outer iterations" ✓ is correct.

- [ ] **NUM-4 (medium). §6.2.3 "matching the optimal cost" (l. 2300–2307)
  overclaims.** Ground truth (`region_split/results.md` §5): from B, ReLU² costs
  10.3 vs true 10.2 (match ✓); from A it takes the correct branch but costs
  **57.9 vs 26.2** — right decision, over-energetic execution. The thesis body
  never reports the closed-loop costs at all. **Fix:** add the cost table (true
  26.2/10.2; gaussian 298.5/278.9; softplus 73013.3/66996.7; leaky ReLU
  869.1/776.9; ReLU² 57.9/10.3; ReLU⁵ 235.9/232.4) and qualify the summary
  sentence ("matches the optimal cost from B; from A it takes the correct branch
  at roughly twice the optimal cost").

- [ ] **NUM-5 (medium). Transect prose ignores leaky ReLU** (l. 2057–2061 and
  caption l. 2080–2086): the figure includes leaky ReLU, whose fitted
  $n\cdot\nabla V$ is a visible *staircase* (piecewise-constant — its derivative
  breaks too); "ReLU² … is the only model that develops a visible kink … while
  the smooth activations necessarily interpolate" silently lumps leaky ReLU with
  the smooth models. `results.md` §4.4 has the correct description (step
  function, plateau ≈ −30…−42). Rewrite the two sentences to cover all five
  curves.

- [ ] **NUM-6 (medium). Two table columns are not backed by the curated results
  docs:** `tab:alg1_gradient` L²-trained column (8.61e-2 / 3.59e-2 / 3.05e-2 and
  5.63e-1 / 4.68e-1 / 4.64e-1) and `tab:alg1_sparsity` L²-trained neuron counts
  (58 / 75 / 105). They are plausibly from the same MLflow sweep, but nothing in
  `experiments/01_vdp/*/results.md` records them (the L² fixed-point cells are not
  in the curated tables). **Fix:** verify against the run records (MLflow) or
  regenerate via `analysis.py` before submission; add them to results.md so the
  paper cites curated numbers only (per repo convention).

- [ ] **NUM-7 (medium). Pendulum problem parameters are never stated** — the
  reader cannot reproduce §6.2: $m = l = 1$, $b = 0.1$, $g = 9.8$,
  $q_1 = q_2 = 1$, $r = 1$ (from `src/OpenLoop/pendulum/problem.py`), 2000
  backward-PMP characteristics, integration cap $V \le 100$, basin restriction cap
  50, pad/collar band within 0.5 of the arms (from the generator + README). Also
  l. 1955 "stops when $V$ hits a predefined ceiling" conflates the integration cap
  (100) with the basin cap (50). Add one parameter sentence/table and the two caps.
  Similarly the VDP horizon/grid are given ✓ but add the Legendre basis size $N_b$.

- [ ] **NUM-8 (minor). Dataset-description nuance (l. 1963–1966):** the band
  construction is described as "tiled by ±2π … kept only if its branch value beats
  the competing branch's locally extrapolated value" — per the generator README,
  survivors are additionally required to match their own branch's lower envelope,
  and the band splits into a near-side pad (300) + far-side collar (600); the
  in-text "two-sided band" description should mention the pad/collar split since
  §6.2's region metrics quote it. Also "3{,}900 samples: a 3{,}000-sample
  in-basin body plus a 900-sample two-sided band" ✓ correct.

- [ ] **NUM-9 (minor). §6.1 unverifiable/odd items:** (i) l. 1443 "Legendre
  polynomial basis" — state the degree; (ii) l. 1444: $\nabla V(x^m) = p^*(0)$ —
  also state $V(x^m) = J(u^*;0,x^m)$ is the *computed* cost ✓ (fine); (iii)
  l. 1438–1441 "The HJB equation at t = 0 is:" — the display is just (HJB)
  restated; delete or make the point (residual check?) explicit.

---

## 3. Structure

- [ ] **STR-1 (critical). Thesis apparatus is missing entirely:** placeholder
  `\title[Review]{Review}`, **no `\maketitle`**, no author, no abstract, no table
  of contents, no declaration of originality, no acknowledgments, no conclusion
  section — the document ends mid-§6.2.3 summary paragraph (l. 2300–2310) and goes
  straight to the bibliography. Note l. 427 says "As introduced in the abstract"
  — there is no abstract. **Fix:** add title page (title/author/advisor/date per
  Prüfungsbüro requirements — confirm the mandated wording with them), abstract
  (draft in §7 below), `\tableofcontents`, a short Conclusion & Outlook section
  (candidate content: summary of the three findings; limitations — $d = 2$
  experiments, compact-$\Omega$ theory, per-sample weighting as the untried
  lever from the oversampling study; outlook — higher-dimensional problems,
  semiconcave/structured models), and the declaration page.

- [ ] **STR-2 (major). The introduction never states the contributions or the
  outline.** L. 192–196 gestures at contributions in four unlabeled sentences
  (with the overclaim flagged in HEAD-1). Add a "Contributions" paragraph with a
  short enumerated list matched 1:1 to sections (theory: existence/optimality/
  finite support for the gradient-augmented functional under assumptions X;
  $k$-homogeneous reformulation and singular-penalty corollaries; algorithms;
  benchmark findings), and an "Outline" paragraph ("Section 2 reviews …").

- [ ] **STR-3 (major). §5 "algorithms" front half re-derives standard material at
  length** (l. 1249–1300: constrained→reduced problem, Lagrangian, adjoint — with
  the errors flagged in MATH-18(vi–viii)). For a 50-page thesis this is the right
  place to *compress*: the adjoint equation is already (TPBVP) from §1; a
  half-page with a citation (Hinze–Pinnau–Ulbrich–Ulbrich or Tröltzsch) suffices.
  The freed space pays for the missing conclusion.

- [ ] **STR-4 (medium). §4 internal organization:** the section starts in the
  measure setting, jumps to the discrete problem (l. 1028–1034) without saying it
  is now finite-sample, has the orphaned "The ReLU^k function" subsubsection
  whose second half (l. 1095–1106) contains the editing debris of MATH-18(i), and
  only Cor 4.2.1 states "Let $\Omega$ be compact" — reconcile with the §3 setting
  (with HEAD-1(a) this becomes uniform). Suggested order: assumption + examples →
  Prop 4.2 (reformulation on the sphere) → the induced $\phi_k$, its failure of
  pen1 → Cor 4.2.1 (optimality) → Cor 4.2.2 (existence, discrete) → Prop 4.3.
  Also §3.1's activation-shape discussion (l. 396–421) and §6's per-activation
  narrative overlap — keep the mechanism discussion in one place (recommend §3.1
  keeps the *assumption-level* content only, the shape/economy story moves to §6
  where the figures are).

- [ ] **STR-5 (medium). Numbering scheme is inconsistent and produces oddities:**
  corollaries are numbered *within theorems* (`\newtheorem{corollary}{Corollary}[theorem]`),
  yielding "Corollary 4.2.1 / 4.2.2" — and since they appear after Proposition
  4.3, the numbers run *backwards* in the text. Remarks are numbered globally
  (Remark 1, 2, … across sections); definitions/assumptions have their own
  per-section counters, so "Assumption 3.1" and "Lemma 3.1" coexist. **Fix
  (standard practice):** one shared counter: `\newtheorem{theorem}{Theorem}[section]`
  then `\newtheorem{lemma}[theorem]{Lemma}` … for proposition, corollary,
  definition, assumption, remark. Everything becomes "3.1, 3.2, …" in order of
  appearance.

- [ ] **STR-6 (medium). Scope statement.** The motivation leans on the curse of
  dimensionality (l. 144–152, l. 229), but every experiment is $d = 2$. That is
  fine for a thesis benchmarking *representation* questions — but say so once
  ("high-dimensional deployment is outside the scope; the $d=2$ benchmarks isolate
  the switching-set representation question") to preempt the obvious committee
  question.

- [ ] **STR-7 (minor). Section/subsection title style:** `\section{introduction}`,
  `\section{numerical Example}` etc. — amsart small-caps the section heads so
  casing survives; unify to sentence case or title case ("Introduction",
  "Preliminaries", "The sparse learning problem", "Numerical examples"). Fix
  "Finite Spport" (l. 637) → "Finite Support". Subsubsection heads: `The $ReLU^2$
  fits the discontinuous gradient` (l. 2151) has math-italic ReLU and reads like a
  caption — "ReLU² fits the discontinuous gradient" with `\reluk` macro.

- [ ] **STR-8 (minor). A notation section is needed** (thesis-standard): the
  same symbols carry multiple meanings — $\omega$ (neuron parameter) vs $\omega =
  \dot\theta$ (pendulum velocity, §6.2); $k$ (control dimension in (P), ReLU
  power, sequence index in Cor 4.2.2's proof); $N$ (network width, dataset size
  at l. 1300, and $\mathcal{N}$ the operator); sample counts $M$ / $K$ (l. 1309,
  1455) / $N_x$ (l. 1340); target $y$ vs $V$. Rename the pendulum velocity to
  $\dot\theta$ (or the neuron parameter to $\theta$… no — keep $\omega$ for
  neurons, use $\dot\theta$ in §6.2), unify sample count to $M$, and add a
  notation table after the introduction.

---

## 4. Wording (full-polish pass: patterns; the fix phase edits section by section)

- [ ] **WORD-1 (major). Typo density is high (~45 unique):** developе→develop,
  poplular→popular, estabilised→established, traning→training,
  propertyies→properties, appraoch→approach, validataion→validation,
  pipelind→pipeline, sparcification→sparsification, magnitute→magnitude,
  representated→represented, garanteed→guaranteed, follwing→following,
  proportion→proposition (l. 317), specifid→specified, atomes→atoms, Spport→
  Support, swtiching→switching, stabalizes→stabilizes, Evoluation→Evolution
  (caption l. 1851), hypeparameter→hyperparameter, Algorithem→Algorithm
  (l. 1462), "to some extend"→extent, "Frank-Wolf"→Frank–Wolfe, "notorious
  difficult"→notoriously difficult, "a feedback a feedback" (l. 142), "other
  other" (l. 317), "fails satisfy" (l. 1026), "we can to some extend solve the
  problem by over." (l. 2090, see MATH-18), "neurones"→neurons (l. 196, 213),
  "envolop" (label), "1 global optimal solution" → "one global minimizer".
  A full spell-pass will be part of the wording batch.

- [ ] **WORD-2 (medium). Grammar patterns to fix globally:** subject–verb
  agreement ("the number of neurons don't serve", "our current network
  approximate", "ReLU² demonstrate"); missing articles ("solving HJB function is
  notorious difficult"); "w.r.t." in prose → "with respect to"; digits for small
  counts ("2 neighboring neurons", "3 classes") → words; "l.s.c." used as a noun
  everywhere ("prove the weak-* l.s.c. of") → "lower semicontinuity" at first use,
  then l.s.c.; inconsistent British/US spelling (synthesise/synthesized,
  behaviour/behavior) → pick US.

- [ ] **WORD-3 (medium). De-AI / de-hype pass (targeted, not blanket):** the §6
  prose imported from results.md carries a repeated rhetorical cadence that reads
  generated: "Raising the power buys sparsity for free", "This accuracy is not
  free: the Gaussian pays with…", "Algorithm 2 dominates", "separates from the
  field almost immediately", "the regime the sparse rectified atoms are built
  for", "Gradient augmentation is decisive and uniform in k", triadic
  em-dash-chained sentences throughout §6.2. Keep the content, convert to sober
  claims with the numbers attached; cap em-dashes at ~one per paragraph. In the
  intro: "we develope a data-driven method that harnesses the precision of the
  gradient training and sparsity promoted by the nonconvex penalty at a higher
  level" (l. 192) — say concretely what is done; "The proposition shows the power
  and limitation of the Barron space" (l. 312) → state which limitation;
  "It is in particular effective when the system is high-dimensional" (l. 229) —
  unsupported here, delete or cite.

- [ ] **WORD-4 (medium). Consistency decisions to make once and enforce:**
  (i) target symbol $V$ everywhere (drop $y$ from l. 840, 850, 871…);
  (ii) $\sigma(x;\omega)$ argument order; (iii) penalty symbols: $\phi$ (abstract),
  $\phi_{\log,\gamma}$ (log), $\psi_k$ or $\phi_k$ (fractional) — currently
  $\phi_1$, $\phi_k$, $\psi_k$, $\psi_5$, "$\phi_{1}$ (meaning log with γ=1?)" in
  §6.2 captions all float around; define $\phi_{\log,\gamma}$ and $\psi_k(z) =
  |z|^{2/(k+1)}$ once in §5 and use only those; (iv) "switching set" vs
  "switching curve" vs "switching band" vs "switching tube" — define each once
  (set = the geometric locus; band = the training-data slice; tube = the
  evaluation region d ≤ 0.3) and use consistently.

- [ ] **WORD-5 (minor). Assorted sentence repairs flagged during the read**
  (will be itemized in the fix batch): l. 104 "We assume the minimizer exists."
  (floating — attach to the standing assumptions); l. 140 "more robust compared
  to the open-loop method" (than); l. 147–151 citation-glosses "(Kunisch et al.,
  2021)" → the authors are Azmi, Kalise, and Kunisch — say "Azmi, Kalise and
  Kunisch \cite{azmi2021optimal}" and "the authors" not "the author" (also
  l. 1300); l. 154 "the sought for approximating function" → "the approximating
  function"; l. 217 "The general appraoch follows:" → "The approach has four
  steps:"; l. 235–240 the "linear inverse problem find μ such that 𝒩μ = f"
  framing appears once and is never used again — either use it or cut it;
  l. 1037 "contracted on the unit sphere" → "projected onto"; l. 1247 "we trained
  the network" (tense) and "The process is repeated a fixed time." → "a fixed
  number of times"; l. 1322 "To find the global maximal is a NP-hard problem" →
  "Finding the global maximum is NP-hard in general [cite or soften]"; l. 2051
  "In Figure X, a normal cross-section of the switching curve through the densest
  data region." (no verb); l. 2091 "The assumption is:" → "The hypothesis is
  that…". Also the intro citation list for data generation (l. 220) includes
  \cite{barzilai1988two} — the BB step-size paper is not a data-generation
  method; it belongs only at l. 1300/1442 where BB updates are meant.

---

## 5. Plots & layout

- [ ] **PLOT-1 (medium). Number formatting is inconsistent across tables:**
  `tab:alg1_gradient`/`tab:alg1_sparsity`/`tab:rs_oversampling` use `8.61e-2`
  (via `\text{e-}`) while `tab:alg1_alpha`/`tab:alg2_gradient` use
  $3.57 \times 10^{-1}$. Pick one (recommend $m.mm \times 10^{-e}$, or siunitx
  `\num{8.61e-2}` everywhere) — one macro, applied to all six tables.

- [ ] **PLOT-2 (medium). Unused/legacy figures in `papar/plot/`:** 7 files are
  not referenced: `H1_activation.png`, `L2_activation.png`, `L2_power.png`,
  `activation.png` (legacy, no experiment source — delete), and
  `near_far_dumbbell.png`, `error_vs_distance_value.png`,
  `error_vs_distance_gradient.png` (current region_split outputs). The dumbbell
  figure directly supports §6.2's "which atoms fit best" claim (ReLU² dominates
  both regions; leaky ReLU runner-up) that currently rests on the frontier alone
  — **consider adding it** to §6.2.2 rather than deleting; the error-vs-distance
  pair backs the "interior price" discussion if NUM-4/§3-price content is kept.

- [ ] **PLOT-3 (minor). 3D view-angle uniformity:** house style fixes
  elev=15/azim=−105 for 3D panels; `v.png`/`dv.png` (from `00_openloop/vdp`,
  which follows the external paper's figure code) use a different view than the
  `surface_*.png` family. Acceptable if intentional (data-only vs learned-surface
  families), but the thesis puts them pages apart — verify the two families each
  look internally consistent; regenerate `v.png` at the house angle if you want
  uniformity (fix goes through `experiments/00_openloop/vdp/generate.py`).

- [ ] **PLOT-4 (minor). Five-across subfigure row** (l. 2008–2049, 0.19\textwidth
  panels): at the 11pt re-layout with standard margins these panels shrink
  further; check legibility after TEX-1 and consider a 3+2 arrangement (the
  results.md layout) if tick labels become unreadable.

- [ ] **PLOT-5 (minor). Caption practice:** captions correctly carry the
  identifying info (house rule ✓). Two repairs: Fig. `fig:rs_surfaces` caption
  (l. 2045–2047) — "the $\mathrm{ReLU}^{k}$ functions are trained with the $\psi_k$ penalty,
  while the others with $\phi_1$" — lowercase sentence start, and $\phi_1$ here
  means the log penalty at γ=1 (see WORD-4(iii)); Fig. 1 caption fixes are
  NUM-2. Also `fig:summary_weights_raw` caption (l. 1877): "Dot color indicates
  sign, dot size outer weights magnitute" — typo + grammar; and l. 1863/1869
  sub-captions use `$e^{-x^2} + \alpha\,\phi_1$" style formulas as panel labels —
  fine, but make the penalty symbols match WORD-4(iii).

- [ ] **PLOT-6 (minor). Transect true-PMP spike:** the true-PMP curve in
  `transect_normal_gradient.png` has a rectangular excursion near s ≈ −0.28
  (envelope nearest-neighbor artifact). Either mention it in the caption
  ("the spike at s ≈ −0.3 is a nearest-neighbour artifact of the envelope
  reference") or mask it in `region_split/analysis.py` — currently a sharp-eyed
  reader will ask.

---

## 6. LaTeX code

- [ ] **TEX-1 (major). Regulation-conformant layout:** switch to 11pt, drop
  `\linespread{1.3}` (or ≤1.05), replace the 15mm `geometry` with standard
  thesis margins (~2.5cm; binding offset per Prüfungsbüro). Keep amsart or move
  to `amsbook`/`scrreprt` — amsart at 11pt is fine for a math thesis. **Then
  re-check the ≤50-page cap** — the current 46 pp at wide text/12pt/1.3-spread
  will land somewhere new after reflow + added front matter/conclusion; §5
  compression (STR-3) is the reserve lever.

- [ ] **TEX-2 (medium). Preamble hygiene:** remove unused packages — `tikz`,
  `ulem`, `multicol`, `multirow`, `marvosym`, `stmaryrd` (all have zero uses;
  verified) — and `url` (hyperref covers it); load `hyperref` (+`bookmark`)
  *after* amsmath/amsthm/amssymb (currently before, l. 21 vs 26); drop the
  `\DeclareMathAlphabet{\mathcal}` override unless intentional; `fleqn` — decide
  (flush-left displays are unusual in math theses; default centering
  recommended); add `microtype`.

- [ ] **TEX-3 (medium). Operators:** `\DeclareMathOperator*` is only right for
  operators taking limits underneath (`\esssup`); `dist`, `span`, `sign`,
  `supp`, `dom`, `cl`, `ran`, `atom`, `cont` should be unstarred. The `\div`
  redefinition shadows the TeX primitive — harmless as done, but `\dvg` or
  `\operatorname{div}` inline avoids the `\AtBeginDocument` dance.

- [ ] **TEX-4 (medium). References & citations:** `\bibliography{ref.bib}` →
  `\bibliography{ref}`; `plain` style with numeric labels is fine, or `alpha`
  for a thesis; make all equation references `\eqref` (currently 7 `\eqref` vs
  11 `(\ref{...})`); replace the hard-coded "Theorem~5.2" (MATH-18(ii));
  prefix labels systematically (`thm:`, `lem:`, `prop:`, `cor:`, `eq:`, `sec:`)
  — currently `envolop`, `l.s.c.` (dots in a label), `opt_cdt`, `pen1` etc.;
  add non-breaking ties before every `\ref`/`\cite` ("Theorem~\ref{…}").
  Consider `cleveref` (`\cref`) to stop hand-typing "Theorem"/"Assumption".

- [ ] **TEX-5 (medium). Theorem environments:** implement the shared counter of
  STR-5; delete the redundant `\renewcommand{\thetheorem}` /
  `\renewcommand{\thedefinition}` lines; `remark` should get the shared counter
  too; the unnumbered `example` env is unused — remove.

- [ ] **TEX-6 (minor). Display-math mechanics:** blank lines around displays
  create stray paragraph breaks (multiple sites, e.g. after l. 374, 1035);
  punctuation at display ends is inconsistent (commas/periods missing or doubled
  — sweep once); `\bigg(...\bigg)` used where `\Big` or `\left...\right` is
  appropriate (and around single symbols, e.g. l. 527); `\quad\\` as a proof
  opener (l. 451, 722, 1200) → `\leavevmode\\` or just start the text;
  `$\text{e-}2$` constructs → PLOT-1's macro; `*` as multiplication (l. 708) →
  `\cdot` or juxtaposition; `\dots` in the ReLU^k gradient display (l. 1090)
  → `\vdots`; the tag `\tag{$P_{\Phi_1}$}` etc. ✓ good practice, keep.

- [ ] **TEX-7 (minor). Algorithms:** `algorithmic` (all-caps `\STATE`) is the
  legacy package — `algpseudocode` is the maintained successor (mechanical
  rename); Algorithm 1/2 bodies use inline math with text subscripts
  (`N_{\mathrm{trial}}` ✓ good); the `$t < T$` loop bound collides with the
  time-horizon $T$ of (P) — rename to $T_{\mathrm{out}}$ or `maxit`.

- [ ] **TEX-8 (minor). Misc:** `\texorpdfstring` used correctly for the §4 head ✓;
  add it for any math that lands in bookmarks after retitling; `\allowdisplaybreaks`
  ✓ fine; `\raggedbottom` ✓ fine for a thesis; hyperref `colorlinks=true,
  linkcolor=blue` — for the printed/submitted copy switch to `hidelinks` or a
  dark link color.

---

## 7. Title & abstract proposal

**Proposed title (recommended):**

> Gradient-Augmented Sparse Regression for Hamilton–Jacobi–Bellman Value
> Functions: Shallow Networks with Nonconvex Regularization

Alternatives:
1. *Nonconvex Sparse Regularization for Gradient-Augmented Learning of HJB Value
   Functions by Shallow Neural Networks*
2. *Sparse Shallow Networks for Optimal Feedback Control: Nonconvex Penalties,
   Gradient-Augmented Training, and the Switching-Set Limit*
3. (short) *Sparse Neural Approximation of HJB Value Functions with Nonconvex
   Regularization*

**Abstract draft** (≈170 words; assumes HEAD-1(a) so the theory claims are honest):

> We study the approximation of value functions of nonlinear optimal control
> problems from data generated by Pontryagin's maximum principle, which supplies
> values and gradients of the value function along open-loop trajectories. The
> approximating class is a shallow neural network, formulated as an integral
> network over a measure on a compact parameter set, trained with a
> gradient-augmented ($H^1$) least-squares loss and a nonconvex sparsity
> penalty. For this functional we prove existence of global minimizers,
> necessary and sufficient optimality conditions, and finite support of local
> minimizers, extending the nonconvex regularization theory of Pieper and
> Petrosyan to gradient-augmented losses and a broader activation class. For
> positively $k$-homogeneous activations we derive an equivalent
> sphere-constrained problem with a fractional-power penalty
> $|c|^{2/(k+1)}$ and establish optimality conditions and existence despite the
> singular penalty derivative. A node-insertion algorithm with a semismooth
> Newton inner solve implements both penalty classes. On a smooth benchmark (Van
> der Pol) the method attains near-optimal feedback at a fraction of the neurons
> of convex $\ell^1$ training; on the pendulum swing-up, whose value function has
> a gradient discontinuity along a switching set, only piecewise-quadratic
> ($\mathrm{ReLU}^2$) atoms both fit the kink and yield a working feedback law
> across it — locating the method's representation limit at nonsmooth structure
> rather than data coverage.

---

## Suggested fix batches (after your approval markup)

1. **Batch M (math):** HEAD-1(a) reframing + MATH-1…-5 proof repairs + MATH-8,
   -10…-19 local fixes; MATH-6/-7 rewrite of §2.2.
2. **Batch N (numerics):** NUM-1…-9 (text/caption updates; add cost table;
   parameter sentence; verify NUM-6 against MLflow; results.md update for the
   two uncurated columns).
3. **Batch S (structure):** STR-1 front matter + conclusion; STR-2 contributions/
   outline; STR-3 compression; STR-4 §4 reorder; STR-5 counter unification (with
   TEX-5); STR-7/-8 titles + notation table.
4. **Batch W (wording):** WORD-1…-5, section by section (5–6 passes).
5. **Batch P/T (plots + LaTeX):** PLOT-1…-6, TEX-1…-8; `make paper-figures`
   after any `analysis.py` change; rebuild and re-check the 50-page cap.
