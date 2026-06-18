# T2 — Separation theorem on the exponential distance function (draft)

Status: research draft, 2026-06-10. Flagship result of the semiconcave-approximation
theory program (ladder below). Chat-derived; rigor gaps are listed explicitly at the end.

## The program in one paragraph

Goal: prove why the semiconcave architecture approximates HJB value functions with
fewer neurons than signed shallow ridge networks. The unifying yardstick is the
dictionary variation norm $\gamma_{\mathcal{D}}(f) = \min \|\mu\|_M$ over representations $f = \int \varphi \, d\mu$, $\varphi \in \mathcal{D}$:
finite $\gamma$ gives the dimension-free Maurey/Barron rate $\gamma \cdot N^{-1/2 - (2k+1)/(2d)}$ (Yang–Zhou),
infinite $\gamma$ forces the curse; for ReLU ridges $\gamma$ is *exactly* the $\mathcal{R}$-norm
$\|f\|_{\mathcal{R}} = \gamma_d \|\partial_b^{d+1} \mathcal{R}\{f\}\|_1$ (Ongie et al., Radon transform $\mathcal{R}$). "Fewer neurons" must
therefore mean: for HJB value functions $V$, the min/semiconcave variation norm is small
while the ridge variation norm is large or infinite. The PDAP/representer layer
(nonconvexity references) converts variation norms into literal atom counts.

Theorem ladder:

| # | Statement | Status |
|---|---|---|
| T1 | 1-D: $\mathrm{TV}(V'') \le 2C|\Omega| + 2L$ for $C$-semiconcave $L$-Lipschitz $V$, independent of #shocks; cone representation automatically nonneg | **proved** — `t1-1d-budget-lemma.md` |
| T2 | Separation on $v_d$: $2d$ exact atoms (min architecture) vs $\|v_d\|_{\mathcal{R}} = \infty$ (ridge nets) | **this note; (B) proved for d = 2**, $d \ge 3$ modulo gaps |
| T2′ | Incomparability converse: a single smooth ridge $s(w \cdot x)$ costs 1 atom / 1-D rates for ridge nets, but $\Omega(n^{-2/d})$ for shared-Hessian min-quadratic nets | **proved** (unconstrained class, Lemma P) — `t4-minplus-capacity.md` |
| T3 | Cone-optimality: minimal-TV signed ridge measure of a ridge-convex $g$ is nonnegative ($\implies c_i \ge 0$ constraint is free) | open |
| T4 | HJB capacity, rank-conditioned: min-plus rank $m$ conserved by LQ flows $\implies$ $m$ atoms exact (DDM Prop 1; McE07); generic $C^2$ targets face $\Omega(n^{-2/d})$ for shared-Hessian atoms (GMQ Thm 3.3) — **no uniform no-curse theorem exists**; favorable HJB class = Hessian-saturated (Bregman-volume attenuation) | framed — `t4-minplus-capacity.md` |
| T5 | PDAP sparsity under the cone: atom count via one-sided dual certificate contact structure | open |

Architecture ground truth (this repo, `src/models/semiconcave.py`):
`SemiconcaveModel` is $V(x) = \tfrac{C}{2}\|x\|^2 - g(x)$, $g(x) = \sum_i c_i\, \sigma(w_i \cdot x + b_i)^p$, $c_i \ge 0$ —
the Legendre-dual form ($g$ convex $\implies V$ $C$-semiconcave), but with **ridge-convex** $g$.
Since convex $g$ = sup of affine maps, $V$ = inf of paraboloids with *shared* Hessian $C I$.
The Kunisch–Vásquez-Varas min-architecture (per-branch Hessians) is strictly richer.
T2 separates the min architecture from **all** ridge nets, including both repo models.

## Setup

Target (Kunisch–Vásquez-Varas §5.1), considered on all of $\mathbb{R}^d$:

$$
v_d(x) = \min_{i=1,\dots,2d} \varphi_i(x), \qquad
\varphi_i = e^{-\frac12 |x - e_i|^2}, \quad
\varphi_{d+i} = e^{-\frac12 |x + e_i|^2}.
$$

$v_d$ is a viscosity solution of $H(\nabla v, v) = 0$ with $H(p,a) = |p|^2 + 2 \log(|a|)\, a^2$
for $a \ne 0$, $H(p,0) = |p|^2$ (their Lemma 5.1). Explicit constants, uniform in $d$:

- Lipschitz: $L = \max |\nabla \varphi_i| = e^{-1/2} \approx 0.607$ (at $|x - c| = 1$);
- Hessian operator norm: $\|D^2 \varphi_i\| \le 1$ (attained at the center, $D^2\varphi = -I$);
- semiconcavity: top Hessian eigenvalue $\sup_u (u-1)e^{-u/2} = 2e^{-3/2} =: C^* \approx 0.446$
  (at $|x-c|^2 = 3$), so $v_d$ is $C^*$-semiconcave.

Note: min picks the smallest $\varphi_i$ = the **farthest** center ($\exp \circ (-\tfrac12 \mathrm{dist}^2)$ is
decreasing). The active partition is the fan of polyhedral cones
$K_i = \{x : x \cdot c_i \le x \cdot c_j \ \forall j\}$ ($c_i$ = center of branch $i$), e.g. the branch centered at $-e_1$
is active on the cone $\{x_1 \ge |x_j| \ \forall j\}$ around $+e_1$.

Competing architectures:

- **Min/semiconcave net**: $v_{n,\varepsilon} = \psi_{n,\varepsilon} \circ \Phi_n$, soft-min of $n$ $C^2$ atoms with the
  Moreau regularizer $g_{\varepsilon,M}$ (Kunisch–VV (2.13), Remark 2.2).
- **Shallow ridge net**: $g_\theta(x) = \sum_{i=1}^N a_i\, \sigma_k(w_i \cdot x - b_i) + v \cdot x + c$, $\sigma_k = \mathrm{ReLU}^k$,
  $w_i \in S^{d-1}$, weight cost $\|a\|_1$. Infinite-width closure $= \mathcal{F}_{\sigma_k}(M)$ (Yang–Zhou);
  for $k = 1$ the minimal cost equals the $\mathcal{R}$-norm exactly (Ongie et al., Theorem 1).

## Theorem (min vs ridge separation, $d \ge 2$)

**(A) Upper bound, min architecture — $n = 2d$ atoms, exact kinks.**
With the $2d$ Gaussian atoms and the Moreau soft-min:

1. $\|v_{2d,\varepsilon} - v_d\|_{C(\bar\Omega)} \le (2d-1)\,\varepsilon$  (KVV (2.22));
2. for $\delta > 0$ and $\varepsilon \in \bigl(0, \tfrac{\delta}{2(2d-1)}\bigr)$: $\nabla v_{2d,\varepsilon} = \nabla v_d$ a.e. on $\Omega_\delta$ — *exact*
   gradients off the $\delta$-neighborhood of the kink set. Reason: KVV (4.13) bounds the
   error by $2L\bigl(g'_\varepsilon(0) + 1 - g'_\varepsilon(\delta)^{2d-1}\bigr)$, and the Moreau regularizer has
   $g'_\varepsilon(0) = 0$ and $g'_\varepsilon(s) = 1$ for $s \ge \varepsilon$; within the stated range $\varepsilon < \delta$, so the
   bound is exactly $0$;
3. $\|\nabla v_{2d,\varepsilon} - \nabla v_d\|_{L^p(\Omega)} \lesssim \mathrm{poly}(d, |\Omega|) \cdot \bigl((2d-1)\varepsilon\bigr)^{1/(1+p)}$  (KVV (4.9), exact atoms);
4. $H(\nabla v_{2d,\varepsilon}, v_{2d,\varepsilon}) \to 0$ uniformly on $\Omega_\delta$, pointwise on $\bar\Omega$  (KVV Cor 3.1, (3.23);
   condition (3.20) verified below).

Neuron count **$2d$ — linear in $d$** — with the semiconcavity constant $\le C^*$ preserved.

**(B) Lower bound, ridge architecture — infinite representational cost.** For $d \ge 2$:

$$
\|v_d\|_{\mathcal{R}} = +\infty,
\qquad \text{and since } \nabla v_d(\infty) = 0:
\qquad \bar{\mathcal{R}}(v_d) = \bar{\mathcal{R}}_1(v_d) = +\infty.
$$

Consequently (Ongie Thm 1, Thm 2, defs (7)/(12)): every sequence of finite-width
shallow ReLU networks converging to $v_d$ uniformly on compacta — with or without the
free linear unit — has total weight cost $C(\theta_k) \to \infty$. In particular $v_d \notin \mathcal{F}_{\sigma_1}(M)$
for any $M < \infty$: the dimension-free variation-space rate is unavailable at any budget.

**(C) Extensions.** The same ray-decay computation excludes $v_d$ from $\mathcal{F}_{\sigma_k}(M)$ for
every $k \ge 1$ (covers the repo's power-2 activations), and adding a global quadratic
$\tfrac{C}{2}|x|^2$ leaves the obstruction untouched (its Laplacian is constant) — so (B) applies
to the repo's cone-constrained `SemiconcaveModel` ridge expansion as well, not just
`SignedModel`.

## Proof of (A) — assembly from KVV

Only (3.20) needs checking: at each $x$ and each active index $i$, $\nabla\varphi_i(x) \in D^* v_d(x)$.
Each active cone $K_i$ is full-dimensional and every boundary point of $K_i$ is a limit of
interior points; on $\mathrm{int}\, K_i$, $v_d = \varphi_i$ is smooth with $\nabla v_d = \nabla \varphi_i$. Approaching $x$ from
$\mathrm{int}\, K_i$ gives $\nabla\varphi_i(x) \in D^* v_d(x)$. Points 1–4 are then verbatim instances of KVV
(2.22), (4.13), (4.9), (3.23) with $\varphi_{i,m} = \varphi_i$ (zero atom-approximation error). $\blacksquare$

## Proof of (B) — rigorous for $d = 2$; general $d$ is the program

Throughout, Fourier convention $\hat f(\xi) = \int f(x)\, e^{-i \xi \cdot x} dx$, and $F_b$ denotes the
1-D Fourier transform in the offset variable $b$. Write $v := v_2 \in L^1(\mathbb{R}^2) \cap \mathrm{Lip}(\mathbb{R}^2)$;
$v$ is even, so $\hat v$ is real and even.

**Step 1: singular part of $\Delta v$.** $v$ is smooth off the two diagonals
$D_1 = \{x_1 = x_2\}$ (tangent $\tau = (1,1)/\sqrt2$, normal $\nu = (1,-1)/\sqrt2$) and
$D_2 = \{x_1 = -x_2\}$ (tangent $\nu$, normal $\tau$). The active tie on $D_1$ at $x = t\tau\sqrt2 \equiv (t,t)$
is between the branches centered at $-e_1, -e_2$ for $t > 0$ (min picks the farthest
centers) and at $+e_1, +e_2$ for $t < 0$; the gradient-jump computation gives, on both
diagonals by symmetry, the **even, continuous, strictly positive** density

$$
J(t) = \sqrt2 \, e^{-(t^2 + |t| + \frac12)},
\qquad
\Delta v = \rho\, dx \;-\; J\, dH^1\big|_{D_1} \;-\; J\, dH^1\big|_{D_2},
$$

with $\rho = \sum_i \mathbf{1}_{K_i} \Delta\varphi_i \in L^1 \cap L^\infty$ (Gaussian decay). $J$ has a derivative kink at
$t = 0$ but is itself continuous; no point mass at the crossing (capacity of a point
is zero for the Laplacian of a Lipschitz function). Since $\Delta v$ is a finite measure
and $v \in L^1$, the identity $\widehat{\Delta v}(\xi) = -|\xi|^2\, \hat v(\xi)$ holds pointwise between continuous
functions. Key transform facts:

$$
\widehat{J\, dH^1|_{D_1}}(\xi) = \hat J(\xi \cdot \tau),
\qquad
\widehat{J\, dH^1|_{D_2}}(\xi) = \hat J(\xi \cdot \nu)
\quad \text{(constant in the resp. normal frequency)},
$$

where $\hat J$ denotes the FT of the **arclength** density $u \mapsto J(u/\sqrt2)$:
$\hat J(s) = \sqrt2\, \hat J_t(\sqrt2\, s)$ — the Jacobian of $dH^1 = \sqrt2\, dt$ on $D_1$ (referee fix 2026-06-11);

$$
\kappa := \hat J(0) = \int_{D_1} J \, dH^1 = \sqrt2 \int J(t)\, dt = 2\sqrt{\pi}\, e^{-1/4}\, \mathrm{erfc}(\tfrac12) \approx 1.32 > 0,
$$

$$
|\hat J(s)| \le \frac{C_J}{1 + s^2} \quad (J \in W^{1,1} \text{ with } J' \in \mathrm{BV};\ C_J \approx 2.5 \text{ after rescaling}),
$$

$$
\hat\rho \in C_0(\mathbb{R}^2): \quad \epsilon(R) := \sup_{|\xi| \ge R} |\hat\rho(\xi)| \to 0 \quad \text{(Riemann–Lebesgue)}.
$$

Fix $\rho_0 \in (0,1]$ with $\hat J(s) \ge \kappa/2$ for $|s| \le 1.1\,\rho_0$.

**Step 2: the pairing identity.** For even real $\psi \in \mathcal{S}(S^1 \times \mathbb{R})$, Ongie's Definition 1
evaluates, via Parseval and the dual-slice formula
$\widehat{\mathcal{R}^*\psi}(\xi) = (2\pi)^{d-1}\, \hat\psi(w_\xi, |\xi|)\, |\xi|^{-(d-1)}$ (even $\psi$):

$$
\begin{aligned}
P(\psi) :=\; & -\gamma_d \,\bigl\langle v, \, (-\Delta)^{(d+1)/2} \mathcal{R}^*\psi \bigr\rangle \\
=\; & -\gamma_d \,(2\pi)^{-1} \int_{\mathbb{R}^d} \hat v(\xi)\, |\xi|^2\, \hat\psi(w_\xi, |\xi|)\, d\xi \\
=\; & \;\;\gamma_d \,(2\pi)^{-1} \int_{S^{d-1}} \int_0^\infty \widehat{\Delta v}(\sigma w)\, \hat\psi(w, \sigma)\, \sigma^{d-1}\, d\sigma\, dw,
\end{aligned}
$$

absolutely convergent ($|\widehat{\Delta v}| \le \|\rho\|_1 + 2\|J\|_1$ bounded; $\hat\psi(w, \cdot)$ uniformly Schwartz).
By Definition 1, $\|v\|_{\mathcal{R}} \ge P(\psi)$ for every admissible $\psi$ ($\|\psi\|_\infty \le 1$, even, Schwartz).

**Step 3: matched-filter witness $\implies$ log-divergence.** Fix $\eta(b) = e^{-b^2/2}$
(so $0 \le \eta \le 1$, $\eta$ even, $\hat\eta = \sqrt{2\pi}\, e^{-\sigma^2/2} > 0$). For parameters
$0 < \theta_{\min} < \theta_0 \le 1/10$, let $\chi : [0,\pi] \to [0,1]$ be smooth, supported in $[\theta_{\min}, \theta_0]$,
$\chi = 1$ on $[2\theta_{\min}, \theta_0/2]$. Parametrize $w(\theta) = \cos\theta\, \nu + \sin\theta\, \tau$ and define the
witness **with frequency matched per angle**:

$$
\psi\bigl(w(\theta), b\bigr) = -\,\chi(\theta)\, \cos\!\Bigl(\frac{\rho_0}{\theta}\, b\Bigr)\, \eta(b),
$$

extended to $\pm(w,b)$ by evenness (set $\psi(-w,-b) = \psi(w,b)$; $\psi$ is even in $b$, so this is
consistent), and $\psi = 0$ off the fan. Then $\psi \in \mathcal{S}(S^1 \times \mathbb{R})$, $\|\psi\|_\infty \le 1$, and

$$
\hat\psi\bigl(w(\theta), \sigma\bigr) = -\tfrac12\, \chi(\theta)\, \Bigl[\, \hat\eta\bigl(\sigma - \tfrac{\rho_0}{\theta}\bigr) + \hat\eta\bigl(\sigma + \tfrac{\rho_0}{\theta}\bigr) \Bigr].
$$

Insert into $P(\psi)$ and split $\widehat{\Delta v} = \hat\rho - \hat J(\sigma\, w \cdot \tau) - \hat J(\sigma\, w \cdot \nu)$ (Step 1):

- **Main term (sheet $D_1$).** For $w = w(\theta)$: $w \cdot \tau = \sin\theta$, and on the window
  $|\sigma - \rho_0/\theta| \le K$ we have $|\sigma \sin\theta - \rho_0| \le K\theta + \rho_0 \theta^2/6 \le 0.1\,\rho_0$ for $\theta_0$ small,
  so $\hat J(\sigma \sin\theta) \ge \kappa/2$ there. With weight $\sigma^{d-1} = \sigma \approx \rho_0/\theta$:

  $$
  \text{contribution per angle} \;\ge\; \frac{\kappa}{2}\cdot\frac{\rho_0}{\theta}\cdot c_K - \text{tails},
  \qquad c_K := \tfrac12 \int_{|u| \le K} \hat\eta(u)\, du > 0,
  $$

  and the Schwartz tails of $\hat\eta$ ($|\sigma - \rho_0/\theta| > K$, plus the $\hat\eta(\sigma + \rho_0/\theta)$ image window)
  cost at most $\varepsilon_K (1 + \rho_0/\theta)$ with $\varepsilon_K \to 0$; choose $K$ with $\varepsilon_K \le \kappa\, c_K/100$.
  Integrating $d\theta$ over $[2\theta_{\min}, \theta_0/2]$:

  $$
  \text{Main} \;\ge\; c_1\, \kappa\, \rho_0 \, \log\!\Bigl(\frac{\theta_0}{4\,\theta_{\min}}\Bigr),
  \qquad c_1 > 0 \text{ absolute}.
  $$

- **Cross sheet ($D_2$).** $w \cdot \nu = \cos\theta \ge 0.99$ on the fan, so $|\hat J(\sigma \cos\theta)| \le 2C_J/\sigma^2$
  and the $\sigma$-weighted window integral is $O(C_J\, \theta / \rho_0)$ per angle — total $O(C_J\, \theta_0^2)$,
  bounded.

- **Regular part.** $|\hat\rho(\sigma w)| \le \epsilon\bigl(\rho_0/(2\theta)\bigr)$ on the central window; choosing $\theta_0$ with
  $\epsilon\bigl(\rho_0/(2\theta_0)\bigr) \le \kappa\, c_K/8$ makes this $\le$ half the main term; Schwartz tails again $O(1)$.

Hence, with $\rho_0, K, \theta_0$ fixed in that order ($\rho_0$ from $\hat J$ alone; $K$ from the $\hat\eta$ tails
vs $\kappa\, c_K$; $\theta_0$ from $\epsilon(\cdot)$ and the window expansion — no circularity), every error
term is either $\le$ half the per-angle main term or an integrable $O(C_J\, \theta)$
cross-term, so

$$
P(\psi_{\theta_{\min}}) \;\ge\; \tfrac12\, c_1\, \kappa\, \rho_0 \, \log\!\Bigl(\frac{\theta_0}{4\,\theta_{\min}}\Bigr) - O(C_J\, \theta_0^2)
\;\longrightarrow\; \infty
\qquad \text{as } \theta_{\min} \downarrow 0.
$$

By Definition 1, $\|v_2\|_{\mathcal{R}} = \infty$. $\blacksquare$ (End-to-end numerical check, referee pass
2026-06-11: $P$ grows by a constant increment $\approx 0.0125$ per halving of $\theta_{\min}$ —
exact log-tracking of the predicted divergence.)

**Step 4: bookkeeping.** The free linear unit and constant affect only $\xi = 0$ and
produce no surface singularity; the even and odd parts of a representing measure
each have TV $\le$ the total (Ongie Prop 7), so restricting to even measures — as
Definition 1's witnesses do — loses no generality; and
$\nabla v(\infty) = \lim_r (c_d\, r^{d-1})^{-1} \oint_{|x| = r} \nabla v \, ds = 0$ by Gaussian decay, so Ongie
Thm 2 gives $\bar{\mathcal{R}}(v_2) = \bar{\mathcal{R}}_1(v_2) = \|v_2\|_{\mathcal{R}} = \infty$ also without the free linear unit. $\blacksquare$

**Interpretation.** Per direction at angle $\theta$ from the kink normal, any
representing measure $\alpha$ must supply $F_b$-content of size $\gtrsim \kappa\, \rho_0/\theta$ at frequency
$\approx \rho_0/\theta$ (this is what the witness extracts), and $\int d\theta/\theta$ diverges — each dyadic
angular band demands a fixed mass quantum. A *constant*-density kink concentrates
$\hat J$ to a delta in the tangential frequency: the fan collapses to one direction and
the demand is a single atom — exactly the representable case. The divergence is
logarithmic: the obstruction is real but marginal, consistent with ridge nets
"coping" numerically at small scales while never converging with bounded norm.

**General $d \ge 3$.** Steps 1–3 are dimension-generic (per-angle demand
$\kappa\, \sigma^{d-1} \cdot \sigma^{-(d-1)}$ integrates to the same $\int d\theta/\theta$ against the $(d-2)$-sphere of
azimuths), but two new technical items appear: faces are *cones*, not full
hyperplanes (their FTs gain edge terms), and the kink density $J_F$ must be derived
on each face. See gaps 1–2.

**(C):** a $\mathrm{ReLU}^k$ atom has ray-FT $\propto \sigma^{-(k+1)}$, so for $k \ge 2$ the per-band supply
shrinks while the demand is unchanged — divergence a fortiori (band at angle $\theta$ now
needs mass $\sim (\rho_0/\theta)^{k-1}\, \kappa$). The quadratic head shifts $\Delta$ by the constant $C d$: no
surface-singular part, Steps 1–3 unaffected. (Rigorous $k \ge 2$ statement: gap 3.) $\blacksquare$

## Remarks

- **$d = 1$ sanity check.** The kink of $v_1$ is an *atom* of $v''$: one ReLU neuron
  reproduces it exactly, $\mathrm{TV}(v_1'') < \infty$, consistent with T1's budget $2C^*|\Omega| + 2L$.
  The dyadic-band divergence has no analogue ($S^0$ is discrete). Separation genuinely
  starts at $d = 2$.
- **Position vs Ongie et al.** Their Prop 5 (compactly supported PWL $\implies \|\cdot\|_{\mathcal{R}} = \infty$)
  does not apply to $v_d$ (not PWL, not compactly supported), so (B) is a new
  instance; it upgrades their Example 5 (pyramid; depth separation) to an
  **architecture separation on an HJB viscosity solution**: the cheap side is not a
  3-layer ReLU net but the semiconcavity-preserving min-network, whose atoms have
  physical meaning (branches = cost-to-go along characteristics, kink set = the
  switching set; Subbotina's representative formula).
- **Smooth parts are cheap — the kinks carry the whole separation.** Ongie Ex. 3:
  fixed-profile radial bumps cost $\sim d^2$ in the $\mathcal{R}$-norm; so the Gaussian branches are
  $\mathrm{poly}(d)$-cheap for ridge nets and the divergence in (B) is purely the shock
  structure. This sharpens the program claim: semiconcave shocks, not smoothness,
  are what ridge dictionaries cannot afford.
- **Quantitative refinement (conjecture).** Truncating the fan at band $m_{\max} \sim
  \log(1/\varepsilon)$ suggests the ridge weight mass needed for sup-error $\varepsilon$ scales like
  $M(\varepsilon) \gtrsim \kappa \log(1/\varepsilon)$, i.e. the proven gap is "$\infty$ vs $2d \cdot \max|a_i|$ at exactness"; a
  neuron-count lower bound at fixed $\varepsilon$ should go through Yang–Zhou's
  pseudo-dimension route applied to a class of $v_d$-type targets.
- **Consequence for this repo (three-way, sharpened by GMQ).** Each architecture
  is blocked on $v_d$ by a different theorem:
  (i) signed ridge (`SignedModel`) and ridge-convex cone model
  (`SemiconcaveModel` with $\sigma^p$ ridges) — blocked by (B)/(C): weight cost $\to \infty$;
  (ii) the *max-affine limit* of `SemiconcaveModel` (min of paraboloids with
  shared Hessian $C I$) — handles the $2d$ kinks but pays $\Omega(n^{-2/d})$ on the smooth
  Gaussian branches ($C^* I - D^2\varphi_i \succeq \varepsilon_0 I$ there, e.g. $(C^*+1) I$ at branch centers $\implies$
  Lemma P Cor 2 in `t4-minplus-capacity.md`, now a proved lower bound);
  (iii) a **per-branch-curvature min-quadratic head** (KVV parametrization with
  quadratic $\xi$; DDM architecture) — exact with $2d$ atoms. Testable PDAP
  prediction: atom counts on $v_d$-type targets scale as (i) growing without bound
  as $\varepsilon \downarrow 0$ with diverging weight mass, (ii) $\approx \varepsilon^{-d/2}$, (iii) saturating at $\approx 2d$.
  See `t4-minplus-capacity.md` for the full two-sided picture.

## Rigor gaps to close

~~Gap "Step 2 disintegration" — closed 2026-06-10~~: the matched-filter witness in
Step 3 works directly inside Ongie's Definition 1; no per-ray disintegration of $\alpha$
is needed. $d = 2$ is now fully rigorous modulo routine constant-chasing
(dual-slice constant, double-cover bookkeeping) and the three items below.

1. **$d \ge 3$ cross-face/edge decay**: faces are cones; edges (codim $\ge 2$) can slow
   their FT decay to $\sim r^{-1}$; make the error-vs-main comparison uniform on the
   fan (the main term is unaffected — the witness sits on one face's normal fan).
2. **General-$d$ kink density**: derive $J_F$ in closed form on each face by rotational
   reduction to the $(c_i, c_j)$ plane (the $d = 2$ computation generalizes), and verify
   $\int_F J_F \, dH^{d-1} > 0$ with the cone (not full-hyperplane) support.
3. **$k \ge 2$ slice criterion**: state and prove the $\mathcal{F}_{\sigma_k}$ analogue of the Step 2
   pairing identity (one page; Yang–Zhou's $\mathcal{F}_{\sigma_k}(M)$ or Parhi–Nowak $\mathcal{R}\mathrm{TV}^{k+1}$ as
   the home), so (C) covers the repo's power-2 activations rigorously.
4. **Constant-chasing pass on Step 2/3**: pin the non-symmetric FT convention
   against Ongie's symmetric one in their (18); fix the double-cover factor
   $c \in \{1, 2\}$ in the dual-slice formula (state whether even $\psi$ is folded to the
   upper half-circle or counted on antipodal pairs); give the honest cross-sheet
   per-angle constant (the $O(C_J\, \theta/\rho_0)$ estimate hides a low-frequency window
   dip); finish the $c_1, c_K, \varepsilon_K$ bookkeeping. (The $\sqrt2$ arclength Jacobian in the
   sheet transform — found by the 2026-06-11 referee pass — is fixed in Step 1.)

## References (local paths under /Users/chaoruiz/Documents/NotePaper/MasterThesis/)

- Kunisch, Vásquez-Varas (2026). *Structure Preserving Approximation of Semiconcave
  Functions*. arXiv:2602.07770. — `representation theorem/semiconcave_approximaton.pdf`
- Ongie, Willett, Soudry, Srebro (2019). *A Function Space View of Bounded Norm
  Infinite Width ReLU Nets*. arXiv:1910.01635. — `representation theorem/1910.01635v1.pdf`
- Yang, Zhou (2025). *Optimal Rates of Approximation by Shallow ReLU^k Neural
  Networks…* Constr. Approx. 62:329–360. — `representation theorem/optimal_rate_Relu^k.pdf`
- Petrosyan, Dereventsov, Webster (2020). *Neural network integral representations
  with the ReLU activation function*. PMLR 107:128–143. — `representation theorem/Relu(15)_IntgRep.pdf`
- Subbotina (2006). *The method of characteristics for Hamilton–Jacobi equations…*
  J. Math. Sci. 135(3). — `PDE/char_HJB.pdf`
- Bach (2017). *Breaking the curse of dimensionality with convex neural networks*.
  JMLR 18. — `CG/breaking the curse of dimensionality.pdf`
- Cannarsa, Sinestrari (2004). *Semiconcave Functions, Hamilton–Jacobi Equations,
  and Optimal Control*. Birkhäuser (KVV ref [6]; Ch. 4 = rectifiability of
  singular sets). — `representation theorem/semiconcave-functions_Cannarsa.pdf`
- McEneaney (2007). *A curse-of-dimensionality-free numerical method for solution
  of certain HJB PDEs*. SIAM J. Control Optim. 46(4):1239–1276.
  — `representation theorem/curse-of-dimensionality-free_McEeany.pdf`
- Gaubert, McEneaney, Qu (2011). *Curse of dimensionality reduction in max-plus
  based approximation methods: theoretical estimates and improved pruning
  algorithms*. CDC-ECC 2011, arXiv:1109.5241. Thms 3.1–3.3: $\Omega(n^{-2/d})$ for
  shared-Hessian quadratic approximation; Bregman-volume constants.
  — `representation theorem/Gaubert–McEneaney–Qu-2011.pdf`
- Darbon, Dower, Meng (2023). *Neural network architectures using min-plus
  algebra for solving certain high-dimensional optimal control problems and
  Hamilton–Jacobi PDEs*. MCSS 35:1–44, arXiv:2105.03336. ⚠ local file is
  misleadingly named — `representation theorem/CDC-max-plus-complexity bounds.pdf`
- Dower, McEneaney. *A max-plus fundamental solution semigroup for a class of
  lossless wave equations* (adjacent: Riccati-propagated quadratic kernels).
  ⚠ also misleadingly named — `representation theorem/curse-free-max-plus_McEneay.pdf`
