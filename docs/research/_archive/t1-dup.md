# T1 — The 1-D semiconcavity budget lemma (proved)

Status: complete proofs, 2026-06-10. Base of the theory ladder in
`t2-separation-draft.md`. Notation: $\Omega = (a_1, a_2) \subset \mathbb{R}$ bounded, $|\Omega| = a_2 - a_1$.

**Standing assumptions.** $V : \bar\Omega \to \mathbb{R}$ is $C$-semiconcave ($x \mapsto V(x) - \tfrac{C}{2}x^2$ concave)
and $L$-Lipschitz.

## Lemma 1 (mass budget)

The distributional second derivative of $V$ is a signed Radon measure of the form

$$
V'' = C\,dx - \mu, \qquad \mu \ge 0 \text{ a finite nonnegative Radon measure on } \Omega,
$$

with

$$
\mu(\Omega) \le C|\Omega| + 2L
\qquad\text{and}\qquad
\mathrm{TV}(V'') = \int (V'')_+ + \int (V'')_- \le 2C|\Omega| + 2L.
$$

Both bounds are **independent of the number of shocks** (gradient jumps) of $V$:
shocks are atoms of $\mu$, and arbitrarily many of them fit inside the fixed budget.

*Proof.* Let $w(x) := V(x) - \tfrac{C}{2}x^2$, concave by assumption. Then $w'$ exists
one-sidedly everywhere, is non-increasing, and $w'' = -\mu$ for a nonnegative Radon
measure $\mu$ (Riesz representation for concave functions). Since $|V'| \le L$ a.e.,

$$
\mu(\Omega) = w'(a_1^+) - w'(a_2^-) = \bigl[V'(a_1^+) - V'(a_2^-)\bigr] + C(a_2 - a_1) \le 2L + C|\Omega|.
$$

Hence $V'' = C\,dx - \mu$ as measures, $(V'')_+ \le C\,dx$ so $\int (V'')_+ \le C|\Omega|$, and
$\int (V'')_- \le \mu(\Omega) \le C|\Omega| + 2L$. Adding the two gives $\mathrm{TV}(V'') \le 2C|\Omega| + 2L$. $\blacksquare$

A downward gradient jump of size $s$ at $x_0$ (a shock) contributes the atom $s\,\delta_{x_0}$
to $\mu$; the budget says $\sum \text{shock sizes} \le C|\Omega| + 2L$, but the *count* is unconstrained.
Upward jumps are impossible (semiconcavity), which is why the budget closes.

## Lemma 2 (exact cone representation = the repo's SemiconcaveModel, p = 1)

With $\mu$ from Lemma 1 and $g(x) := \tfrac{C}{2}x^2 - V(x)$ (convex, $g'' = \mu$):

$$
V(x) = \tfrac{C}{2}x^2 - \beta_0 - \beta_1 (x - a_1) - \int_\Omega (x - s)_+ \, d\mu(s),
\qquad x \in \bar\Omega,
$$

with $\beta_0 = g(a_1)$, $\beta_1 = g'(a_1^+)$. This is exactly `SemiconcaveModel` with ReLU
activation, power $p = 1$, nonnegative outer weights of total mass $\le C|\Omega| + 2L$,
plus an affine correction.

*Proof.* Taylor expansion with measure remainder for the convex $g$:
$g(x) = g(a_1) + g'(a_1^+)(x - a_1) + \int_{(a_1, x]} (x - s)\, d\mu(s)$, and for $x \in \bar\Omega$ the
integrand vanishes for $s > x$, so the integral equals $\int_\Omega (x - s)_+ \, d\mu(s)$.
Substitute $g = \tfrac{C}{2}x^2 - V$. $\blacksquare$

## Lemma 3 (uniqueness $\Rightarrow$ the cone constraint is free in 1-D; T3 for d = 1)

If $V(x) = \tfrac{C}{2}x^2 - \mathrm{affine}(x) - \int_\Omega (x - s)_+ \, d\nu(s)$ on $\Omega$ for **any** signed finite
measure $\nu$, then $\nu = \mu$ on $\Omega$. Consequently the minimal-TV signed ridge
representation of the convex part is automatically nonnegative, and the cone
constraint $c_i \ge 0$ in `SemiconcaveModel` costs nothing in representational power:

$$
\min\{\, \|\nu\|_{\mathrm{TV}} : \text{signed } \nu \text{ representing } g \,\}
= \mu(\Omega)
= \min \text{ over the cone}.
$$

*Proof.* Differentiate the representation twice in $\mathcal{D}'(\Omega)$: the quadratic gives
$C\,dx$, the affine part $0$, and $\frac{d^2}{dx^2} \int (x - s)_+ \, d\nu(s) = \nu$. So $C\,dx - \nu = V'' =
C\,dx - \mu$, i.e. $\nu = \mu \ge 0$. $\blacksquare$

(This settles ladder item T3 trivially in $d = 1$. In $d \ge 2$ the analogous question —
is the Ongie-optimal even measure of a ridge-convex function nonnegative? — is
open and is the cleanest new question of the program.)

## Corollary 4 (shock-independent approximation rate)

There is an absolute constant $c$ such that for every $N \in \mathbb{N}$ there is an $N$-atom
network $V_N$ of the repo's semiconcave form ($p = 1$, nonneg weights) with

$$
\|V - V_N\|_{C(\bar\Omega)} \le c \,(C|\Omega| + 2L)\, |\Omega| \, N^{-2},
$$

uniformly over the class of $C$-semiconcave $L$-Lipschitz functions — no dependence
on the number or position of shocks.

*Proof sketch.* By Lemma 2 it suffices to approximate $x \mapsto \int (x-s)_+ \, d\mu(s)$ with $N$
atoms at mass $\|\mu\| \le C|\Omega| + 2L$. This is the $d = 1$, $k = 1$ variation-space setting:
Yang–Zhou Lemma 2.3 gives sup-norm rate $N^{-1/2 - (2k+1)/(2d)} = N^{-2}$ on
$\mathcal{F}_{\sigma_1}(M)$ with constant $\propto M$; rescale $\Omega$ to unit length ($\mathcal{R}$-norm scaling, Ongie
Prop 2) to get the $|\Omega|$ factor. Alternatively, elementary: place atoms at the
$N$-quantiles of $\mu$ and compare piecewise-linear interpolants. $\blacksquare$

## Why this matters for the program

- It is the $d = 1$ instance of mechanism M1 ("semiconcavity is a one-sided
  second-order budget"): the total price of all shocks is capped by $C|\Omega| + 2L$.
- It shows the separation of T2 is genuinely a $d \ge 2$ phenomenon: in 1-D, ridge
  (ReLU) networks handle semiconcave shock structure perfectly — each shock is a
  single atom — consistent with the discreteness of $S^0$ in the T2 fan argument.
- It justifies the cone constraint exactly (Lemma 3) where it can be justified
  cheaply, isolating T3 ($d \ge 2$) as the real question.

## References

- Yang, Zhou (2025), Constr. Approx. 62 — Lemma 2.3 (rate), Cor 2.4.
  `representation theorem/optimal_rate_Relu^k.pdf`
- Ongie et al. (2019), arXiv:1910.01635 — Prop 2 (scaling), §2 (cost $= \|a\|_1$).
  `representation theorem/1910.01635v1.pdf`
- Savarese et al. (2019) / Petrosyan–Dereventsov–Webster (2020) — 1-D minimal
  representations, $\max(\int |f''|, \dots)$ characterization.
  `representation theorem/Relu(15)_IntgRep.pdf`
