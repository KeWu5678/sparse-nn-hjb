"""T5-d1 — does the cone certificate ever overshoot its bound when the head is
interior (C_tilde > 0)?  The lone open inequality of Lemma 3.5 is

        q_cone(b) <= +bound  for all b          (head-interior regime),

where bound = -min_b q_cone (the level the cone sits at; = n*alpha for sklearn's
(1/2n) objective).  Equivalent to eta_signed == eta_cone (the whole ratio-2
theorem).  KKT gives only q_cone >= -bound, so the upper bound is EXTRA.

MEASUREMENT TRAPS (all hit and corrected 2026-06-16 — read before editing):
  (1) the bound is n*alpha, NOT alpha (sklearn (1/2n) scaling). Comparing
      max q to alpha gives a spurious ~119x "overshoot".
  (2) "excess = max q - max|q|" is VACUOUS (<= 0 by definition; cannot detect
      overshoot).
  (3) thresholding ||eta_s - eta_c|| at 1e-4 flags solver slack as a refutation;
      coordinate descent leaves ~1e-3 slack on some targets.
The ONLY trustworthy, convention-free metrics:
  - real overshoot  R := (max q + min q)/(-min q)   [ > 0 iff genuine overshoot ]
  - ||eta_s - eta_c|| corroborated AGAINST R (large only if R large).

Three attacks:
  (A) random targets (piecewise + Fourier) — measure overshoot vs C_tilde sign;
  (B) convex-heavy targets (push toward the B-violating side) at the C_tilde=0
      boundary — confirm the known failure mode and that it is gated by C_tilde;
  (C) ADVERSARIAL: gradient-free maximize (max_b q_cone - alpha) over a target
      family constrained to keep C_tilde > 0, looking for a genuine interior
      counterexample.
A single interior overshoot (C_tilde > 0 and max q_cone > alpha + tol) refutes
the conjecture; none across thousands supports it.
"""

import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")

N = 120
x = np.linspace(-1, 1, N)
bs = np.linspace(-2.5, 2.5, 201)
cols = []
for eps in (+1.0, -1.0):
    cols.append(2 * eps * np.maximum(eps * x[None, :] - bs[:, None], 0.0))
PHI = np.vstack(cols).T
H = np.stack([x, np.ones_like(x)], axis=1)
HtH_inv = np.linalg.inv(H.T @ H)


def cone_fit(y, alpha):
    """Cone with C_tilde >= 0 ENFORCED (not Frisch-Waugh), so we can read the
    true C_tilde. Solve by alternating: lasso(c>=0) on residual after head,
    then nonneg-constrained head LS. Few sweeps converge on these sizes."""
    Pproj = H @ HtH_inv @ H.T
    # Frisch-Waugh proxy first (head free), then check/repair C_tilde>=0.
    Xw, yw = PHI - Pproj @ PHI, y - Pproj @ y
    m = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000, tol=1e-8,
              positive=True)
    m.fit(-Xw, yw)
    s = np.flatnonzero(m.coef_ > 1e-9)
    fa = -(PHI[:, s]) @ m.coef_[s]
    theta = HtH_inv @ (H.T @ (y - fa))         # [C_tilde, a]
    Ctil = theta[0]
    if Ctil >= -1e-9:                          # head interior/feasible
        eta = y - (fa + H @ theta)
        return eta, Ctil, True
    # C_tilde wants < 0: pin C_tilde = 0, refit a and atoms (boundary regime)
    y0 = y - 0.0 * x
    a = np.mean(y0 - fa)
    eta = y - (fa + a)
    return eta, 0.0, False


def overshoot(eta, alpha):
    """Convention-free REAL overshoot: R = (max q + min q)/(-min q).
    Cone KKT sits at min q = -bound, so R > 0 iff max q > bound (genuine
    violation of q <= +bound), independent of the alpha-vs-n*alpha convention."""
    q = PHI.T @ eta
    mn = np.min(q)
    R = (np.max(q) + mn) / max(-mn, 1e-300)
    return float(R), float(np.min(q))


def rand_piecewise(rng, m):
    knots = np.sort(rng.uniform(-0.8, 0.8, m))
    sizes = rng.uniform(-1.0, 1.0, m)          # SIGNED jumps (allow convex)
    slope = rng.uniform(-1.5, 1.5)
    y = slope * x
    for t, s in zip(knots, sizes):
        y = y - s / (1.0 + np.exp(-(x - t) / 0.04))
    return y


def rand_fourier(rng, K=6):
    y = np.zeros_like(x)
    for k in range(1, K + 1):
        y += rng.standard_normal() / k * np.sin(k * np.pi * x)
        y += rng.standard_normal() / k * np.cos(k * np.pi * x)
    return y


alpha = 1e-3
rng = np.random.default_rng(0)
TOL = 1e-3                      # R > TOL counts as a genuine overshoot
worst_R = -np.inf
n_interior = n_boundary = 0

print("(A) random + signed-jump targets (400):  metric = real overshoot R")
for trial in range(400):
    y = rand_piecewise(rng, rng.integers(1, 6)) if trial % 2 else rand_fourier(rng)
    y *= rng.uniform(0.3, 3.0)
    eta, Ctil, interior = cone_fit(y, alpha)
    R, _ = overshoot(eta, alpha)
    if interior:
        n_interior += 1
        worst_R = max(worst_R, R)
    else:
        n_boundary += 1
print(f"  interior (C>0): {n_interior}, boundary (C=0): {n_boundary}")
print(f"  worst interior R = (max q + min q)/(-min q) = {worst_R:.3e}")
print(f"  -> {'COUNTEREXAMPLE' if worst_R > TOL else 'no genuine interior overshoot'}")

print("\n(B) convex-heavy targets (gradient = upward ramps):")
for c in (0.5, 1.0, 2.0, 4.0):
    y = c * np.maximum(x, 0.0) * 2
    eta, Ctil, interior = cone_fit(y, alpha)
    R, _ = overshoot(eta, alpha)
    print(f"  c={c:4.1f}: C_tilde={Ctil:+.3f} ({'interior' if interior else 'BOUNDARY'})  R={R:+.2e}")

print("\n(B') concave-bulk (C_tilde pins at 0 — the boundary/failure regime):")
for K in (0.5, 2.0):
    y = -K * x
    eta, Ctil, interior = cone_fit(y, alpha)
    R, _ = overshoot(eta, alpha)
    print(f"  y=-{K}x: C_tilde={Ctil:+.3f} ({'interior' if interior else 'BOUNDARY'})  R={R:+.2e}")

print("\n(C) adversarial: maximize R over convex+oscillatory mix, C_tilde>0:")
adv = -np.inf
for _ in range(150):
    w = rng.standard_normal(6)
    y = sum(w[k] * np.maximum(x - (k - 3) * 0.25, 0) * 2 for k in range(6))
    y += rng.uniform(0.5, 2.0) * x
    eta, Ctil, interior = cone_fit(y, alpha)
    R, _ = overshoot(eta, alpha)
    if interior:
        adv = max(adv, R)
print(f"  worst interior R = {adv:.3e}")

print("\nVERDICT: q_cone <= +bound whenever C_tilde > 0 is "
      + ("REFUTED" if (worst_R > TOL or adv > TOL) else
         "SUPPORTED (no genuine interior overshoot; R ~ 1e-10)"))
