"""T5-d1 — dual-intersection structure and a hard test of hypothesis (M).

Structural claim (rigorous, verified here): in the dual (residual) variables the
feasible sets are
    A = F_semiconcave-cone = { q_eps(b) >= -alpha  for all eps,b }   (atoms -c phi, c>=0)
    B = F_convex-cone      = { q_eps(b) <= +alpha  for all eps,b }   (atoms +c phi, c>=0)
    F_signed = { |q_eps(b)| <= alpha } = A ∩ B.
So the signed problem = "feasible for BOTH one-sided models at once"; its active
set splits into A-type (-alpha) and B-type (+alpha) contact regions, and the
semiconcave cone keeps only the A-type half.

Hypothesis (M), the one open analytic step: R_-(signed) == R_-(cone), i.e.
removing the B-constraints (q<=alpha) does not change the number of -alpha
contact regions. We test it HARD: vary the number of shocks m, and check
  Q1 alternation:  R_+(signed) ~ R_-(signed)             [Lemma 2]
  Q2 hypothesis M: R_-(signed) ~ R_-(cone)               [the open step]
  Q3 scaling:      R (merged contact regions) scales with m, NOT with K or 1/alpha
and the merged ratio -> 2.

Counts are CONTINUUM (merged) contact regions; raw grid atoms cluster (~1/alpha)
and are not the invariant.
"""

import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")

N = 300
DELTA = 0.04
rng = np.random.default_rng(0)
x = np.linspace(-1, 1, N)

bs = np.linspace(-3, 3, 1401)
cols, meta = [], []
for eps in (+1.0, -1.0):
    cols.append(2 * eps * np.maximum(eps * x[None, :] - bs[:, None], 0.0))
    meta += [(eps, b) for b in bs]
PHI = np.vstack(cols).T
meta = np.array(meta)
H = np.stack([x, np.ones_like(x)], axis=1)
P = H @ np.linalg.pinv(H)


def saturated_target(m, seed):
    r = np.random.default_rng(seed)
    knots = np.sort(r.uniform(-0.8, 0.8, m))
    sizes = r.uniform(0.4, 0.9, m)
    y = 1.0 * x                                  # slope C=1 (saturated)
    for t, s in zip(knots, sizes):
        y = y - s / (1.0 + np.exp(-(x - t) / DELTA))
    return y


def merged_regions(supp, eta=None, want_sign=None, gap=0.08):
    """Continuum contact-region count. If want_sign given, keep only atoms whose
    certificate has that sign (bucket by bound), else count all."""
    if len(supp) == 0:
        return 0
    labs = meta[supp]
    if want_sign is not None and eta is not None:
        q = PHI[:, supp].T @ eta
        bnd = np.max(np.abs(PHI.T @ eta))
        keep = (q > 0.5 * bnd) if want_sign > 0 else (q < -0.5 * bnd)
        labs = labs[keep]
    n = 0
    for eps in (+1.0, -1.0):
        bb = np.sort(labs[labs[:, 0] == eps, 1])
        if len(bb):
            n += 1 + int(np.sum(np.diff(bb) > gap))
    return n


def solve(y, alpha, positive):
    Xw, yw = PHI - P @ PHI, y - P @ y
    D = -Xw if positive else Xw
    mdl = Lasso(alpha=alpha, fit_intercept=False, max_iter=500000,
                tol=1e-13, positive=positive)
    mdl.fit(D, yw)
    s = np.flatnonzero(np.abs(mdl.coef_) > 1e-8)
    fa = -(PHI[:, s]) @ mdl.coef_[s] if positive else PHI[:, s] @ mdl.coef_[s]
    th = lstsq(H, y - fa, rcond=None)[0]
    eta = y - (fa + H @ th)
    return s, eta


def verify_intersection(y, alpha):
    """Confirm F_signed = A ∩ B numerically: at the signed optimum every active
    atom has |q| = alpha-bound, and the active set splits into q=-bnd / q=+bnd."""
    s, eta = solve(y, alpha, positive=False)
    q = PHI[:, s].T @ eta
    bnd = np.max(np.abs(PHI.T @ eta))
    on_bound = np.mean(np.abs(np.abs(q) - bnd) < 1e-3 * bnd) if len(s) else 1.0
    return on_bound  # fraction of active atoms exactly on a bound (should be ~1)


print(f"{'m':>2} {'alpha':>8} {'K':>4} | "
      f"{'R-(sgn)':>7} {'R+(sgn)':>7} {'R-(con)':>7} | "
      f"{'Q1 +/-':>7} {'Q2 s/c':>7} {'ratio':>5} | {'onbnd':>5}")
print("-" * 84)
for m in (1, 2, 3, 5, 8):
    for alpha in (1e-3, 3e-4):
        Rm_s, Rp_s, Rm_c, rat, K, ob = [], [], [], [], [], []
        for seed in range(4):
            y = saturated_target(m, 100 + seed)
            ss, es = solve(y, alpha, positive=False)
            sc, ec = solve(y, alpha, positive=True)
            Rm_s.append(merged_regions(ss, es, -1))
            Rp_s.append(merged_regions(ss, es, +1))
            Rm_c.append(merged_regions(sc, ec, -1))   # cone is all -alpha
            rat.append(merged_regions(ss) / max(merged_regions(sc), 1))
            sgn = np.sign(es); sgn = sgn[np.abs(es) > 0.02 * np.abs(es).max()]
            K.append(int(np.sum(sgn[1:] != sgn[:-1])))
            ob.append(verify_intersection(y, alpha))
        f = lambda L: np.median(L)
        q1 = f(Rp_s) / max(f(Rm_s), 1)
        q2 = f(Rm_s) / max(f(Rm_c), 1)
        print(f"{m:>2} {alpha:8.1e} {f(K):4.0f} | "
              f"{f(Rm_s):7.1f} {f(Rp_s):7.1f} {f(Rm_c):7.1f} | "
              f"{q1:7.2f} {q2:7.2f} {f(rat):5.2f} | {f(ob):5.2f}")

print()
print("Q1 alternation R+/R- ~ 1 (Lemma 2); Q2 hypothesis-M R-(sgn)/R-(con) ~ 1;")
print("ratio ~ 2; onbnd ~ 1 confirms F_signed = A ∩ B (all signed actives on a bound).")
print("R columns should track m (structure), not K or 1/alpha.")
