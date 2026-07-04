"""T5 in d=1 under GRADIENT training: atom counts of the cone-constrained
(semiconcave) fit vs the signed fit, at matched accuracy.

Setting (matches the repo's models at power p=1, gradient-only data):
  data: y_i = V'(x_i) + noise,  x_1 < ... < x_N  on [-1, 1].
  SIGNED model:  V' fitted by any piecewise-constant v; penalty alpha*TV(v).
      => 1-D fused lasso:  min_v  0.5||y - v||^2 + alpha*sum|v_{i+1} - v_i|.
      Solved EXACTLY via the dual:  v = y - D^T z,
      z = argmin ||y - D^T z||^2  s.t. |z_i| <= alpha    (lsq_linear, box).
  CONE model:    V' = C*x - m,  m nondecreasing (g convex), C >= 0 learnable;
      penalty alpha*(m_N - m_1)  (= total atom mass on the interval).
      For fixed C, with z = y - C*x and u = a1 - m (u nonincreasing):
      min_u 0.5*||z - u||^2 + alpha*(u_1 - u_N)  s.t. u nonincreasing
      = antitonic PAVA projection of z with endpoint shift
        z_1 -> z_1 - alpha,  z_N -> z_N + alpha
      (complete the square on the linear endpoint terms; shift = alpha for the
      0.5-scaled LS, matching the fused-lasso scaling 0.5||.||^2 + alpha*TV —
      referee fix 2026-06-12, previously alpha/2 = silent half-strength).
      Outer: jointly convex in (C, u) => 1-D convex minimization over C.

Atom count = number of jumps of the fitted piecewise-constant gradient
(strictly: increases of m for the cone; nonzero |dv| for signed).

Protocol: targets x noise levels x seeds x alpha-sweep; report atoms needed
to reach matched ORACLE accuracy (RMSE of fitted v vs the true noiseless V').
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar

rng_global = np.random.default_rng(7)
N = 200
JUMP_TOL = 1e-6


# ---------------- exact solvers ----------------
def fused_lasso(y, alpha):
    """Exact 1-D fused lasso via dual box-LS. Returns fitted v."""
    n = len(y)
    # D: (n-1) x n difference matrix; dual: min ||y - D^T z||, |z|<=alpha
    rows = np.repeat(np.arange(n - 1), 2)
    cols = np.empty(2 * (n - 1), dtype=int)
    cols[0::2] = np.arange(n - 1)
    cols[1::2] = np.arange(1, n)
    vals = np.tile([-1.0, 1.0], n - 1)
    from scipy import sparse
    D = sparse.csr_matrix((vals, (rows, cols)), shape=(n - 1, n))
    res = lsq_linear(D.T, y, bounds=(-alpha, alpha),
                     tol=1e-12, lsmr_tol=1e-12, max_iter=500)
    return y - D.T @ res.x


def pava_antitonic(z):
    """Exact LS projection onto nonincreasing sequences (PAVA)."""
    # project -z onto nondecreasing, negate back
    y = -z
    n = len(y)
    level, weight, idx = [], [], []
    for i in range(n):
        level.append(y[i]); weight.append(1.0); idx.append(1)
        while len(level) > 1 and level[-2] > level[-1]:
            w = weight[-2] + weight[-1]
            lv = (level[-2] * weight[-2] + level[-1] * weight[-1]) / w
            level[-2:] = [lv]; weight[-2:] = [w]
            idx[-2:] = [idx[-2] + idx[-1]]
    out = np.concatenate([np.full(k, lv) for lv, k in zip(level, idx)])
    return -out


def cone_fit(x, y, alpha):
    """Cone model: v = C*x - m, m nondecreasing, C >= 0; returns v, C."""
    def inner(C):
        z = y - C * x
        zt = z.copy()
        zt[0] -= alpha          # shift = alpha for the 0.5-scaled objective
        zt[-1] += alpha
        u = pava_antitonic(zt)              # u = a1 - m, nonincreasing
        v = C * x + u
        obj = 0.5 * np.sum((y - v) ** 2) + alpha * (u[0] - u[-1])
        return obj, v

    r = minimize_scalar(lambda C: inner(C)[0], bounds=(0.0, 10.0),
                        method="bounded", options={"xatol": 1e-6})
    obj, v = inner(r.x)
    return v, r.x


def count_atoms_signed(v):
    """Signed model: piecewise-constant gradient; atoms = nonzero jumps."""
    dv = np.diff(v)
    return int(np.sum(np.abs(dv) > JUMP_TOL * max(1.0, np.abs(v).max())))


def count_atoms_cone(v, C, x):
    """Cone model: v = C*x - m; atoms = increases of m = C*dx - dv."""
    dm = C * np.diff(x) - np.diff(v)
    return int(np.sum(dm > JUMP_TOL * max(1.0, np.abs(v).max())))


# ---------------- targets (true gradients V') ----------------
def target_staircase(x, rng):
    """In-cone: V' = x - staircase with 5 shocks (C-semiconcave, C=1)."""
    knots = np.sort(rng.uniform(-0.8, 0.8, 5))
    drops = rng.uniform(0.3, 0.8, 5)
    m = np.zeros_like(x)
    for k, d in zip(knots, drops):
        m += d * (x > k)
    return x - m


def target_smooth(x, rng):
    """Smooth semiconcave: V' = 0.8x - 1.5*tanh(3x) - 0.6*tanh(5(x-0.4))."""
    return 0.8 * x - 1.5 * np.tanh(3 * x) - 0.6 * np.tanh(5 * (x - 0.4))


# ---------------- protocol ----------------
def run(target_fn, name, noise_levels=(0.0, 0.05, 0.15), seeds=5, n_alpha=24):
    print(f"\n### target: {name} ###")
    x = np.linspace(-1, 1, N)
    alphas = np.logspace(-3.2, 0.3, n_alpha)
    for sig in noise_levels:
        # per seed: curves (oracle RMSE vs atoms), then matched-accuracy table
        table = []
        scale = None
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            vtrue = target_fn(x, rng)
            scale = np.sqrt(np.mean(vtrue ** 2))
            y = vtrue + sig * rng.standard_normal(N)
            cs, cc = [], []
            for a in alphas:
                vs = fused_lasso(y, a)
                vc, C = cone_fit(x, y, a)
                cs.append((np.sqrt(np.mean((vs - vtrue) ** 2)),
                           count_atoms_signed(vs)))
                cc.append((np.sqrt(np.mean((vc - vtrue) ** 2)),
                           count_atoms_cone(vc, C, x)))
            # matched ABSOLUTE accuracy: atoms to reach err <= frac * scale
            row = []
            for frac in (0.02, 0.05, 0.10, 0.20):
                thr = frac * scale
                a_s = min((n for e, n in cs if e <= thr), default=None)
                a_c = min((n for e, n in cc if e <= thr), default=None)
                row.append((a_s, a_c))
            table.append(row)
        print(f"  noise sigma = {sig}  (target RMS scale = {scale:.3f}):")
        for j, frac in enumerate((0.02, 0.05, 0.10, 0.20)):
            sgn = [t[j][0] for t in table if t[j][0] is not None]
            con = [t[j][1] for t in table if t[j][1] is not None]
            def fmt(lst, k):
                return (f"{np.median(lst):5.1f} (range {min(lst)}-{max(lst)},"
                        f" {k}/{seeds} reached)") if lst else "   --"
            print(f"    err <= {frac:4.0%}*scale:  signed = "
                  f"{fmt(sgn, len(sgn))}   cone = {fmt(con, len(con))}")


if __name__ == "__main__":
    run(target_staircase, "staircase (in-cone, 5 true shocks)")
    run(target_smooth, "smooth semiconcave")
    print("\nreading: cone < signed at matched accuracy, growing with noise "
          "=> T5-d1 mechanism confirmed; equal => selection story dies in d=1.")
