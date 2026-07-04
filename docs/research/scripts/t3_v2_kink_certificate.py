"""Mean-zero certificate test on v2's kinks: is the head-completed cone model
blocked by straight kinks with NON-CONSTANT jump density?

Targets (all = Laplacian of the convex part g* = (C*/2)|x|^2 - V on Omega):
  S  sharp v2:        Delta g* = (2C* - rho) dx + J dH^1 on both diagonals,
                      J(t) = sqrt(2)*exp(-(t^2+|t|+1/2))  (non-constant!).
  K  const-J control: same a.c. part, same total kink mass, J replaced by its
                      average over the in-Omega diagonal — should be cone+head
                      representable (atoms on the diagonals + background), so
                      the mean-zero LP should be ~0. Isolates non-constancy.
  M2 mollified d=0.2, M1 mollified d=0.1: V = v2 * Gaussian_delta (a.c.
                      Laplacian => Portmanteau-safe regime); traces pi(delta).

LPs per target:
  pure cone      : max <Dg, psi>  s.t. line-integrals(psi) <= 0, |psi| <= 1
  mean-zero      : + constraint int(psi) >= 0  (refutes model with free head;
                   Farkas dual of that row = lambda = curvature price).

Stage 2 (only if sharp mean-zero value is positive): smooth the certificate
(Gaussian, 1.5 cells), re-check feasibility on a 3x denser line family and the
retained objective — the Portmanteau-safe version of the verdict.
"""

import time

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import linprog

METHOD = "highs-ipm"
CSTAR = 2 * np.exp(-1.5)          # semiconcavity constant of v2

# ---------------- coarse grid (must match t3_cone_certificate.py) ----------
S = 3.0
n = 61
h = 2 * S / n
centers = -S + h * (np.arange(n) + 0.5)
X, Y = np.meshgrid(centers, centers, indexing="ij")
ncell = n * n

CENTERS4 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

def v2(X, Y):
    r2 = [(X - c[0]) ** 2 + (Y - c[1]) ** 2 for c in CENTERS4]
    return np.exp(-0.5 * np.maximum.reduce(r2) * 1.0) if False else \
        np.minimum.reduce([np.exp(-0.5 * r) for r in r2])

def rho_ac(X, Y):
    """a.c. part of Delta v2: (r^2-2)*phi for the ACTIVE (farthest) branch."""
    r2 = np.stack([(X - c[0]) ** 2 + (Y - c[1]) ** 2 for c in CENTERS4])
    act = r2.max(axis=0)                      # farthest center <=> min phi
    return (act - 2.0) * np.exp(-0.5 * act)

def J_density(t):
    return np.sqrt(2.0) * np.exp(-(t ** 2 + np.abs(t) + 0.5))

# ---------------- objective vectors ----------------
def obj_sharp(const_J=False):
    ac = (2 * CSTAR - rho_ac(X, Y)) * h * h            # a.c. part of Dg*
    dep = np.zeros((n, n))
    Jvals = J_density(centers)
    Jbar = Jvals.mean()
    for i in range(n):
        jd1 = Jbar if const_J else Jvals[i]
        dep[i, i] += jd1 * np.sqrt(2.0) * h            # D1: (t,t)
        jd2 = Jbar if const_J else Jvals[i]
        dep[i, n - 1 - i] += jd2 * np.sqrt(2.0) * h    # D2: (t,-t)
    c = ac + dep
    print(f"    a.c. min = {ac.min():.4f} (convexity check, want >= ~0); "
          f"kink mass = {dep.sum():.3f}")
    return c.ravel()

def obj_mollified(delta):
    """Delta g*_delta = 2C* - Delta(v2 * G_delta), fine-grid numerics."""
    hf = h / 4.0
    pad = 4 * delta + 4 * hf
    m = int(round(2 * (S + pad) / hf))
    cf = -(S + pad) + hf * (np.arange(m) + 0.5)
    Xf, Yf = np.meshgrid(cf, cf, indexing="ij")
    Vf = gaussian_filter(v2(Xf, Yf), sigma=delta / hf, mode="nearest")
    L = (np.roll(Vf, 1, 0) + np.roll(Vf, -1, 0) + np.roll(Vf, 1, 1)
         + np.roll(Vf, -1, 1) - 4 * Vf) / hf ** 2      # Delta v2_delta
    Dg = 2 * CSTAR - L                                  # density of Dg*
    # crop to [-S,S]^2 and block-sum 4x4 fine cells -> coarse cells * h^2
    i0 = int(round((pad) / hf))
    Dg = Dg[i0:i0 + 4 * n, i0:i0 + 4 * n]
    coarse = Dg.reshape(n, 4, n, 4).sum(axis=(1, 3)) * hf * hf
    print(f"    mollified delta={delta}: density min = {coarse.min()/h/h:.4f}")
    return coarse.ravel()

# ---------------- line matrix (same construction as t3_cone_certificate) ---
def build_line_matrix(n_theta=90, db=h, ds=None):
    ds = h / 2.0 if ds is None else ds
    rows, cols, vals = [], [], []
    r = 0
    bmax = S * np.sqrt(2.0)
    for j in range(n_theta):
        th = np.pi * j / n_theta
        nx, ny = np.cos(th), np.sin(th)
        tx, ty = -ny, nx
        for b in np.arange(-bmax, bmax + 1e-9, db):
            tlo, thi = -1e18, 1e18
            ok = True
            for (pn, pt) in ((b * nx, tx), (b * ny, ty)):
                if abs(pt) < 1e-12:
                    if abs(pn) > S:
                        ok = False
                        break
                else:
                    a1, a2 = (-S - pn) / pt, (S - pn) / pt
                    tlo = max(tlo, min(a1, a2)); thi = min(thi, max(a1, a2))
            if not ok or thi - tlo < 3 * ds:
                continue
            ts = np.arange(tlo + ds / 2, thi, ds)
            px = b * nx + ts * tx
            py = b * ny + ts * ty
            ix = np.clip(((px + S) / h).astype(int), 0, n - 1)
            iy = np.clip(((py + S) / h).astype(int), 0, n - 1)
            w = np.bincount(ix * n + iy, minlength=ncell) * ds
            nz = np.nonzero(w)[0]
            rows.extend([r] * len(nz)); cols.extend(nz.tolist())
            vals.extend(w[nz].tolist())
            r += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(r, ncell))

print("building line matrix ...", flush=True)
A = build_line_matrix()
print(f"  lines: {A.shape[0]}, nnz: {A.nnz}", flush=True)
row_mz = sparse.csr_matrix(-np.full((1, ncell), h * h))
A_mz = sparse.vstack([A, row_mz]).tocsr()

def solve(name, c_obj, mean_zero):
    A_ub = A_mz if mean_zero else A
    t0 = time.time()
    res = linprog(-c_obj, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                  bounds=(-1, 1), method=METHOD)
    val = -res.fun
    rel = val / np.abs(c_obj).sum()
    lam = ""
    if mean_zero and res.status == 0:
        marg = res.ineqlin.marginals[-1]
        lam = f"   lambda(dual of mz-row) = {abs(marg):.4f}"
    print(f"[{name}{' +mz' if mean_zero else ''}] value = {val:.5f}  "
          f"rel = {rel:.2%}  [{time.time()-t0:.0f}s]{lam}", flush=True)
    return val, rel, (res.x if val > 1e-6 else None)

print("\n=== target S: sharp v2 ===", flush=True)
cS = obj_sharp()
solve("S", cS, False)
vS, rS, psiS = solve("S", cS, True)

print("\n=== target K: const-J control ===", flush=True)
cK = obj_sharp(const_J=True)
solve("K", cK, False)
vK, rK, _ = solve("K", cK, True)

print("\n=== target M2: mollified delta=0.2 ===", flush=True)
cM2 = obj_mollified(0.2)
vM2, rM2, _ = solve("M2", cM2, True)

print("\n=== target M1: mollified delta=0.1 ===", flush=True)
cM1 = obj_mollified(0.1)
vM1, rM1, _ = solve("M1", cM1, True)

print("\n=== summary (mean-zero, the model-with-head question) ===")
print(f"  sharp v2      : {rS:.2%}   (obstruction iff >> control)")
print(f"  const-J ctrl  : {rK:.2%}   (expected ~ 0)")
print(f"  mollified 0.2 : {rM2:.2%}")
print(f"  mollified 0.1 : {rM1:.2%}")

# ---------------- stage 2: smoothed certificate, denser family -------------
if psiS is not None:
    print("\n=== stage 2: smoothed sharp-v2 certificate, 3x denser lines ===",
          flush=True)
    psi = gaussian_filter(psiS.reshape(n, n), sigma=1.5).ravel()
    psi /= max(1.0, np.abs(psi).max())
    A3 = build_line_matrix(n_theta=270, db=h / 3.0, ds=h / 4.0)
    li = A3 @ psi
    print(f"  objective retained = {float(cS @ psi):.5f} "
          f"(bang-bang was {vS:.5f})")
    print(f"  max line-integral over dense family = {li.max():.5f} "
          f"(violations: {(li > 1e-9).sum()}/{A3.shape[0]})")
    print(f"  int(psi) = {float(psi.sum() * h * h):.5f}")
