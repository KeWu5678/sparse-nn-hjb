"""Measure the curvature price pi(g) = min{ lambda >= 0 : Delta g + lambda*1
in the discrete cone of nonneg line superpositions } — and its MESH SCALING.

Instrument (per target, per mesh): two-phase LP over y >= 0 (line weights),
lambda >= 0, slacks s+- >= 0 with cell equalities
    A^T y - lambda*h^2*1 - s+ + s- = c_obj.
Phase 1: minimize sum(s+ + s-)            -> residual floor r* (quadrature).
Phase 2: minimize lambda  s.t. sum(s) <= 1.05*r* + 1e-9.

Diagnostic: lambda(h) flat under refinement => finite continuum price pi;
lambda(h) growing => pi = infinity revealing itself as mesh-divergence
(expected for SHARP v2 kinks by the structure theorem; NOT expected for the
smoothed junction / mollified targets).

Configs: junction & sharp-v2 at n=61 AND n=91; mollified v2 (0.2, 0.1) at n=61.
"""

import time

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import linprog

METHOD = "highs-ipm"
S = 3.0
CSTAR = 2 * np.exp(-1.5)
CENTERS4 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


# ---------------- grid-parametrized pieces ----------------
def make_grid(n):
    h = 2 * S / n
    c = -S + h * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(c, c, indexing="ij")
    return h, c, X, Y


def v2(X, Y):
    r2 = [(X - c[0]) ** 2 + (Y - c[1]) ** 2 for c in CENTERS4]
    return np.minimum.reduce([np.exp(-0.5 * r) for r in r2])


def rho_ac(X, Y):
    r2 = np.stack([(X - c[0]) ** 2 + (Y - c[1]) ** 2 for c in CENTERS4])
    act = r2.max(axis=0)
    return (act - 2.0) * np.exp(-0.5 * act)


def J_density(t):
    return np.sqrt(2.0) * np.exp(-(t ** 2 + np.abs(t) + 0.5))


def obj_sharp(n):
    h, cgrid, X, Y = make_grid(n)
    c = (2 * CSTAR - rho_ac(X, Y)) * h * h
    Jv = J_density(cgrid)
    for i in range(n):
        c[i, i] += Jv[i] * np.sqrt(2.0) * h
        c[i, n - 1 - i] += Jv[i] * np.sqrt(2.0) * h
    return c.ravel()


def obj_junction(n, eta=0.3):
    h, _, X, Y = make_grid(n)
    angs = np.deg2rad([90.0, 210.0, 330.0])
    V = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    Z = np.stack([(V[i, 0] * X + V[i, 1] * Y) / eta for i in range(3)])
    Z -= Z.max(axis=0, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=0, keepdims=True)
    mx = sum(P[i] * V[i, 0] for i in range(3))
    my = sum(P[i] * V[i, 1] for i in range(3))
    return ((1.0 - (mx ** 2 + my ** 2)) / eta * h * h).ravel()


def obj_mollified(n, delta):
    h, _, _, _ = make_grid(n)
    hf = h / 4.0
    pad = 4 * delta + 4 * hf
    m = int(round(2 * (S + pad) / hf))
    cf = -(S + pad) + hf * (np.arange(m) + 0.5)
    Xf, Yf = np.meshgrid(cf, cf, indexing="ij")
    Vf = gaussian_filter(v2(Xf, Yf), sigma=delta / hf, mode="nearest")
    L = (np.roll(Vf, 1, 0) + np.roll(Vf, -1, 0) + np.roll(Vf, 1, 1)
         + np.roll(Vf, -1, 1) - 4 * Vf) / hf ** 2
    Dg = 2 * CSTAR - L
    i0 = int(round(pad / hf))
    Dg = Dg[i0:i0 + 4 * n, i0:i0 + 4 * n]
    return (Dg.reshape(n, 4, n, 4).sum(axis=(1, 3)) * hf * hf).ravel()


def build_line_matrix(n, n_theta=None):
    h = 2 * S / n
    n_theta = int(1.5 * n) if n_theta is None else n_theta
    ds = h / 2.0
    rows, cols, vals = [], [], []
    r = 0
    bmax = S * np.sqrt(2.0)
    for j in range(n_theta):
        th = np.pi * j / n_theta
        nx, ny = np.cos(th), np.sin(th)
        tx, ty = -ny, nx
        for b in np.arange(-bmax, bmax + 1e-9, h):
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
            w = np.bincount(ix * n + iy, minlength=n * n) * ds
            nz = np.nonzero(w)[0]
            rows.extend([r] * len(nz)); cols.extend(nz.tolist())
            vals.extend(w[nz].tolist())
            r += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(r, n * n))


# ---------------- two-phase min-lambda ----------------
def min_lambda(name, n, c_obj, A):
    h = 2 * S / n
    nl, nc = A.shape
    # variables: [ y (nl) | lambda (1) | s+ (nc) | s- (nc) ]
    At = A.T.tocsr()
    Aeq = sparse.hstack([
        At,
        sparse.csr_matrix(-np.full((nc, 1), h * h)),
        -sparse.eye(nc, format="csr"),
        sparse.eye(nc, format="csr"),
    ]).tocsr()
    nv = nl + 1 + 2 * nc
    bounds = [(0, None)] * nv

    # phase 1: min total slack
    c1 = np.zeros(nv); c1[nl + 1:] = 1.0
    t0 = time.time()
    r1 = linprog(c1, A_eq=Aeq, b_eq=c_obj, bounds=bounds, method=METHOD)
    if r1.status != 0:
        print(f"[{name} n={n}] phase1 FAILED: {r1.message}", flush=True)
        return None
    rstar = r1.fun
    rel_res = rstar / np.abs(c_obj).sum()
    # phase 2: min lambda subject to slack <= 1.05 r*
    c2 = np.zeros(nv); c2[nl] = 1.0
    Aub = sparse.csr_matrix(c1.reshape(1, -1))
    r2 = linprog(c2, A_eq=Aeq, b_eq=c_obj,
                 A_ub=Aub, b_ub=np.array([1.05 * rstar + 1e-9]),
                 bounds=bounds, method=METHOD)
    dt = time.time() - t0
    if r2.status != 0:
        print(f"[{name} n={n}] phase2 FAILED: {r2.message}", flush=True)
        return None
    lam = r2.x[nl]
    mass = r2.x[:nl].sum()
    print(f"[{name} n={n}] lambda_min = {lam:.4f}   residual floor = "
          f"{rel_res:.2%} of ||c||_1   cone mass = {mass:.2f}   [{dt:.0f}s]",
          flush=True)
    return lam


# Round 2 (2026-06-11): n=91 retired — IPM cost is cubic in CELL count (the
# normal equations are dense: every cell pair is coupled by a line), so n=91
# costs 11x n=61 per iteration. Cheaper evidence: extend the delta-curve
# (power law lambda ~ delta^-alpha) + one modest mesh check at n=76 (~2x).
# Round-1 results (n=61): junction 2.3732, moll-0.2 0.2912, moll-0.1 2.1130,
# sharp-v2 25.1239 (all residual floors 0.00%).
# Round 3: n=76 retired too — IPM iteration count exploded (phase 1 alone
# >77 min; iteration counts on near-degenerate sharp targets are not
# predictable). Same mesh-scaling signal by COARSENING: n=45 vs known n=61.
CONFIGS = [
    ("sharp-v2", 45),
]


def run_config(cfg):
    """Worker: builds its own matrix and objective, runs the two-phase LP."""
    name, n = cfg
    if name == "junction":
        c_obj = obj_junction(n)
    elif name == "sharp-v2":
        c_obj = obj_sharp(n)
    elif name.startswith("moll-"):
        c_obj = obj_mollified(n, float(name.split("-")[1]))
    else:
        raise ValueError(name)
    A = build_line_matrix(n)
    print(f"[{name} n={n}] matrix ready: lines {A.shape[0]}, nnz {A.nnz}",
          flush=True)
    return cfg, min_lambda(name, n, c_obj, A)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ROUND1 = {("junction", 61): 2.3732, ("moll-0.2", 61): 0.2912,
              ("moll-0.1", 61): 2.1130, ("sharp-v2", 61): 25.1239,
              ("moll-0.4", 61): 0.0, ("moll-0.3", 61): 0.0}
    results = dict(ROUND1)
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(run_config, cfg) for cfg in CONFIGS]
        for f in as_completed(futs):
            cfg, lam = f.result()
            results[cfg] = lam

    print("\n=== curvature prices pi (lambda_min) ===")
    for (name, n), lam in sorted(results.items()):
        if lam is not None:
            print(f"  {name:9s} n={n}: {lam:.4f}")
    meshes = sorted(n for (nm, n) in results if nm == "sharp-v2"
                    and results[(nm, n)] is not None)
    if len(meshes) >= 2:
        print("\nsharp-v2 mesh scaling (lambda should GROW with resolution"
              " if pi = infinity):")
        for n in meshes:
            print(f"  n={n} (h={6.0/n:.3f}): lambda = "
                  f"{results[('sharp-v2', n)]:.4f}")
    ds = [(0.4, results.get(("moll-0.4", 61))),
          (0.3, results.get(("moll-0.3", 61))),
          (0.2, results.get(("moll-0.2", 61))),
          (0.1, results.get(("moll-0.1", 61)))]
    pts = [(d, l) for d, l in ds if l]
    if len(pts) >= 2:
        import math
        print("\ndelta-curve lambda(delta):")
        for d, l in pts:
            print(f"  delta={d}: {l:.4f}")
        a = [(math.log(l2 / l1) / math.log(d1 / d2))
             for (d1, l1), (d2, l2) in zip(pts, pts[1:])]
        print(f"  local exponents alpha (lambda ~ delta^-alpha): "
              f"{['%.2f' % x for x in a]}")
