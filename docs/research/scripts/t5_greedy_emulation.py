"""Can greedy (PDAP-style) insertion DISCOVER the head emulation at p=2?

Theorem A(iii): at p=2 the signed model can match the cone+head model's
capacity within 2 atoms in d=1, via (x)_+^2 + (-x)_+^2 = x^2. But that parity
needs a COORDINATED PAIR of atoms whose individual marginal gain is poor.
Question: does greedy insertion (max |dual certificate|, the repo's PDAP
insertion logic) ever find it — or is the capacity parity algorithmically
unreachable, making the observed sparsity gap an ALGORITHMIC separation?

Setting (d=1, gradient training, p=2):
  target: mollified saturated semiconcave gradient
          y(x) = C*x - sum_j s_j * sigmoid((x - t_j)/delta),  C=1, m=3 shocks.
  atom gradient features: phi_{eps,b}(x) = d/dx (eps*x - b)_+^2
                                         = 2*eps*(eps*x - b)_+.
  SIGNED model: sum c_i phi_i, c free. Dictionary b in [-3, 3] (generous:
      includes near-affine atoms on the domain), eps = +-1.
  CONE+HEAD:    C~*x + a - sum c_i phi_i, c_i >= 0, C~ >= 0, a free.

Greedy loop (OMP-with-refit = PDAP insertion + exact coefficient polish):
  residual -> select atom maximizing |<r, phi>|/||phi|| (signed, two-sided)
              or  <r, -phi>/||phi||  (cone, one-sided: atoms enter as -c*phi)
  -> refit all coefficients exactly (lsq_linear with per-column bounds)
  -> record oracle RMSE vs #atoms.

Diagnostics: slope of fitted gradient on the shock-free bulk (does signed
build the slope-C component = emulation found?) + list of selected atoms.
"""

import numpy as np
from scipy.optimize import lsq_linear

N = 240
C_TRUE = 1.0
DELTA = 0.04
K_MAX = 40
rng = np.random.default_rng(3)

x = np.linspace(-1, 1, N)
shocks = np.array([-0.45, 0.1, 0.6])
sizes = np.array([0.7, 0.5, 0.8])


def target(x):
    y = C_TRUE * x
    for t, s in zip(shocks, sizes):
        y = y - s / (1.0 + np.exp(-(x - t) / DELTA))
    return y


# dictionary
bs = np.linspace(-3, 3, 601)
feats, labels = [], []
for eps in (+1.0, -1.0):
    F = 2 * eps * np.maximum(eps * x[None, :] - bs[:, None], 0.0)  # (B, N)
    feats.append(F)
    labels += [(eps, b) for b in bs]
PHI = np.vstack(feats)                      # (2B, N)
PHI_norm = np.linalg.norm(PHI, axis=1)
PHI_norm[PHI_norm == 0] = np.inf
n_atoms_dict = PHI.shape[0]


def greedy(y, vtrue, mode, k_max=K_MAX):
    """mode: 'signed' | 'cone'. Returns list of (n_atoms, rmse), selections."""
    free_cols = []
    if mode == "cone":
        free_cols = [x.copy(), np.ones_like(x)]      # head C~*x (>=0), a free
    active = []          # indices into PHI
    out, sel = [], []

    def refit():
        cols, lo, hi = [], [], []
        for j, fc in enumerate(free_cols):
            cols.append(fc)
            lo.append(0.0 if (mode == "cone" and j == 0) else -np.inf)
            hi.append(np.inf)
        for i in active:
            cols.append(-PHI[i] if mode == "cone" else PHI[i])
            lo.append(0.0 if mode == "cone" else -np.inf)
            hi.append(np.inf)
        if not cols:
            return np.zeros_like(x), np.array([])
        A = np.stack(cols, axis=1)
        r = lsq_linear(A, y, bounds=(np.array(lo), np.array(hi)),
                       tol=1e-12, max_iter=300)
        return A @ r.x, r.x

    fit, coef = refit()
    out.append((0, float(np.sqrt(np.mean((fit - vtrue) ** 2)))))
    for k in range(1, k_max + 1):
        r = y - fit
        scores = (PHI @ r) / PHI_norm
        if mode == "cone":
            scores = -scores                 # atoms enter with -phi
        else:
            scores = np.abs(scores)
        scores[active] = -np.inf
        i = int(np.argmax(scores))
        if scores[i] <= 1e-12:
            break
        active.append(i)
        sel.append(labels[i])
        fit, coef = refit()
        nz = int(np.sum(np.abs(coef[len(free_cols):]) > 1e-9))
        out.append((nz, float(np.sqrt(np.mean((fit - vtrue) ** 2)))))
    return out, sel, fit


def bulk_slope(fit):
    """slope of fitted gradient on the largest shock-free interval."""
    mask = (x > 0.18) & (x < 0.52)           # between shocks 2 and 3
    A = np.stack([x[mask], np.ones(mask.sum())], axis=1)
    sl, _ = np.linalg.lstsq(A, fit[mask], rcond=None)[0]
    return sl


for sigma in (0.0, 0.05):
    vtrue = target(x)
    y = vtrue + sigma * rng.standard_normal(N)
    print(f"\n#### noise sigma = {sigma} ####")
    for mode in ("cone", "signed"):
        out, sel, fit = greedy(y, vtrue, mode)
        # atoms needed for thresholds
        scale = np.sqrt(np.mean(vtrue ** 2))
        line = []
        for frac in (0.10, 0.05, 0.02, 0.01):
            k = next((n for n, e in out if e <= frac * scale), None)
            line.append(f"{frac:4.0%}: {k if k is not None else '--':>3}")
        print(f"[{mode:6s}] atoms to reach err<=frac*scale   " + "   ".join(line))
        print(f"         final rmse after {out[-1][0]} atoms = {out[-1][1]:.4f}"
              f"   bulk slope of fit = {bulk_slope(fit):+.3f} (target {C_TRUE})")
        if mode == "signed":
            eps_list = [e for e, b in sel[:12]]
            print("         first 12 selections (eps, b): "
                  + ", ".join(f"({int(e):+d},{b:.2f})" for e, b in sel[:12]))
            n_pos = sum(1 for e in eps_list if e > 0)
            print(f"         eps balance in first 12: {n_pos}+ / "
                  f"{12 - n_pos}-   (emulation pair needs matched +/- atoms"
                  f" with near-equal coefficients)")
