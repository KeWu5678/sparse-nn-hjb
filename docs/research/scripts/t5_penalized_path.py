"""Regularized-path comparison at p=2 (the PDAP-faithful version of the
greedy test): signed = lasso path over atom gradients; cone+head = nonneg
lasso over -atoms with the head columns {x, 1} UNPENALIZED (projected out by
Frisch-Waugh: residualize y and the atom features against span{x, 1}, then
positive-lasso; the head coefficients are recovered implicitly).

KNOWN LIMITATION (referee 2026-06-12): the Frisch-Waugh formulation leaves
C~ >= 0 UNENFORCED (head slope free both signs). On the saturated target the
violation occurs at 14/297 path points, all off-threshold (C~ ~ +10 at the
reported thresholds), so the headline numbers stand; for concave-bulk targets
this formulation would overpower the cone model (cf. Lemma B2's condition).
Also note: reported atom counts exclude the cone's 2 free head params.

Same target/dictionary as t5_greedy_emulation.py. Exact LARS paths
(sklearn.linear_model.lars_path, lasso variant). Atom count = #nonzero along
the path; oracle RMSE computed by refitting... no: we evaluate the PATH
solutions themselves (penalized coefficients), which is what PDAP-with-alpha
returns before SSN polish; we also report debiased (refit-on-support) curves,
which is what PDAP-with-SSN-polish returns. Both views printed.

Readout: atoms needed for oracle gradient-RMSE <= frac*scale, signed vs cone,
path view and debiased view, sigma in {0, 0.05}.
"""

import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import lars_path

N = 240
C_TRUE = 1.0
DELTA = 0.04
rng = np.random.default_rng(3)

x = np.linspace(-1, 1, N)
shocks = np.array([-0.45, 0.1, 0.6])
sizes = np.array([0.7, 0.5, 0.8])


def target(x):
    y = C_TRUE * x
    for t, s in zip(shocks, sizes):
        y = y - s / (1.0 + np.exp(-(x - t) / DELTA))
    return y


bs = np.linspace(-3, 3, 601)
cols = []
for eps in (+1.0, -1.0):
    cols.append(2 * eps * np.maximum(eps * x[None, :] - bs[:, None], 0.0))
PHI = np.vstack(cols).T                      # (N, 2B): atom gradient features

H = np.stack([x, np.ones_like(x)], axis=1)   # head columns (gradient space)
P = H @ np.linalg.pinv(H)                    # projector onto span{x, 1}


def eval_path(name, X, y, vtrue, positive, project_head, max_atoms=60):
    """Run lasso path; return list (n_atoms, rmse_path, rmse_debiased)."""
    Xw, yw = (X - P @ X, y - P @ y) if project_head else (X, y)
    alphas, _, coefs = lars_path(Xw, yw, method="lasso",
                                 positive=positive, max_iter=600)
    out = []
    seen = {}
    for j in range(coefs.shape[1]):
        c = coefs[:, j]
        supp = np.flatnonzero(np.abs(c) > 1e-10)
        n = len(supp)
        if n == 0 or n > max_atoms:
            continue
        # path solution (+ implicit head refit if projected)
        fit = X[:, supp] @ c[supp]
        if project_head:
            th = lstsq(H, y - fit, rcond=None)[0]
            fitp = fit + H @ th
        else:
            fitp = fit
        e_path = float(np.sqrt(np.mean((fitp - vtrue) ** 2)))
        # debiased: LS refit on support (with head if cone)
        A = np.concatenate([X[:, supp], H], axis=1) if project_head \
            else X[:, supp]
        cf = lstsq(A, y, rcond=None)[0]
        e_db = float(np.sqrt(np.mean((A @ cf - vtrue) ** 2)))
        if n not in seen or e_db < seen[n][1]:
            seen[n] = (e_path, e_db)
    return sorted((n, ep, ed) for n, (ep, ed) in seen.items())


for sigma in (0.0, 0.05):
    vtrue = target(x)
    y = vtrue + sigma * rng.standard_normal(N)
    scale = np.sqrt(np.mean(vtrue ** 2))
    print(f"\n#### noise sigma = {sigma} ####")
    res_signed = eval_path("signed", PHI, y, vtrue, False, False)
    res_cone = eval_path("cone", -PHI, y, vtrue, True, True)
    for view, idx in (("path (PDAP pre-polish)", 1),
                      ("debiased (PDAP+SSN)", 2)):
        line_s, line_c = [], []
        for frac in (0.10, 0.05, 0.02, 0.01):
            thr = frac * scale
            ks = next((n for n, *e in res_signed if e[idx - 1] <= thr), None)
            kc = next((n for n, *e in res_cone if e[idx - 1] <= thr), None)
            line_s.append(f"{frac:4.0%}:{ks if ks else '--':>3}")
            line_c.append(f"{frac:4.0%}:{kc if kc else '--':>3}")
        print(f"  [{view}]")
        print(f"    signed : " + "  ".join(line_s))
        print(f"    cone   : " + "  ".join(line_c))
