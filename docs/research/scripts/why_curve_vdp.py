"""The quantifiable WHY: n(epsilon) curves for cone vs signed on real VDP (repo
PDAP). Prediction (Theorem A-d): n_signed ~ 1/eps (must build the bulk curvature
from flat ReLU pieces), n_cone ~ const (head supplies curvature; atoms only carry
the O(1) convex correction). Hence n_signed/n_cone = Theta(1/eps) -> diverges.

Confirm by the SLOPE of log n vs log(1/eps): signed slope ~ +1 (or steeper),
cone slope ~ 0. No subtraction (that probe is confounded) -- just the two curves.
"""
import sys
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = Path('/Users/chaoruiz/Documents/Repos/SparseNNforHJB'); sys.path.insert(0, str(ROOT))
import src.config.store  # noqa
from src.data import load_value_samples, normalize_value_samples, split_value_samples
from src.models import build_model
from src.PDAP import PDAP

def cfg(kind, a, gamma):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        return compose(config_name='config', overrides=[
            f'model={kind}','data=vdp','model.activation=softplus',
            f'model.gamma={gamma}',f'model.alpha={a}','model.power=1.0'])

def fit(kind, a, gamma, data):
    tr,va = split_value_samples(data, data and 0.9)
    m = build_model(cfg(kind,a,gamma), data['x'].shape[1])
    c = cfg(kind,a,gamma)
    h = PDAP(c).fit(m, tr, va, num_iterations=c.training.num_iterations,
                    num_insertion=c.training.num_insertion, max_insert=c.training.max_insert,
                    amp_tol=c.training.prune_amp_tol, verbose=False)
    s = h.summary_metrics(); return s['best_neurons'], s['rel_grad_val']

raw = load_value_samples('VDP_beta_0.1_grid_30x30.npy')
data,_ = normalize_value_samples(raw)
GAMMA = 0.0
alphas = np.logspace(-1.0, -4.5, 16)
print(f"# real VDP, softplus, gamma={GAMMA}, repo PDAP")
print(f"{'kind':7} {'alpha':>9} {'n':>4} {'relGrad':>8}", flush=True)
curves = {}
for kind in ('semiconcave','signed'):
    pts = []
    t0=time.time()
    for a in alphas:
        n, gr = fit(kind, a, GAMMA, data); pts.append((gr, n))
        print(f"{kind:7} {a:9.2e} {n:4d} {gr:8.3f}", flush=True)
    curves[kind] = pts
    print(f"  [{kind}: {time.time()-t0:.0f}s]", flush=True)

# n at matched accuracy targets + slope of log n vs log(1/eps)
print("\n# n(eps) at matched accuracy (min neurons reaching relGrad <= target)")
print(f"{'target':>7} {'cone':>5} {'signed':>7} {'ratio':>6}")
tgts = [0.35,0.30,0.25,0.22,0.20,0.18]
rows=[]
def minn(pts,t):
    ok=[n for gr,n in pts if gr<=t]; return min(ok) if ok else None
for t in tgts:
    nc=minn(curves['semiconcave'],t); ns=minn(curves['signed'],t)
    r = (ns/nc) if (nc and ns) else None
    rows.append((t,nc,ns))
    print(f"{t:7.2f} {str(nc):>5} {str(ns):>7} {('%.1f'%r) if r else '--':>6}")
# slope fit on points where both defined
pts_both=[(t,nc,ns) for t,nc,ns in rows if nc and ns]
if len(pts_both)>=2:
    le=np.log([1/t for t,_,_ in pts_both])
    sc=np.log([nc for _,nc,_ in pts_both]); ss=np.log([ns for _,_,ns in pts_both])
    slope_c=np.polyfit(le,sc,1)[0]; slope_s=np.polyfit(le,ss,1)[0]
    print(f"\nslope d(log n)/d(log 1/eps):  cone={slope_c:+.2f}  signed={slope_s:+.2f}")
    print("Theorem A-d predicts: signed >~ +1 (build bulk curvature), cone ~ 0 (head free).")
    print(f"=> ratio exponent ~ {slope_s-slope_c:+.2f} (n_signed/n_cone = Theta(eps^-{slope_s-slope_c:.2f}))")
