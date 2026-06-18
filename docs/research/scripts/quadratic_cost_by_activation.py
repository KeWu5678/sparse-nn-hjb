"""N_sigma(Q, eps): how many SIGNED atoms does the real PDAP need to fit a PURE
quadratic V = (C/2)||x||^2 (g=0) to gradient accuracy eps, per activation?

This isolates the head's capacity advantage with NO ReLU assumption and NO
synthetic-target confound: cone fits Q with 0 atoms (head exact), so
n_signed(eps) = N_sigma(Q, eps) exactly. The slope d(log n)/d(log 1/eps) per
activation tells us:
  slope ~ +1  (like ReLU)   -> quadratic is expensive  -> head = CAPACITY advantage
  slope ~ 0  / tiny n        -> quadratic is cheap       -> the real gap is SELECTION
Decisive for whether Theorem A-d generalizes to the activations we actually use.
"""
import sys, time
import numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
ROOT = Path('/Users/chaoruiz/Documents/Repos/SparseNNforHJB'); sys.path.insert(0, str(ROOT))
import src.config.store  # noqa
from src.data import normalize_value_samples, split_value_samples
from src.models import build_model
from src.PDAP import PDAP

# pure-quadratic data on a grid (normalized box), C=1
m=30; t=np.linspace(-1,1,m); X1,X2=np.meshgrid(t,t)
P=np.stack([X1.ravel(),X2.ravel()],1).astype(np.float64)
C=1.0
data={'x':P, 'v':(0.5*C*(P**2).sum(1)).reshape(-1,1), 'dv':C*P}
data,_=normalize_value_samples(data)   # match train.py preprocessing exactly

def cfg(kind, act, a):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        return compose(config_name='config', overrides=[
            f'model={kind}','data=vdp',f'model.activation={act}',
            'model.gamma=0', f'model.alpha={a}','model.power=1.0'])

def fit(kind, act, a):
    c=cfg(kind,act,a); tr,va=split_value_samples(data, c.data.train_fraction)
    md=build_model(c, 2)
    h=PDAP(c).fit(md,tr,va,num_iterations=c.training.num_iterations,
                  num_insertion=c.training.num_insertion,max_insert=c.training.max_insert,
                  amp_tol=c.training.prune_amp_tol,verbose=False)
    s=h.summary_metrics(); return s['best_neurons'], s['rel_grad_val']

# timing probe
t0=time.time(); n,gr=fit('signed','softplus',1e-3); dt=time.time()-t0
ACTS=['softplus','tanh','gelu_squared','gaussian','matern52']
ALPHAS=np.logspace(-1.0,-4.0,10)
print(f"PROBE 1 fit = {dt:.1f}s; full ~ {dt*len(ACTS)*len(ALPHAS):.0f}s ({dt*len(ACTS)*len(ALPHAS)/60:.1f} min)", flush=True)
if dt>15: print("ABORT slow"); raise SystemExit

print(f"\n# n_signed(eps) = N_sigma(pure quadratic, eps), repo PDAP, gamma=0")
import math
for act in ACTS:
    pts=[]
    for a in ALPHAS:
        n,gr=fit('signed',act,a); pts.append((gr,n))
    # n at matched accuracy + slope
    def minn(t):
        ok=[n for gr,n in pts if gr<=t]; return min(ok) if ok else None
    tg=[0.20,0.15,0.10,0.07]
    ns={t:minn(t) for t in tg}
    both=[(t,ns[t]) for t in tg if ns[t]]
    slope=float('nan')
    if len(both)>=2:
        le=np.log([1/t for t,_ in both]); ln=np.log([n for _,n in both])
        slope=np.polyfit(le,ln,1)[0]
    nstr=" ".join(f"{t}:{ns[t]}" for t in tg)
    print(f"  {act:14} n(eps)[{nstr}]  best={min(g for g,_ in pts):.3f}  slope={slope:+.2f}", flush=True)
print("\nread per activation: slope ~ +1 => quadratic expensive => HEAD=capacity (Thm A-d holds);")
print("                     small n / slope ~0 => quadratic cheap => gap is SELECTION, not head.")
