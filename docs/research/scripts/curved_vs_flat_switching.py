"""Test the d>=2 negative result: the repo cone (ReLU-ridge) model inherits the
curvature budget only for FLAT switching sets, not CURVED ones.

V = 1/2||x||^2 - g, two convex corrections g on [-2,2]^2:
  FLAT   : g = (x1)_+ + (x2)_+        (flat switching = axes; gamma+ finite)
  CURVED : g = (||x||-1)_+            (circular switching; gamma+ = inf by T2)
Both have finite curvature budget (Prop B). Predict: cone N(eps) bounded for FLAT,
growing for CURVED -- confirming gamma+(g)=inf on curved switching sets (the gap in
semiconcave-rate.md (d)).  Faithful repo PDAP; timing-probed.
"""
import sys, time, numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
ROOT = Path('/Users/chaoruiz/Documents/Repos/SparseNNforHJB'); sys.path.insert(0, str(ROOT))
import src.config.store  # noqa
from src.data import normalize_value_samples, split_value_samples
from src.models import build_model
from src.PDAP import PDAP

m=34; t=np.linspace(-2,2,m); X1,X2=np.meshgrid(t,t)
P=np.stack([X1.ravel(),X2.ravel()],1).astype(np.float64); r=np.linalg.norm(P,axis=1)
def make(kind):
    if kind=='flat':
        g  = np.maximum(P[:,0],0)+np.maximum(P[:,1],0)
        gg = np.stack([(P[:,0]>0).astype(float),(P[:,1]>0).astype(float)],1)
    else:  # curved: (||x||-1)_+
        g  = np.maximum(r-1,0)
        out=(r>1); xhat=P/np.maximum(r,1e-9)[:,None]
        gg = xhat*out[:,None]
    V  = 0.5*(P**2).sum(1) - g
    Vg = P - gg
    d={'x':P,'v':V.reshape(-1,1),'dv':Vg}; d,_=normalize_value_samples(d); return d

def cfg(a):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        return compose(config_name='config', overrides=[
            'model=semiconcave','data=vdp','model.activation=relu','model.gamma=0',
            f'model.alpha={a}','model.power=1.0'])
def fit(a, data):
    c=cfg(a); tr,va=split_value_samples(data,c.data.train_fraction); md=build_model(c,2)
    try:
        h=PDAP(c).fit(md,tr,va,num_iterations=c.training.num_iterations,num_insertion=c.training.num_insertion,
                      max_insert=c.training.max_insert,amp_tol=c.training.prune_amp_tol,verbose=False)
    except RuntimeError:  # no atoms accepted at this alpha => head-only fit
        return 0, 9.9
    s=h.summary_metrics(); return s['best_neurons'], s['rel_grad_val']

dflat=make('flat'); dcurv=make('curved')
t0=time.time(); n,gr=fit(1e-2,dflat); dt=time.time()-t0
ALPHAS=np.logspace(-1.5,-5.0,10)
print(f"PROBE 1 fit={dt:.1f}s; full ~{dt*2*len(ALPHAS):.0f}s",flush=True)
if dt>20: print("ABORT slow"); raise SystemExit

def curve(data,label):
    print(f"\n# {label}: cone (ReLU ridge) N(eps)",flush=True)
    pts=[]
    for a in ALPHAS:
        n,gr=fit(a,data); pts.append((gr,n)); print(f"  alpha={a:.1e} n={n:3d} relGrad={gr:.3f}",flush=True)
    return pts
def report(pts,label):
    def minn(tg): ok=[n for g,n in pts if g<=tg]; return min(ok) if ok else None
    tgs=[0.20,0.15,0.10,0.05]
    print(f"  {label} N(eps): "+" ".join(f"{tg}:{minn(tg)}" for tg in tgs)+f"  best={min(g for g,_ in pts):.3f}")
    return [(tg,minn(tg)) for tg in tgs]

pf=curve(dflat,'FLAT switching  g=(x1)_+ + (x2)_+')
pc=curve(dcurv,'CURVED switching g=(||x||-1)_+')
print("\n=== verdict ===")
rf=report(pf,'FLAT  '); rc=report(pc,'CURVED')
import numpy as np
def slope(rows):
    b=[(tg,n) for tg,n in rows if n]
    return np.polyfit(np.log([1/tg for tg,_ in b]),np.log([n for _,n in b]),1)[0] if len(b)>=2 else float('nan')
print(f"slope d(log n)/d(log 1/eps):  FLAT={slope(rf):+.2f}  CURVED={slope(rc):+.2f}")
print("predict: FLAT ~0 (bounded, gamma+ finite); CURVED >0 (growing, gamma+ = inf, T2)")
