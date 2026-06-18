"""d=2 current-goal test, SMOOTH convex g (the hard case, no switching set).
V = C/2|x|^2 - g, g = (1/b)logsumexp(b x1, b x2, 0) (smooth convex, bounded Hessian).
C chosen so D^2V = C I - D^2g >~ 1.5 I (V strongly curved). Faithful repo PDAP.
Measure N(semiconcave) vs N(signed) across accuracy; check semiconcave more sparse and
compare to the curvature ratio  ∫|D^2 g| / ∫|D^2 V|  (the d=1 predictor).
"""
import sys, time, numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
ROOT = Path('/Users/chaoruiz/Documents/Repos/SparseNNforHJB'); sys.path.insert(0,str(ROOT))
import src.config.store  # noqa
from src.data import normalize_value_samples, split_value_samples
from src.models import build_model
from src.PDAP import PDAP

m=32; t=np.linspace(-1,1,m); X1,X2=np.meshgrid(t,t)
P=np.stack([X1.ravel(),X2.ravel()],1).astype(np.float64); N=P.shape[0]
C=2.0; b=2.0
z=np.stack([b*P[:,0], b*P[:,1], np.zeros(N)],1)
M=z.max(1); lse=M+np.log(np.exp(z-M[:,None]).sum(1)); g=lse/b
sm=np.exp(z-M[:,None]); sm=sm/sm.sum(1,keepdims=True)   # softmax weights
gg=np.stack([sm[:,0], sm[:,1]],1)                        # grad g
V=0.5*C*(P**2).sum(1)-g; Vg=C*P-gg
data={'x':P,'v':V.reshape(-1,1),'dv':Vg}; data,_=normalize_value_samples(data)

# curvature integrals (predictor): D^2g = (b)(diag(s)-s s^T) over softmax; |D^2g|~tr=b*sum s_i(1-s_i)
trD2g = b*(sm*(1-sm)).sum(1)            # tr D^2 g >=0
trD2V = (2*C - trD2g)                    # tr(C I - D^2 g), I in 2D contributes 2C
intD2g=trD2g.mean(); intD2V=np.abs(trD2V).mean()
print(f"# predictor: ∫|D^2 g| ~ {intD2g:.3f}   ∫|D^2 V| ~ {intD2V:.3f}   ratio={intD2g/intD2V:.3f}")

def cfg(kind,a):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'),version_base=None):
        return compose(config_name='config',overrides=[f'model={kind}','data=vdp',
            'model.activation=relu','model.gamma=0',f'model.alpha={a}','model.power=1.0'])
def fit(kind,a):
    c=cfg(kind,a); tr,va=split_value_samples(data,c.data.train_fraction); md=build_model(c,2)
    try:
        h=PDAP(c).fit(md,tr,va,num_iterations=c.training.num_iterations,num_insertion=c.training.num_insertion,
                      max_insert=c.training.max_insert,amp_tol=c.training.prune_amp_tol,verbose=False)
    except RuntimeError: return 0,9.9
    s=h.summary_metrics(); return s['best_neurons'],s['rel_grad_val']

t0=time.time(); n,gr=fit('semiconcave',1e-2); dt=time.time()-t0
ALPHAS=np.logspace(-1.0,-4.0,8)
print(f"PROBE {dt:.1f}s; full ~{dt*2*len(ALPHAS):.0f}s",flush=True)
if dt>20: print("ABORT"); raise SystemExit
def sweep(kind):
    print(f"# {kind}",flush=True); pts=[]
    for a in ALPHAS:
        n,gr=fit(kind,a); pts.append((gr,n)); print(f"  a={a:.1e} n={n:3d} relGrad={gr:.3f}",flush=True)
    return pts
ps=sweep('semiconcave'); pz=sweep('signed')
def minn(pts,tg): ok=[n for g,n in pts if g<=tg]; return min(ok) if ok else None
print("\n# N at matched accuracy")
print(f"{'relGrad':>8} {'semi':>5} {'signed':>7} {'ratio':>6}")
for tg in (0.20,0.15,0.10,0.07):
    a=minn(ps,tg); s=minn(pz,tg); r=(a/s) if (a and s) else None
    print(f"{tg:8.2f} {str(a):>5} {str(s):>7} {(f'{r:.2f}' if r else '--'):>6}")
print(f"\npredict semi/signed ~ curvature ratio {intD2g/intD2V:.2f}; semiconcave more sparse if <1")
