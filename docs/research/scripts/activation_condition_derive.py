"""Derive the activation condition from FAITHFUL data (repo PDAP), not by guessing.

Hypothesis to TEST (not assert): the cone advantage on real V is driven by how hard
it is for signed sigma-atoms to build the quadratic that the head supplies free.
So measure, per activation, at MATCHED accuracy:
  (Q)  N(signed, pure quadratic 1/2||x||^2, eps)   = "quadratic cost" / head value
  (S)  N(signed, VDP, eps)
  (C)  N(cone,   VDP, eps)
advantage = S/C. Test: does advantage correlate with Q?  If yes, the condition is
"sigma-atoms have high quadratic cost", stated precisely. Multi-property tags printed
so the condition can be read off (monotone? bounded? growth? sigma'' sign).
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

# --- data: pure quadratic Q and real VDP ---
m=30; t=np.linspace(-1,1,m); X1,X2=np.meshgrid(t,t); P=np.stack([X1.ravel(),X2.ravel()],1).astype(np.float64)
dataQ={'x':P,'v':(0.5*(P**2).sum(1)).reshape(-1,1),'dv':P}; dataQ,_=normalize_value_samples(dataQ)
dataV,_=normalize_value_samples(load_value_samples('VDP_beta_0.1_grid_30x30.npy'))

def cfg(kind, act, a, p):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        return compose(config_name='config', overrides=[
            f'model={kind}','data=vdp',f'model.activation={act}','model.gamma=0',
            f'model.alpha={a}',f'model.power={p}'])
def fit(kind, act, a, p, data):
    c=cfg(kind,act,a,p); tr,va=split_value_samples(data,c.data.train_fraction); md=build_model(c,2)
    h=PDAP(c).fit(md,tr,va,num_iterations=c.training.num_iterations,num_insertion=c.training.num_insertion,
                  max_insert=c.training.max_insert,amp_tol=c.training.prune_amp_tol,verbose=False)
    s=h.summary_metrics(); return s['best_neurons'], s['rel_grad_val']
def minN(kind, act, p, data, tgt, alphas):
    best=None; bestreach=9.9
    for a in alphas:
        n,gr=fit(kind,act,a,p,data); bestreach=min(bestreach,gr)
        if gr<=tgt and (best is None or n<best): best=n
    return best, bestreach

ALPHAS=np.logspace(-1.0,-4.0,7)
# (activation, power, property tags: monotone, bounded, growth, sigma'' sign)
ACTS=[('softplus',1.0,'monotone,linear-growth,convex(σ″>0)'),
      ('tanh',    1.0,'monotone,bounded,σ″ sign-changes'),
      ('gaussian',1.0,'NON-monotone(bump),bounded,σ″ sign-changes'),
      ('matern52',1.0,'NON-monotone(bump),bounded'),
      ('gelu_squared',1.0,'monotone-ish,QUADRATIC-growth')]
TQ=0.15; TV=0.30
print(f"# faithful repo PDAP, matched accuracy (Q:relGrad<={TQ}, VDP:relGrad<={TV})", flush=True)
print(f"{'activation':13} {'N_sig(Q)':>9} {'Qreach':>7} | {'N_sig(VDP)':>10} {'N_cone(VDP)':>11} {'adv=S/C':>8}  props", flush=True)
rows=[]
for act,p,props in ACTS:
    t0=time.time()
    nQ,rQ = minN('signed',act,p,dataQ,TQ,ALPHAS)
    nS,_  = minN('signed',act,p,dataV,TV,ALPHAS)
    nC,_  = minN('semiconcave',act,p,dataV,TV,ALPHAS)
    adv = (nS/nC) if (nS and nC) else None
    rows.append((act,nQ,rQ,nS,nC,adv,props))
    qstr = str(nQ) if nQ else f">{len(ALPHAS)}a(@{rQ:.2f})"
    print(f"{act:13} {qstr:>9} {rQ:7.2f} | {str(nS):>10} {str(nC):>11} {str(round(adv,1) if adv else None):>8}  {props}  [{time.time()-t0:.0f}s]", flush=True)
print("\n# TEST: does quadratic cost N_sig(Q) [or Q-unreachable] track the advantage S/C?")
print("# high quad cost (can't fit Q) -> head essential -> large advantage?")
