"""T3 (cone-free) check in 2D: for a convex g with finite R-norm, is the minimal
variation-norm (‖c‖_1) signed ReLU representation automatically NONNEGATIVE?
I.e. is γ⁺(g) = γ(g)?  This is a VARIATION-NORM question (min ‖c‖_1), faithful to
dictionary basis-pursuit (unlike neuron COUNT). If γ⁺ ≈ γ, T3 supported ⟹ the 1-D
separation argument lifts to d≥2.

Compare min ‖c‖_1 over signed vs nonnegative c, at matched fit error, for several
convex g on [-1,1]^2 (affine part free/unpenalized).
"""
import numpy as np
from sklearn.linear_model import Lasso
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

m = 22; t = np.linspace(-1,1,m); X1,X2 = np.meshgrid(t,t)
P = np.stack([X1.ravel(), X2.ravel()],1); N = P.shape[0]
def relu(z): return np.maximum(z,0.)

# ReLU dictionary
nd = 36; ang = np.linspace(0,2*np.pi,nd,endpoint=False)
Wd = np.stack([np.cos(ang),np.sin(ang)],1)
nb = 31; bd = np.linspace(-1.5,1.5,nb)
atoms=[(w,b) for w in Wd for b in bd]; A=len(atoms)
Phi=np.zeros((N,A))
for k,(w,b) in enumerate(atoms): Phi[:,k]=relu(P@w-b)
# free affine columns [x1,x2,1]
H=np.stack([P[:,0],P[:,1],np.ones(N)],1); Pp=H@np.linalg.pinv(H)
Phw=Phi-Pp@Phi

convex_targets = {
  "sqrt(1+|x|^2) (curved Hessian)": np.sqrt(1+ (P**2).sum(1)),
  "logsumexp(2x1,2x2,0)/2":        0.5*np.log(np.exp(2*P[:,0])+np.exp(2*P[:,1])+1),
  "max(x1,x2,0) (polyhedral convex)": np.maximum.reduce([P[:,0],P[:,1],np.zeros(N)]),
  "0.5|x|^2 (pure quadratic)":     0.5*(P**2).sum(1),
}
def minl1(target, pos):
    y=target-Pp@target
    best=None
    for a in np.logspace(-2.5,-5,10):
        mdl=Lasso(alpha=a,fit_intercept=False,max_iter=60000,tol=1e-9,positive=pos)
        mdl.fit(Phw,y); c=mdl.coef_
        fit=Phw@c; rel=np.linalg.norm(fit-y)/np.linalg.norm(y)
        l1=np.abs(c).sum()
        if rel<=0.05:
            if best is None or l1<best[0]: best=(l1,rel,int((np.abs(c)>1e-8).sum()))
    return best

print("T3 cone-free 2D: min ‖c‖_1 signed vs nonneg, matched fit (rel<=0.05)")
print(f"{'convex g':34} {'γ(signed)':>10} {'γ+(nonneg)':>11} {'γ+/γ':>7}  verdict")
for name,tg in convex_targets.items():
    s=minl1(tg,False); p=minl1(tg,True)
    if s and p:
        ratio=p[0]/s[0]
        v = "CONE-FREE" if ratio<1.05 else ("cone penalty" if ratio<5 else "CONE EXPENSIVE")
        print(f"{name:34} {s[0]:10.3f} {p[0]:11.3f} {ratio:7.2f}  {v}")
    else:
        print(f"{name:34}  (one model couldn't reach 5% — signed={s}, nonneg={p})")
print("\nT3 holds (γ+≈γ) ⟹ cone constraint free for convex g ⟹ 1-D separation lifts to d≥2")
