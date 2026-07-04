"""Verify the Radon-convexity lemma: g convex ⟹ R{g}(w,b)=∫_{w·x=b} g ds is CONVEX in b
for every direction w (i.e. ∂_b^2 R{g} = ∫_{line} wᵀ D²g w ds ≥ 0). The d≥2 generalization
of g''≥0. Used by cone_free_convex.md (T3). Nonconvex g is a control (should fail)."""
import numpy as np


def radon_b(g_fn, w, bs, S=400, L=3.0):
    wp = np.array([-w[1], w[0]]); s = np.linspace(-L, L, S)
    return np.array([np.mean(g_fn(b*w[None,:] + s[:,None]*wp[None,:]))*(2*L) for b in bs])

def d2(R, bs):
    h = bs[1]-bs[0]; return (R[2:]-2*R[1:-1]+R[:-2])/h**2

bs = np.linspace(-1.5, 1.5, 80)
convex = {
  "0.5|x|^2":               lambda X: 0.5*(X**2).sum(1),
  "sqrt(1+|x|^2)":          lambda X: np.sqrt(1+(X**2).sum(1)),
  "x1^4/4+x2^2/2":          lambda X: X[:,0]**4/4 + X[:,1]**2/2,
  "logsumexp(2x1,2x2,0)/2": lambda X: 0.5*np.log(np.exp(2*X[:,0])+np.exp(2*X[:,1])+1),
}
control = {"sin(3x1)+0.5|x|^2 (NONconvex)": lambda X: np.sin(3*X[:,0]) + 0.5*(X**2).sum(1)}

print("Radon-convexity: min over w,b of ∂_b^2 R{g}  (>=0 ⟺ R convex in b ⟺ lemma holds)")
for name, g in {**convex, **control}.items():
    m = min(d2(radon_b(g, np.array([np.cos(a),np.sin(a)]), bs), bs).min()
            for a in np.linspace(0, np.pi, 12, endpoint=False))
    tag = "CONVEX in b ✓" if m > -0.05 else "NOT convex (expected: nonconvex g)"
    print(f"  {name:34} min ∂_b^2 R = {m:+.3f}   {tag}")
