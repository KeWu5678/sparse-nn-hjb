"""Leg 2 (general curved V) — numerical check of the load-bearing curvature step:
   for V with D^2V >= cI everywhere, on any cell K,
     ∫_K |∇V - <∇V>_K|^2  >=  c^2 ∫_K |x - x̄_K|^2.
(The analytic proof: V = c/2|x|^2 + h, h convex; cross term ∫(x-x̄)·∇h >= 0 by
monotonicity of ∇h. This script confirms it on a NON-quadratic curved V.)
"""
import numpy as np
rng = np.random.default_rng(0)
d = 2; c = 0.7
# V = c/2|x|^2 + sum a_i softplus(w_i·x - b_i), a_i>=0 (convex h) => D^2V >= cI
K = 6
W = rng.standard_normal((K, d)); W /= np.linalg.norm(W, axis=1, keepdims=True)
a = rng.uniform(0.2, 1.5, K); b = rng.uniform(-1, 1, K)

def gradV(X):
    G = c * X.copy()
    for i in range(K):
        s = 1 / (1 + np.exp(-(X @ W[i] - b[i])))
        G += a[i] * s[:, None] * W[i][None, :]
    return G

def hess_min_eig(X):
    mn = 1e9
    for x in X[::50]:
        H = c * np.eye(d)
        for i in range(K):
            z = x @ W[i] - b[i]; s = 1 / (1 + np.exp(-z)); sp = s * (1 - s)
            H += a[i] * sp * np.outer(W[i], W[i])
        mn = min(mn, np.linalg.eigvalsh(H)[0])
    return mn

worst = 1e9
for _ in range(2000):
    ctr = rng.uniform(-0.8, 0.8, d); sz = rng.uniform(0.05, 0.5)
    pts = ctr + rng.uniform(-sz/2, sz/2, (4000, d))
    G = gradV(pts); lhs = np.mean(((G - G.mean(0))**2).sum(1))
    rhs = c**2 * np.mean(((pts - pts.mean(0))**2).sum(1))
    worst = min(worst, lhs / rhs)

print(f"min eig D^2V (sampled) = {hess_min_eig(rng.uniform(-1,1,(4000,d))):.3f}  (>= c={c})")
print(f"worst ratio  ∫|∇V-<∇V>|² / (c²∫|x-x̄|²)  over 2000 cells = {worst:.3f}  (>=1 ⟹ inequality holds)")
