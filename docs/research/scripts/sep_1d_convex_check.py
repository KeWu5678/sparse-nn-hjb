"""1-D current-goal check (σ=ReLU, gradient-L²). V = C/2 x² − g, g convex on [-1,1].
In 1-D a signed ReLU sum's derivative = ANY n-step piecewise-constant; the semiconcave
model's derivative = C·x + (n-step piecewise-constant) [head exact]. So:
  e_signed(n) = best n-step L² approx error of V' = Cx − g'
  e_semi(n)   = best n-step L² approx error of g'     [head removes Cx exactly]
Both ~ K/n. Predict K_semi/K_signed = (free-knot L¹ const) = bud(g)/∫|C−g''| = ∫g''/∫|C−g''|.
Verify via exact DP optimal n-step approximation. "more sparse" ⟺ bud(g) < ∫|C−g''|.
"""
import numpy as np

N = 600; x = np.linspace(-1, 1, N); dx = x[1]-x[0]

def best_nstep_err(h, n):
    """Exact min L2 error (squared, integrated) of best n-piece constant approx of h on grid,
    via DP over partitions into n contiguous segments. Cost of segment [i,j) = sum (h-mean)^2*dx."""
    # prefix sums for segment SSE
    P1 = np.concatenate([[0], np.cumsum(h)])
    P2 = np.concatenate([[0], np.cumsum(h*h)])
    def sse(i, j):  # [i,j)
        c = j - i
        s = P1[j]-P1[i]; s2 = P2[j]-P2[i]
        return (s2 - s*s/c) * dx
    INF = 1e18
    # DP[k][i] = min SSE to cover [0,i) with k segments
    dp = np.full((n+1, N+1), INF); dp[0,0] = 0
    arg = np.zeros((n+1, N+1), int)
    for k in range(1, n+1):
        for i in range(k, N+1):
            best = INF; ba = 0
            # last segment [j,i)
            lo = k-1
            for j in range(lo, i):
                v = dp[k-1, j] + sse(j, i)
                if v < best: best = v; ba = j
            dp[k, i] = best; arg[k, i] = ba
    return np.sqrt(max(dp[n, N], 0.0))

def run(name, gpp_fn, C):
    # g'' >= 0 given on grid; integrate to g', and form V' = C x - g'
    gpp = gpp_fn(x)
    gp = np.concatenate([[0], np.cumsum(0.5*(gpp[1:]+gpp[:-1]))*dx])  # g'(x), g'(-1)=0
    gp = gp[:N]
    Vp = C*x - gp
    bud = np.trapezoid(gpp, x)                 # ∫ g''
    sig_var = np.trapezoid(np.abs(C - gpp), x) # ∫ |C - g''| = ∫|V''|
    ns = [4, 8, 16, 32]
    es = [best_nstep_err(gp, n) for n in ns]    # semi
    ez = [best_nstep_err(Vp, n) for n in ns]    # signed
    Ks = np.mean([e*n for e, n in zip(es, ns)])
    Kz = np.mean([e*n for e, n in zip(ez, ns)])
    print(f"\n{name}  (C={C})")
    print(f"  bud(g)=∫g'' = {bud:.3f}   ∫|C-g''| = {sig_var:.3f}   predict K_semi/K_signed = {bud/sig_var:.3f}")
    print(f"  measured  K_semi={Ks:.3f}  K_signed={Kz:.3f}  ratio={Ks/Kz:.3f}")
    print(f"  semi err (n=4..32): {[f'{e:.3f}' for e in es]}")
    print(f"  sign err (n=4..32): {[f'{e:.3f}' for e in ez]}")
    print(f"  => MORE SPARSE (semi<signed): {bud < sig_var}")

# C = sup V'' = sup(C - g'') ; choose C = max g'' so that min V''=... actually set C and report
run("uniform g'' = 0.5 (V''=C-0.5, V convex if C>0.5)", lambda x: 0.5*np.ones_like(x), C=1.0)
run("g'' spike (nearly-linear V except center)", lambda x: 3.0*np.exp(-(x/0.1)**2), C=3.0)
run("g'' = 0.9*C broad (g strongly curved)", lambda x: 0.9*np.ones_like(x), C=1.0)
run("flat g: g''=0 (pure quadratic V) ", lambda x: 0.0*x + 1e-9, C=1.0)
run("g = |x|-like kink (g''=delta approx at 0)", lambda x: 8.0*np.exp(-(x/0.05)**2), C=1.0)
