# Theoretical Side Remarks

## Equivalent Formulations of Regularized Problems (p. 4 of nonconvex regularization paper)

The paragraph on page 4 invokes the classical **Lagrangian duality / Lagrange multiplier equivalence** in convex optimization. There are three equivalent ways to write the same regularized problem:

**1. Penalized form (Lagrangian relaxation):**
```
min  loss(N) + alpha * ||c||_l1
```

**2. Norm-constrained form (Tikhonov):**
```
min  loss(N)     subject to  ||c||_l1 <= M
```

**3. Fidelity-constrained form (Ivanov regularization):**
```
min  ||c||_l1    subject to  ||N_{omega,c} - f|| <= delta
```

Under mild conditions these three trace out the **same Pareto frontier** between fit and regularization — for each value of alpha there exists an M and a delta giving the same solution, and vice versa.

### Why the author mentions this

The theoretical results about solution structure and sparsity derived for the penalized form (P_l1) automatically carry over to both constrained variants. The author broadens the scope of applicability without re-proving anything.

### On delta

delta is not defined earlier in the paper. It is introduced here purely to name the hyperparameter of the third (fidelity-constrained) formulation, completing the analogy:
- alpha parametrizes the penalized form
- M parametrizes the norm-constrained form
- delta parametrizes the fidelity-constrained form
