"""Insertion strategies for the PDAP outer loop.

An insertion strategy proposes new atoms (inner weights/biases) to add to the
current support, given the data and the current residual.  Both strategies use
multistart L-BFGS on the fidelity derivative, but their parameter domains and
acceptance tests differ:

  * ``profile_threshold`` — for a nonhomogeneous dictionary, sample inside the
    theorem/numerical radius, jointly refine ``omega=(a,b)``, discard final
    points outside that radius, remove Euclidean near-duplicates, and apply the
    configured profile threshold.  There is no direction-then-radius code path.
  * ``finite_step`` — accept candidates with a profitable finite step, i.e. where
    min_c Delta J(c; omega) < 0; returns the
    optimal outer weight c* alongside each atom.  Used for the q<1 penalty.

The strategies are pure functions: the caller supplies the precomputed residual
(so the "zero network" first iteration is just residual = -target), the shared
problem and its penalty.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from ..models import Penalty, SparseModel

# Numerical bound for random starts and final candidate filtering.
FIXED_LOG_CLAMP = 5.0

logger = logging.getLogger(__name__)

__all__ = [
    "profile_threshold",
    "finite_step",
]


# ---------------------------------------------------------------------------- #
# Strategy 1: profile-threshold acceptance
# ---------------------------------------------------------------------------- #
def profile_threshold(
    problem: SparseModel,
    penalty: Penalty,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    max_insert: int = 15, merge_tol: float = 1e-2,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = True,
    lbfgs_lr: float = 1e-2, lbfgs_steps: int = 200,
    insert_init: str = "warm_start",
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Select candidates for Algorithm 1 or the ReLU--L1 baseline.

    Normalized Algorithm 1 accepts ``|P(omega)| / w_p(omega) > alpha``.
    The ReLU--L1 baseline has sphere-normalized inner parameters but no moment
    normalization, so it accepts ``|P(omega)| > alpha``.
    Both use the implemented log/L1 family with ``L_phi = phi'(0+) = 1``;
    for a general penalty the threshold would be ``alpha * L_phi``.

    Candidates are ranked by their margin above the applicable threshold. For
    Algorithm 1 this is the certificate violation
    ``Delta(mu, omega) = |P(omega)| / w_p(omega) - alpha``.

    Returns ``(W, b, c)``; ``c`` is ``None`` unless ``insert_init="guaranteed"``,
    in which case it carries the theorem's per-atom coefficient.
    """
    d_dim = problem.input_dim

    a_t, b_t, discarded_outside = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    guaranteed = insert_init == "guaranteed"

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    margins: List[float] = []
    accepted_c: List[float] = []
    for a_i, b_i in zip(a_t, b_t):
        slope, curvature = _slope_and_curvature(problem, residual, a_i, b_i)
        v = abs(slope)
        scale = float(problem.moment_weight_of_atoms(a_i, b_i))
        threshold = penalty.alpha * scale
        # Retain the physical-profile comparison, including its rounding.
        if v <= threshold or (guaranteed and curvature < 1e-30):
            continue
        accepted_a.append(a_i.detach())
        accepted_b.append(b_i.detach())
        margins.append(v / scale - penalty.alpha)
        if guaranteed:
            magnitude = (v - threshold) / curvature
            accepted_c.append(-magnitude if slope > 0 else magnitude)

    best = _best_indices(margins, max_insert)
    accepted_a = [accepted_a[i] for i in best]
    accepted_b = [accepted_b[i] for i in best]
    if guaranteed:
        accepted_c = [accepted_c[i] for i in best]

    if verbose:
        rule = (
            "|P|/w_p above alpha*L_phi" if problem.normalized
            else "|P| above alpha"
        )
        logger.debug(
            "Candidate search  sampled=%d  unique=%d  accepted=%d/%d  "
            "rule=%s (init=%s)",
            N, a_t.shape[0], len(accepted_a), max_insert, rule, insert_init,
        )
        if discarded_outside:
            logger.debug(
                "Candidate search discarded %d refined candidate(s) outside the "
                "search radius",
                discarded_outside,
            )
        if not accepted_a and a_t.shape[0] == 0:
            logger.debug(
                "Candidate search retained no distinct candidate after geometric filters"
            )
        elif not accepted_a:
            logger.debug("No retained candidate clears the insertion threshold")

    W, b = _as_arrays(accepted_a, accepted_b, d_dim)
    return W, b, np.array(accepted_c, dtype=np.float64) if guaranteed else None


# ---------------------------------------------------------------------------- #
# Strategy 2: finite-step acceptance
# ---------------------------------------------------------------------------- #
def finite_step(
    problem: SparseModel,
    penalty: Penalty,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    max_insert: int = 15, merge_tol: float = 1e-2,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = True,
    lbfgs_lr: float = 1e-2, lbfgs_steps: int = 200,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept atoms with a profitable finite step (Delta J(c*) < 0); return c* too."""
    d_dim = problem.input_dim

    a_t, b_t, _ = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_c: List[float] = []
    decreases: List[float] = []
    for a_i, b_i in zip(a_t, b_t):
        slope, curvature = _slope_and_curvature(problem, residual, a_i, b_i)
        if curvature < 1e-30 or abs(slope) < 1e-30:
            continue
        c_star = penalty.insertion_weight(slope, curvature)
        if c_star == 0.0:
            continue
        # The one-atom objective increment, eq. algorithm2-increment.
        dJ = (
            slope * c_star + 0.5 * curvature * c_star**2
            + penalty.scalar_penalty(c_star)
        )
        if dJ >= 0.0:
            continue
        accepted_a.append(a_i.detach())
        accepted_b.append(b_i.detach())
        accepted_c.append(c_star)
        decreases.append(-dJ)

    best = _best_indices(decreases, max_insert)
    accepted_a = [accepted_a[i] for i in best]
    accepted_b = [accepted_b[i] for i in best]
    accepted_c = [accepted_c[i] for i in best]

    if verbose:
        logger.debug(
            "Candidate search  sampled=%d  unique=%d  accepted=%d/%d  "
            "rule=adding atom reduces objective (q=%.3f)",
            N, a_t.shape[0], len(accepted_a), max_insert, problem.penalty_exponent,
        )

    W, b = _as_arrays(accepted_a, accepted_b, d_dim)
    return W, b, np.array(accepted_c, dtype=np.float64)


# ---------------------------------------------------------------------------- #
# Candidate search shared by both strategies
# ---------------------------------------------------------------------------- #
def _generate_candidates(
    problem: SparseModel,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    merge_tol: float,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]],
    lbfgs_lr: float = 1e-2,
    lbfgs_steps: int = 200,
    radius: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return distinct locally refined maximizers of the score |P(omega)| / w_p(omega).

    Homogeneous activations are searched on ``S^d``.  For nonhomogeneous
    activations, initial points are sampled inside ``radius`` and all components
    of ``omega=(a,b)`` are then optimized jointly without a constraint.  When
    optimized nonhomogeneous points outside the sampling radius are discarded
    rather than projected back.  Homogeneous activations retain the normalized
    sphere search used by Algorithm 2. Also return how many refined candidates
    were discarded outside the radius.
    """
    r_max = float(radius) if radius is not None else math.exp(FIXED_LOG_CLAMP)
    normalized_residual = _normalize_residual(residual)
    a_t, b_t = _sample_starts(problem, sample_sphere, N, r_max)
    a_t, b_t, discarded_outside = _refine_and_deduplicate(
        problem, normalized_residual, a_t, b_t,
        r_max=r_max, merge_tol=merge_tol, steps=lbfgs_steps, lr=lbfgs_lr,
    )
    a_t, b_t = _reject_support_repeats(problem, a_t, b_t, existing_atoms)
    return a_t, b_t, discarded_outside


def _normalize_residual(
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the residual for the L-BFGS search (MATLAB find_max:385)."""
    X, res_v, res_dv = residual
    res_norm = torch.sqrt(res_v.pow(2).sum() + res_dv.pow(2).sum()).clamp_min(1e-30)
    return X, res_v / res_norm, res_dv / res_norm


def _sample_starts(
    problem: SparseModel,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample N search starts.

    Homogeneous models use the sampled directions directly; nonhomogeneous
    models give each direction a log-uniform radius (ADR 0006).
    """
    a_t, b_t = sample_sphere(N)
    if not problem.use_sphere:
        lo, hi = math.log(math.exp(-3.0)), math.log(max(r_max, math.exp(-3.0) * 1.001))
        u = torch.rand(a_t.shape[0], dtype=torch.float64)
        r = torch.exp(lo + (hi - lo) * u)
        a_t = a_t * r.unsqueeze(1)
        b_t = b_t * r
    return a_t, b_t


def _refine_and_deduplicate(
    problem: SparseModel,
    normalized_residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a_t: torch.Tensor, b_t: torch.Tensor, *,
    r_max: float, merge_tol: float, steps: int, lr: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Refine the starts and remove duplicates; also count those discarded outside r_max.

    A nonhomogeneous search uses one joint solve from each random start, then
    filters and deduplicates once.  Algorithm 2 retains its repeated sphere
    refinement after merges.
    """
    use_sphere = problem.use_sphere
    discarded_outside = 0
    refinement_rounds = 5 if use_sphere else 1
    for _ in range(refinement_rounds):
        a_t, b_t = _maximize_score(problem, normalized_residual, a_t, b_t, steps=steps, lr=lr)
        if not use_sphere:
            U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
            inside = torch.linalg.vector_norm(U, dim=1) <= r_max
            discarded_outside += int((~inside).sum().item())
            a_t, b_t = a_t[inside], b_t[inside]
        n_before = a_t.shape[0]
        a_t, b_t = _remove_duplicates(a_t, b_t, use_sphere=use_sphere, merge_tol=merge_tol)
        if a_t.shape[0] == n_before:
            break
    return a_t, b_t, discarded_outside


def _maximize_score(
    problem: SparseModel,
    normalized_residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a_batch: torch.Tensor, b_batch: torch.Tensor, *,
    steps: int, lr: float, eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locally maximize the score |P(omega)| / w_p(omega) from each starting point.

    For a positively homogeneous activation the parameter is gauge-fixed to
    the sphere, so the profile is maximized over directions and the iterate
    is renormalized after every step.  For a nonhomogeneous activation the
    radial variable is a genuine shape parameter and is optimized *jointly*
    with the direction: maximizing over directions at |omega| = 1 and then
    over the radius along the winning direction is a strictly weaker search,
    because the profile does not factor into radial and directional parts,
    so the best direction at unit radius need not be the direction of a
    joint maximizer.

    The nonhomogeneous solve is unconstrained.  Its caller may subsequently
    discard a final point outside the radius used to sample its start.
    """
    use_sphere = problem.use_sphere
    d_dim = problem.input_dim
    X = normalized_residual[0]
    results_a, results_b = [], []
    for a0, b0 in zip(a_batch, b_batch):
        w = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            w_s = w / w.norm().clamp_min(eps) if use_sphere else w
            features = problem.kernel(X, w_s[:d_dim].reshape(1, -1), w_s[d_dim].reshape(1))
            profile = problem.jac_obj(*features, normalized_residual)[0]
            obj = profile.abs() / problem.moment_weight_of_atoms(w_s[:d_dim], w_s[d_dim])
            (-obj).backward()
            return -obj

        opt.step(closure)
        w_s = (w / w.norm().clamp_min(eps)).detach() if use_sphere else w.detach()
        results_a.append(w_s[:d_dim])
        results_b.append(w_s[d_dim:d_dim + 1])
    return torch.stack(results_a), torch.stack(results_b).reshape(-1)


def _remove_duplicates(
    a_cands: torch.Tensor, b_cands: torch.Tensor, *, use_sphere: bool, merge_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep each candidate unless it duplicates an earlier kept candidate."""
    n = a_cands.shape[0]
    if n <= 1:
        return a_cands, b_cands
    U = torch.cat([a_cands, b_cands.reshape(-1, 1)], dim=1)
    if use_sphere:
        unit = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        duplicate = unit @ unit.T > 1.0 - merge_tol
    else:
        duplicate = torch.cdist(U, U) <= merge_tol
    keep = torch.ones(n, dtype=torch.bool, device=U.device)
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j] and duplicate[i, j]:
                    keep[j] = False
    return a_cands[keep], b_cands[keep]


def _reject_support_repeats(
    problem: SparseModel,
    a_t: torch.Tensor, b_t: torch.Tensor,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop candidates that numerically repeat an atom of the current support.

    The fractional-power one-atom increment assumes a distinct new location.
    Reject only numerical repetitions of the current support: nearby sphere
    points are distinct atoms and must remain admissible.  In particular, do
    not reuse the much wider candidate-to-candidate deduplication tolerance.
    The k=1 ReLU--L1 baseline retains its established candidate behavior.
    Existing atoms are used only for this check; they are not search starts.
    """
    has_support = existing_atoms is not None and existing_atoms[0].shape[0] > 0
    if problem.use_sphere and problem.penalty_exponent < 1.0 and has_support and a_t.shape[0] > 0:
        W_exist, b_exist = existing_atoms
        U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
        existing_unit = U_exist / U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        distinct = torch.all(
            # 1e-8 is a numerical-repeat gap only, deliberately far tighter than
            # merge_tol, the candidate-to-candidate deduplication (ADR 0013).
            U @ existing_unit.T <= 1.0 - 1e-8,
            dim=1,
        )
        a_t, b_t = a_t[distinct], b_t[distinct]
    return a_t, b_t


# ---------------------------------------------------------------------------- #
# Candidate evaluation shared by both strategies
# ---------------------------------------------------------------------------- #
def _slope_and_curvature(
    problem: SparseModel,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a: torch.Tensor, b: torch.Tensor,
) -> Tuple[float, float]:
    """The fitting term's slope and curvature in a new atom's coefficient, at zero."""
    features = problem.kernel(residual[0], a.reshape(1, -1), b.reshape(1))
    slope = float(problem.jac_obj(*features, residual)[0])
    curvature = float(problem.hess_obj(*features)[0, 0])
    return slope, curvature


def _best_indices(gains: List[float], max_insert: int) -> List[int]:
    """Indices of the max_insert largest gains, best first; all, in order, if they fit."""
    indices = list(range(len(gains)))
    if len(gains) > max_insert:
        indices = sorted(indices, key=lambda i: gains[i], reverse=True)[:max_insert]
    return indices


def _as_arrays(
    accepted_a: List[torch.Tensor], accepted_b: List[torch.Tensor], d_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack the accepted atoms into W (n, d_dim) and b (n,), including n = 0."""
    W = np.empty((0, d_dim), dtype=np.float64)
    b = np.empty((0,), dtype=np.float64)
    if accepted_a:
        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
    return W, b
