"""Insertion strategies for the PDAP outer loop.

An insertion strategy proposes new atoms (inner weights/biases) to add to the
current support, given the data and the current residual.  Both strategies use
multistart L-BFGS on the profile P(omega), but their parameter domains and
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

# Numerical bound for random initial points and final candidate filtering.
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
    num_of_initial_pts: int,
    max_insert: int = 15, merge_tol: float = 1e-2,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = True,
    lbfgs_lr: float = 1e-2, lbfgs_steps: int = 200,
    insert_init: str = "warm_start",
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Select candidates for Algorithm 1; accepts ``|P(omega)| / w_p(omega) > alpha``.
    Candidates are ranked by their margin above the applicable threshold.
    """
    input_dim = problem.input_dim

    a, b, discarded_outside = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, num_of_initial_pts=num_of_initial_pts,
        merge_tol=merge_tol,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    guaranteed = insert_init == "guaranteed"

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    margins: List[float] = []
    accepted_c: List[float] = []
    for a_i, b_i in zip(a, b):
        slope, curvature = _slope_and_curvature(problem, residual, a_i, b_i)
        abs_slope = abs(slope)
        moment_weight = float(problem.moment_weight_of_atoms(a_i, b_i))
        threshold = penalty.alpha * moment_weight
        # Retain the physical-profile comparison, including its rounding.
        if abs_slope <= threshold or (guaranteed and curvature < 1e-30):
            continue
        accepted_a.append(a_i.detach())
        accepted_b.append(b_i.detach())
        margins.append(abs_slope / moment_weight - penalty.alpha)
        if guaranteed:
            magnitude = (abs_slope - threshold) / curvature
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
            num_of_initial_pts, a.shape[0], len(accepted_a), max_insert, rule, insert_init,
        )
        if discarded_outside:
            logger.debug(
                "Candidate search discarded %d refined candidate(s) outside the "
                "search radius",
                discarded_outside,
            )
        if not accepted_a and a.shape[0] == 0:
            logger.debug(
                "Candidate search retained no distinct candidate after geometric filters"
            )
        elif not accepted_a:
            logger.debug("No retained candidate clears the insertion threshold")

    c = np.array(accepted_c, dtype=np.float64) if guaranteed else None
    return *_as_arrays(accepted_a, accepted_b, input_dim), c


# ---------------------------------------------------------------------------- #
# Strategy 2: finite-step acceptance
# ---------------------------------------------------------------------------- #
def finite_step(
    problem: SparseModel,
    penalty: Penalty,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    num_of_initial_pts: int,
    max_insert: int = 15, merge_tol: float = 1e-2,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = True,
    lbfgs_lr: float = 1e-2, lbfgs_steps: int = 200,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept atoms with a profitable finite step (Delta J(c*) < 0); return c* too."""
    input_dim = problem.input_dim

    a, b, _ = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, num_of_initial_pts=num_of_initial_pts,
        merge_tol=merge_tol,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_c: List[float] = []
    decreases: List[float] = []
    for a_i, b_i in zip(a, b):
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
            num_of_initial_pts, a.shape[0], len(accepted_a), max_insert, problem.penalty_exponent,
        )

    return *_as_arrays(accepted_a, accepted_b, input_dim), np.array(accepted_c, dtype=np.float64)


# ---------------------------------------------------------------------------- #
# Candidate search shared by both strategies
# ---------------------------------------------------------------------------- #
def _generate_candidates(
    problem: SparseModel,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    num_of_initial_pts: int,
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
    search_radius = float(radius) if radius is not None else math.exp(FIXED_LOG_CLAMP)
    normalized_residual = _normalize_residual(residual)
    a, b = _sample_initial_pts(problem, sample_sphere, num_of_initial_pts, search_radius)
    a, b, discarded_outside = _refine_and_deduplicate(
        problem, normalized_residual, a, b,
        search_radius=search_radius, merge_tol=merge_tol, steps=lbfgs_steps, lr=lbfgs_lr,
    )
    a, b = _reject_support_repeats(problem, a, b, existing_atoms)
    return a, b, discarded_outside


def _normalize_residual(
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the residual for the L-BFGS search (MATLAB find_max:385)."""
    X, residual_value, residual_gradient = residual
    residual_norm = torch.sqrt(
        residual_value.pow(2).sum() + residual_gradient.pow(2).sum()
    ).clamp_min(1e-30)
    return X, residual_value / residual_norm, residual_gradient / residual_norm


def _sample_initial_pts(
    problem: SparseModel,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    num_of_initial_pts: int,
    search_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample num_of_initial_pts initial points.

    Homogeneous models use the sampled directions directly; nonhomogeneous
    models give each direction a log-uniform radius (ADR 0006).
    """
    a, b = sample_sphere(num_of_initial_pts)
    if not problem.use_sphere:
        log_r_min = math.log(math.exp(-3.0))
        log_r_max = math.log(max(search_radius, math.exp(-3.0) * 1.001))
        uniform = torch.rand(a.shape[0], dtype=torch.float64)
        radii = torch.exp(log_r_min + (log_r_max - log_r_min) * uniform)
        a = a * radii.unsqueeze(1)
        b = b * radii
    return a, b


def _refine_and_deduplicate(
    problem: SparseModel,
    normalized_residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a: torch.Tensor, b: torch.Tensor, *,
    search_radius: float, merge_tol: float, steps: int, lr: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Refine the initial points, drop duplicates, count those discarded outside the radius.

    A nonhomogeneous search uses one joint solve from each random initial
    point, then filters and deduplicates once.  Algorithm 2 retains its
    repeated sphere refinement after merges.
    """
    use_sphere = problem.use_sphere
    discarded_outside = 0
    refinement_rounds = 5 if use_sphere else 1
    for _ in range(refinement_rounds):
        a, b = _maximize_score(problem, normalized_residual, a, b, steps=steps, lr=lr)
        if not use_sphere:
            omega = torch.cat([a, b.reshape(-1, 1)], dim=1)
            inside = torch.linalg.vector_norm(omega, dim=1) <= search_radius
            discarded_outside += int((~inside).sum().item())
            a, b = a[inside], b[inside]
        n_before = a.shape[0]
        a, b = _remove_duplicates(a, b, use_sphere=use_sphere, merge_tol=merge_tol)
        if a.shape[0] == n_before:
            break
    return a, b, discarded_outside


def _maximize_score(
    problem: SparseModel,
    normalized_residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a: torch.Tensor, b: torch.Tensor, *,
    steps: int, lr: float, eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locally maximize the score |P(omega)| / w_p(omega) from each initial point.

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
    discard a final point outside the radius used to sample its initial point.
    """
    use_sphere = problem.use_sphere
    input_dim = problem.input_dim
    results_a, results_b = [], []
    for a0, b0 in zip(a, b):
        omega = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([omega], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            omega_s = omega / omega.norm().clamp_min(eps) if use_sphere else omega
            profile = problem.jac_obj(
                normalized_residual, omega_s[:input_dim].reshape(1, -1), omega_s[input_dim].reshape(1),
            )[0]
            score = profile.abs() / problem.moment_weight_of_atoms(
                omega_s[:input_dim], omega_s[input_dim],
            )
            (-score).backward()
            return -score

        opt.step(closure)
        omega_s = (omega / omega.norm().clamp_min(eps)).detach() if use_sphere else omega.detach()
        results_a.append(omega_s[:input_dim])
        results_b.append(omega_s[input_dim:input_dim + 1])
    return torch.stack(results_a), torch.stack(results_b).reshape(-1)


def _remove_duplicates(
    a: torch.Tensor, b: torch.Tensor, *, use_sphere: bool, merge_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep each candidate unless it duplicates an earlier kept candidate."""
    n = a.shape[0]
    if n <= 1:
        return a, b
    omega = torch.cat([a, b.reshape(-1, 1)], dim=1)
    if use_sphere:
        unit = omega / omega.norm(dim=1, keepdim=True).clamp_min(1e-12)
        duplicate = unit @ unit.T > 1.0 - merge_tol
    else:
        duplicate = torch.cdist(omega, omega) <= merge_tol
    keep = torch.ones(n, dtype=torch.bool, device=omega.device)
    for i in range(n):
        if keep[i]:
            for j in range(i + 1, n):
                if keep[j] and duplicate[i, j]:
                    keep[j] = False
    return a[keep], b[keep]


def _reject_support_repeats(
    problem: SparseModel,
    a: torch.Tensor, b: torch.Tensor,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop candidates that numerically repeat an atom of the current support.

    The fractional-power one-atom increment assumes a distinct new location.
    Reject only numerical repetitions of the current support: nearby sphere
    points are distinct atoms and must remain admissible.  In particular, do
    not reuse the much wider candidate-to-candidate deduplication tolerance.
    The k=1 ReLU--L1 baseline retains its established candidate behavior.
    Existing atoms are used only for this check; they are not initial points.
    """
    has_support = existing_atoms is not None and existing_atoms[0].shape[0] > 0
    if problem.use_sphere and problem.penalty_exponent < 1.0 and has_support and a.shape[0] > 0:
        W_exist, b_exist = existing_atoms
        support_omega = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
        support_unit = support_omega / support_omega.norm(dim=1, keepdim=True).clamp_min(1e-12)
        omega = torch.cat([a, b.reshape(-1, 1)], dim=1)
        unit = omega / omega.norm(dim=1, keepdim=True).clamp_min(1e-12)
        distinct = torch.all(
            # 1e-8 is a numerical-repeat gap only, deliberately far tighter than
            # merge_tol, the candidate-to-candidate deduplication (ADR 0013).
            unit @ support_unit.T <= 1.0 - 1e-8,
            dim=1,
        )
        a, b = a[distinct], b[distinct]
    return a, b


# ---------------------------------------------------------------------------- #
# Candidate evaluation shared by both strategies
# ---------------------------------------------------------------------------- #
def _slope_and_curvature(
    problem: SparseModel,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a: torch.Tensor, b: torch.Tensor,
) -> Tuple[float, float]:
    """The fitting term's slope and curvature in a new atom's coefficient, at zero."""
    slope = float(problem.jac_obj(residual, a.reshape(1, -1), b.reshape(1))[0])
    curvature = float(problem.hess_obj(residual[0], a.reshape(1, -1), b.reshape(1))[0, 0])
    return slope, curvature


def _best_indices(gains: List[float], max_insert: int) -> List[int]:
    """Indices of the max_insert largest gains, best first; all, in order, if they fit."""
    indices = list(range(len(gains)))
    if len(gains) > max_insert:
        indices = sorted(indices, key=lambda i: gains[i], reverse=True)[:max_insert]
    return indices


def _as_arrays(
    accepted_a: List[torch.Tensor], accepted_b: List[torch.Tensor], input_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack the accepted atoms into W (n, input_dim) and b (n,), including n = 0."""
    W = np.empty((0, input_dim), dtype=np.float64)
    b = np.empty((0,), dtype=np.float64)
    if accepted_a:
        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
    return W, b
