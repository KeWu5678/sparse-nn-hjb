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

from ..problems import Proximal, SparseProblem

# Numerical bound for random starts and final candidate filtering.
FIXED_LOG_CLAMP = 5.0

logger = logging.getLogger(__name__)

__all__ = [
    "profile_threshold",
    "finite_step",
]


def _generate_candidates(
    problem: SparseProblem,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    merge_tol: float,
    two_sided: bool,
    existing_atoms: Optional[Tuple[torch.Tensor, torch.Tensor]],
    lbfgs_lr: float = 1e-2,
    lbfgs_steps: int = 200,
    radius: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return distinct locally refined maximizers of the configured search score.

    Homogeneous activations are searched on ``S^d``.  For nonhomogeneous
    activations, initial points are sampled inside ``radius`` and all components
    of ``omega=(a,b)`` are then optimized jointly without a constraint.  When
    optimized nonhomogeneous points outside the sampling radius are discarded
    rather than projected back.  Homogeneous activations retain the normalized
    sphere search used by Algorithm 2.
    """
    use_sphere = problem.use_sphere
    d_dim = problem.input_dim

    # Normalize residual for the L-BFGS direction search (MATLAB find_max:385).
    X, res_v, res_dv = residual
    res_norm = torch.sqrt(res_v.pow(2).sum() + res_dv.pow(2).sum()).clamp_min(1e-30)
    normalized_residual = X, res_v / res_norm, res_dv / res_norm

    def maximize_batch(
        a_batch: torch.Tensor, b_batch: torch.Tensor,
        steps: int = 200, lr: float = 1e-2, eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Locally maximize the configured profile score from each starting point.

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
        results_a, results_b = [], []
        for a0, b0 in zip(a_batch, b_batch):
            w = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
            opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                w_s = w / w.norm().clamp_min(eps) if use_sphere else w
                features = problem.kernel(X, w_s[:d_dim].reshape(1, -1), w_s[d_dim].reshape(1))
                profile = problem.jac_obj(*features, normalized_residual)[0]
                score = profile.abs() if two_sided else profile
                obj = score / problem.moment_weight_of_atoms(w_s[:d_dim], w_s[d_dim])
                (-obj).backward()
                return -obj

            opt.step(closure)
            w_s = (w / w.norm().clamp_min(eps)).detach() if use_sphere else w.detach()
            results_a.append(w_s[:d_dim])
            results_b.append(w_s[d_dim:d_dim + 1])
        return torch.stack(results_a), torch.stack(results_b).reshape(-1)

    def merge(
        a_cands: torch.Tensor, b_cands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge the duplicated (close) atoms.
        """
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

    # Homogeneous models use the sampled directions directly;
    # nonhomogeneous models give each direction a log-uniform radius (ADR 0006).
    a_t, b_t = sample_sphere(N)
    existing_unit = None
    r_max = float(radius) if radius is not None else math.exp(FIXED_LOG_CLAMP)
    if not use_sphere:
        lo, hi = math.log(math.exp(-3.0)), math.log(max(r_max, math.exp(-3.0) * 1.001))
        u = torch.rand(a_t.shape[0], dtype=torch.float64)
        r = torch.exp(lo + (hi - lo) * u)
        a_t = a_t * r.unsqueeze(1)
        b_t = b_t * r
    # Existing atoms are used only to reject numerical repeats after refinement;
    # they are not added to the optimization starts.
    if use_sphere and existing_atoms is not None:
        W_exist, b_exist = existing_atoms
        if W_exist.shape[0] > 0:
            U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
            U_exist = U_exist / U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
            existing_unit = U_exist

    # A nonhomogeneous search uses one joint solve from each random start, then
    # filters and deduplicates once.  Algorithm 2 retains its repeated sphere
    # refinement after merges.
    n_after = a_t.shape[0]
    discarded_outside = 0
    refinement_rounds = 5 if use_sphere else 1
    for _ in range(refinement_rounds):
        a_t, b_t = maximize_batch(a_t, b_t, steps=lbfgs_steps, lr=lbfgs_lr)
        if not use_sphere:
            U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
            inside = torch.linalg.vector_norm(U, dim=1) <= r_max
            discarded_outside += int((~inside).sum().item())
            a_t, b_t = a_t[inside], b_t[inside]
        n_before = a_t.shape[0]
        a_t, b_t = merge(a_t, b_t)
        n_after = a_t.shape[0]
        if n_before == n_after:
            break

    # The fractional-power one-atom increment assumes a distinct new location.
    # Reject only numerical repetitions of the current support: nearby sphere
    # points are distinct atoms and must remain admissible.  In particular, do
    # not reuse the much wider candidate-to-candidate deduplication tolerance.
    # The k=1 ReLU--L1 baseline retains its established candidate behavior.
    if use_sphere and problem.penalty_exponent < 1.0 and existing_unit is not None and a_t.shape[0] > 0:
        U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        distinct = torch.all(
            # 1e-8 is a numerical-repeat gap only, deliberately far tighter than
            # merge_tol, the candidate-to-candidate deduplication (ADR 0013).
            U @ existing_unit.T <= 1.0 - 1e-8,
            dim=1,
        )
        a_t, b_t = a_t[distinct], b_t[distinct]
        n_after = a_t.shape[0]

    return a_t, b_t, n_after, discarded_outside


# ---------------------------------------------------------------------------- #
# Strategy 1: profile-threshold acceptance
# ---------------------------------------------------------------------------- #
def profile_threshold(
    problem: SparseProblem,
    proximal: Proximal,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *,
    sample_sphere: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    N: int,
    max_insert: int = 15, merge_tol: float = 1e-2,
    two_sided: bool = True,
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

    a_t, b_t, n_after, discarded_outside = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol, two_sided=two_sided,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    guaranteed = insert_init == "guaranteed"
    if guaranteed and not two_sided:
        raise ValueError(
            "insert_init='guaranteed' requires a two-sided signed coefficient"
        )

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_scores: List[float] = []
    accepted_c: List[float] = []
    for a_i, b_i in zip(a_t, b_t):
        features = problem.kernel(residual[0], a_i.reshape(1, -1), b_i.reshape(1))
        slope = float(problem.jac_obj(*features, residual)[0])
        curvature = float(problem.hess_obj(*features)[0, 0])
        v = abs(slope) if two_sided else slope
        scale = float(problem.moment_weight_of_atoms(a_i, b_i))
        threshold = proximal.alpha * scale
        # Retain the physical-profile comparison, including its rounding.
        if v <= threshold or (guaranteed and curvature < 1e-30):
            continue
        accepted_a.append(a_i.detach())
        accepted_b.append(b_i.detach())
        accepted_scores.append(v / scale - proximal.alpha)
        if guaranteed:
            magnitude = (v - threshold) / curvature
            accepted_c.append(-magnitude if slope > 0 else magnitude)

    if len(accepted_scores) > max_insert:
        order = sorted(range(len(accepted_scores)), key=lambda i: accepted_scores[i], reverse=True)[:max_insert]
        accepted_a = [accepted_a[i] for i in order]
        accepted_b = [accepted_b[i] for i in order]
        if guaranteed:
            accepted_c = [accepted_c[i] for i in order]

    if verbose:
        rule = (
            "|P|/w_p above alpha*L_phi" if problem.normalized
            else "|P| above alpha"
        )
        logger.debug(
            "Candidate search  sampled=%d  unique=%d  accepted=%d/%d  "
            "rule=%s (init=%s)",
            N, n_after, len(accepted_a), max_insert, rule, insert_init,
        )
        if discarded_outside:
            logger.debug(
                "Candidate search discarded %d refined candidate(s) outside the "
                "search radius",
                discarded_outside,
            )
        if not accepted_a and n_after == 0:
            logger.debug(
                "Candidate search retained no distinct candidate after geometric filters"
            )
        elif not accepted_a:
            logger.debug("No retained candidate clears the insertion threshold")

    if len(accepted_a) == 0:
        empty_c = np.empty((0,), dtype=np.float64) if guaranteed else None
        return (
            np.empty((0, d_dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            empty_c,
        )
    return (
        torch.stack(accepted_a, dim=0).detach().cpu().numpy(),
        torch.stack(accepted_b, dim=0).detach().cpu().numpy(),
        np.array(accepted_c, dtype=np.float64) if guaranteed else None,
    )


# ---------------------------------------------------------------------------- #
# Strategy 2: finite-step acceptance
# ---------------------------------------------------------------------------- #
def finite_step(
    problem: SparseProblem,
    proximal: Proximal,
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

    a_t, b_t, n_after, _discarded_outside = _generate_candidates(
        problem, residual, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol, two_sided=True,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_c: List[float] = []
    accepted_dJ: List[float] = []
    for a_i, b_i in zip(a_t, b_t):
        features = problem.kernel(residual[0], a_i.reshape(1, -1), b_i.reshape(1))
        slope = float(problem.jac_obj(*features, residual)[0])
        curvature = float(problem.hess_obj(*features)[0, 0])
        if curvature < 1e-30 or abs(slope) < 1e-30:
            continue
        magnitude = proximal.fractional_prox(abs(slope) / curvature, proximal.alpha / curvature)
        if magnitude <= 0.0:
            continue
        c_star = magnitude if slope < 0 else -magnitude
        # The insertion increment uses the exact power, without the tiny
        # magnitude floor needed when differentiating the full penalty.
        dJ = (
            slope * c_star + 0.5 * curvature * c_star**2
            + proximal.alpha * abs(c_star)**proximal.penalty_exponent
        )
        if dJ >= 0.0:
            continue
        accepted_a.append(a_i.detach())
        accepted_b.append(b_i.detach())
        accepted_c.append(c_star)
        accepted_dJ.append(dJ)

    if len(accepted_dJ) > max_insert:
        order = sorted(range(len(accepted_dJ)), key=lambda i: accepted_dJ[i])[:max_insert]
        accepted_a = [accepted_a[i] for i in order]
        accepted_b = [accepted_b[i] for i in order]
        accepted_c = [accepted_c[i] for i in order]

    if verbose:
        logger.debug(
            "Candidate search  sampled=%d  unique=%d  accepted=%d/%d  "
            "rule=adding atom reduces objective (q=%.3f)",
            N, n_after, len(accepted_a), max_insert, proximal.penalty_exponent,
        )

    if len(accepted_a) == 0:
        return (
            np.empty((0, d_dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        torch.stack(accepted_a, dim=0).detach().cpu().numpy(),
        torch.stack(accepted_b, dim=0).detach().cpu().numpy(),
        np.array(accepted_c, dtype=np.float64),
    )
