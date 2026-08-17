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
    min_c Delta J(c; omega) < 0 (see :func:`solve_insertion_weight`); returns the
    optimal outer weight c* alongside each atom.  Used for the q<1 penalty.

The strategies are pure functions: the caller supplies the precomputed residual
(so the "zero network" first iteration is just residual = -target) and the model
configuration.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from ..SSN.prox import power_prox
from .moment import moment_weight

logger = logging.getLogger(__name__)

__all__ = ["profile_threshold", "finite_step", "solve_insertion_weight"]


# ---------------------------------------------------------------------------- #
# Shared dual-profile evaluation
# ---------------------------------------------------------------------------- #
def _neuron_value_grad(
    X: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor], power: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """sigma(x.a+b)^p and its input-gradient, both differentiable w.r.t. (a, b)."""
    X_cand = X.detach().clone().requires_grad_(True)
    pre = X_cand @ a.reshape(-1) + b.reshape(())
    act = activation(pre).reshape(-1, 1)
    neuron_v = act ** power
    neuron_dv = torch.autograd.grad(
        outputs=neuron_v.sum(), inputs=X_cand,
        create_graph=True, retain_graph=True,
    )[0]
    return neuron_v, neuron_dv


def _profile_value(
    X, a, b, activation, power, w1, w2, Kx, res_v, res_dv, two_sided: bool,
) -> torch.Tensor:
    """Empirical fidelity derivative P_mu^M(omega), absolute when two-sided."""
    neuron_v, neuron_dv = _neuron_value_grad(X, a, b, activation, power)
    val_part = (neuron_v * res_v).sum() / Kx
    grad_part = (neuron_dv * res_dv).sum() / Kx
    signed = w1 * val_part + w2 * grad_part
    return torch.abs(signed) if two_sided else signed


# ---------------------------------------------------------------------------- #
# Shared candidate generation
# ---------------------------------------------------------------------------- #
def _generate_candidates(
    X, residual_v, residual_dv, *,
    activation, power, loss_weights, sample_sphere, N,
    merge_tol, two_sided, use_sphere, existing_atoms,
    lbfgs_lr=1e-2, lbfgs_steps=200, moment_beta=0.0, moment_order=2.0,
    normalized=False, radius=None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return distinct locally refined maximizers of the configured search score.

    Homogeneous activations are searched on ``S^d``.  For nonhomogeneous
    activations, initial points are sampled inside ``radius`` and all components
    of ``omega=(a,b)`` are then optimized jointly without a constraint.  When
    optimized nonhomogeneous points outside the sampling radius are discarded
    rather than projected back.  Homogeneous activations retain the normalized
    sphere search used by Algorithm 2.
    """
    K, d_dim = X.shape
    Kx = K  # M, the sample count: the empirical fidelity is 1/(2M) * sum_m
    w1, w2 = loss_weights

    # Normalize residual for the L-BFGS direction search (MATLAB find_max:385).
    res_norm = torch.sqrt(residual_v.pow(2).sum() + residual_dv.pow(2).sum()).clamp_min(1e-30)
    res_v_n = residual_v / res_norm
    res_dv_n = residual_dv / res_norm

    def maximize_batch(a_batch, b_batch, steps=200, lr=1e-2, eps=1e-12):
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
                joint = not use_sphere
                w_s = w if joint else w / w.norm().clamp_min(eps)
                obj = _profile_value(X, w_s[:d_dim], w_s[d_dim], activation, power,
                                     w1, w2, Kx, res_v_n, res_dv_n, two_sided)
                if joint and normalized:
                    obj = obj / moment_weight(w_s[:d_dim], w_s[d_dim], moment_order)
                elif joint and moment_beta > 0.0:
                    # Search the same margin used by the additive-moment
                    # acceptance test.  The profile uses the normalized residual,
                    # so beta is put in those units as well.
                    obj = obj - (moment_beta / res_norm) * moment_weight(
                        w_s[:d_dim], w_s[d_dim], moment_order
                    )
                (-obj).backward()
                return -obj

            opt.step(closure)
            joint = not use_sphere
            w_s = w.detach() if joint else (w / w.norm().clamp_min(eps)).detach()
            results_a.append(w_s[:d_dim])
            results_b.append(w_s[d_dim:d_dim + 1])
        return torch.stack(results_a), torch.stack(results_b).reshape(-1)

    def merge(a_cands, b_cands):
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

    # Step 1: random starts.  Only the homogeneous sphere search injects the
    # existing support.
    #
    # The homogeneous search is posed on the sphere, so its starting points live
    # there.  The nonhomogeneous search ranges over the ball of radius
    # ``radius``, and its starting points carry a radius too -- starting every
    # trajectory at |omega| = 1 would bias the search toward that shell.
    #
    # The radius is drawn log-uniformly, not uniformly in volume.  Uniform in
    # the ball puts r = R * U^(1/(d+1)), whose median already sits at ~0.79 R;
    # with R = e^5 nearly every start lands in the far field, where the
    # normalized profile |P|/w_p is flat and vanishing, and the local solve
    # cannot climb back.  Scale, not volume, is the meaningful spread for a
    # shape parameter that ranges over decades.
    a_t, b_t = sample_sphere(N)
    existing_unit = None
    if not use_sphere:
        r_max = float(radius) if radius is not None else math.exp(5.0)
        lo, hi = math.log(math.exp(-3.0)), math.log(max(r_max, math.exp(-3.0) * 1.001))
        u = torch.rand(a_t.shape[0], dtype=torch.float64)
        r = torch.exp(lo + (hi - lo) * u)
        a_t = a_t * r.unsqueeze(1)
        b_t = b_t * r
    if use_sphere and existing_atoms is not None:
        W_exist, b_exist = existing_atoms
        if W_exist.shape[0] > 0:
            U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
            U_exist = U_exist / U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
            existing_unit = U_exist
            n_exist = U_exist.shape[0]
            if n_exist > N // 2:
                U_exist = U_exist[torch.randperm(n_exist)[:N // 2]]
            a_t = torch.cat([a_t, U_exist[:, :d_dim]], dim=0)
            b_t = torch.cat([b_t, U_exist[:, d_dim]], dim=0)

    # A nonhomogeneous search uses one joint solve from each random start, then
    # filters and deduplicates once.  Algorithm 2 retains its repeated sphere
    # refinement after merges.
    n_after = a_t.shape[0]
    refinement_rounds = 5 if use_sphere else 1
    for _ in range(refinement_rounds):
        a_t, b_t = maximize_batch(a_t, b_t, steps=lbfgs_steps, lr=lbfgs_lr)
        if not use_sphere:
            U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
            r_max = float(radius) if radius is not None else math.exp(5.0)
            inside = torch.linalg.vector_norm(U, dim=1) <= r_max
            a_t, b_t = a_t[inside], b_t[inside]
        n_before = a_t.shape[0]
        a_t, b_t = merge(a_t, b_t)
        n_after = a_t.shape[0]
        if n_before == n_after:
            break

    # Existing atoms are useful L-BFGS starts for the fractional-power method,
    # but its one-atom increment assumes a distinct new location.  Drop final
    # k>1 candidates that return to the current support.  The k=1 ReLU--L1
    # baseline intentionally retains its established candidate behavior.
    if use_sphere and power > 1.0 and existing_unit is not None and a_t.shape[0] > 0:
        U = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        distinct = torch.all(U @ existing_unit.T <= 1.0 - merge_tol, dim=1)
        a_t, b_t = a_t[distinct], b_t[distinct]
        n_after = a_t.shape[0]

    return a_t, b_t, n_after


# ---------------------------------------------------------------------------- #
# Strategy 1: profile-threshold acceptance
# ---------------------------------------------------------------------------- #
def profile_threshold(
    X, residual_v, residual_dv, *,
    activation, power, loss_weights, alpha, sample_sphere, N,
    max_insert=15, merge_tol=1e-2, two_sided=True, use_sphere=True,
    existing_atoms=None, verbose=True,
    lbfgs_lr=1e-2, lbfgs_steps=200, moment_beta=0.0, moment_order=2.0,
    normalized=False, insert_init="warm_start", radius=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Accept atoms whose derivative magnitude clears the insertion threshold.

    Three acceptance rules, selected by the objective in force:

      * plain (``moment_beta = 0``, not normalized) -- the classical
        ``|P(omega)| > alpha``.
      * additive moment (``moment_beta > 0``) -- the optional rule
        ``|P(omega)| > alpha + beta*w_p(omega)``, so a distant candidate must clear
        a higher bar.
      * normalized (``normalized=True``) -- the revised paper's
        ``|P_p(omega)| > alpha*L_phi`` with ``P_p = P/w_p``.  ``L_phi = phi'(0+) = 1``
        for the whole log family, so the threshold is just ``alpha``.

    Candidates are ranked by their margin above the threshold, which in the
    normalized case is the certificate violation
    ``Delta(mu,omega) = max{|P_p(omega)| - alpha*L_phi, 0}``.

    Returns ``(W, b, c)``; ``c`` is ``None`` unless ``insert_init="guaranteed"``,
    in which case it carries the theorem's per-atom coefficient.
    """
    K, d_dim = X.shape
    Kx = K  # M, the sample count (empirical fidelity is 1/(2M) * sum_m)
    w1, w2 = loss_weights

    a_t, b_t, n_after = _generate_candidates(
        X, residual_v, residual_dv, activation=activation, power=power,
        loss_weights=loss_weights, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol, two_sided=two_sided, use_sphere=use_sphere,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        moment_beta=moment_beta, moment_order=moment_order,
        normalized=normalized, radius=radius,
    )

    guaranteed = insert_init == "guaranteed"
    if guaranteed and not two_sided:
        raise ValueError(
            "insert_init='guaranteed' is the signed theorem step; the one-sided "
            "(semiconcave) convention has no counterpart in the paper"
        )

    res_v_flat = residual_v.reshape(-1)
    res_dv_flat = residual_dv.reshape(-1)

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_scores: List[float] = []
    accepted_c: List[float] = []
    with torch.enable_grad():
        for a_i, b_i in zip(a_t, b_t):
            neuron_v, neuron_dv = _neuron_value_grad(X, a_i, b_i, activation, power)
            S_val = neuron_v.detach().reshape(-1)
            S_grad = neuron_dv.detach().reshape(-1)
            # P(omega): the Gateaux derivative of the fidelity in the direction
            # delta_omega, in unnormalized (physical-coefficient) units.
            p_signed = float(
                (w1 / Kx) * S_val.dot(res_v_flat) + (w2 / Kx) * S_grad.dot(res_dv_flat)
            )
            v = abs(p_signed) if two_sided else p_signed
            w_p = float(moment_weight(a_i, b_i, moment_order))

            # The acceptance threshold, expressed on the unnormalized profile so the
            # three rules share one comparison.  Normalized: |P|/w_p > alpha*L_phi
            # with L_phi = phi'(0+) = 1.  Additive: |P| > alpha + beta*w_p.
            if normalized:
                threshold = alpha * w_p
                score = v / w_p - alpha          # Delta(mu, omega)
            else:
                threshold = alpha + (moment_beta * w_p if moment_beta > 0.0 else 0.0)
                score = v - threshold
            if v <= threshold:
                continue

            accepted_a.append(a_i.detach())
            accepted_b.append(b_i.detach())
            accepted_scores.append(score)
            if guaranteed:
                # Exact minimizer of the increment bound
                #   c*P + c^2*||K||^2/2 + (threshold)|c|,
                # i.e. the theorem's step with the per-neuron curvature
                # A_p = ||K_p||^2 in place of the uniform B_p^2.
                S_sq = float(
                    (w1 / Kx) * S_val.dot(S_val) + (w2 / Kx) * S_grad.dot(S_grad)
                )
                if S_sq < 1e-30:
                    accepted_a.pop(); accepted_b.pop(); accepted_scores.pop()
                    continue
                magnitude = (v - threshold) / S_sq
                accepted_c.append(-magnitude if p_signed > 0 else magnitude)

    if len(accepted_scores) > max_insert:
        order = sorted(range(len(accepted_scores)), key=lambda i: accepted_scores[i], reverse=True)[:max_insert]
        accepted_a = [accepted_a[i] for i in order]
        accepted_b = [accepted_b[i] for i in order]
        if guaranteed:
            accepted_c = [accepted_c[i] for i in order]

    if verbose:
        rule = (
            "|P|/w_p above alpha*L_phi" if normalized
            else "|P| above alpha+beta*w_p"
        )
        logger.debug(
            "Candidate search  sampled=%d  unique=%d  accepted=%d/%d  "
            "rule=%s (alpha=%.2e, moment_beta=%.2e, init=%s)",
            N, n_after, len(accepted_a), max_insert, rule, alpha, moment_beta, insert_init,
        )

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
def solve_insertion_weight(
    p_omega: float, S_sq: float, alpha: float, q: float,
) -> Optional[Tuple[float, float]]:
    """Minimize the actual one-atom objective increment.

    The increment is ``c*p_omega + S_sq*c^2/2 + alpha*|c|^q``.  After choosing
    the sign opposite ``p_omega``, its magnitude is the global proximal point
    at input ``|p_omega|/S_sq`` with proximal scale ``alpha/S_sq``.
    """
    if S_sq < 1e-30:
        return None
    abs_p = abs(p_omega)
    if abs_p < 1e-30:
        return None

    prox_input = torch.tensor([abs_p / S_sq], dtype=torch.float64)
    magnitude = float(power_prox(prox_input, alpha / S_sq, q=q)[0])
    if magnitude <= 0.0:
        return None

    dJ = -abs_p * magnitude + 0.5 * S_sq * magnitude ** 2 + alpha * magnitude ** q
    if dJ >= 0.0:
        return None
    sign = 1.0 if p_omega < 0 else -1.0
    return (sign * magnitude, dJ)


def finite_step(
    X, residual_v, residual_dv, *,
    activation, power, loss_weights, alpha, sample_sphere, N,
    max_insert=15, merge_tol=1e-2, use_sphere=True,
    existing_atoms=None, verbose=True,
    lbfgs_lr=1e-2, lbfgs_steps=200,
    radius=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept atoms with a profitable finite step (Delta J(c*) < 0); return c* too."""
    K, d_dim = X.shape
    Kx = K  # M, the sample count (empirical fidelity is 1/(2M) * sum_m)
    w1, w2 = loss_weights
    q = 2.0 / (power + 1.0)

    a_t, b_t, n_after = _generate_candidates(
        X, residual_v, residual_dv, activation=activation, power=power,
        loss_weights=loss_weights, sample_sphere=sample_sphere, N=N,
        merge_tol=merge_tol, two_sided=True, use_sphere=use_sphere,
        existing_atoms=existing_atoms, lbfgs_lr=lbfgs_lr, lbfgs_steps=lbfgs_steps,
        radius=radius,
    )

    res_v_flat = residual_v.reshape(-1)
    res_dv_flat = residual_dv.reshape(-1)

    accepted_a: List[torch.Tensor] = []
    accepted_b: List[torch.Tensor] = []
    accepted_c: List[float] = []
    accepted_dJ: List[float] = []
    with torch.enable_grad():
        for a_i, b_i in zip(a_t, b_t):
            neuron_v, neuron_dv = _neuron_value_grad(X, a_i, b_i, activation, power)
            S_val = neuron_v.detach().reshape(-1)
            S_grad = neuron_dv.detach().reshape(-1)
            p_omega = float((w1 / Kx) * S_val.dot(res_v_flat) + (w2 / Kx) * S_grad.dot(res_dv_flat))
            S_sq = float((w1 / Kx) * S_val.dot(S_val) + (w2 / Kx) * S_grad.dot(S_grad))
            result = solve_insertion_weight(p_omega, S_sq, alpha, q)
            if result is not None:
                c_star, dJ = result
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
            "rule=adding atom reduces objective (alpha=%.2e, q=%.3f)",
            N, n_after, len(accepted_a), max_insert, alpha, q,
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
