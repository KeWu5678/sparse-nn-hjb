"""Insertion strategies for the PDAP outer loop.

An insertion strategy proposes new atoms (inner weights/biases) to add to the
current support, given the data and the current residual.  Both strategies share
the same candidate generation (sample S^d, L-BFGS-maximize the dual profile,
iterative cosine-merge dedup, optional per-direction rescale); they differ only
in the *acceptance* test:

  * ``profile_threshold`` — accept candidates whose dual profile exceeds ``alpha``.
    Signed networks use the two-sided test ``|p|>alpha``; convex (semiconcave)
    models use the one-sided ``p>alpha`` (convex atoms carry nonnegative mass).
  * ``finite_step`` — accept candidates with a profitable finite step, i.e. where
    min_c Delta J(c; omega) < 0 (see :func:`solve_insertion_weight`); returns the
    optimal outer weight c* alongside each atom.  Used for the q<1 penalty.

The strategies are pure functions: the caller supplies the precomputed residual
(so the "zero network" first iteration is just residual = -target) and the model
configuration.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

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
    """Dual profile p_t(omega) = w1/Kx <S,res_v> + w2/Kx <dS,res_dv> (abs if two_sided)."""
    neuron_v, neuron_dv = _neuron_value_grad(X, a, b, activation, power)
    val_part = (neuron_v * res_v).sum() / Kx
    grad_part = (neuron_dv * res_dv).sum() / Kx
    signed = w1 * val_part + w2 * grad_part
    return torch.abs(signed) if two_sided else signed


# ---------------------------------------------------------------------------- #
# Shared candidate generation: sample -> (maximize -> merge) x5 -> rescale
# ---------------------------------------------------------------------------- #
def _generate_candidates(
    X, residual_v, residual_dv, *,
    activation, power, loss_weights, sample_sphere, N,
    merge_tol, two_sided, use_sphere, existing_atoms,
    lbfgs_lr=1e-2, lbfgs_steps=200, moment_beta=0.0, moment_order=2.0,
    normalized=False, radius=None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (a_t, b_t, n_after_merge): distinct dual-profile maximisers on S^d."""
    K, d_dim = X.shape
    Kx = K  # M, the sample count: the empirical fidelity is 1/(2M) * sum_m
    w1, w2 = loss_weights

    # Normalize residual for the L-BFGS direction search (MATLAB find_max:385).
    res_norm = torch.sqrt(residual_v.pow(2).sum() + residual_dv.pow(2).sum()).clamp_min(1e-30)
    res_v_n = residual_v / res_norm
    res_dv_n = residual_dv / res_norm

    def maximize_batch(a_batch, b_batch, steps=200, lr=1e-2, eps=1e-12):
        results_a, results_b = [], []
        for a0, b0 in zip(a_batch, b_batch):
            w = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
            opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                w_s = w / w.norm().clamp_min(eps)
                obj = _profile_value(X, w_s[:d_dim], w_s[d_dim], activation, power,
                                     w1, w2, Kx, res_v_n, res_dv_n, two_sided)
                (-obj).backward()
                return -obj

            opt.step(closure)
            w_s = (w / w.norm().clamp_min(eps)).detach()
            results_a.append(w_s[:d_dim])
            results_b.append(w_s[d_dim:d_dim + 1])
        return torch.stack(results_a), torch.stack(results_b).reshape(-1)

    def merge(a_cands, b_cands):
        n = a_cands.shape[0]
        if n <= 1:
            return a_cands, b_cands
        U = torch.cat([a_cands, b_cands.reshape(-1, 1)], dim=1)
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = U @ U.T
        keep = torch.ones(n, dtype=torch.bool)
        for i in range(n):
            if keep[i]:
                for j in range(i + 1, n):
                    if keep[j] and sim[i, j] > 1.0 - merge_tol:
                        keep[j] = False
        return a_cands[keep], b_cands[keep]

    # Step 1: candidates = random samples + existing support.
    a_t, b_t = sample_sphere(N)
    if existing_atoms is not None:
        W_exist, b_exist = existing_atoms
        if W_exist.shape[0] > 0:
            U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
            U_exist = U_exist / U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
            n_exist = U_exist.shape[0]
            if n_exist > N // 2:
                U_exist = U_exist[torch.randperm(n_exist)[:N // 2]]
            a_t = torch.cat([a_t, U_exist[:, :d_dim]], dim=0)
            b_t = torch.cat([b_t, U_exist[:, d_dim]], dim=0)

    # Step 2: iterative optimize + merge (MATLAB find_max:390-414).
    n_after = a_t.shape[0]
    for _ in range(5):
        a_t, b_t = maximize_batch(a_t, b_t, steps=lbfgs_steps, lr=lbfgs_lr)
        n_before = a_t.shape[0]
        a_t, b_t = merge(a_t, b_t)
        n_after = a_t.shape[0]
        if n_before == n_after:
            break

    # Step 2.5: non-sphere activations also optimise a per-direction scale r>0.
    # What the scale search maximises depends on the objective:
    #   * plain              -- the raw profile |P(r*d)|, confined only by the clamp.
    #   * additive moment    -- the net margin |P(r*d)| - beta*w_p(r*d), so a real
    #                           fidelity-vs-moment tradeoff confines r.  The profile
    #                           here is against the *normalized* residual, so the
    #                           moment cost is divided by res_norm to share units
    #                           with it; argmax is unchanged in raw units.
    #   * normalized moment  -- the normalized profile |P_p(r*d)| = |P|/w_p, which is
    #                           the quantity the revised paper's insertion condition
    #                           tests and is confined by p > s_1 on its own.
    # ``radius`` caps the search; None keeps the historical exp(5) clamp
    # (docs/adr/0006), and the theorem radius enters as min(R(mu), exp(5)).
    if not use_sphere:
        moment_scale = float(moment_beta) / float(res_norm) if moment_beta > 0.0 else 0.0
        log_hi = 5.0 if radius is None else float(np.log(max(float(radius), np.exp(-3.0))))
        scaled_a, scaled_b = [], []
        for a_hat, b_hat in zip(a_t, b_t):
            s = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
            opt_s = torch.optim.LBFGS([s], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")
            a_h, b_h = a_hat.detach(), b_hat.detach()

            def closure_s():
                opt_s.zero_grad()
                r = torch.exp(s.clamp(-3, log_hi))
                obj = _profile_value(X, r * a_h, r * b_h, activation, power,
                                     w1, w2, Kx, res_v_n, res_dv_n, two_sided)
                if normalized:
                    obj = obj / moment_weight(r * a_h, r * b_h, moment_order)
                elif moment_scale > 0.0:
                    obj = obj - moment_scale * moment_weight(r * a_h, r * b_h, moment_order)
                (-obj).backward()
                return -obj

            opt_s.step(closure_s)
            best_r = torch.exp(s.clamp(-3, log_hi)).detach()
            scaled_a.append((best_r * a_hat).detach())
            scaled_b.append((best_r * b_hat).detach())
        a_t = torch.stack(scaled_a)
        b_t = torch.stack(scaled_b).reshape(-1)

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
    """Accept atoms whose dual profile clears the insertion threshold.

    Three acceptance rules, selected by the objective in force:

      * plain (``moment_beta = 0``, not normalized) -- the classical
        ``|P(omega)| > alpha``.
      * additive moment (``moment_beta > 0``) -- the preserved comparator's
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
# Strategy 2: finite-step acceptance (q < 1)
# ---------------------------------------------------------------------------- #
def solve_insertion_weight(
    p_omega: float, S_sq: float, alpha: float, q: float,
    newton_tol: float = 1e-12, max_iter: int = 50,
) -> Optional[Tuple[float, float]]:
    """Solve min_c Delta J(c; omega) and return (c*, Delta J(c*)), or None.

    Delta J(c; omega) = c * p_omega + (1/2) c^2 S_sq + (alpha/q) |c|^q.
    For q >= 1 this reduces to the classical criterion |p| > alpha.
    """
    if S_sq < 1e-30:
        return None
    abs_p = abs(p_omega)
    if abs_p < 1e-30:
        return None

    if q >= 1.0:
        if abs_p <= alpha:
            return None
        c_opt = (abs_p - alpha) / S_sq
        dJ = -abs_p * c_opt + 0.5 * S_sq * c_opt ** 2 + alpha * c_opt
        sign = 1.0 if p_omega < 0 else -1.0
        return (sign * c_opt, dJ)

    # q < 1: finite-step.  h(c) = S^2 c + alpha c^{q-1} has a min at c_star.
    c_star = (alpha * (1.0 - q) / S_sq) ** (1.0 / (2.0 - q))
    h_min = S_sq * c_star + alpha * c_star ** (q - 1.0)
    if abs_p < h_min:
        return None

    # Newton on F(c) = S^2 c + alpha c^{q-1} - |p| = 0, right branch c > c_star.
    c = max(abs_p / S_sq, 2.0 * c_star)
    for _ in range(max_iter):
        F = S_sq * c + alpha * c ** (q - 1.0) - abs_p
        dF = S_sq + alpha * (q - 1.0) * c ** (q - 2.0)
        if abs(dF) < 1e-30:
            break
        c_new = c - F / dF
        if c_new <= c_star:
            c_new = 0.5 * (c + c_star)
        c = c_new
        if abs(F) < newton_tol * (abs_p + 1e-30):
            break

    dJ = -abs_p * c + 0.5 * S_sq * c ** 2 + (alpha / q) * c ** q
    if dJ >= 0:
        return None
    sign = 1.0 if p_omega < 0 else -1.0
    return (sign * c, dJ)


def finite_step(
    X, residual_v, residual_dv, *,
    activation, power, loss_weights, alpha, sample_sphere, N,
    max_insert=15, merge_tol=1e-2, use_sphere=True,
    existing_atoms=None, verbose=True,
    lbfgs_lr=1e-2, lbfgs_steps=200, newton_tol=1e-12, newton_max_iter=50,
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
            result = solve_insertion_weight(
                p_omega, S_sq, alpha, q,
                newton_tol=newton_tol, max_iter=newton_max_iter,
            )
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
