"""Tests for the parameter-moment regularizer axis (``beta * Psi_p``).

Covers the pure primitives (``moment_weight`` / ``moment_penalty``), the config
schema defaults, the ``PDAP`` guardrail that confines the axis to its valid
regime, and the SSN integration via an analytic weighted-L1 fixed point.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.config.schema import ExperimentConfig, ModelConfig
from src.PDAP import PDAP
from src.PDAP.moment import moment_penalty, moment_weight


# --------------------------------------------------------------------------- #
# Pure primitives
# --------------------------------------------------------------------------- #
def test_moment_weight_p2_is_one_plus_squared_norm() -> None:
    W = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([0.0, 2.0], dtype=torch.float64)
    # |omega|^2 = |a|^2 + b^2 -> [25, 4]; w_2 = 1 + that.
    got = moment_weight(W, b, p=2.0)
    assert torch.allclose(got, torch.tensor([26.0, 5.0], dtype=torch.float64))


def test_moment_weight_general_p() -> None:
    W = torch.tensor([[3.0, 4.0]], dtype=torch.float64)  # |a| = 5
    b = torch.tensor([12.0], dtype=torch.float64)        # |omega| = 13
    got = moment_weight(W, b, p=3.0)
    assert math.isclose(float(got.item()), 1.0 + 13.0**3, rel_tol=1e-12)


def test_moment_weight_single_atom_scalar_bias() -> None:
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    got = moment_weight(a, torch.tensor(0.0, dtype=torch.float64), p=2.0)
    assert math.isclose(float(got), 2.0, rel_tol=1e-12)


def test_moment_weight_is_differentiable_in_scale() -> None:
    # The insertion scale search backprops through w_p; check the gradient.
    s = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0], dtype=torch.float64)
    w = moment_weight(s * a, s * b, p=2.0)  # = 1 + s^2
    w.backward()
    assert math.isclose(float(s.grad), 2.0 * 0.5, rel_tol=1e-9)  # d/ds (1+s^2) = 2s


def test_moment_penalty_value_and_empty() -> None:
    c = torch.tensor([2.0, -3.0], dtype=torch.float64)
    w = torch.tensor([1.5, 2.0], dtype=torch.float64)
    # beta * (1.5*2 + 2*3) = 0.1 * 9 = 0.9
    assert math.isclose(float(moment_penalty(c, w, beta=0.1)), 0.9, rel_tol=1e-12)
    assert float(moment_penalty(torch.zeros(0), torch.zeros(0), beta=1.0)) == 0.0


def test_moment_penalty_beta_zero_is_zero() -> None:
    c = torch.tensor([5.0, -7.0], dtype=torch.float64)
    w = torch.tensor([3.0, 9.0], dtype=torch.float64)
    assert float(moment_penalty(c, w, beta=0.0)) == 0.0


# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #
def test_schema_moment_defaults_off() -> None:
    m = ModelConfig()
    assert m.moment_beta == 0.0
    assert m.moment_order == 2.0


# --------------------------------------------------------------------------- #
# PDAP guardrail: moment axis is confined to signed + non-sphere + q=1 + profile
# --------------------------------------------------------------------------- #
def _cfg(**model_kwargs) -> ExperimentConfig:
    return ExperimentConfig(model=ModelConfig(**model_kwargs))


def test_guard_allows_valid_moment_config() -> None:
    # softplus is non-sphere; power=1 (q=1); signed + profile are defaults.
    PDAP(_cfg(activation="softplus", power=1.0, insertion="profile", moment_beta=1e-3))


def test_guard_off_by_default_allows_sphere() -> None:
    # beta=0 (default) must not trip the guard even for a sphere activation.
    PDAP(_cfg(activation="relu", power=1.0))


def test_guard_rejects_sphere_activation() -> None:
    with pytest.raises(ValueError, match="non-sphere"):
        PDAP(_cfg(activation="relu", power=1.0, moment_beta=1e-3))


def test_guard_rejects_nonunit_power() -> None:
    with pytest.raises(ValueError, match="power"):
        PDAP(_cfg(activation="softplus", power=2.0, moment_beta=1e-3))


def test_guard_rejects_finite_step_insertion() -> None:
    with pytest.raises(ValueError, match="profile"):
        PDAP(_cfg(activation="softplus", power=1.0, insertion="finite_step", moment_beta=1e-3))


def test_guard_rejects_semiconcave_kind() -> None:
    with pytest.raises(ValueError, match="signed"):
        PDAP(_cfg(kind="semiconcave", activation="softplus", power=1.0, moment_beta=1e-3))


# --------------------------------------------------------------------------- #
# SSN integration: analytic weighted-L1 fixed point.
#
# For the scalar problem  min_u (1/2)(u - ref)^2 + alpha|u| + kappa|u|,
# the minimizer is soft-thresholding at (alpha + kappa):
#     u* = sign(ref) * max(|ref| - (alpha + kappa), 0).
# The moment term must shift the effective threshold by exactly kappa.
# --------------------------------------------------------------------------- #
def _run_scalar_ssn(ref: float, alpha: float, kappa: float, steps: int = 8) -> float:
    from src.SSN import SSN

    theta = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    moment_vec = None if kappa == 0.0 else torch.tensor([kappa], dtype=torch.float64)
    opt = SSN(
        [theta], alpha=alpha, gamma=0.0, th=0.5, lr=1.0, power=1.0,
        method="levenberg_marquardt", moment_vec=moment_vec,
    )
    opt.data_hessian = torch.tensor([[1.0]], dtype=torch.float64)
    ref_t = torch.tensor([ref], dtype=torch.float64)

    def closure():
        opt.zero_grad()
        data = 0.5 * (theta - ref_t).pow(2).sum()
        pen = alpha * theta.abs().sum()
        if moment_vec is not None:
            pen = pen + (moment_vec * theta.abs()).sum()
        return data + pen

    for _ in range(steps):
        opt.step(closure)
    return float(theta.detach().item())


def test_ssn_moment_shifts_soft_threshold() -> None:
    ref, alpha, kappa = 1.0, 0.1, 0.15
    got = _run_scalar_ssn(ref, alpha, kappa)
    expected = math.copysign(max(abs(ref) - (alpha + kappa), 0.0), ref)  # 0.75
    assert math.isclose(got, expected, abs_tol=1e-8)


def test_ssn_without_moment_recovers_plain_soft_threshold() -> None:
    ref, alpha = 1.0, 0.1
    got = _run_scalar_ssn(ref, alpha, kappa=0.0)
    assert math.isclose(got, abs(ref) - alpha, abs_tol=1e-8)  # 0.9


def test_ssn_moment_can_threshold_to_zero() -> None:
    # threshold alpha+kappa = 0.5 > |ref| = 0.3  ->  u* = 0.
    got = _run_scalar_ssn(ref=0.3, alpha=0.2, kappa=0.3)
    assert math.isclose(got, 0.0, abs_tol=1e-8)


# --------------------------------------------------------------------------- #
# End-to-end: the moment path runs through the full PDAP loop (build model ->
# get_atoms -> moment_weight -> SSN moment_vec -> moment-aware insertion), and
# it confines the atom scale relative to the same run without the moment term.
# --------------------------------------------------------------------------- #
def _tiny_data():
    x = torch.linspace(-1.0, 1.0, 21, dtype=torch.float64).reshape(-1, 1)
    V = (x.pow(2)).reshape(-1, 1)          # smooth target
    dV = (2.0 * x).reshape(-1, 1)
    return (x, V, dV)


def _fit_softplus(moment_beta: float):
    from src.models import build_model

    torch.manual_seed(0)
    cfg = ExperimentConfig(
        model=ModelConfig(
            kind="signed", insertion="profile", activation="softplus", power=1.0,
            alpha=1e-4, gamma=1.0, moment_beta=moment_beta, moment_order=2.0,
        )
    )
    data = _tiny_data()
    model = build_model(cfg, input_dim=1)
    history = PDAP(cfg).fit(
        model, data, data, num_iterations=3, num_insertion=40, verbose=False,
    )
    W, b, _ = model.get_atoms()
    max_omega = float(torch.sqrt((W * W).sum(dim=1) + b * b).max()) if W.shape[0] else 0.0
    return history, model.n_neurons, max_omega


def test_moment_path_runs_end_to_end_and_confines_scale() -> None:
    hist_off, n_off, omega_off = _fit_softplus(moment_beta=0.0)
    hist_on, n_on, omega_on = _fit_softplus(moment_beta=1e-1)

    # Both complete with a finite support and finite recorded objective.
    assert n_on >= 1 and n_off >= 1
    assert math.isfinite(hist_on.train_loss[-1]) and math.isfinite(hist_off.train_loss[-1])
    assert math.isfinite(omega_on)
    # The moment term prices out large-scale atoms: the confined run's largest
    # parameter norm does not exceed the unconstrained run's (with a small slack).
    assert omega_on <= omega_off + 1e-9
