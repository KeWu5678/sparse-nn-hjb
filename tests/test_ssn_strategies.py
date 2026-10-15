"""Acceptance, parameter restoration, and radius contracts through SSN.step(closure)."""

import pytest
import torch
from ssn_test_problem import closure_for, synthetic_correction

from src.SSN import SSN


def _make_optimizer(method="levenberg_marquardt", *, curvature=1.0, **options):
    coefficient = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    penalty, _, hess_obj = synthetic_correction(
        lambda u: 0.5 * curvature * u.square().sum(),
        torch.tensor([[curvature]], dtype=torch.float64),
        alpha=0.0, gamma=0.0,
    )
    optimizer = SSN([coefficient], penalty, hess_obj, method=method, **options)
    return optimizer, coefficient


@pytest.mark.parametrize("max_ls_iter", [0, 1])
def test_damped_newton_accepts_the_last_allowed_trial(max_ls_iter):
    optimizer, coefficient = _make_optimizer(max_ls_iter=max_ls_iter)
    calls = 0

    def objective(u):
        nonlocal calls
        calls += 1
        loss = 0.5 * u.square().sum()
        # Script trial rejection to test the retry limit independently of damping.
        if 1 < calls < max_ls_iter + 2:
            return loss + 1.0
        return loss

    loss = optimizer.step(closure_for(optimizer, coefficient, objective))

    assert calls == max_ls_iter + 2  # Initial loss, initial trial, then retries.
    assert optimizer.last_step_success
    assert coefficient.item() == pytest.approx(0.0, abs=2e-12)
    torch.testing.assert_close(loss, 0.5 * coefficient.square().sum())


@pytest.mark.parametrize("method", ["levenberg_marquardt", "steihaug_cg"])
@pytest.mark.parametrize("trial_loss", [1.0, float("nan"), float("-inf")])
def test_rejection_restores_parameters_and_returns_original_loss(method, trial_loss):
    optimizer, coefficient = _make_optimizer(method, max_ls_iter=2)
    original = coefficient.detach().clone()
    losses = []

    def objective(u):
        loss = 0.5 * u.square().sum()
        if losses:
            loss = loss + trial_loss
        losses.append(loss)
        return loss

    loss = optimizer.step(closure_for(optimizer, coefficient, objective))

    torch.testing.assert_close(loss, losses[0], rtol=0, atol=0, equal_nan=True)
    assert not optimizer.last_step_success
    torch.testing.assert_close(coefficient, original, rtol=0, atol=0)
    assert len(losses) == (4 if method == "levenberg_marquardt" else 2)
    if method == "steihaug_cg":
        assert optimizer.state[coefficient]["sigma"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("method", "failure_call"),
    [("levenberg_marquardt", 2), ("levenberg_marquardt", 3), ("steihaug_cg", 2)],
)
@pytest.mark.parametrize("exception_type", [ValueError, torch.linalg.LinAlgError])
def test_trial_exception_restores_parameters_and_cause(method, failure_call, exception_type):
    optimizer, coefficient = _make_optimizer(method, max_ls_iter=2)
    original = coefficient.detach().clone()
    error = exception_type("objective failed")
    calls = 0

    def objective(u):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            # The failing trial is installed in the parameters while evaluated.
            assert not torch.equal(coefficient.detach(), original)
            raise error
        loss = 0.5 * u.square().sum()
        return loss if calls == 1 else loss + 1.0

    with pytest.raises(exception_type, match="objective failed") as caught:
        optimizer.step(closure_for(optimizer, coefficient, objective))

    assert caught.value is error
    assert not optimizer.last_step_success
    torch.testing.assert_close(coefficient, original, rtol=0, atol=0)


@pytest.mark.parametrize("failure_call", [1, 2])
@pytest.mark.parametrize("exception_type", [RuntimeError, torch.linalg.LinAlgError])
def test_linear_solve_exception_restores_parameters_and_propagates(
    monkeypatch, failure_call, exception_type
):
    optimizer, coefficient = _make_optimizer(max_ls_iter=2)
    original = coefficient.detach().clone()
    error = exception_type("linear solve failed")
    solve = torch.linalg.solve
    solve_calls = 0
    value_calls = 0

    def failing_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        if solve_calls == failure_call:
            raise error
        return solve(*args, **kwargs)

    def objective(u):
        nonlocal value_calls
        value_calls += 1
        loss = 0.5 * u.square().sum()
        return loss if value_calls == 1 else loss + 1.0

    monkeypatch.setattr(torch.linalg, "solve", failing_solve)
    with pytest.raises(exception_type, match="linear solve failed") as caught:
        optimizer.step(closure_for(optimizer, coefficient, objective))

    assert caught.value is error
    assert not optimizer.last_step_success
    torch.testing.assert_close(coefficient, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("curvature", "sigma", "expected_sigma", "expected_coefficient"),
    [
        (1.0, 0.1, 0.2, 0.9),       # Radius truncation.
        (1.0, 2.0, 2.4, 0.0),       # Interior step.
        (1.0, 9.0, 10.0, 0.0),      # Radius growth is capped.
        (-1.0, 0.1, 0.12, 1.1),     # Negative curvature also reaches the boundary.
    ],
)
def test_accepted_trust_region_step_updates_persistent_radius(
    curvature, sigma, expected_sigma, expected_coefficient
):
    optimizer, coefficient = _make_optimizer("steihaug_cg", curvature=curvature, sigmamax=10.0)
    optimizer.state[coefficient]["sigma"] = sigma

    def objective(u):
        return 1.0 + 0.5 * curvature * u.square().sum()

    loss = optimizer.step(closure_for(optimizer, coefficient, objective))

    assert optimizer.last_step_success
    assert coefficient.item() == pytest.approx(expected_coefficient)
    assert optimizer.state[coefficient]["sigma"] == pytest.approx(expected_sigma)
    torch.testing.assert_close(loss, objective(coefficient))


def test_a_closure_without_backward_is_rejected():
    optimizer, coefficient = _make_optimizer()
    original = coefficient.detach().clone()

    def closure():
        optimizer.zero_grad()
        return 0.5 * coefficient.square().sum()

    with pytest.raises(RuntimeError, match="must call backward"):
        optimizer.step(closure)

    assert not optimizer.last_step_success
    torch.testing.assert_close(coefficient, original, rtol=0, atol=0)
