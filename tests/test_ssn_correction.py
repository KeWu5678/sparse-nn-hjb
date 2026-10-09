"""A complete correction runs its budget and owns final acceptance."""

import pytest
import torch
from ssn_test_problem import synthetic_correction

from src.SSN import SSN


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_one_step_runs_the_configured_complete_correction(iterations):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: .5 * u.square().sum(), torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, max_iter=iterations)
    loss = optimizer.step()

    assert optimizer.iterations_run == iterations
    assert optimizer.failed_steps == 0
    assert optimizer.correction_accepted
    assert float(loss.detach()) <= .5
    torch.testing.assert_close(loss, coefficient_loss(coefficient))
    assert coefficient.item() == pytest.approx(1. if iterations == 0 else 0., abs=1e-11)


@pytest.mark.parametrize("guard", [True, False])
def test_final_guard_rejects_a_worsening_trial_allowed_by_local_tolerance(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    # Newton moves toward this concave objective's maximum. A permissive local
    # tolerance allows that trial; the complete correction must decide separately.
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: 2 - .5 * u.square().sum(), -torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN(
        [coefficient], proximal, coefficient_loss, hess_obj, max_iter=1,
        tolerance_ls=2., correction_guard=guard,
    )
    loss = optimizer.step()

    assert optimizer.last_step_success
    assert optimizer.correction_accepted is not guard
    assert coefficient.item() == pytest.approx(1. if guard else 0., abs=1e-11)
    assert float(loss.detach()) == pytest.approx(1.5 if guard else 2.)
    torch.testing.assert_close(loss, coefficient_loss(coefficient))


@pytest.mark.parametrize("guard", [True, False])
def test_nonfinite_final_loss_restores_starting_coefficients(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))

    def fitting(u):
        return torch.where(u.abs() < .1, u.new_tensor(-torch.inf), .5 * u.square()).sum()

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, max_iter=1, correction_guard=guard)
    loss = optimizer.step()

    assert not optimizer.correction_accepted
    assert coefficient.item() == 1.
    assert loss.item() == .5


def test_exception_after_an_accepted_iteration_restores_the_whole_correction():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    error = RuntimeError("second iteration failed")

    def fitting(u):
        if u.requires_grad and u.abs().max() < 1e-6:
            raise error
        return .5 * u.square().sum()

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, max_iter=2)
    with pytest.raises(RuntimeError, match="second iteration failed") as caught:
        optimizer.step()

    assert caught.value is error
    assert optimizer.iterations_run == 1
    assert not optimizer.correction_accepted
    assert coefficient.item() == 1.


def test_complete_trust_region_correction_continues_after_a_rejected_trial():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    trials = 0

    def fitting(u):
        nonlocal trials
        value = .5 * u.square().sum()
        if not u.requires_grad:
            trials += 1
            # Reject the second trial to exercise radius reduction and recovery
            # within one correction, after an earlier iterate was accepted.
            if trials == 2:
                return value + 1.
        return value

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, method="steihaug_cg", max_iter=4)
    loss = optimizer.step()

    assert optimizer.iterations_run == trials == 4
    assert optimizer.failed_steps == 1
    assert optimizer.last_step_success and optimizer.correction_accepted
    assert coefficient.item() == pytest.approx(.78)
    assert optimizer.state[coefficient]["sigma"] == pytest.approx(.16)
    torch.testing.assert_close(loss, .5 * coefficient.square().sum())
