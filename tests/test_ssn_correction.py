"""A complete correction runs its SSN budget and owns final acceptance."""

import pytest
import torch
from ssn_test_problem import synthetic_correction

from src.config.schema import TrainingConfig
from src.PDAP import PDAP
from src.PDAP.correction import correct_coefficients
from src.SSN import SSN


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_correction_runs_the_configured_budget(iterations):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: .5 * u.square().sum(), torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj)
    diagnostics = correct_coefficients(
        optimizer, coefficient, coefficient_loss, iterations=iterations, correction_guard=True,
    )

    assert diagnostics["ssn_iterations"] == iterations
    assert diagnostics["ssn_failed_steps"] == 0
    assert diagnostics["correction_accepted"]
    assert float(coefficient_loss(coefficient).detach()) <= .5
    assert coefficient.item() == pytest.approx(1. if iterations == 0 else 0., abs=1e-11)


@pytest.mark.parametrize("guard", [True, False])
def test_final_guard_rejects_a_worsening_trial_allowed_by_local_tolerance(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    # Newton moves toward this concave objective's maximum. A permissive local
    # tolerance allows that trial; the complete correction must decide separately.
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: 2 - .5 * u.square().sum(), -torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, tolerance_ls=2.)
    diagnostics = correct_coefficients(
        optimizer, coefficient, coefficient_loss, iterations=1, correction_guard=guard,
    )

    assert optimizer.last_step_success
    assert diagnostics["correction_accepted"] is not guard
    assert coefficient.item() == pytest.approx(1. if guard else 0., abs=1e-11)
    assert float(coefficient_loss(coefficient).detach()) == pytest.approx(1.5 if guard else 2.)


@pytest.mark.parametrize("guard", [True, False])
def test_nonfinite_final_loss_restores_starting_coefficients(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))

    def fitting(u):
        return torch.where(u.abs() < .1, u.new_tensor(-torch.inf), .5 * u.square()).sum()

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj)
    diagnostics = correct_coefficients(
        optimizer, coefficient, coefficient_loss, iterations=1, correction_guard=guard,
    )

    assert not diagnostics["correction_accepted"]
    assert coefficient.item() == 1.
    assert coefficient_loss(coefficient).item() == .5


def test_exception_after_an_accepted_iteration_restores_the_whole_correction():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    error = RuntimeError("second iteration failed")

    def fitting(u):
        if u.requires_grad and u.abs().max() < 1e-6:
            raise error
        return .5 * u.square().sum()

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj)
    with pytest.raises(RuntimeError, match="second iteration failed") as caught:
        correct_coefficients(optimizer, coefficient, coefficient_loss, iterations=2, correction_guard=True)

    assert caught.value is error
    assert not optimizer.last_step_success
    assert coefficient.item() == 1.


def test_complete_trust_region_correction_continues_after_a_rejected_trial():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    evaluations = 0

    def fitting(u):
        nonlocal evaluations
        value = .5 * u.square().sum()
        if not u.requires_grad:
            evaluations += 1
            # The correction first evaluates its starting objective. Reject the
            # second trial after it to exercise radius reduction and recovery
            # within one correction, after an earlier iterate was accepted.
            if evaluations == 3:
                return value + 1.
        return value

    proximal, coefficient_loss, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, method="steihaug_cg")
    diagnostics = correct_coefficients(
        optimizer, coefficient, coefficient_loss, iterations=4, correction_guard=True,
    )

    assert diagnostics["ssn_iterations"] == evaluations - 1 == 4
    assert diagnostics["ssn_failed_steps"] == 1
    assert optimizer.last_step_success and diagnostics["correction_accepted"]
    assert coefficient.item() == pytest.approx(.78)
    assert optimizer.state[coefficient]["sigma"] == pytest.approx(.16)


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_pdap_rejects_an_invalid_iteration_budget(iterations):
    proximal, _, _ = synthetic_correction(lambda u: u.sum(), torch.eye(1, dtype=torch.float64), alpha=0.)

    with pytest.raises(ValueError, match="fit_outer_iterations must be a nonnegative integer"):
        PDAP(TrainingConfig(fit_outer_iterations=iterations), proximal)
