"""A complete correction runs its SSN budget and owns final acceptance."""

import pytest
import torch
from ssn_test_problem import closure_for, synthetic_correction

from src.config.schema import TrainingConfig
from src.PDAP import PDAP
from src.PDAP.ssn_refine import ssn_refine_outerweights
from src.SSN import SSN


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_correction_runs_the_configured_budget(iterations):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    penalty, fitting, hess_obj = synthetic_correction(
        lambda u: .5 * u.square().sum(), torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN([coefficient], penalty, hess_obj)
    diagnostics = ssn_refine_outerweights(
        optimizer, coefficient, closure_for(optimizer, coefficient, fitting),
        iterations=iterations, correction_guard=True,
    )

    assert diagnostics["ssn_iterations"] == iterations
    assert diagnostics["ssn_failed_steps"] == 0
    assert diagnostics["correction_accepted"]
    assert float(fitting(coefficient).detach()) <= .5
    assert coefficient.item() == pytest.approx(1. if iterations == 0 else 0., abs=1e-11)


@pytest.mark.parametrize("guard", [True, False])
def test_final_guard_rejects_a_worsening_trial_allowed_by_local_tolerance(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    # Newton moves toward this concave objective's maximum. A permissive local
    # tolerance allows that trial; the complete correction must decide separately.
    penalty, fitting, hess_obj = synthetic_correction(
        lambda u: 2 - .5 * u.square().sum(), -torch.eye(1, dtype=torch.float64), alpha=0.,
    )
    optimizer = SSN([coefficient], penalty, hess_obj, tolerance_ls=2.)
    diagnostics = ssn_refine_outerweights(
        optimizer, coefficient, closure_for(optimizer, coefficient, fitting),
        iterations=1, correction_guard=guard,
    )

    assert optimizer.last_step_success
    assert diagnostics["correction_accepted"] is not guard
    assert coefficient.item() == pytest.approx(1. if guard else 0., abs=1e-11)
    assert float(fitting(coefficient).detach()) == pytest.approx(1.5 if guard else 2.)


@pytest.mark.parametrize("guard", [True, False])
def test_nonfinite_final_loss_restores_starting_coefficients(guard):
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    calls = 0

    def fitting(u):
        nonlocal calls
        calls += 1
        value = .5 * u.square().sum()
        # The step's own objective (after the starting one) is NaN with a finite
        # gradient, so every trial is rejected and the step returns that NaN.
        return value + torch.nan if calls == 2 else value

    penalty, _, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], penalty, hess_obj, max_ls_iter=2)
    diagnostics = ssn_refine_outerweights(
        optimizer, coefficient, closure_for(optimizer, coefficient, fitting),
        iterations=1, correction_guard=guard,
    )

    assert not diagnostics["correction_accepted"]
    assert coefficient.item() == 1.
    assert fitting(coefficient).item() == .5


def test_exception_after_an_accepted_iteration_restores_the_whole_correction():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    error = RuntimeError("second iteration failed")
    calls = 0

    def fitting(u):
        nonlocal calls
        calls += 1
        # Calls: starting objective, first iteration and its accepted trial,
        # then the second iteration's own evaluation.
        if calls == 4:
            raise error
        return .5 * u.square().sum()

    penalty, _, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], penalty, hess_obj)
    with pytest.raises(RuntimeError, match="second iteration failed") as caught:
        ssn_refine_outerweights(
            optimizer, coefficient, closure_for(optimizer, coefficient, fitting),
            iterations=2, correction_guard=True,
        )

    assert caught.value is error
    assert not optimizer.last_step_success
    assert coefficient.item() == 1.


def test_complete_trust_region_correction_continues_after_a_rejected_trial():
    coefficient = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    calls = 0

    def fitting(u):
        nonlocal calls
        calls += 1
        value = .5 * u.square().sum()
        # Calls: the starting objective, then an evaluation and one trial per
        # trust-region iteration. Reject the second trial to exercise radius
        # reduction and recovery after an earlier iterate was accepted.
        if calls == 5:
            return value + 1.
        return value

    penalty, _, hess_obj = synthetic_correction(fitting, torch.eye(1, dtype=torch.float64), alpha=0.)
    optimizer = SSN([coefficient], penalty, hess_obj, method="steihaug_cg")
    diagnostics = ssn_refine_outerweights(
        optimizer, coefficient, closure_for(optimizer, coefficient, fitting),
        iterations=4, correction_guard=True,
    )

    assert calls == 1 + 2 * 4
    assert diagnostics["ssn_iterations"] == 4
    assert diagnostics["ssn_failed_steps"] == 1
    assert optimizer.last_step_success and diagnostics["correction_accepted"]
    assert coefficient.item() == pytest.approx(.78)
    assert optimizer.state[coefficient]["sigma"] == pytest.approx(.16)


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_pdap_rejects_an_invalid_iteration_budget(iterations):
    penalty, _, _ = synthetic_correction(lambda u: u.sum(), torch.eye(1, dtype=torch.float64), alpha=0.)

    with pytest.raises(ValueError, match="fit_outer_iterations must be a nonnegative integer"):
        PDAP(TrainingConfig(fit_outer_iterations=iterations), penalty)
