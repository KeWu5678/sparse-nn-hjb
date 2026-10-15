"""Analytic cases for the Newton step, truncation, and residual correction."""

import math

import pytest
import torch

from src.SSN.mpcg import MPCG


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sigma", [0.0, 1.0])
@pytest.mark.parametrize("use_prox_derivative", [False, True])
def test_zero_rhs_returns_a_converged_zero_step(dtype, sigma, use_prox_derivative):
    jac_normal = torch.eye(2, dtype=dtype)
    newton_rhs = torch.zeros(2, dtype=dtype)
    jac_prox = torch.diag(torch.tensor([1.0, 0.0], dtype=dtype)) if use_prox_derivative else None

    step, stop_reason = MPCG(jac_normal, jac_prox).solve(newton_rhs, sigma=sigma)

    torch.testing.assert_close(step, newton_rhs)
    assert stop_reason == "convgd"


@pytest.mark.parametrize("use_prox_derivative", [False, True])
def test_positive_definite_system_reaches_the_newton_step(use_prox_derivative):
    jac_normal = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    newton_rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    jac_prox = torch.eye(2, dtype=torch.float64) if use_prox_derivative else None

    step, stop_reason = MPCG(jac_normal, jac_prox).solve(newton_rhs)

    torch.testing.assert_close(step, newton_rhs.new_tensor([1 / 11, 7 / 11]))
    assert stop_reason == "convgd"


def test_non_symmetric_normal_map_with_non_projector_derivative():
    # jac_prox @ jac_normal is symmetric positive definite, although jac_normal is not
    # symmetric and the proximal derivative is not an orthogonal projection.
    jac_normal = torch.tensor([[4.0, 0.75], [1.5, 1.75]], dtype=torch.float64)
    newton_rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    jac_prox = torch.diag(torch.tensor([1.5, 0.75], dtype=torch.float64))
    originals = [value.clone() for value in (jac_normal, newton_rhs, jac_prox)]

    step, stop_reason = MPCG(jac_normal, jac_prox).solve(newton_rhs)

    torch.testing.assert_close(step, newton_rhs.new_tensor([2 / 47, 52 / 47]))
    assert stop_reason == "convgd"
    for value, original in zip((jac_normal, newton_rhs, jac_prox), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("sigma", [-1.0, 0.0, 0.5])
@pytest.mark.parametrize("curvature", [-1.0, 0.0])
def test_nonpositive_curvature_moves_to_boundary_only_with_a_trust_region(sigma, curvature):
    jac_normal = curvature * torch.eye(2, dtype=torch.float64)
    newton_rhs = torch.tensor([1.0, 0.0], dtype=torch.float64)

    step, stop_reason = MPCG(jac_normal).solve(newton_rhs, sigma=sigma)

    distance = max(0.0, sigma)
    torch.testing.assert_close(step, newton_rhs.new_tensor([distance, 0.0]))
    assert stop_reason == "negdef"


@pytest.mark.parametrize(
    ("sigma", "expected_reason"), [(0.5, "radius"), (2.0, "convgd")]
)
def test_trust_region_truncates_only_when_the_trial_step_exceeds_the_radius(
    sigma, expected_reason
):
    jac_normal = torch.eye(2, dtype=torch.float64)
    newton_rhs = torch.tensor([2.0, 0.0], dtype=torch.float64)

    step, stop_reason = MPCG(jac_normal).solve(newton_rhs, sigma=sigma)

    torch.testing.assert_close(step, newton_rhs.new_tensor([sigma, 0.0]))
    assert stop_reason == expected_reason


@pytest.mark.parametrize(
    ("maxit", "expected_step"),
    [
        (0, [3 / 5, 3 / 5]),
        (1, [13 / 15, 7 / 15]),
    ],
)
def test_iteration_limit_still_applies_the_final_residual_correction(
    maxit, expected_step
):
    jac_normal = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
    newton_rhs = torch.ones(2, dtype=torch.float64)

    step, stop_reason = MPCG(jac_normal).solve(newton_rhs, maxit=maxit)

    torch.testing.assert_close(step, newton_rhs.new_tensor(expected_step))
    assert stop_reason == "maxitr"


@pytest.mark.parametrize("sigma", [0.0, 1.9])
def test_final_correction_resolves_the_residual_in_the_kernel_of_jac_prox(sigma):
    jac_normal = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    newton_rhs = torch.tensor([2.0, 3.0], dtype=torch.float64)
    jac_prox = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))

    step, stop_reason = MPCG(jac_normal, jac_prox).solve(newton_rhs, sigma=sigma)

    # CG reaches [1, 1.5], where jac_prox @ r = 0 but r != 0. The final
    # correction continues toward the solution [1, 2], subject to the radius.
    second_coordinate = math.sqrt(sigma**2 - 1) if sigma > 0 else 2.0
    torch.testing.assert_close(step, newton_rhs.new_tensor([1.0, second_coordinate]))
    assert stop_reason == ("radius" if sigma > 0 else "convgd")


def test_boundary_exit_with_a_singular_proximal_derivative():
    jac_normal = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    newton_rhs = torch.tensor([2.0, 3.0], dtype=torch.float64)
    jac_prox = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))

    step, stop_reason = MPCG(jac_normal, jac_prox).solve(newton_rhs, sigma=1.0)

    torch.testing.assert_close(step, newton_rhs / math.sqrt(13))
    assert stop_reason == "radius"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("rtol", "expected_step"),
    [
        (0.5, [89 / 98, 499 / 980, 263 / 980]),
        (1e-3, [1.0, 0.5, 0.25]),
    ],
)
def test_predicted_reduction_tolerance_controls_the_step(dtype, rtol, expected_step):
    jac_normal = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=dtype))
    newton_rhs = torch.ones(3, dtype=dtype)

    step, stop_reason = MPCG(jac_normal).solve(newton_rhs, rtol=rtol)

    # The loose tolerance stops CG after two iterations, while the residual
    # remains above its threshold. Both exits apply the final correction.
    torch.testing.assert_close(step, newton_rhs.new_tensor(expected_step))
    assert stop_reason == "convgd"


def test_solver_reuse_starts_each_system_from_a_zero_step():
    jac_normal = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
    solver = MPCG(jac_normal)

    cases = [
        ([1.0, 1.0], 0.0, [1.0, 0.5], "convgd"),
        ([2.0, 0.0], 0.5, [0.5, 0.0], "radius"),
        ([0.0, 0.0], 0.0, [0.0, 0.0], "convgd"),
        ([2.0, 2.0], 0.0, [2.0, 1.0], "convgd"),
    ]
    for rhs, sigma, expected_step, expected_reason in cases:
        step, stop_reason = solver.solve(jac_normal.new_tensor(rhs), sigma=sigma)

        torch.testing.assert_close(step, jac_normal.new_tensor(expected_step))
        assert stop_reason == expected_reason
