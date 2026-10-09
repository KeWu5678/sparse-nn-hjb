"""The proximal splitting and SSN's normal map built on it."""

import pytest
import torch
from ssn_test_problem import correction_optimizer, synthetic_correction

from src.SSN import SSN


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("power", "gamma"), [(1.0, 0.7), (2.0, 0.0), (3.0, 0.0)])
def test_problem_smooth_derivatives_match_the_splitting(dtype, power, gamma):
    A = torch.tensor([[2.0, 0.4, -0.2], [0.1, 1.0, 0.5]], dtype=dtype)
    target = torch.tensor([0.2, -0.3], dtype=dtype)
    mask = torch.tensor([True, False, True])
    alpha = 0.1

    def data_loss(u):
        residual = A @ u - target
        return 0.5 * (residual @ residual)

    proximal, coefficient_loss, hess_obj = synthetic_correction(
        data_loss, A.T @ A, alpha=alpha, gamma=gamma, power=power, penalized_mask=mask,
    )
    u = torch.tensor([-0.6, 0.8, 1.2], dtype=dtype)
    original = u.clone()
    optimizer = correction_optimizer(proximal, coefficient_loss, hess_obj, u)

    def smooth(u):
        if gamma == 0.0:
            return data_loss(u)
        magnitude = u[mask].abs()
        remainder = -0.5 * magnitude + torch.log1p(2 * gamma * magnitude) / (4 * gamma)
        return data_loss(u) + alpha * remainder.sum()

    jac_loss = torch.func.grad(coefficient_loss)(u)
    with torch.no_grad():
        grad = optimizer.jac_obj_augmented(u, jac_loss)
        hessian = optimizer.hess_obj_augmented(u)
    torch.testing.assert_close(grad, torch.func.grad(smooth)(u))
    torch.testing.assert_close(hessian, torch.func.hessian(smooth)(u))
    torch.testing.assert_close(u, original, rtol=0, atol=0)
    assert not u.requires_grad
    assert grad.dtype == hessian.dtype == dtype
    assert not grad.requires_grad

    # The full objective remains a differentiable Torch function of its input.
    expected_grad = torch.func.grad(data_loss)(u)
    exponent = 2.0 / (power + 1.0)
    if power == 1.0:
        slope = 0.5 + 0.5 / (1 + 2 * gamma * u[mask].abs())
    else:
        slope = exponent * u[mask].abs().pow(exponent - 1)
    expected_grad[mask] += alpha * slope * u[mask].sign()
    torch.testing.assert_close(torch.func.grad(coefficient_loss)(u), expected_grad)


def test_problem_keeps_the_selected_hess_obj_augmented_at_zero():
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: 0.5 * u.square().sum(), torch.eye(1, dtype=torch.float64),
        alpha=0.1, gamma=0.7,
    )
    u = torch.zeros(1, dtype=torch.float64)
    optimizer = correction_optimizer(proximal, coefficient_loss, hess_obj, u)
    jac_loss = torch.func.grad(coefficient_loss)(u)
    torch.testing.assert_close(optimizer.jac_obj_augmented(u, jac_loss), u)
    torch.testing.assert_close(
        optimizer.hess_obj_augmented(u), torch.tensor([[0.93]], dtype=torch.float64),
    )


@pytest.mark.parametrize(("power", "gamma"), [(1.0, 0.7), (2.0, 0.0), (3.0, 0.0)])
def test_problem_normal_jacobian_matches_finite_differences(power, gamma):
    A = torch.tensor([[2.0, 0.4, -0.2], [0.1, 1.0, 0.5]], dtype=torch.float64)
    target = torch.tensor([0.2, -0.3], dtype=torch.float64)

    def data_loss(u):
        residual = A @ u - target
        return 0.5 * (residual @ residual)

    proximal, coefficient_loss, hess_obj = synthetic_correction(
        data_loss, A.T @ A, alpha=0.1, gamma=gamma, power=power,
        penalized_mask=torch.tensor([True, True, False]),
    )
    # Active, inactive, and unpenalized coordinates, away from switching points.
    q = torch.tensor([-1.2, 0.01, 0.8], dtype=torch.float64)
    optimizer = correction_optimizer(proximal, coefficient_loss, hess_obj, q, prox_scale=0.1 / 1.3)
    lambda_ = proximal.alpha / optimizer.param_groups[0]["inverse_step"]

    def normal(q):
        u = proximal.prox(q, lambda_)
        jac_loss = torch.func.grad(coefficient_loss)(u)
        grad = optimizer.jac_obj_augmented(u, jac_loss)
        return optimizer.normal_map(q, u, grad)

    u = proximal.prox(q, lambda_)
    jac_prox = proximal.jac_prox(q, lambda_, prox_image=u)
    jac_normal = optimizer.jac_normal(u, jac_prox)
    epsilon = 1e-5
    perturbations = epsilon * torch.eye(q.numel(), dtype=q.dtype)
    finite_differences = torch.stack([
        (normal(q + dq) - normal(q - dq)) / (2 * epsilon) for dq in perturbations
    ], dim=1)

    torch.testing.assert_close(jac_normal, finite_differences, rtol=1e-6, atol=1e-8)
    assert not torch.allclose(jac_normal, jac_normal.T)


@pytest.mark.parametrize("method", ["levenberg_marquardt", "steihaug_cg"])
def test_accepted_trials_leave_parameters_untouched_until_commit(method):
    coefficient = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float64))
    original = coefficient.detach().clone()
    evaluated = []

    def data_loss(u):
        torch.testing.assert_close(coefficient, original, rtol=0, atol=0)
        evaluated.append(u.detach().clone())
        return 0.5 * u.square().sum()

    proximal, coefficient_loss, hess_obj = synthetic_correction(
        data_loss, torch.eye(2, dtype=torch.float64), alpha=0.0, gamma=0.0,
    )
    optimizer = SSN([coefficient], proximal, coefficient_loss, hess_obj, method=method, max_iter=1)
    loss = optimizer.step()

    assert optimizer.last_step_success
    assert len(evaluated) == 2
    torch.testing.assert_close(evaluated[0], original, rtol=0, atol=0)
    assert not torch.equal(evaluated[1], original)
    torch.testing.assert_close(coefficient, evaluated[1], rtol=0, atol=0)
    torch.testing.assert_close(loss, 0.5 * coefficient.square().sum())


def test_problem_applies_coordinate_masks_without_mutating_the_input():
    proximal, coefficient_loss, hess_obj = synthetic_correction(
        lambda u: 0.5 * u.square().sum(), torch.eye(3, dtype=torch.float64),
        alpha=0.25, gamma=0.0,
        penalized_mask=torch.tensor([True, True, False]),
        nonneg_mask=torch.tensor([False, True, False]),
    )
    u = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(coefficient_loss(u), u.new_tensor(15.0))
    torch.testing.assert_close(proximal.prox(u, 0.25), u.new_tensor([1.75, 0.0, 4.0]))
    torch.testing.assert_close(u, u.new_tensor([2.0, -3.0, 4.0]), rtol=0, atol=0)
