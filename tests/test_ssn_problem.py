"""The explicit-vector problem interface and its SSN splitting."""

import pytest
import torch
from ssn_test_problem import SyntheticProblem

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

    problem = SyntheticProblem(
        data_loss, A.T @ A, alpha=alpha, gamma=gamma, power=power, penalized_mask=mask,
    )
    u = torch.tensor([-0.6, 0.8, 1.2], dtype=dtype)
    original = u.clone()

    def smooth(u):
        if gamma == 0.0:
            return data_loss(u)
        magnitude = u[mask].abs()
        remainder = -0.5 * magnitude + torch.log1p(2 * gamma * magnitude) / (4 * gamma)
        return data_loss(u) + alpha * remainder.sum()

    jac_loss = torch.func.grad(problem.loss)(u)
    with torch.no_grad():
        grad = problem.jac_obj_augmented(u, jac_loss)
        hessian = problem.hess_obj_augmented(u)
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
    torch.testing.assert_close(torch.func.grad(problem.loss)(u), expected_grad)


def test_problem_keeps_the_selected_hess_obj_augmented_at_zero():
    problem = SyntheticProblem(
        lambda u: 0.5 * u.square().sum(), torch.eye(1, dtype=torch.float64),
        alpha=0.1, gamma=0.7,
    )
    u = torch.zeros(1, dtype=torch.float64)
    jac_loss = torch.func.grad(problem.loss)(u)
    torch.testing.assert_close(problem.jac_obj_augmented(u, jac_loss), u)
    torch.testing.assert_close(
        problem.hess_obj_augmented(u), torch.tensor([[0.93]], dtype=torch.float64),
    )


@pytest.mark.parametrize(("power", "gamma"), [(1.0, 0.7), (2.0, 0.0), (3.0, 0.0)])
def test_problem_normal_jacobian_matches_finite_differences(power, gamma):
    A = torch.tensor([[2.0, 0.4, -0.2], [0.1, 1.0, 0.5]], dtype=torch.float64)
    target = torch.tensor([0.2, -0.3], dtype=torch.float64)

    def data_loss(u):
        residual = A @ u - target
        return 0.5 * (residual @ residual)

    problem = SyntheticProblem(
        data_loss, A.T @ A, alpha=0.1, gamma=gamma, power=power,
        penalized_mask=torch.tensor([True, True, False]),
    )
    inverse_step = 1.3
    # Active, inactive, and unpenalized coordinates, away from switching points.
    q = torch.tensor([-1.2, 0.01, 0.8], dtype=torch.float64)

    def normal(q):
        u = problem.prox(q, inverse_step)
        jac_loss = torch.func.grad(problem.loss)(u)
        grad = problem.jac_obj_augmented(u, jac_loss)
        return problem.normal_map(q, u, grad, inverse_step)

    u = problem.prox(q, inverse_step)
    jac_prox = problem.jac_prox(q, inverse_step, prox_result=u)
    jac_normal = problem.jac_normal(u, jac_prox, inverse_step)
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

    problem = SyntheticProblem(
        data_loss, torch.eye(2, dtype=torch.float64), alpha=0.0, gamma=0.0,
    )
    optimizer = SSN([coefficient], problem, method=method)
    loss = optimizer.step()

    assert optimizer.last_step_success
    assert len(evaluated) == 2
    torch.testing.assert_close(evaluated[0], original, rtol=0, atol=0)
    assert not torch.equal(evaluated[1], original)
    torch.testing.assert_close(coefficient, evaluated[1], rtol=0, atol=0)
    torch.testing.assert_close(loss, 0.5 * coefficient.square().sum())


def test_problem_applies_coordinate_masks_without_mutating_the_input():
    problem = SyntheticProblem(
        lambda u: 0.5 * u.square().sum(), torch.eye(3, dtype=torch.float64),
        alpha=0.25, gamma=0.0,
        penalized_mask=torch.tensor([True, True, False]),
        nonneg_mask=torch.tensor([False, True, False]),
    )
    u = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(problem.loss(u), u.new_tensor(15.0))
    torch.testing.assert_close(problem.prox(u, 1.0), u.new_tensor([1.75, 0.0, 4.0]))
    torch.testing.assert_close(u, u.new_tensor([2.0, -3.0, 4.0]), rtol=0, atol=0)
