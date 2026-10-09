"""External samples and temporary coefficient calculations used by PDAP and SSN."""

import pytest
import torch
from ssn_test_problem import capture_coefficient_solve, correction_optimizer, proximal_for

from src.config import resolve_model, validate_training
from src.config.activations import ACTIVATIONS
from src.data import make_dataloader
from src.eval import metrics
from src.models import Proximal, SparseModel
from src.PDAP.warmstart import _warm_start_kernel


def _samples():
    X = torch.tensor([[-1.0, 0.2], [0.3, -0.4], [0.8, 0.9]], dtype=torch.float64)
    V = X[:, :1].square() + 0.1
    dV = torch.cat((2 * X[:, :1], torch.zeros_like(X[:, :1])), dim=1)
    return X, V, dV


def _problem(activation="relu", power=1.0, **specifications):
    config = resolve_model({
        "activation": activation, "power": power, "moment_order": 2.0, **specifications,
    }, 2)
    validate_training(config)
    problem = SparseModel(config)
    problem.set_atoms(
        torch.tensor([[0.5, -0.3], [0.1, 0.7]], dtype=torch.float64),
        torch.tensor([0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.6, 0.8], dtype=torch.float64),
    )
    return problem, proximal_for(config, problem)


def test_obj_is_the_weighted_h1_quadratic():
    problem = SparseModel(resolve_model({"loss_weights": (.4, 1.3)}, 2))
    value = torch.tensor([[2.], [3.]], dtype=torch.float64, requires_grad=True)
    gradient = value.new_tensor([[1., 2.], [2., 1.]], requires_grad=True)
    target = torch.ones_like(value), torch.zeros_like(gradient)

    obj = problem.obj((value, gradient), target)

    torch.testing.assert_close(obj, value.new_tensor(3.75))
    dvalue, dgradient = torch.autograd.grad(obj, (value, gradient))
    torch.testing.assert_close(dvalue, value.new_tensor([[.2], [.4]]))
    torch.testing.assert_close(dgradient, value.new_tensor([[.65, 1.3], [1.3, .65]]))


def test_coefficient_gradient_and_full_hessian_match_the_scalar_objective():
    problem = SparseModel(resolve_model({"loss_weights": (.4, 1.3)}, 2))
    Kv = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    Kg = Kv.new_tensor([[1., 0.], [0., 2.], [-1., 2.], [2., 3.]])
    u = Kv.new_tensor([.2, -.3])
    target = Kv.new_tensor([[.5], [1.]]), Kv.new_tensor([[1., -.5], [0., 2.]])

    def fitting(u):
        return problem.obj(((Kv @ u).reshape(2, 1), (Kg @ u).reshape(2, 2)), target)

    residual = Kv.new_zeros((2, 2)), (Kv @ u).reshape(2, 1) - target[0], (Kg @ u).reshape(2, 2) - target[1]
    jac = problem.jac_obj(Kv, Kg, residual)
    hess = problem.hess_obj(Kv, Kg)

    torch.testing.assert_close(jac, torch.func.grad(fitting)(u))
    torch.testing.assert_close(hess, torch.func.hessian(fitting)(u))
    torch.testing.assert_close(hess, Kv.new_tensor([[5.9, 5.4], [5.4, 15.05]]))
    direction = Kv.new_tensor([1., -2.])
    delta = fitting(u + direction) - fitting(u)
    torch.testing.assert_close(delta, jac @ direction + .5 * direction @ hess @ direction)


@pytest.mark.parametrize("alpha", [0., .3])
def test_sparsity_functional_remains_unweighted_and_respects_masks(alpha):
    proximal = Proximal(
        alpha, 0.0, 1.0, penalized_mask=torch.tensor([True, True, False]),
        nonneg_mask=torch.tensor([False, True, False]),
    )
    u = torch.tensor([-2., -3., 100.], dtype=torch.float64)
    torch.testing.assert_close(proximal.sparsity_functional(u), u.new_tensor(2.))


@pytest.mark.parametrize("power", [1.5, 4.0, 1.0 + 5e-13, 2.0 + 5e-13, 3.0 - 5e-13])
def test_training_config_rejects_unsupported_powers(power):
    with pytest.raises(ValueError, match="supports activation powers 1, 2, 3"):
        validate_training(resolve_model({"power": power}, 2))


@pytest.mark.parametrize(
    "power,exponent,alpha,value,expected,derivative",
    [
        (1.0, 1.0, 1.0, 2.0, 1.0, 1.0),
        (2.0, 2.0 / 3.0, 1.5, 8.5, 8.0, 48.0 / 47.0),
        (3.0, 0.5, 1.0, 4.25, 4.0, 32.0 / 31.0),
    ],
)
def test_validated_power_selects_the_proximal_map_and_derivative(
    power, exponent, alpha, value, expected, derivative,
):
    config = resolve_model({"power": power, "alpha": alpha}, 2)
    problem = SparseModel(config)
    assert problem.penalty_exponent == exponent
    proximal = proximal_for(config, problem)
    q = torch.tensor([-value, value], dtype=torch.float64)
    prox = proximal.prox(q, alpha)
    jac = proximal.jac_prox(q, alpha, prox_image=prox)
    torch.testing.assert_close(prox, q.new_tensor([-expected, expected]))
    torch.testing.assert_close(jac, derivative * torch.eye(2, dtype=q.dtype))


@pytest.mark.parametrize(
    "activation,power,gamma",
    [
        ("relu", 1.0, 0.7),
        ("softplus", 1.0, 0.7),
        ("relu", 2.0, 0.0),
        ("relu", 3.0, 0.0),
    ],
)
def test_samples_define_the_objective_and_its_derivatives(monkeypatch, activation, power, gamma):
    data = _samples()
    loader = make_dataloader(data)
    problem, proximal = _problem(activation, power, alpha=0.1, gamma=gamma, loss_weights=(0.4, 1.3))
    before = {name: tensor.clone() for name, tensor in problem.state_dict().items()}
    loss, hess_obj, u = capture_coefficient_solve(monkeypatch, problem, proximal, loader)
    moment_weight = problem.moment_weight_of_atoms()
    value_jacobian, gradient_jacobian = problem.kernel(data[0])

    def fitting(u):
        c = u / moment_weight
        prediction = value_jacobian @ c, gradient_jacobian @ c
        return problem.obj(prediction, (data[1].reshape(-1), data[2].reshape(-1)))

    Vp, dVp = problem(data[0])
    expected_obj = (0.4 * (Vp - data[1]).square().sum() + 1.3 * (dVp - data[2]).square().sum()) / (
        2 * len(data[0])
    )
    torch.testing.assert_close(problem.obj((Vp, dVp), data[1:]), expected_obj)
    torch.testing.assert_close(fitting(u), expected_obj)
    torch.testing.assert_close(hess_obj, torch.func.hessian(fitting)(u))
    obj_augmented = lambda u: fitting(u) + proximal.correction(u)
    assert float(loss(u)) == pytest.approx(metrics(problem, proximal, loader)["train_loss"], rel=1e-12)

    optimizer = correction_optimizer(proximal, loss, hess_obj, u)
    jac_loss = torch.func.grad(loss)(u)
    torch.testing.assert_close(
        optimizer.jac_obj_augmented(u, jac_loss), torch.func.grad(obj_augmented)(u)
    )
    torch.testing.assert_close(
        optimizer.hess_obj_augmented(u), torch.func.hessian(obj_augmented)(u)
    )
    if power == 1.0:
        torch.testing.assert_close(loss(u), obj_augmented(u) + 0.1 * u.abs().sum())
    else:
        torch.testing.assert_close(obj_augmented(u), fitting(u), rtol=0, atol=0)
    for name, tensor in problem.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_one_problem_handles_empty_and_replaced_supports(monkeypatch):
    data = _samples()
    loader = make_dataloader(data)
    config = resolve_model({"activation": "softplus", "alpha": 0.01, "moment_order": 2.0}, 2)
    validate_training(config)
    problem = SparseModel(config)
    proximal = proximal_for(config, problem)
    prediction = problem(data[0])
    residual = data[0], *(p - t for p, t in zip(prediction, data[1:]))
    torch.testing.assert_close(residual[1], -data[1])
    torch.testing.assert_close(residual[2], -data[2])
    initial_loss = metrics(problem, proximal, loader)["train_loss"]
    a = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    b = a.new_tensor(0.1, requires_grad=True)
    Kv, Kg = problem.kernel(data[0], a.reshape(1, -1), b.reshape(1))
    score = problem.jac_obj(Kv, Kg, residual)[0].abs() / problem.moment_weight_of_atoms(a, b)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(score, (a, b)))

    for width in (0, 2, 1, 3):
        W = torch.arange(1, 2 * width + 1, dtype=torch.float64).reshape(width, 2) / 5
        problem.set_atoms(W, torch.zeros(width, dtype=W.dtype), torch.ones(width, dtype=W.dtype))
        loss, hess_obj, u = capture_coefficient_solve(monkeypatch, problem, proximal, loader)
        assert hess_obj.shape == (width, width)
        assert float(loss(u)) == pytest.approx(metrics(problem, proximal, loader)["train_loss"], rel=1e-12)

    problem.set_atoms(W[:0], W.new_zeros(0), W.new_zeros(0))
    assert metrics(problem, proximal, loader)["train_loss"] == pytest.approx(initial_loss)


@pytest.mark.parametrize("activation,power", [("relu", 1.), ("relu", 2.), ("relu", 3.), ("softplus", 1.)])
def test_candidate_objective_gradient_retains_inner_weight_derivatives(activation, power):
    problem, _ = _problem(activation, power, loss_weights=(.4, 1.3))
    X, V, dV = _samples()
    prediction = problem(X)
    residual = X, prediction[0] - V, prediction[1] - dV
    omega = X.new_tensor([.7, -.2, .4], requires_grad=True)
    def candidate_gradient(omega):
        Kv, Kg = problem.kernel(X, omega[:2].reshape(1, -1), omega[2:].reshape(1))
        return problem.jac_obj(Kv, Kg, residual)[0]

    slope = candidate_gradient(omega)
    derivative = torch.autograd.grad(slope, omega)[0]

    # Differentiate the fitting loss of an independently constructed enlarged network.
    x = X.clone().requires_grad_(True)
    neuron = ACTIVATIONS[activation][0](x @ omega[:2] + omega[2]).reshape(-1, 1) ** power
    neuron_grad = torch.autograd.grad(neuron.sum(), x, create_graph=True)[0]
    c = X.new_zeros((), requires_grad=True)
    fitting = (
        .4 * (residual[1] + c * neuron).square().sum()
        + 1.3 * (residual[2] + c * neuron_grad).square().sum()
    ) / (2 * len(X))
    torch.testing.assert_close(slope, torch.autograd.grad(fitting, c)[0])

    step = 1e-6
    finite_differences = []
    for i in range(omega.numel()):
        delta = torch.zeros_like(omega)
        delta[i] = step
        plus, minus = omega.detach() + delta, omega.detach() - delta
        finite_differences.append((
            candidate_gradient(plus) - candidate_gradient(minus)
        ) / (2 * step))
    torch.testing.assert_close(derivative, torch.stack(finite_differences), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("activation,actual", [("leaky_relu", [.01, 1.]), ("relu2", [0., 4.])])
def test_warmstart_preserves_the_historical_sphere_derivative_convention(activation, actual):
    problem = SparseModel(resolve_model({"activation": activation}, 1))
    X = torch.tensor([[-1.], [2.]], dtype=torch.float64)
    W, b = X.new_ones((1, 1)), X.new_zeros(1)
    values, gradients = problem.kernel(X, W, b)
    legacy_values, legacy_gradients = _warm_start_kernel(problem, X, W, b)
    torch.testing.assert_close(values, legacy_values, rtol=0, atol=0)
    torch.testing.assert_close(gradients[:, 0], X.new_tensor(actual), rtol=0, atol=0)
    torch.testing.assert_close(legacy_gradients[:, 0], X.new_tensor([0., 1.]), rtol=0, atol=0)
