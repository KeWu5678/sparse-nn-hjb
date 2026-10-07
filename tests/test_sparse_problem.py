"""External samples and temporary coefficient calculations used by PDAP and SSN."""

import pytest
import torch

from src.config.activations import ACTIVATIONS
from src.data import make_dataloader
from src.eval import metrics
from src.problems import SparseProblem


def _samples():
    X = torch.tensor([[-1.0, 0.2], [0.3, -0.4], [0.8, 0.9]], dtype=torch.float64)
    V = X[:, :1].square() + 0.1
    dV = torch.cat((2 * X[:, :1], torch.zeros_like(X[:, :1])), dim=1)
    return X, V, dV


def _problem(activation="relu", power=1.0, **specifications):
    problem = SparseProblem(2, activation=activation, power=power, **specifications)
    problem.set_atoms(
        torch.tensor([[0.5, -0.3], [0.1, 0.7]], dtype=torch.float64),
        torch.tensor([0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.6, 0.8], dtype=torch.float64),
    )
    return problem


def test_obj_is_the_weighted_h1_quadratic():
    problem = SparseProblem(2, loss_weights=(.4, 1.3))
    value = torch.tensor([[2.], [3.]], dtype=torch.float64, requires_grad=True)
    gradient = value.new_tensor([[1., 2.], [2., 1.]], requires_grad=True)
    target = torch.ones_like(value), torch.zeros_like(gradient)

    obj = problem.obj((value, gradient), target)

    torch.testing.assert_close(obj, value.new_tensor(3.75))
    dvalue, dgradient = torch.autograd.grad(obj, (value, gradient))
    torch.testing.assert_close(dvalue, value.new_tensor([[.2], [.4]]))
    torch.testing.assert_close(dgradient, value.new_tensor([[.65, 1.3], [1.3, .65]]))


@pytest.mark.parametrize("alpha", [0., .3])
def test_sparsity_functional_remains_unweighted_and_respects_masks(alpha):
    problem = SparseProblem(
        1, alpha=alpha, penalized_mask=torch.tensor([True, True, False]),
        nonneg_mask=torch.tensor([False, True, False]),
    )
    u = torch.tensor([-2., -3., 100.], dtype=torch.float64)
    torch.testing.assert_close(problem.sparsity_functional(u), u.new_tensor(2.))


@pytest.mark.parametrize(
    "power",
    [-1.0, 0.0, float("nan"), float("inf"), -float("inf")],
)
def test_constructor_rejects_invalid_prediction_powers(power):
    with pytest.raises(ValueError, match="power must be positive and finite"):
        SparseProblem(2, power=power)


@pytest.mark.parametrize("power", [1.5, 4.0, 1.0 + 5e-13, 2.0 + 5e-13, 3.0 - 5e-13])
def test_correction_rejects_unsupported_powers(power):
    problem = SparseProblem(2, power=power)
    with pytest.raises(ValueError, match="supports activation powers 1, 2, 3"):
        problem.prepare_coefficients(make_dataloader(_samples()))


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
    problem = SparseProblem(2, power=power, alpha=alpha)
    assert problem.penalty_exponent == exponent
    q = torch.tensor([-value, value], dtype=torch.float64)
    prox = problem.prox(q, 1.0)
    jac = problem.jac_prox(q, 1.0, prox_result=prox)
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
def test_samples_define_the_objective_and_its_derivatives(activation, power, gamma):
    data = _samples()
    loader = make_dataloader(data)
    problem = _problem(activation, power, alpha=0.1, gamma=gamma, loss_weights=(0.4, 1.3))
    before = {name: tensor.clone() for name, tensor in problem.state_dict().items()}
    prepared = problem.prepare_coefficients(loader)
    u = problem.coefficient_coordinates(problem.c.detach(), prepared)
    Vp, dVp = problem.forward_with_gradient(data[0])
    expected_obj = (0.4 * (Vp - data[1]).square().sum() + 1.3 * (dVp - data[2]).square().sum()) / (
        2 * len(data[0])
    )
    torch.testing.assert_close(problem.obj((Vp, dVp), data[1:]), expected_obj)
    torch.testing.assert_close(prepared.obj(u), expected_obj)
    torch.testing.assert_close(prepared.hess_obj, torch.func.hessian(prepared.obj)(u))
    torch.testing.assert_close(problem.model_coefficients(u, prepared), problem.c.detach())
    loss = lambda u: problem.loss(u, prepared)
    obj_augmented = lambda u: problem.obj_augmented(u, prepared)
    assert float(loss(u)) == pytest.approx(metrics(problem, loader)["train_loss"], rel=1e-12)

    jac_loss = torch.func.grad(loss)(u)
    torch.testing.assert_close(
        problem.jac_obj_augmented(u, jac_loss), torch.func.grad(obj_augmented)(u)
    )
    torch.testing.assert_close(
        problem.hess_obj_augmented(u, prepared), torch.func.hessian(obj_augmented)(u)
    )
    if power == 1.0:
        torch.testing.assert_close(loss(u), obj_augmented(u) + 0.1 * u.abs().sum())
    else:
        torch.testing.assert_close(obj_augmented(u), prepared.obj(u), rtol=0, atol=0)
    for name, tensor in problem.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_one_problem_handles_empty_and_replaced_supports():
    data = _samples()
    loader = make_dataloader(data)
    problem = SparseProblem(2, activation="softplus", alpha=0.01)
    prediction = problem.forward_with_gradient(data[0])
    residual = data[0], *(p - t for p, t in zip(prediction, data[1:]))
    torch.testing.assert_close(residual[1], -data[1])
    torch.testing.assert_close(residual[2], -data[2])
    initial_loss = metrics(problem, loader)["train_loss"]
    a = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    b = a.new_tensor(0.1, requires_grad=True)
    score = problem.profile(a, b, residual).abs() / problem.normalization_weight(a, b)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(score, (a, b)))

    preparations = []
    for width in (0, 2, 1, 3):
        W = torch.arange(1, 2 * width + 1, dtype=torch.float64).reshape(width, 2) / 5
        problem.set_atoms(W, torch.zeros(width, dtype=W.dtype), torch.ones(width, dtype=W.dtype))
        prepared = problem.prepare_coefficients(loader)
        preparations.append(prepared)
        u = problem.coefficient_coordinates(problem.c.detach(), prepared)
        assert prepared.hess_obj.shape == (width, width)
        assert float(problem.loss(u, prepared)) == pytest.approx(metrics(problem, loader)["train_loss"], rel=1e-12)
    assert len({id(prepared) for prepared in preparations}) == 4

    problem.set_atoms(W[:0], W.new_zeros(0), W.new_zeros(0))
    assert metrics(problem, loader)["train_loss"] == pytest.approx(initial_loss)


@pytest.mark.parametrize("activation,power", [("relu", 1.), ("relu", 2.), ("relu", 3.), ("softplus", 1.)])
def test_candidate_profile_matches_fidelity_derivative_and_inner_weight_gradient(activation, power):
    problem = _problem(activation, power, loss_weights=(.4, 1.3))
    X, V, dV = _samples()
    prediction = problem.forward_with_gradient(X)
    residual = X, prediction[0] - V, prediction[1] - dV
    omega = X.new_tensor([.7, -.2, .4], requires_grad=True)
    profile = problem.profile(omega[:2], omega[2], residual)
    derivative = torch.autograd.grad(profile, omega)[0]

    # Differentiate the fitting loss of an independently constructed enlarged network.
    x = X.clone().requires_grad_(True)
    neuron = ACTIVATIONS[activation][0](x @ omega[:2] + omega[2]).reshape(-1, 1) ** power
    neuron_grad = torch.autograd.grad(neuron.sum(), x, create_graph=True)[0]
    c = X.new_zeros((), requires_grad=True)
    fitting = (
        .4 * (residual[1] + c * neuron).square().sum()
        + 1.3 * (residual[2] + c * neuron_grad).square().sum()
    ) / (2 * len(X))
    torch.testing.assert_close(profile, torch.autograd.grad(fitting, c)[0])

    step = 1e-6
    finite_differences = []
    for i in range(omega.numel()):
        delta = torch.zeros_like(omega)
        delta[i] = step
        plus, minus = omega.detach() + delta, omega.detach() - delta
        finite_differences.append((
            problem.profile(plus[:2], plus[2], residual)
            - problem.profile(minus[:2], minus[2], residual)
        ) / (2 * step))
    torch.testing.assert_close(derivative, torch.stack(finite_differences), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("activation,actual", [("leaky_relu", [.01, 1.]), ("relu2", [0., 4.])])
def test_warmstart_preserves_the_historical_sphere_derivative_convention(activation, actual):
    problem = SparseProblem(1, activation=activation)
    X = torch.tensor([[-1.], [2.]], dtype=torch.float64)
    W, b = X.new_ones((1, 1)), X.new_zeros(1)
    values, gradients = problem.jac_forward_with_gradient(X, W, b)
    legacy_values, legacy_gradients = problem.jac_forward_with_gradient(X, W, b, warm_start=True)
    torch.testing.assert_close(values, legacy_values, rtol=0, atol=0)
    torch.testing.assert_close(gradients[:, 0], X.new_tensor(actual), rtol=0, atol=0)
    torch.testing.assert_close(legacy_gradients[:, 0], X.new_tensor([0., 1.]), rtol=0, atol=0)
