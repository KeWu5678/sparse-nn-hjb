"""SparseModel is the network, with exact atoms and native module behavior."""

import pytest
import torch

from src.config import get_activation, make_penalty, resolve_model, validate_model_config
from src.config.schema import ExperimentConfig
from src.models import SparseModel
from src.PDAP import PDAP


def _atoms(n=4, d=2):
    W = torch.randn(n, d, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(n, dtype=torch.float64)
    c = torch.rand(n, dtype=torch.float64)
    return W, b, c


def test_exact_atom_assignment_preserves_rng_and_module_state():
    atoms = _atoms()
    rng = torch.random.get_rng_state().clone()
    problem = SparseModel(resolve_model({}, 2))
    problem.set_atoms(*atoms)

    assert isinstance(problem, torch.nn.Module)
    assert set(dict(problem.named_parameters())) == {"c"}
    assert set(dict(problem.named_buffers())) == {"inner_weight", "inner_bias"}
    assert set(problem.state_dict()) == {"inner_weight", "inner_bias", "c"}
    assert problem.n_neurons == 4
    for actual, expected in zip((problem.inner_weight, problem.inner_bias, problem.c.detach()), atoms):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    problem.set_atoms(*(atom[:2] for atom in atoms))
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    atoms[0].zero_()
    assert problem.inner_weight.any()


def test_problem_predicts_values_and_input_gradients():
    x = torch.randn(6, 2, dtype=torch.float64)
    problem = SparseModel(resolve_model({}, 2))
    problem.set_atoms(*_atoms())

    value, gradient = problem(x)

    assert value.shape == (6, 1)
    assert gradient.shape == (6, 2)
    assert not x.requires_grad


@pytest.mark.parametrize("activation", ["relu", "relu2", "softplus"])
def test_registered_callable_preserves_named_prediction_and_training(activation):
    named_config = resolve_model({"activation": activation, "moment_order": 2.}, 2)
    direct_config = resolve_model({"activation": get_activation(activation), "moment_order": 2.}, 2)
    validate_model_config(named_config)
    validate_model_config(direct_config)
    named, direct = SparseModel(named_config), SparseModel(direct_config)
    atoms = _atoms()
    x = torch.tensor([[-1., 2.], [3., -4.]], dtype=torch.float64)
    for problem in (named, direct):
        problem.set_atoms(*atoms)
    assert direct.activation_name == named.activation_name == activation
    assert direct.use_sphere == named.use_sphere
    assert direct.moment_order == named.moment_order
    for actual, expected in zip(direct(x), named(x)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gradient_only_loss_backpropagates_to_coefficients_and_input_graph():
    problem = SparseModel(resolve_model({"power": 2., "loss_weights": (0., 1.)}, 2))
    problem.set_atoms(
        torch.tensor([[2., -1.]], dtype=torch.float64),
        torch.tensor([.5], dtype=torch.float64),
        torch.tensor([3.], dtype=torch.float64),
    )
    base = torch.tensor([[.5, .5]], dtype=torch.float64, requires_grad=True)
    x = 2 * base

    value, gradient = problem(x)
    loss = problem.obj(x, (torch.zeros_like(value), torch.zeros_like(gradient)))
    dc, dx = torch.autograd.grad(loss, (problem.c, base))

    torch.testing.assert_close(value, x.new_tensor([[6.75]]))
    torch.testing.assert_close(gradient, x.new_tensor([[18., -9.]]))
    torch.testing.assert_close(loss, x.new_tensor(202.5))
    torch.testing.assert_close(dc, x.new_tensor([135.]))
    torch.testing.assert_close(dx, x.new_tensor([[1080., -540.]]))

    with torch.no_grad():
        reported_value, reported_gradient = problem(x)
    torch.testing.assert_close(reported_value, value)
    torch.testing.assert_close(reported_gradient, gradient)
    assert not reported_value.requires_grad and not reported_gradient.requires_grad


def test_empty_support_predicts_and_prepares_zero_features():
    problem = SparseModel(resolve_model({}, 2))
    x = torch.ones(3, 2, dtype=torch.float64, requires_grad=True)

    value, gradient = problem(x)
    assert problem.n_neurons == 0
    assert problem.inner_weight.shape == (0, 2)
    assert problem.inner_bias.shape == problem.c.shape == (0,)
    assert value.shape == (3, 1)
    assert gradient.shape == x.shape
    assert not value.any() and not gradient.any()
    dc, dx = torch.autograd.grad(gradient.square().sum(), (problem.c, x))
    assert dc.shape == (0,)
    torch.testing.assert_close(dx, gradient)
    with torch.no_grad():
        empty_value, empty_gradient = problem(x)
    torch.testing.assert_close(empty_value, value)
    torch.testing.assert_close(empty_gradient, gradient)
    assert not empty_value.requires_grad and not empty_gradient.requires_grad
    Phi_v, Phi_g = problem.kernel(x)
    assert Phi_v.shape == (3, 0)
    assert Phi_g.shape == (6, 0)


def test_module_conversion_and_forward_gradients_follow_exact_atoms():
    problem = SparseModel(resolve_model({}, 2))
    assert problem.c.dtype == problem.inner_weight.dtype == problem.inner_bias.dtype == torch.float64
    problem.float()
    problem.set_atoms(torch.tensor([[2., -1.]]), torch.tensor([.5]), torch.tensor([3.]))
    assert problem.c.dtype == problem.inner_weight.dtype == problem.inner_bias.dtype == torch.float32
    x = torch.tensor([[1., 1.], [-1., 1.]], requires_grad=True)

    value, gradient = problem(x)
    torch.testing.assert_close(value, x.new_tensor([[4.5], [0.]]))
    dc, dx = torch.autograd.grad(value.sum(), (problem.c, x))
    torch.testing.assert_close(dc, x.new_tensor([1.5]))
    torch.testing.assert_close(dx, x.new_tensor([[6., -3.], [0., 0.]]))
    torch.testing.assert_close(gradient, dx)


def test_setting_and_pruning_atoms_preserves_float64_coefficients():
    config = resolve_model({}, 2)
    problem = SparseModel(config)
    W, b, _ = _atoms(n=2)
    c = torch.tensor([1.123456789012345, -1.0000000001], dtype=torch.float64)
    problem.set_atoms(W, b, c)
    trainer = PDAP(ExperimentConfig().training, make_penalty(config))
    assert trainer._prune(problem, amp_tol=1e-8) == 0
    for actual, expected in zip((problem.inner_weight, problem.inner_bias, problem.c.detach()), (W, b, c)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float64 for parameter in problem.parameters())


@pytest.mark.parametrize("specifications,message", [
    ({"activation": torch.relu, "power": 4.}, "supports activation powers 1, 2, 3"),
    ({"activation": "softplus"}, "explicit moment_order"),
    ({"activation": "softplus", "moment_order": 1.}, "moment_order must exceed"),
    ({"activation": lambda x: x.square()}, "requires a registered activation"),
])
def test_prediction_only_settings_remain_usable_without_training_validation(specifications, message):
    config = resolve_model(specifications, 2)
    problem = SparseModel(config)
    problem.set_atoms(
        torch.tensor([[1., 0.]], dtype=torch.float64),
        torch.tensor([0.], dtype=torch.float64),
        torch.tensor([2.], dtype=torch.float64),
    )
    x = torch.tensor([[2., 3.]], dtype=torch.float64)
    value, gradient = problem(x)
    if specifications["activation"] == "softplus":
        torch.testing.assert_close(value, 2 * torch.nn.functional.softplus(x[:, :1]))
        torch.testing.assert_close(gradient, torch.cat((2 * torch.sigmoid(x[:, :1]), x.new_zeros((1, 1))), dim=1))
    elif specifications["activation"] is torch.relu:
        torch.testing.assert_close(value, x.new_tensor([[32.]]))
        torch.testing.assert_close(gradient, x.new_tensor([[64., 0.]]))
    else:
        torch.testing.assert_close(value, x.new_tensor([[8.]]))
        torch.testing.assert_close(gradient, x.new_tensor([[8., 0.]]))
    before = {name: tensor.clone() for name, tensor in problem.state_dict().items()}
    with pytest.raises(ValueError, match=message):
        validate_model_config(config)
    for name, tensor in problem.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
