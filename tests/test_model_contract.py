"""SparseProblem is the network, with exact atoms and native module behavior."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.schema import ExperimentConfig
from src.PDAP import PDAP
from src.problems import SparseProblem


def _atoms(n=4, d=2):
    W = torch.randn(n, d, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(n, dtype=torch.float64)
    c = torch.rand(n, dtype=torch.float64)
    return W, b, c


def test_exact_atom_assignment_preserves_rng_and_module_state():
    atoms = _atoms()
    rng = torch.random.get_rng_state().clone()
    problem = SparseProblem(2)
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
    problem = SparseProblem(2)
    problem.set_atoms(*_atoms())

    value, gradient = problem.forward_with_gradient(x)

    assert value.shape == (6, 1)
    assert gradient.shape == (6, 2)


def test_empty_support_predicts_and_prepares_zero_features():
    problem = SparseProblem(2)
    x = torch.ones(3, 2, dtype=torch.float64, requires_grad=True)

    value, gradient = problem.forward_with_gradient(x)
    assert problem.n_neurons == 0
    assert problem.inner_weight.shape == (0, 2)
    assert problem.inner_bias.shape == problem.c.shape == (0,)
    assert value.shape == (3, 1)
    assert gradient.shape == x.shape
    assert not value.any() and not gradient.any()
    torch.testing.assert_close(torch.autograd.grad(problem(x).sum(), x)[0], gradient)
    Phi_v, Phi_g = problem.jac_forward_with_gradient(x)
    assert Phi_v.shape == (3, 0)
    assert Phi_g.shape == (6, 0)


def test_module_conversion_and_forward_gradients_follow_exact_atoms():
    problem = SparseProblem(2)
    assert problem.c.dtype == problem.inner_weight.dtype == problem.inner_bias.dtype == torch.float64
    problem.float()
    problem.set_atoms(torch.tensor([[2., -1.]]), torch.tensor([.5]), torch.tensor([3.]))
    assert problem.c.dtype == problem.inner_weight.dtype == problem.inner_bias.dtype == torch.float32
    x = torch.tensor([[1., 1.], [-1., 1.]], requires_grad=True)

    value = problem(x)
    torch.testing.assert_close(value, x.new_tensor([[4.5], [0.]]))
    dc, dx = torch.autograd.grad(value.sum(), (problem.c, x))
    torch.testing.assert_close(dc, x.new_tensor([1.5]))
    torch.testing.assert_close(dx, x.new_tensor([[6., -3.], [0., 0.]]))
    predicted_value, predicted_gradient = problem.forward_with_gradient(x)
    torch.testing.assert_close(predicted_value, value)
    torch.testing.assert_close(predicted_gradient, dx)


def test_setting_and_pruning_atoms_preserves_float64_coefficients():
    problem = SparseProblem(2)
    W, b, _ = _atoms(n=2)
    c = torch.tensor([1.123456789012345, -1.0000000001], dtype=torch.float64)
    problem.set_atoms(W, b, c)
    trainer = PDAP(ExperimentConfig().training)
    assert trainer._prune(problem, amp_tol=1e-8) == 0
    for actual, expected in zip((problem.inner_weight, problem.inner_bias, problem.c.detach()), (W, b, c)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float64 for parameter in problem.parameters())


def test_historical_prediction_power_is_rejected_before_training_reads_data():
    class UnavailableSamples(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise AssertionError("unsupported correction must be rejected before sample access")

    problem = SparseProblem(2, activation=torch.relu, power=4.)
    problem.set_atoms(torch.tensor([[1., 0.]]), torch.tensor([0.]), torch.tensor([2.]))
    x = torch.tensor([[2., 3.]], dtype=torch.float64)
    value, gradient = problem.forward_with_gradient(x)
    torch.testing.assert_close(value, x.new_tensor([[32.]]))
    torch.testing.assert_close(gradient, x.new_tensor([[64., 0.]]))
    before = {name: tensor.clone() for name, tensor in problem.state_dict().items()}
    loader = DataLoader(UnavailableSamples(), generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="supports activation powers 1, 2, 3"):
        PDAP(ExperimentConfig().training).fit(
            problem, loader, num_iterations=1, num_insertion=2, verbose=False,
        )
    for name, tensor in problem.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
