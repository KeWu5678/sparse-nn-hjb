"""Full-data mathematics must be independent of batch delivery."""

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.config.activations import get_activation
from src.config.schema import TrainingConfig
from src.data import make_dataloader
from src.models import SignedModel
from src.PDAP import PDAP
from src.problems import SparseProblem


def _linear_case():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    data = x, x.new_tensor([[1.], [2.], [4.]]), x.new_tensor([[1., 2.]]).repeat(3, 1)
    model = SignedModel(activation=torch.relu, power=1., verbose=False)
    model.set_atoms(torch.eye(2, dtype=x.dtype), x.new_full((2,), .5), x.new_tensor([1., 2.]))
    return data, model


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_prepared_objective_matches_worked_value_gradient_and_hessian(batch_size):
    data, model = _linear_case()
    problem = SparseProblem(input_dim=2, alpha=.1)
    prepared = problem.prepare_coefficients(model, make_dataloader(data, batch_size=batch_size))
    c = model.get_atoms()[2]

    # Residuals are (1.5, 1.5, .5), and the target gradients are exact.
    assert float(prepared.obj(c)) == pytest.approx(19 / 24)
    assert float(problem.loss(c, prepared)) == pytest.approx(19 / 24 + .3)
    torch.testing.assert_close(torch.func.grad(prepared.obj)(c), c.new_tensor([1.25, 1.25]))
    expected_hessian = c.new_tensor([[31 / 12, 5 / 4], [5 / 4, 31 / 12]])
    torch.testing.assert_close(prepared.hess_obj, expected_hessian)
    torch.testing.assert_close(torch.func.hessian(prepared.obj)(c), expected_hessian)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_full_data_metrics_weight_samples_and_add_penalty_once(batch_size):
    data, model = _linear_case()
    problem = SparseProblem(input_dim=2, alpha=.1)
    loader = make_dataloader(data, batch_size=batch_size)
    metrics = problem.metrics(model, loader)
    assert problem.model_loss(model, loader) == pytest.approx(19 / 24 + .3)
    for split in ("train", "val"):
        assert metrics[f"{split}_loss"] == pytest.approx(19 / 24 + .3)
        assert metrics[f"err_l2_{split}"] == pytest.approx((19 / 84) ** .5)
        assert metrics[f"err_grad_{split}"] == 0.
        assert metrics[f"err_h1_{split}"] == pytest.approx((19 / 144) ** .5)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_candidate_profile_combines_signed_batches_before_acceptance(batch_size):
    x = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    data = x, x.new_tensor([[-1.], [3.], [-2.]]), torch.zeros_like(x)
    model = SignedModel(activation=torch.relu, power=1., verbose=False)
    model.input_dim = 1
    problem = SparseProblem(input_dim=1, alpha=.05)
    residuals = problem.residual(model, make_dataloader(data, batch_size=batch_size))
    a, b = x.new_tensor([1.], requires_grad=True), x.new_tensor(.25, requires_grad=True)

    profile = problem.profile(a, b, residuals)
    # The signed value contributions are 1.25, -6.75, 6.5; derivative residuals vanish.
    assert float(profile.detach()) == pytest.approx(1 / 3)
    grad_a, grad_b = torch.autograd.grad(profile, (a, b))
    torch.testing.assert_close(grad_a, a.new_tensor([1 / 3]))
    torch.testing.assert_close(grad_b, b.new_tensor(0.))
    score, coefficient = problem.profile_insertion(a.detach(), b.detach(), residuals, guaranteed=True)
    assert score == pytest.approx(17 / 60)
    assert coefficient == pytest.approx(-68 / 1615)


@pytest.mark.parametrize("activation,power", [("relu", 1.), ("relu", 2.), ("relu", 3.), ("softplus", 1.)])
def test_chunking_preserves_correction_candidate_and_warmstart_calculations(activation, power):
    x = torch.tensor([[-1., .2], [.3, -.4], [.8, .9], [-.2, .7], [.5, -.8]], dtype=torch.float64)
    data = x, x.square().sum(dim=1, keepdim=True), 2 * x
    model = SignedModel(activation=get_activation(activation), power=power, verbose=False)
    W = x.new_tensor([[.5, -.3], [.1, .7]])
    b, c = x.new_tensor([.2, .3]), x.new_tensor([-.6, .8])
    model.set_atoms(W, b, c)
    problem = SparseProblem(2, activation=activation, power=power, alpha=.01, loss_weights=(.4, 1.3))
    full, chunked = make_dataloader(data), make_dataloader(data, batch_size=2)
    reference = problem.prepare_coefficients(model, full)
    prepared = problem.prepare_coefficients(model, chunked)
    u = problem.coefficient_coordinates(c, reference)
    torch.testing.assert_close(prepared.obj(u), reference.obj(u))
    torch.testing.assert_close(prepared.hess_obj, reference.hess_obj)
    torch.testing.assert_close(torch.func.grad(lambda v: problem.loss(v, prepared))(u),
        torch.func.grad(lambda v: problem.loss(v, reference))(u))
    assert problem.metrics(model, chunked) == pytest.approx(problem.metrics(model, full))

    full_residual, chunked_residual = problem.residual(model, full), problem.residual(model, chunked)
    a = x.new_tensor([.2, -.3], requires_grad=True)
    bias = x.new_tensor(.1, requires_grad=True)
    reference_score = problem.search_score(a, bias, full_residual)
    score = problem.search_score(a, bias, chunked_residual)
    torch.testing.assert_close(score, reference_score)
    for got, expected in zip(torch.autograd.grad(score, (a, bias)), torch.autograd.grad(reference_score, (a, bias))):
        torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        problem.warmstart_profiles(W, b, chunked_residual),
        problem.warmstart_profiles(W, b, full_residual),
    )
    direction = x.new_tensor([-.2, .4])
    torch.testing.assert_close(
        problem.warmstart_coefficients(W, b, direction, chunked_residual),
        problem.warmstart_coefficients(W, b, direction, full_residual),
    )
    assert problem.certificate_radius(chunked_residual) == pytest.approx(problem.certificate_radius(full_residual))


def test_prepared_objective_survives_later_data_and_support_changes():
    data, model = _linear_case()
    problem = SparseProblem(2, alpha=.1)
    loader = make_dataloader(data)
    prepared = problem.prepare_coefficients(model, loader)
    c = model.get_atoms()[2]
    expected_loss, expected_hessian = problem.loss(c, prepared).clone(), prepared.hess_obj.clone()
    data[1].add_(10)
    model.set_atoms(data[0].new_tensor([[2., -1.]]), c.new_tensor([.1]), c.new_tensor([.5]))
    later = problem.prepare_coefficients(model, loader)
    assert later.hess_obj.shape == (1, 1)
    torch.testing.assert_close(problem.loss(c, prepared), expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(prepared.hess_obj, expected_hessian, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [None, 2])
def test_pdap_records_one_full_data_result_per_outer_iteration(batch_size):
    torch.manual_seed(41)
    x = torch.tensor([[.2], [.4], [.7], [1.], [1.3]], dtype=torch.float64)
    data = x, x.square(), 2 * x
    model = SignedModel(activation=torch.relu, power=1., verbose=False)
    model.input_dim = 1
    problem = SparseProblem(1, alpha=1e-4)
    loader = make_dataloader(data, batch_size=batch_size)
    training = TrainingConfig(fit_outer_iterations=2, lbfgs_steps=5)
    history = PDAP(training).fit(model, problem, loader, num_iterations=2, num_insertion=4, verbose=False)
    assert len(history.train_loss) == 2
    for iteration, loss in enumerate(history.train_loss):
        history.restore_model(model, iteration)
        value, gradient = model.predict_tensors(x)
        fitting = ((value - data[1]).square().sum() + (gradient - data[2]).square().sum()) / 10
        penalty = 1e-4 * model.get_atoms()[2].abs().sum()
        assert loss == pytest.approx(float(fitting + penalty), rel=1e-12)


@pytest.mark.parametrize("verbose", [False, True])
def test_logging_accepts_a_finite_loader_without_a_sample_count(verbose):
    class Samples(IterableDataset):
        def __iter__(self):
            x = torch.zeros(2, 1, dtype=torch.float64)
            return iter(zip(x, x, x))

    loader = DataLoader(Samples(), batch_size=2, generator=torch.Generator().manual_seed(0))
    model = SignedModel(activation=torch.relu, power=1., verbose=False)
    model.input_dim = 1
    history = PDAP(TrainingConfig()).fit(
        model, SparseProblem(1), loader, num_iterations=0, num_insertion=1, verbose=verbose,
    )
    assert history.train_loss == [0.]
