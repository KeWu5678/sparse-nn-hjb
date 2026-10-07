"""Full-data mathematics must be independent of batch delivery."""

import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    IterableDataset,
    SequentialSampler,
    TensorDataset,
)

from src.config.schema import TrainingConfig
from src.data import full_batch, make_dataloader
from src.PDAP import PDAP, insertion
from src.PDAP.warmstart import warm_start
from src.problems import SparseProblem


def _linear_case(**specifications):
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    data = x, x.new_tensor([[1.], [2.], [4.]]), x.new_tensor([[1., 2.]]).repeat(3, 1)
    problem = SparseProblem(2, **specifications)
    problem.set_atoms(torch.eye(2, dtype=x.dtype), x.new_full((2,), .5), x.new_tensor([1., 2.]))
    return data, problem


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_prepared_objective_matches_worked_value_gradient_and_hessian(batch_size):
    data, problem = _linear_case(alpha=.1)
    prepared = problem.prepare_coefficients(make_dataloader(data, batch_size=batch_size))
    c = problem.c.detach()

    # Residuals are (1.5, 1.5, .5), and the target gradients are exact.
    assert float(prepared.obj(c)) == pytest.approx(19 / 24)
    assert float(problem.loss(c, prepared)) == pytest.approx(19 / 24 + .3)
    torch.testing.assert_close(torch.func.grad(prepared.obj)(c), c.new_tensor([1.25, 1.25]))
    expected_hessian = c.new_tensor([[31 / 12, 5 / 4], [5 / 4, 31 / 12]])
    torch.testing.assert_close(prepared.hess_obj, expected_hessian)
    torch.testing.assert_close(torch.func.hessian(prepared.obj)(c), expected_hessian)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_full_data_metrics_weight_samples_and_add_penalty_once(batch_size):
    data, problem = _linear_case(alpha=.1)
    loader = make_dataloader(data, batch_size=batch_size)
    metrics = problem.metrics(loader)
    for split in ("train", "val"):
        assert metrics[f"{split}_loss"] == pytest.approx(19 / 24 + .3)
        assert metrics[f"err_l2_{split}"] == pytest.approx((19 / 84) ** .5)
        assert metrics[f"err_grad_{split}"] == 0.
        assert metrics[f"err_h1_{split}"] == pytest.approx((19 / 144) ** .5)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_candidate_profile_combines_signed_batches_before_acceptance(monkeypatch, batch_size):
    x = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    data = x, x.new_tensor([[-1.], [3.], [-2.]]), torch.zeros_like(x)
    problem = SparseProblem(input_dim=1, alpha=.05)
    residuals = problem.residual(make_dataloader(data, batch_size=batch_size))
    a, b = x.new_tensor([1.], requires_grad=True), x.new_tensor(.25, requires_grad=True)

    profile = problem.profile(a, b, residuals)
    # The signed value contributions are 1.25, -6.75, 6.5; derivative residuals vanish.
    assert float(profile.detach()) == pytest.approx(1 / 3)
    grad_a, grad_b = torch.autograd.grad(profile, (a, b))
    torch.testing.assert_close(grad_a, a.new_tensor([1 / 3]))
    torch.testing.assert_close(grad_b, b.new_tensor(0.))
    monkeypatch.setattr(insertion, "_generate_candidates", lambda *args, **kwargs: (
        a.detach().reshape(1, 1), b.detach().reshape(1), 1, 0,
    ))
    W, bias, coefficient = insertion.profile_threshold(
        problem, residuals, sample_sphere=lambda _: (a, b), N=1,
        insert_init="guaranteed", verbose=False,
    )
    torch.testing.assert_close(torch.as_tensor(W[0]), a)
    assert bias[0] == .25
    assert coefficient[0] == pytest.approx(-68 / 1615)
    W, _, _ = insertion.profile_threshold(
        SparseProblem(1, alpha=.34), residuals, sample_sphere=lambda _: (a, b), N=1,
        insert_init="guaranteed", verbose=False,
    )
    assert len(W) == 0


@pytest.mark.parametrize("activation,power", [("relu", 1.), ("relu", 2.), ("relu", 3.), ("softplus", 1.)])
def test_chunking_preserves_correction_candidate_and_warmstart_calculations(activation, power):
    x = torch.tensor([[-1., .2], [.3, -.4], [.8, .9], [-.2, .7], [.5, -.8]], dtype=torch.float64)
    data = x, x.square().sum(dim=1, keepdim=True), 2 * x
    W = x.new_tensor([[.5, -.3], [.1, .7]])
    b, c = x.new_tensor([.2, .3]), x.new_tensor([-.6, .8])
    problem = SparseProblem(2, activation=activation, power=power, alpha=.01, loss_weights=(.4, 1.3))
    problem.set_atoms(W, b, c)
    full, chunked = make_dataloader(data), make_dataloader(data, batch_size=2)
    reference = problem.prepare_coefficients(full)
    prepared = problem.prepare_coefficients(chunked)
    u = problem.coefficient_coordinates(c, reference)
    torch.testing.assert_close(prepared.obj(u), reference.obj(u))
    torch.testing.assert_close(prepared.hess_obj, reference.hess_obj)
    torch.testing.assert_close(torch.func.grad(lambda v: problem.loss(v, prepared))(u),
        torch.func.grad(lambda v: problem.loss(v, reference))(u))
    assert problem.metrics(chunked) == pytest.approx(problem.metrics(full))

    full_residual, chunked_residual = problem.residual(full), problem.residual(chunked)
    a = x.new_tensor([.2, -.3], requires_grad=True)
    bias = x.new_tensor(.1, requires_grad=True)
    reference_score = problem.profile(a, bias, full_residual).abs() / problem.coefficient_scale(a, bias)
    score = problem.profile(a, bias, chunked_residual).abs() / problem.coefficient_scale(a, bias)
    torch.testing.assert_close(score, reference_score)
    for got, expected in zip(torch.autograd.grad(score, (a, bias)), torch.autograd.grad(reference_score, (a, bias))):
        torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        warm_start(W, b, chunked_residual, problem),
        warm_start(W, b, full_residual, problem),
    )
    assert problem.certificate_radius(chunked_residual) == pytest.approx(problem.certificate_radius(full_residual))


def test_prepared_objective_survives_later_data_and_support_changes():
    data, problem = _linear_case(alpha=.1)
    loader = make_dataloader(data)
    prepared = problem.prepare_coefficients(loader)
    c = problem.c.detach()
    expected_loss, expected_hessian = problem.loss(c, prepared).clone(), prepared.hess_obj.clone()
    data[1].add_(10)
    problem.set_atoms(data[0].new_tensor([[2., -1.]]), c.new_tensor([.1]), c.new_tensor([.5]))
    later = problem.prepare_coefficients(loader)
    assert later.hess_obj.shape == (1, 1)
    torch.testing.assert_close(problem.loss(c, prepared), expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(prepared.hess_obj, expected_hessian, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [None, 2])
def test_pdap_records_one_full_data_result_per_outer_iteration(batch_size):
    torch.manual_seed(41)
    x = torch.tensor([[.2], [.4], [.7], [1.], [1.3]], dtype=torch.float64)
    data = x, x.square(), 2 * x
    problem = SparseProblem(1, alpha=1e-4)
    loader = make_dataloader(data, batch_size=batch_size)
    training = TrainingConfig(fit_outer_iterations=2, lbfgs_steps=5)
    history = PDAP(training).fit(problem, loader, num_iterations=2, num_insertion=4, verbose=False)
    assert len(history.train_loss) == 2
    for iteration, loss in enumerate(history.train_loss):
        history.restore_model(problem, iteration)
        value, gradient = problem.forward_with_gradient(x)
        fitting = ((value - data[1]).square().sum() + (gradient - data[2]).square().sum()) / 10
        penalty = 1e-4 * problem.c.detach().abs().sum()
        assert loss == pytest.approx(float(fitting + penalty), rel=1e-12)


@pytest.mark.parametrize("verbose", [False, True])
def test_logging_accepts_a_finite_loader_without_a_sample_count(verbose):
    class Samples(IterableDataset):
        def __iter__(self):
            x = torch.zeros(2, 1, dtype=torch.float64)
            return iter(zip(x, x, x))

    loader = DataLoader(Samples(), batch_size=2, generator=torch.Generator().manual_seed(0))
    history = PDAP(TrainingConfig()).fit(
        SparseProblem(1), loader, num_iterations=0, num_insertion=1, verbose=verbose,
    )
    assert history.train_loss == [0.]


@pytest.mark.parametrize("drop_last", [False, True])
def test_native_batch_sampler_keeps_full_data_evaluation_and_no_drop_contract(drop_last):
    data, problem = _linear_case(alpha=.1)
    dataset = TensorDataset(*data)
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(SequentialSampler(dataset), 2, drop_last),
        generator=torch.Generator().manual_seed(17),
    )
    rng = torch.get_rng_state().clone()
    if drop_last:
        with pytest.raises(ValueError, match="without dropping samples"):
            problem.prepare_coefficients(loader)
    else:
        for actual, expected in zip(full_batch(loader), data):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        prepared = problem.prepare_coefficients(loader)
        reference = problem.prepare_coefficients(make_dataloader(data))
        torch.testing.assert_close(prepared.obj(problem.c), reference.obj(problem.c))
        torch.testing.assert_close(prepared.hess_obj, reference.hess_obj)
        assert problem.metrics(loader) == pytest.approx(problem.metrics(make_dataloader(data)))
    assert torch.equal(torch.get_rng_state(), rng)
