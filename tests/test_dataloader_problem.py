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
from src.eval import metrics
from src.PDAP import PDAP, insertion
from src.PDAP.warmstart import warm_start
from src.problems import SparseProblem


def _linear_case(**specifications):
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    data = x, x.new_tensor([[1.], [2.], [4.]]), x.new_tensor([[1., 2.]]).repeat(3, 1)
    problem = SparseProblem(2, moment_order=2.0, **specifications)
    problem.set_atoms(torch.eye(2, dtype=x.dtype), x.new_full((2,), .5), x.new_tensor([1., 2.]))
    return data, problem


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_prepared_objective_matches_worked_value_gradient_and_hessian(batch_size):
    data, problem = _linear_case(alpha=.1)
    obj, hess_obj, _ = problem.prepare_coefficients(make_dataloader(data, batch_size=batch_size))
    c = problem.c.detach()

    # Residuals are (1.5, 1.5, .5), and the target gradients are exact.
    assert float(obj(c)) == pytest.approx(19 / 24)
    assert float(problem.loss(c, obj)) == pytest.approx(19 / 24 + .3)
    torch.testing.assert_close(torch.func.grad(obj)(c), c.new_tensor([1.25, 1.25]))
    expected_hessian = c.new_tensor([[31 / 12, 5 / 4], [5 / 4, 31 / 12]])
    torch.testing.assert_close(hess_obj, expected_hessian)
    torch.testing.assert_close(torch.func.hessian(obj)(c), expected_hessian)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_full_data_metrics_weight_samples_and_add_penalty_once(batch_size):
    data, problem = _linear_case(alpha=.1)
    loader = make_dataloader(data, batch_size=batch_size)
    values = metrics(problem, loader)
    for split in ("train", "val"):
        assert values[f"{split}_loss"] == pytest.approx(19 / 24 + .3)
        assert values[f"err_l2_{split}"] == pytest.approx((19 / 84) ** .5)
        assert values[f"err_grad_{split}"] == 0.
        assert values[f"err_h1_{split}"] == pytest.approx((19 / 144) ** .5)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_candidate_profile_combines_signed_batches_before_acceptance(monkeypatch, batch_size):
    x = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    data = x, x.new_tensor([[-1.], [3.], [-2.]]), torch.zeros_like(x)
    problem = SparseProblem(input_dim=1, alpha=.05)
    X, V, dV = full_batch(make_dataloader(data, batch_size=batch_size))
    residuals = X, -V, -dV  # The current network is empty.
    a, b = x.new_tensor([1.], requires_grad=True), x.new_tensor(.25, requires_grad=True)

    features = problem.jac_forward(X, a.reshape(1, -1), b.reshape(1))
    profile = problem.jac_obj(*features, residuals)[0]
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
    problem = SparseProblem(
        2, activation=activation, power=power, alpha=.01,
        loss_weights=(.4, 1.3), moment_order=2.0,
    )
    problem.set_atoms(W, b, c)
    full, chunked = make_dataloader(data), make_dataloader(data, batch_size=2)
    reference_obj, reference_hessian, reference_weight = problem.prepare_coefficients(full)
    obj, hess_obj, normalization_weight = problem.prepare_coefficients(chunked)
    u = c * reference_weight
    torch.testing.assert_close(obj(u), reference_obj(u))
    torch.testing.assert_close(hess_obj, reference_hessian)
    torch.testing.assert_close(normalization_weight, reference_weight)
    torch.testing.assert_close(torch.func.grad(lambda v: problem.loss(v, obj))(u),
        torch.func.grad(lambda v: problem.loss(v, reference_obj))(u))
    assert metrics(problem, chunked) == pytest.approx(metrics(problem, full))

    residuals = []
    for loader in (full, chunked):
        X, V, dV = full_batch(loader)
        Vp, dVp = problem(X)
        residuals.append((X.detach(), (Vp - V).detach(), (dVp - dV).detach()))
    full_residual, chunked_residual = residuals
    a = x.new_tensor([.2, -.3], requires_grad=True)
    bias = x.new_tensor(.1, requires_grad=True)
    features = problem.jac_forward(full_residual[0], a.reshape(1, -1), bias.reshape(1))
    reference_score = problem.jac_obj(*features, full_residual)[0].abs() / problem.normalization_weight(a, bias)
    features = problem.jac_forward(chunked_residual[0], a.reshape(1, -1), bias.reshape(1))
    score = problem.jac_obj(*features, chunked_residual)[0].abs() / problem.normalization_weight(a, bias)
    torch.testing.assert_close(score, reference_score)
    for got, expected in zip(torch.autograd.grad(score, (a, bias)), torch.autograd.grad(reference_score, (a, bias))):
        torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        warm_start(W, b, chunked_residual, problem),
        warm_start(W, b, full_residual, problem),
    )
    trainer = PDAP(TrainingConfig(radial_cap="theorem"))
    assert trainer._search_radius(problem, chunked_residual) == pytest.approx(
        trainer._search_radius(problem, full_residual),
    )


@pytest.mark.parametrize("activation", ["relu", "softplus"])
def test_prepared_objective_survives_later_data_and_support_changes(activation):
    data, problem = _linear_case(alpha=.1, activation=activation)
    loader = make_dataloader(data)
    obj, hess_obj, normalization_weight = problem.prepare_coefficients(loader)
    u = problem.c.detach() * normalization_weight
    expected_loss, expected_hessian = problem.loss(u, obj).clone(), hess_obj.clone()
    expected_weight = normalization_weight.clone()
    data[1].add_(10)
    problem.set_atoms(data[0].new_tensor([[2., -1.]]), u.new_tensor([.1]), u.new_tensor([.5]))
    _, later_hessian, _ = problem.prepare_coefficients(loader)
    assert later_hessian.shape == (1, 1)
    torch.testing.assert_close(problem.loss(u, obj), expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(hess_obj, expected_hessian, rtol=0, atol=0)
    torch.testing.assert_close(normalization_weight, expected_weight, rtol=0, atol=0)


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
        with torch.no_grad():
            value, gradient = problem(x)
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
        obj, hess_obj, _ = problem.prepare_coefficients(loader)
        reference_obj, reference_hessian, _ = problem.prepare_coefficients(make_dataloader(data))
        torch.testing.assert_close(obj(problem.c), reference_obj(problem.c))
        torch.testing.assert_close(hess_obj, reference_hessian)
        assert metrics(problem, loader) == pytest.approx(metrics(problem, make_dataloader(data)))
    assert torch.equal(torch.get_rng_state(), rng)
