"""The data-owned problem contract used by both PDAP and SSN."""

import gc
import weakref

import pytest
import torch

from src.config.activations import get_activation
from src.models import SignedModel
from src.PDAP.ssn_solve import SolverConfig, ssn_solve
from src.problems import SparseProblem


def _samples():
    X = torch.tensor([[-1.0, 0.2], [0.3, -0.4], [0.8, 0.9]], dtype=torch.float64)
    V = X[:, :1].square() + 0.1
    dV = torch.cat((2 * X[:, :1], torch.zeros_like(X[:, :1])), dim=1)
    return X, V, dV


def _model(activation="relu", power=1.0):
    model = SignedModel(activation=get_activation(activation), power=power, verbose=False)
    model.set_atoms(
        torch.tensor([[0.5, -0.3], [0.1, 0.7]], dtype=torch.float64),
        torch.tensor([0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.6, 0.8], dtype=torch.float64),
    )
    return model


@pytest.mark.parametrize(
    "power",
    [-1.0, 0.0, 1.5, 4.0, 1.0 + 5e-13, 2.0 + 5e-13, 3.0 - 5e-13,
     float("nan"), float("inf"), -float("inf")],
)
def test_constructor_rejects_unsupported_powers(power):
    with pytest.raises(ValueError, match="supports activation powers 1, 2, 3"):
        SparseProblem(_samples(), power=power)


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
    problem = SparseProblem(_samples(), power=power, alpha=alpha)
    problem.prepare_coefficients(_model(power=power))
    assert problem.penalty_exponent == exponent
    q = torch.tensor([-value, value], dtype=torch.float64)
    prox = problem.prox(q, inverse_step=1.0)
    jac = problem.jac_prox(q, inverse_step=1.0, prox_result=prox)
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
    model = _model(activation, power)
    problem = SparseProblem(
        data, activation=activation, power=power, alpha=0.1, gamma=gamma, loss_weights=(0.4, 1.3)
    )
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="prepare_coefficients"):
        problem.obj(model.get_atoms()[2])

    problem.prepare_coefficients(model)
    u = problem.coefficient_coordinates(model.get_atoms()[2])
    Vp, dVp = model.predict_tensors(data[0])
    expected_obj = (0.4 * (Vp - data[1]).square().sum() + 1.3 * (dVp - data[2]).square().sum()) / (
        2 * len(data[0])
    )
    torch.testing.assert_close(problem.obj(u), expected_obj)
    torch.testing.assert_close(problem.hess_obj(), torch.func.hessian(problem.obj)(u))
    torch.testing.assert_close(problem.model_coefficients(u), model.get_atoms()[2])
    assert float(problem.loss(u)) == pytest.approx(problem.model_loss(model), rel=1e-12)

    jac_loss = torch.func.grad(problem.loss)(u)
    torch.testing.assert_close(
        problem.jac_obj_augmented(u, jac_loss), torch.func.grad(problem.obj_augmented)(u)
    )
    torch.testing.assert_close(
        problem.hess_obj_augmented(u), torch.func.hessian(problem.obj_augmented)(u)
    )
    if power == 1.0:
        torch.testing.assert_close(problem.loss(u), problem.obj_augmented(u) + 0.1 * u.abs().sum())
    else:
        torch.testing.assert_close(problem.obj_augmented(u), problem.obj(u), rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_one_problem_handles_empty_models_and_replaced_supports(monkeypatch):
    # The problem exists before the model, and is reused across all corrections.
    data = _samples()
    problem = SparseProblem(data, activation="softplus", alpha=0.01)
    model = SignedModel(activation=get_activation("softplus"), power=1.0, verbose=False)
    model.input_dim = 2
    residual = problem.residual(model)
    torch.testing.assert_close(residual[0], -data[1])
    torch.testing.assert_close(residual[1], -data[2])
    initial_loss = problem.model_loss(model)
    assert initial_loss == problem.metrics(model)["train_loss"]
    a = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    b = a.new_tensor(0.1, requires_grad=True)
    score = problem.search_score(a, b, residual)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(score, (a, b)))

    import importlib

    integration = importlib.import_module("src.PDAP.ssn_solve")
    SSN = integration.SSN
    optimizers = []

    def record_optimizer(params, supplied_problem, **options):
        assert supplied_problem is problem
        optimizer = SSN(params, supplied_problem, **options)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(integration, "SSN", record_optimizer)
    for width in (2, 1, 3):
        W = torch.arange(1, 2 * width + 1, dtype=torch.float64).reshape(width, 2) / 5
        model.set_atoms(W, torch.zeros(width, dtype=W.dtype), torch.ones(width, dtype=W.dtype))
        # Full-model evaluation uses the current support even before cache refresh.
        assert problem.model_loss(model) == problem.metrics(model)["train_loss"]
        ssn_solve(model, problem, SolverConfig(), iterations=1)
        u = problem.coefficient_coordinates(model.get_atoms()[2])
        assert problem.hess_obj().shape == (width, width)
        assert float(problem.loss(u)) == pytest.approx(problem.model_loss(model), rel=1e-12)
        assert optimizers[-1].param_groups[0]["params"][0].numel() == width
    assert len({id(optimizer) for optimizer in optimizers}) == 3

    model.set_atoms(W[:0], W.new_zeros(0), W.new_zeros(0))
    assert problem.model_loss(model) == pytest.approx(initial_loss)


def test_preparation_does_not_retain_the_model():
    problem = SparseProblem(_samples())
    model = _model()
    reference = weakref.ref(model)
    problem.prepare_coefficients(model)
    del model
    gc.collect()
    assert reference() is None
    assert torch.isfinite(problem.obj(torch.ones(2, dtype=torch.float64)))
