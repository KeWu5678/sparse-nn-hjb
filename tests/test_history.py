import math
import pickle

import numpy as np
import pytest
import torch
from ssn_test_problem import proximal_for

from src.config import resolve_model, validate_training
from src.data import ValueSampleNormalizer, make_dataloader
from src.eval import metrics as evaluate
from src.eval import relative_errors
from src.models import SparseModel
from src.PDAP.history import History


def test_history_summary_metrics_uses_best_iteration() -> None:
    history = History(
        err_l2_train=[0.4, 0.2],
        err_l2_val=[0.5, 0.25],
        err_grad_train=[0.6, 0.3],
        err_grad_val=[0.7, 0.35],
        err_h1_train=[0.8, 0.4],
        err_h1_val=[0.9, 0.45],
        inner_weights=[
            {"weight": torch.zeros(3, 2), "bias": torch.zeros(3)},
            {"weight": torch.zeros(5, 2), "bias": torch.zeros(5)},
        ],
        best_iteration=1,
        final_neurons=6,
    )

    assert history.summary_metrics() == {
        "rel_l2_train": 0.2,
        "rel_l2_val": 0.25,
        "rel_grad_train": 0.3,
        "rel_grad_val": 0.35,
        "rel_h1_train": 0.4,
        "rel_h1_val": 0.45,
        "best_iteration": 1,
        "best_neurons": 5,
        "final_neurons": 6,
        # This fixture records no losses, so no outer iteration ran.
        "iterations": 0,
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_restoring_a_zero_measure_snapshot_empties_an_initialized_model(legacy) -> None:
    """A model that already carries atoms must not keep them across the restore.

    This is the terminal zero-measure run of ``PDAP.fit``: under strong
    regularization the initial insertion accepts no atom, and the recorded model
    has empty support. Historical zero-measure snapshots had an empty state_dict;
    current snapshots explicitly register empty inner_weight, inner_bias, and c tensors.
    """
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    config = resolve_model({"activation": "tanh", "moment_order": 2.0}, 2)
    empty = SparseModel(config)
    samples = (x, *empty(x))
    history = History()
    history.record(empty, evaluate(empty, proximal_for(config, empty), make_dataloader(samples)))
    if legacy:
        history.model_states[0] = {}

    restored = SparseModel(resolve_model({"activation": "tanh"}, 2))
    restored.set_atoms(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.tensor([3.0, -4.0], dtype=torch.float64),
    )
    assert restored.n_neurons == 2

    history.restore_model(restored)

    assert restored.n_neurons == 0
    value, gradient = restored(x)
    assert torch.allclose(value, torch.zeros_like(value))
    assert torch.allclose(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("state_keys", [
    ("inner_weight", "inner_bias", "c"),
    ("W", "b", "c"),
    ("hidden.weight", "hidden.bias", "output.weight"),
    None,
])
def test_restoring_recorded_state_preserves_predictions(state_keys):
    inner_weight = torch.tensor([[1., 2.]], dtype=torch.float64)
    inner_bias = torch.tensor([.5], dtype=torch.float64)
    c = torch.tensor([[3.]], dtype=torch.float64)
    state = {} if state_keys is None else dict(zip(state_keys, (inner_weight, inner_bias, c.reshape(-1))))
    if state_keys is not None and state_keys[0] == "hidden.weight":
        state["output.weight"] = c
        state["output.bias"] = torch.zeros(1, dtype=c.dtype)
    history = History(
        inner_weights=[{"weight": inner_weight, "bias": inner_bias}], outer_weights=[c],
        model_states=[state],
    )

    problem = history.restore_model(SparseModel(resolve_model({}, 2)))
    value, gradient = problem(inner_weight.new_tensor([[1., 1.]]))

    torch.testing.assert_close(value, inner_weight.new_tensor([[10.5]]))
    torch.testing.assert_close(gradient, inner_weight.new_tensor([[3., 6.]]))
    assert set(problem.state_dict()) == {"inner_weight", "inner_bias", "c"}


def test_recorded_checkpoints_stay_independent_of_models_and_each_other():
    atoms = {
        "inner_weight": torch.tensor([[1., 2.]], dtype=torch.float64),
        "inner_bias": torch.tensor([.5], dtype=torch.float64),
        "c": torch.tensor([3.], dtype=torch.float64),
    }
    config = resolve_model({}, 2)
    problem = SparseModel(config)
    problem.set_atoms(atoms["inner_weight"], atoms["inner_bias"], atoms["c"])
    x = torch.tensor([[1., 1.]], dtype=torch.float64)
    loader = make_dataloader((x, *problem(x)))
    history = History()
    for _ in range(2):
        history.record(problem, evaluate(problem, proximal_for(config, problem), loader))
        with torch.no_grad():
            for tensor in problem.state_dict().values():
                tensor.add_(1.)

    history = pickle.loads(pickle.dumps(history))
    for i, state in enumerate(history.model_states):
        restored = history.restore_model(SparseModel(resolve_model({}, 2)), iteration=i)
        with torch.no_grad():
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, atoms[name] + i, rtol=0, atol=0)
                tensor.zero_()
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, atoms[name] + i, rtol=0, atol=0)
        torch.testing.assert_close(history.inner_weights[i]["weight"], state["inner_weight"])
        torch.testing.assert_close(history.inner_weights[i]["bias"], state["inner_bias"])
        torch.testing.assert_close(history.outer_weights[i], state["c"].reshape(1, -1))


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_history_summary_records_normalized_objective_decomposition(alpha) -> None:
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    c = torch.tensor([50.0, -45.0, 5.0], dtype=torch.float64)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    config = resolve_model({"activation": "tanh", "alpha": alpha, "gamma": 0.0,
                            "loss_weights": (1.0, 1.0), "moment_order": 2.0}, 2)
    problem = SparseModel(config)
    proximal = proximal_for(config, problem)
    problem.set_atoms(W, b, c)
    samples = (x, *problem(x))

    history = History()
    history.record(problem, evaluate(problem, proximal, make_dataloader(samples)))
    metrics = history.summary_metrics()

    # The full moment is 50*(1+1) + 45*(1+4) + 5*(1+25) = 455.
    assert math.isclose(metrics["phi_1"], 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["psi_p"], 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["alpha_phi_1"], alpha * 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["radius_r95"], 2.0, rel_tol=1e-12)
    assert math.isclose(metrics["radius_max"], 5.0, rel_tol=1e-12)
    assert math.isclose(metrics["total_variation"], 100.0, rel_tol=1e-12)

    # Targets equal predictions, so both fidelity channels vanish and the
    # complete objective is exactly the normalized-measure penalty.
    assert metrics["data_value_term_train"] == 0.0
    assert metrics["data_gradient_term_train"] == 0.0
    assert math.isclose(metrics["objective_train"], alpha * 455.0, rel_tol=1e-12)
    assert math.isclose(
        metrics["objective_train"],
        metrics["data_value_term_train"]
        + metrics["data_gradient_term_train"]
        + metrics["alpha_phi_1"],
        rel_tol=1e-12,
    )


@pytest.mark.parametrize("power", [1., 2., 3.])
def test_sphere_penalty_and_reporting_do_not_depend_on_historical_moment_order(power):
    x = torch.tensor([[.2, -.3], [.5, .7]], dtype=torch.float64)
    inner_weight, inner_bias, c = torch.eye(2, dtype=x.dtype), x.new_zeros(2), x.new_tensor([-2., 3.])
    expected_penalty = .1 * c.abs().pow(2 / (power + 1)).sum()
    for order in (None, 2., 4.):
        config = resolve_model({"power": power, "alpha": .1, "moment_order": order}, 2)
        validate_training(config)
        problem = SparseModel(config)
        problem.set_atoms(inner_weight, inner_bias, c)
        assert problem.moment_order is None
        torch.testing.assert_close(problem.moment_weight_of_atoms(), torch.ones_like(c))
        history = History()
        history.record(problem, evaluate(problem, proximal_for(config, problem), make_dataloader((x, *problem(x)))))
        history = pickle.loads(pickle.dumps(history))
        summary = history.summary_metrics()
        assert history.psi_p == [None] and summary["psi_p"] is None
        assert summary["total_variation"] == 5.
        assert summary["objective_train"] == pytest.approx(float(expected_penalty), rel=1e-12)
        # Existing archives keep their recorded numeric diagnostic unchanged.
        history.psi_p[0] = 10.
        assert pickle.loads(pickle.dumps(history)).summary_metrics()["psi_p"] == 10.


def test_physical_reporting_does_not_change_objectives_snapshots_or_selection():
    normalizer = ValueSampleNormalizer(np.array([2.0, 8.0]), 5.0)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)
    v = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    dv = torch.tensor([[0.7, -0.6], [1.2, 0.5]], dtype=torch.float64)
    data = x, v, dv
    config = resolve_model({
        "activation": "tanh", "alpha": 0.1, "gamma": 0.2, "moment_order": 2.0,
    }, 2)
    problem = SparseModel(config)
    proximal = proximal_for(config, problem)
    loader = make_dataloader(data)
    unchanged = History()
    # Old pickles predate the optional reporting transform.
    del unchanged.reporting_normalizer
    unchanged = pickle.loads(pickle.dumps(unchanged))
    physical = History(reporting_normalizer=normalizer)
    for coefficient in (1.0, 2.0):
        problem.set_atoms(
            torch.tensor([[1.0, 0.5]], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([coefficient], dtype=torch.float64),
        )
        unchanged.record(problem, evaluate(problem, proximal, loader))
        physical.record(problem, evaluate(problem, proximal, loader, reporting_normalizer=normalizer))
        expected = relative_errors(
            *normalizer.denormalize_tensors(*problem(x)),
            *normalizer.denormalize_tensors(v, dv),
        )
        assert (physical.err_l2_val[-1], physical.err_grad_val[-1], physical.err_h1_val[-1]) == expected

    assert physical.err_grad_val != pytest.approx(unchanged.err_grad_val)
    assert physical.err_h1_val != pytest.approx(unchanged.err_h1_val)
    for name in (
        "train_loss", "val_loss", "data_loss_train", "data_loss_val",
        "value_loss_train", "value_loss_val", "gradient_loss_train", "gradient_loss_val",
        "sparsity_functional", "psi_p", "alpha_phi", "total_variation", "radius_r95", "radius_max",
        "best_iteration", "best_train_loss", "best_neurons",
    ):
        assert getattr(physical, name) == getattr(unchanged, name)
    assert physical.best_iteration == int(np.argmin(physical.train_loss))
    for expected_state, state in zip(unchanged.model_states, physical.model_states):
        assert state.keys() == expected_state.keys()
        for name in state:
            assert torch.equal(state[name], expected_state[name])
    for expected_weights, weights in zip(unchanged.outer_weights, physical.outer_weights):
        assert torch.equal(weights, expected_weights)
    for expected_weights, weights in zip(unchanged.inner_weights, physical.inner_weights):
        assert torch.equal(weights["weight"], expected_weights["weight"])
        assert torch.equal(weights["bias"], expected_weights["bias"])
