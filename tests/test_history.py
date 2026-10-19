import io
import math

import numpy as np
import pytest
import torch

from src.config import make_penalty, resolve_model, validate_model_config
from src.data import ValueSampleNormalizer, make_dataloader
from src.eval import evaluate_checkpoint, relative_errors
from src.models import SparseNetwork
from src.PDAP.history import History


def _round_trip(history: History) -> History:
    """Save and load a History the way ``src.tracking`` and ``src.results`` do."""
    buffer = io.BytesIO()
    torch.save(history.to_dict(), buffer)
    buffer.seek(0)
    return History.from_dict(torch.load(buffer, weights_only=True))


def test_summary_is_the_selected_row_with_run_totals() -> None:
    history = History(
        rows=[{"objective_train": 2.0, "rel_h1_val": 0.9, "neurons": 3},
              {"objective_train": 1.0, "rel_h1_val": 0.45, "neurons": 5}],
        diagnostics=[{}, {"cap_of_theorem_applied": True}],
        run={"final_neurons": 6, "solver_seconds": 1.5},
    )

    assert history.summary() == {
        "objective_train": 1.0, "rel_h1_val": 0.45, "neurons": 5, "best_iteration": 1,
        "final_neurons": 6, "solver_seconds": 1.5, "cap_of_theorem_applied": 1.0,
    }


@pytest.mark.parametrize(("objective", "best"), [
    ([3.0, 1.0, 1.0, 2.0], 1),
    ([math.nan, 2.0, 1.5], 2),
    ([math.nan, math.inf], 0),
])
def test_selection_is_the_first_least_finite_objective(objective, best) -> None:
    history = History(rows=[{"objective_train": value} for value in objective])

    assert history.best_iteration == best


def test_restoring_a_zero_measure_snapshot_empties_an_initialized_network() -> None:
    """A network that already carries atoms must not keep them across the restore.

    This is the terminal zero-measure run of ``PDAP.fit``: under strong
    regularization the initial insertion accepts no atom, and the recorded network
    has empty support.
    """
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    config = resolve_model({"activation": "tanh", "moment_order": 2.0}, 2)
    empty = SparseNetwork(config)
    samples = (x, *empty(x))
    history = History()
    history.record(empty, evaluate_checkpoint(empty, make_penalty(config), make_dataloader(samples)))

    restored = SparseNetwork(resolve_model({"activation": "tanh"}, 2))
    restored.set_atoms(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.tensor([3.0, -4.0], dtype=torch.float64),
    )
    assert restored.n_neurons == 2

    _round_trip(history).restore(restored)

    assert restored.n_neurons == 0
    value, gradient = restored(x)
    assert torch.allclose(value, torch.zeros_like(value))
    assert torch.allclose(gradient, torch.zeros_like(gradient))


def test_recorded_checkpoints_stay_independent_of_models_and_each_other():
    atoms = {
        "inner_weight": torch.tensor([[1., 2.]], dtype=torch.float64),
        "inner_bias": torch.tensor([.5], dtype=torch.float64),
        "c": torch.tensor([3.], dtype=torch.float64),
    }
    config = resolve_model({}, 2)
    network = SparseNetwork(config)
    network.set_atoms(atoms["inner_weight"], atoms["inner_bias"], atoms["c"])
    x = torch.tensor([[1., 1.]], dtype=torch.float64)
    loader = make_dataloader((x, *network(x)))
    history = History()
    for _ in range(2):
        history.record(network, evaluate_checkpoint(network, make_penalty(config), loader))
        with torch.no_grad():
            for tensor in network.state_dict().values():
                tensor.add_(1.)

    history = _round_trip(history)
    for i, state in enumerate(history.states):
        restored = history.restore(SparseNetwork(resolve_model({}, 2)), iteration=i)
        with torch.no_grad():
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, atoms[name] + i, rtol=0, atol=0)
                tensor.zero_()
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, atoms[name] + i, rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_summary_records_the_normalized_objective_decomposition(alpha) -> None:
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    c = torch.tensor([50.0, -45.0, 5.0], dtype=torch.float64)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    config = resolve_model({"activation": "tanh", "alpha": alpha, "gamma": 0.0,
                            "loss_weights": (1.0, 1.0), "moment_order": 2.0}, 2)
    network = SparseNetwork(config)
    penalty = make_penalty(config)
    network.set_atoms(W, b, c)
    samples = (x, *network(x))

    history = History()
    history.record(network, evaluate_checkpoint(network, penalty, make_dataloader(samples)))
    metrics = history.summary()

    # The full moment is 50*(1+1) + 45*(1+4) + 5*(1+25) = 455.
    assert math.isclose(metrics["sparsity_functional"], 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["alpha_phi"], alpha * 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["radius_r95"], 2.0, rel_tol=1e-12)
    assert math.isclose(metrics["radius_max"], 5.0, rel_tol=1e-12)
    assert math.isclose(metrics["total_variation"], 100.0, rel_tol=1e-12)
    assert metrics["neurons"] == 3

    # Targets equal predictions, so both fidelity channels vanish and the
    # complete objective is exactly the normalized-measure penalty.
    assert metrics["value_loss_train"] == 0.0
    assert metrics["gradient_loss_train"] == 0.0
    assert metrics["data_loss_train"] == 0.0
    assert math.isclose(metrics["objective_train"], alpha * 455.0, rel_tol=1e-12)
    assert math.isclose(
        metrics["objective_train"], metrics["data_loss_train"] + metrics["alpha_phi"], rel_tol=1e-12,
    )


@pytest.mark.parametrize("power", [1., 2., 3.])
def test_sphere_penalty_and_reporting_do_not_depend_on_historical_moment_order(power):
    x = torch.tensor([[.2, -.3], [.5, .7]], dtype=torch.float64)
    inner_weight, inner_bias, c = torch.eye(2, dtype=x.dtype), x.new_zeros(2), x.new_tensor([-2., 3.])
    expected_penalty = .1 * c.abs().pow(2 / (power + 1)).sum()
    for order in (None, 2., 4.):
        config = resolve_model({"power": power, "alpha": .1, "moment_order": order}, 2)
        validate_model_config(config)
        network = SparseNetwork(config)
        network.set_atoms(inner_weight, inner_bias, c)
        assert network.moment_order is None
        torch.testing.assert_close(network.moment_weight_of_atoms(), torch.ones_like(c))
        history = History()
        history.record(network, evaluate_checkpoint(
            network, make_penalty(config), make_dataloader((x, *network(x)))))
        summary = _round_trip(history).summary()
        assert summary["total_variation"] == 5.
        assert summary["objective_train"] == pytest.approx(float(expected_penalty), rel=1e-12)


def test_physical_reporting_changes_only_the_relative_errors():
    normalizer = ValueSampleNormalizer(np.array([2.0, 8.0]), 5.0)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)
    v = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    dv = torch.tensor([[0.7, -0.6], [1.2, 0.5]], dtype=torch.float64)
    config = resolve_model({
        "activation": "tanh", "alpha": 0.1, "gamma": 0.2, "moment_order": 2.0,
    }, 2)
    network = SparseNetwork(config)
    penalty = make_penalty(config)
    loader = make_dataloader((x, v, dv))
    unchanged, physical = History(), History()
    for coefficient in (1.0, 2.0):
        network.set_atoms(
            torch.tensor([[1.0, 0.5]], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([coefficient], dtype=torch.float64),
        )
        unchanged.record(network, evaluate_checkpoint(network, penalty, loader))
        physical.record(network, evaluate_checkpoint(network, penalty, loader, normalizer=normalizer))
        expected = relative_errors(
            *normalizer.denormalize_tensors(*network(x)),
            *normalizer.denormalize_tensors(v, dv),
        )
        assert {kind: physical.rows[-1][f"rel_{kind}_val"] for kind in expected} == expected

    errors = {f"rel_{kind}_{split}" for kind in ("l2", "grad", "h1") for split in ("train", "val")}
    for before, after in zip(unchanged.rows, physical.rows):
        assert before["rel_h1_val"] != pytest.approx(after["rel_h1_val"])
        assert {k: v for k, v in before.items() if k not in errors} == {
            k: v for k, v in after.items() if k not in errors}
    assert physical.best_iteration == unchanged.best_iteration
    for expected_state, state in zip(unchanged.states, physical.states):
        assert state.keys() == expected_state.keys()
        for name in state:
            assert torch.equal(state[name], expected_state[name])
