import math

import torch

from src.models.signed import SignedModel
from src.PDAP.history import History
from src.PDAP.ssn_solve import Objective


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


def test_restoring_a_zero_measure_snapshot_empties_an_initialized_model() -> None:
    """A model that already carries atoms must not keep them across the restore.

    This is the terminal zero-measure run of ``PDAP.fit``: under strong
    regularization the initial insertion accepts no atom, and the recorded model
    is the one ``build_model`` returned -- layerless, so its ``state_dict`` is
    empty and only the emptiness of the support identifies the snapshot.
    """
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    empty = SignedModel(activation=torch.tanh, power=1.0, verbose=False)
    empty.input_dim = 2  # what build_model does; set_atoms is never reached
    assert dict(empty.state_dict()) == {}
    samples = (x, *empty.predict_tensors(x))
    history = History()
    history.record(empty, Objective(), samples, samples)

    restored = SignedModel(activation=torch.tanh, power=1.0, verbose=False)
    restored.set_atoms(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.tensor([3.0, -4.0], dtype=torch.float64),
    )
    assert restored.get_atoms()[0].shape[0] == 2

    history.restore_model(restored)

    assert restored.get_atoms()[0].shape[0] == 0
    value, gradient = restored.predict_tensors(x)
    assert torch.allclose(value, torch.zeros_like(value))
    assert torch.allclose(gradient, torch.zeros_like(gradient))


def test_history_summary_records_normalized_objective_decomposition() -> None:
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.zeros(3, dtype=torch.float64)
    c = torch.tensor([50.0, -45.0, 5.0], dtype=torch.float64)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64)

    model = SignedModel(activation=torch.tanh, power=1.0, verbose=False)
    model.set_atoms(W, b, c)
    v, dv = model.predict_tensors(x)
    samples = (x, v, dv)
    objective = Objective(
        alpha=0.1,
        gamma=0.0,
        loss_weights=(1.0, 1.0),
        moment_order=2.0,
        normalized=True,
    )

    history = History()
    history.record(model, objective, samples, samples)
    metrics = history.summary_metrics()

    # gamma=0 gives Phi_1(mu_p) = Psi_2(mu)
    # = 50*(1+1) + 45*(1+4) + 5*(1+25) = 455.
    assert math.isclose(metrics["phi_1"], 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["psi_p"], 455.0, rel_tol=1e-12)
    assert math.isclose(metrics["alpha_phi_1"], 45.5, rel_tol=1e-12)
    assert math.isclose(metrics["radius_r95"], 2.0, rel_tol=1e-12)
    assert math.isclose(metrics["radius_max"], 5.0, rel_tol=1e-12)
    assert math.isclose(metrics["total_variation"], 100.0, rel_tol=1e-12)

    # Targets equal predictions, so both fidelity channels vanish and the
    # complete objective is exactly the normalized-measure penalty.
    assert metrics["data_value_term_train"] == 0.0
    assert metrics["data_gradient_term_train"] == 0.0
    assert math.isclose(metrics["objective_train"], 45.5, rel_tol=1e-12)
    assert math.isclose(
        metrics["objective_train"],
        metrics["data_value_term_train"]
        + metrics["data_gradient_term_train"]
        + metrics["alpha_phi_1"],
        rel_tol=1e-12,
    )
