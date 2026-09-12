import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import split_value_samples
from src.eval import relative_errors
from src.results import (
    evaluate_history,
    history_atoms,
    load_run,
    load_run_for_artifact,
    model_from_history,
    predict_physical,
    validation_metrics,
    validation_samples,
    value_surface_grid,
)


@pytest.fixture
def record_path(tmp_path):
    samples = {
        "x": np.arange(1, 21, dtype=float).reshape(10, 2),
        "v": np.arange(1, 11, dtype=float).reshape(10, 1),
        "dv": np.ones((10, 2)),
    }
    data_path = tmp_path / "data.npz"
    np.savez(data_path, **samples)
    history = SimpleNamespace(
        best_iteration=1,
        inner_weights=[{"weight": torch.tensor([[1., 2.]]), "bias": torch.tensor([.5])}
                       for _ in range(2)],
        outer_weights=[torch.tensor([[1.]]), torch.tensor([[3.]])],
        err_h1_val=[123., 456.],
    )
    artifact = tmp_path / "result_example.pkl"
    with artifact.open("wb") as stream:
        pickle.dump(history, stream)
    path = tmp_path / "example.json"
    record = {
        "run_id": "example", "status": "completed",
        "config": {
            "model": {"kind": "signed", "activation": "relu", "power": 1.,
                      "normalized": True, "moment_order": 99.},
            "data": {"path": str(data_path), "normalize": True, "train_fraction": .6},
            "env": {"seed": 42},
        },
        "normalization": {"x_scale": [2., 4.], "v_scale": 8.},
        "artifacts": [{"name": "fit_history", "path": artifact.name}],
    }
    path.write_text(json.dumps(record))
    return path


def test_saved_run_uses_recorded_scaling_not_dataset_maxima(record_path):
    run = load_run(record_path)
    with torch.no_grad():
        v, dv = predict_physical(run, [[2., 4.]])
    torch.testing.assert_close(v, torch.tensor([[84.]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[12., 12.]], dtype=torch.float64))
    assert v.shape == (1, 1) and dv.shape == (1, 2)
    assert not v.requires_grad and not dv.requires_grad
    assert run.iteration == 1
    # Objective/measure normalization is not another transform of V.
    assert run.record["config"]["model"]["moment_order"] == 99.


def test_checkpoint_override_does_not_change_saved_model(record_path):
    run = load_run(record_path)
    before = {k: v.clone() for k, v in run.model.state_dict().items()}
    v, dv = predict_physical(run, [[2., 4.]], iteration=0)
    torch.testing.assert_close(v, torch.tensor([[28.]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[4., 4.]], dtype=torch.float64))
    assert run.iteration == 1
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(IndexError, match="checkpoint"):
        predict_physical(run, [[2., 4.]], iteration=-1)


def test_loading_and_validation_split_leave_rng_unchanged(record_path):
    np.random.seed(173)
    torch.manual_seed(173)
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    run = load_run(record_path)
    actual = validation_samples(run)
    assert np.random.get_state()[2:] == np_state[2:]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

    with np.load(run.record["config"]["data"]["path"]) as data:
        samples = {k: data[k] for k in ("x", "v", "dv")}
    np.random.seed(42)
    _, expected = split_value_samples(samples, .6)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rescoring_ignores_stored_normalized_errors(record_path):
    run = load_run(record_path)
    raw = record_path.read_bytes()
    x, v, dv = validation_samples(run)
    expected = relative_errors(*predict_physical(run, x), v, dv)
    metrics = validation_metrics(run)
    assert list(metrics.values()) == list(expected)
    curve = evaluate_history(run)
    np.testing.assert_array_equal(curve["iteration"], [0, 1])
    np.testing.assert_array_equal(curve["neurons"], [1, 1])
    assert curve["rel_h1"][1] == expected[2]
    assert curve["rel_h1"][0] == relative_errors(*predict_physical(run, x, 0), v, dv)[2]
    assert run.history.err_h1_val == [123., 456.]
    assert record_path.read_bytes() == raw


def test_normalization_disabled_is_identity(record_path):
    record = json.loads(record_path.read_text())
    record["normalization"] = None
    record["config"]["data"]["normalize"] = False
    record_path.write_text(json.dumps(record))
    v, dv = predict_physical(load_run(record_path), [[1., 1.]])
    torch.testing.assert_close(v, torch.tensor([[10.5]], dtype=torch.float64))
    torch.testing.assert_close(dv, torch.tensor([[3., 6.]], dtype=torch.float64))


@pytest.mark.parametrize("normalization", [None, {"x_scale": [2., 0.], "v_scale": 8.},
                                        {"x_scale": [2.], "v_scale": 8.}])
def test_missing_or_invalid_normalization_is_not_guessed(record_path, normalization):
    record = json.loads(record_path.read_text())
    record["normalization"] = normalization
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="normalization"):
        load_run(record_path)


def test_legacy_containers_and_artifact_entrypoint(record_path):
    run = load_run_for_artifact(record_path.with_name("result_example.pkl"))
    w, b, c = history_atoms([vars(run.history)])
    assert w.shape == (1, 2) and b.shape == (1,) and c.shape == (1, 1)
    net = model_from_history([vars(run.history)], "relu", 1.)
    torch.testing.assert_close(net.output.weight, run.model.output.weight)


def test_relocated_artifact_does_not_read_old_declared_copy(record_path):
    old = record_path.with_name("old.pkl")
    old.write_bytes(b"unreadable old artifact")
    record = json.loads(record_path.read_text())
    record["artifacts"][0]["path"] = str(old)
    record_path.write_text(json.dumps(record))
    run = load_run_for_artifact(record_path.with_name("result_example.pkl"))
    assert run.iteration == 1
    torch.testing.assert_close(predict_physical(run, [[2., 4.]])[0],
                               torch.tensor([[84.]], dtype=torch.float64))

    # The transform must be checked against the actual supplied artifact.
    run.history.inner_weights[1]["weight"] = np.ones((1, 3))
    with record_path.with_name("result_example.pkl").open("wb") as stream:
        pickle.dump(run.history, stream)
    with pytest.raises(ValueError, match="normalization"):
        load_run_for_artifact(record_path.with_name("result_example.pkl"))


def test_relocated_record_prefers_its_adjacent_fit_over_an_existing_old_copy(record_path):
    old = record_path.with_name("old.pkl")
    old.write_bytes(b"unreadable old artifact")
    record = json.loads(record_path.read_text())
    record["artifacts"][0]["path"] = str(old)
    record_path.write_text(json.dumps(record))

    run = load_run(record_path)
    torch.testing.assert_close(predict_physical(run, [[2., 4.]])[0],
                               torch.tensor([[84.]], dtype=torch.float64))


def test_empty_support_and_empty_prediction_batch(record_path):
    run = load_run(record_path)
    v, dv = predict_physical(run, np.empty((0, 2)))
    assert v.shape == (0, 1) and dv.shape == (0, 2)
    run.history.inner_weights[0] = {"weight": np.empty((0, 2)), "bias": np.empty(0)}
    run.history.outer_weights[0] = np.empty((1, 0))
    with pytest.warns(UserWarning, match="zero-element"):
        v, dv = predict_physical(run, [[1., 2.]], iteration=0)
    assert torch.count_nonzero(v) == 0 and torch.count_nonzero(dv) == 0


def test_surface_grid_uses_same_unclipped_prediction(record_path):
    run = load_run(record_path)
    x0, x1, value = value_surface_grid(run, x_range=(0., 2.), y_range=(0., 4.), grid_n=3)
    assert x0.shape == x1.shape == value.shape == (3, 3)
    assert value[-1, -1] == 84.  # Above the paper's display ceiling, not clamped.
    expected, _ = predict_physical(run, np.column_stack([x0.ravel(), x1.ravel()]))
    np.testing.assert_array_equal(value.ravel(), expected.numpy().ravel())


def test_explicit_legacy_recovery_must_reproduce_saved_metrics(record_path):
    record = json.loads(record_path.read_text())
    record.pop("normalization")
    with np.load(record["config"]["data"]["path"]) as samples:
        indices = np.random.RandomState(42).permutation(10)[6:]
        x = samples["x"][indices] / [19., 20.]
        truth = samples["v"][indices] / 10.
        grad_truth = samples["dv"][indices] * np.array([19., 20.]) / 10.
    predicted = 3. * (x[:, :1] + 2. * x[:, 1:] + .5)
    grad_predicted = np.tile([3., 6.], (4, 1))
    ev = np.sum((predicted - truth) ** 2)
    eg = np.sum((grad_predicted - grad_truth) ** 2)
    tv = np.sum(truth ** 2)
    tg = np.sum(grad_truth ** 2)
    record["metrics"] = [{"values": {
        "rel_l2_val": float(np.sqrt(ev / tv)),
        "rel_grad_val": float(np.sqrt(eg / tg)),
        "rel_h1_val": float(np.sqrt((ev + eg) / (tv + tg))),
    }}]
    record_path.write_text(json.dumps(record))
    raw = record_path.read_bytes()
    with pytest.raises(ValueError, match="no recorded normalization"):
        load_run(record_path)
    run = load_run(record_path, recover_legacy_normalization=True)
    assert run.normalization_recovered
    np.testing.assert_array_equal(run.normalizer.x_scale, [19., 20.])
    assert run.normalizer.v_scale == 10.
    assert record_path.read_bytes() == raw

    record["metrics"][0]["values"]["rel_grad_val"] *= 1.1
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="did not reproduce"):
        load_run(record_path, recover_legacy_normalization=True)
