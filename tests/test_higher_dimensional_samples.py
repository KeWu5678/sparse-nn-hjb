"""The common data boundary must retain all state and costate coordinates."""

import numpy as np
import pytest

from src.data import load_value_samples, normalize_value_samples
from src.OpenLoop.allen_cahn.sampling import sobol_initial_states
from src.OpenLoop.value_samples import ValueSamples


@pytest.mark.parametrize("dimension", [2, 6, 10])
def test_sample_roundtrip_and_gradient_chain_rule(tmp_path, dimension):
    states = np.arange(1, 4 * dimension + 1, dtype=float).reshape(4, dimension)
    values = np.sum(states**2, axis=1)
    samples = ValueSamples(x=states, v=values, dv=2 * states)
    path = samples.save_npz(tmp_path / "samples.npz")
    loaded = load_value_samples(path)
    normalized, transform = normalize_value_samples(loaded)
    np.testing.assert_allclose(
        normalized["dv"], 2 * states * transform.x_scale / transform.v_scale,
    )
    recovered_value, recovered_gradient = transform.denormalize_prediction(
        normalized["v"], normalized["dv"],
    )
    np.testing.assert_allclose(recovered_value.ravel(), samples.v)
    np.testing.assert_allclose(recovered_gradient, samples.dv)
    restored = ValueSamples.load_npz(path)
    np.testing.assert_array_equal(restored.x, states)


@pytest.mark.parametrize("dimension", [6, 10])
def test_empty_and_concatenated_samples_keep_dimension(dimension):
    empty = ValueSamples(np.empty((0, dimension)), np.empty(0), np.empty((0, dimension)))
    one = ValueSamples(np.ones((1, dimension)), np.ones(1), np.zeros((1, dimension)))
    assert empty.x.shape == (0, dimension)
    assert ValueSamples.concatenate([empty, one]).x.shape == (1, dimension)


def test_legacy_empty_samples_remain_two_dimensional():
    assert ValueSamples([], [], []).x.shape == (0, 2)
    assert ValueSamples.concatenate([]).x.shape == (0, 2)


@pytest.mark.parametrize("dimension", [6, 10])
def test_sobol_sampling_reproducible_bounded_full_rank(dimension):
    states = sobol_initial_states(32, dimension)
    np.testing.assert_array_equal(states, sobol_initial_states(32, dimension))
    assert not np.array_equal(states, sobol_initial_states(32, dimension, seed=43))
    assert states.shape == (32, dimension)
    assert np.max(np.abs(states)) < 1
    assert np.linalg.matrix_rank(states - states.mean(axis=0)) == dimension


@pytest.mark.parametrize("count", [0, 1, 3, 10, True])
def test_sobol_rejects_nonbalanced_sample_counts(count):
    with pytest.raises(ValueError, match="power of two"):
        sobol_initial_states(count, 6)
