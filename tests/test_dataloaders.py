from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import full_batch, make_dataloader, make_dataloaders


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loader_defaults_to_one_complete_batch(dtype):
    samples = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype),
        torch.tensor([[10.0], [20.0], [30.0]], dtype=dtype),
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=dtype),
    )

    loader = make_dataloader(samples)
    batches = list(loader)

    assert isinstance(loader, DataLoader)
    assert len(batches) == 1
    for actual, expected in zip(batches[0], samples):
        torch.testing.assert_close(actual, expected)


def test_loader_does_not_change_global_torch_random_state():
    samples = (torch.ones(5, 2), torch.ones(5, 1), torch.zeros(5, 2))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        state_before = torch.get_rng_state().clone()

        loader = make_dataloader(samples, batch_size=2, seed=17)
        list(loader)
        list(loader)

        assert torch.equal(torch.get_rng_state(), state_before)


@pytest.mark.parametrize(
    "samples, message",
    [
        ((torch.empty(0, 2), torch.empty(0, 1), torch.empty(0, 2)), "non-empty"),
        ((torch.ones(3), torch.ones(3, 1), torch.ones(3)), "x must have shape"),
        ((torch.ones(3, 2), torch.ones(3), torch.ones(3, 2)), "v must have shape"),
        ((torch.ones(3, 2), torch.ones(2, 1), torch.ones(3, 2)), "v must have shape"),
        ((torch.ones(3, 2), torch.ones(3, 1), torch.ones(3, 1)), "dv must have shape"),
    ],
)
def test_loader_rejects_empty_or_misaligned_samples(samples, message):
    with pytest.raises(ValueError, match=message):
        make_dataloader(samples)


def test_train_and_validation_loaders_keep_the_seeded_split_and_sample_alignment():
    samples = {
        "x": np.arange(5.0).reshape(-1, 1),
        "v": np.arange(10.0, 15.0).reshape(-1, 1),
        "dv": np.arange(20.0, 25.0).reshape(-1, 1),
    }
    random_state = np.random.get_state()
    try:
        np.random.seed(0)
        train_loader, valid_loader = make_dataloaders(
            samples, train_fraction=0.6, batch_size=2, seed=99,
        )
    finally:
        np.random.set_state(random_state)

    for loader, expected in (
        (train_loader, [[2.0, 0.0], [1.0]]),
        (valid_loader, [[3.0, 4.0]]),
    ):
        for _ in range(2):
            batches = list(loader)
            assert [x[:, 0].tolist() for x, _, _ in batches] == expected
            for x, v, dv in batches:
                torch.testing.assert_close(v, x + 10.0)
                torch.testing.assert_close(dv, x + 20.0)
                assert x.dtype == v.dtype == dv.dtype == torch.float64


def test_validation_iteration_leaves_training_random_state_unchanged():
    samples = {"x": np.ones((4, 2)), "v": np.ones((4, 1)), "dv": np.zeros((4, 2))}
    random_state = np.random.get_state()
    try:
        train_loader, valid_loader = make_dataloaders(samples, train_fraction=0.5, seed=7)
    finally:
        np.random.set_state(random_state)
    train_state = train_loader.generator.get_state().clone()

    list(valid_loader)

    assert torch.equal(train_loader.generator.get_state(), train_state)


@pytest.mark.parametrize("batch_size,num_workers", [(None, 0), (2, 0), (2, 1)])
def test_full_batch_collects_aligned_samples_in_delivery_order(batch_size, num_workers):
    samples = (
        torch.arange(10, dtype=torch.float64).reshape(5, 2),
        torch.arange(20, 25, dtype=torch.float64).reshape(5, 1),
        torch.arange(30, 40, dtype=torch.float64).reshape(5, 2),
    )

    result = full_batch(make_dataloader(samples, batch_size=batch_size, num_workers=num_workers))

    for actual, expected in zip(result, samples):
        torch.testing.assert_close(actual, expected)


def test_full_batch_reuses_the_single_delivered_batch():
    samples = (torch.ones(3, 2), torch.ones(3, 1), torch.zeros(3, 2))
    loader = DataLoader(TensorDataset(*samples), batch_size=3, collate_fn=lambda _: samples)

    assert all(actual is expected for actual, expected in zip(full_batch(loader), samples))


@pytest.mark.parametrize("batch_size,drop_last,count", [(None, False, 3), (2, True, 3), (2, False, 0)])
def test_full_batch_rejects_unbatched_incomplete_or_empty_delivery(batch_size, drop_last, count):
    samples = (torch.ones(count, 2), torch.ones(count, 1), torch.zeros(count, 2))
    loader = DataLoader(TensorDataset(*samples), batch_size=batch_size, drop_last=drop_last)

    with pytest.raises(ValueError):
        full_batch(loader)
