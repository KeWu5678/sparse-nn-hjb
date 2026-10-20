from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import ValueSampleNormalizer, load_value_samples
from src.tables import format_table


def test_load_value_samples_accepts_structured_npy(tmp_path):
    path = tmp_path / "samples.npy"
    dtype = [("x", "f8", (2,)), ("dv", "f8", (2,)), ("v", "f8")]
    data = np.zeros(3, dtype=dtype)
    data["x"] = [[1, 2], [3, 4], [5, 6]]
    data["v"] = [10, 20, 30]
    data["dv"] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    np.save(path, data)

    samples = load_value_samples(path)

    assert samples.x.shape == (3, 2)
    assert samples.v.shape == (3,)
    assert samples.dv.shape == (3, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_denormalization_preserves_dtype_device_and_autograd(dtype):
    normalizer = ValueSampleNormalizer(np.array([2.0, 4.0]), 8.0)
    v = torch.tensor([[1.0]], dtype=dtype, requires_grad=True)
    dv = torch.tensor([[0.75, 2.5]], dtype=dtype, requires_grad=True)

    v_phys, dv_phys = normalizer.denormalize_tensors(v, dv)

    assert v_phys.dtype == dv_phys.dtype == dtype
    assert v_phys.device == v.device and dv_phys.device == dv.device
    torch.testing.assert_close(v_phys, torch.tensor([[8.0]], dtype=dtype))
    torch.testing.assert_close(dv_phys, torch.tensor([[3.0, 5.0]], dtype=dtype))
    (v_phys.sum() + dv_phys.sum()).backward()
    torch.testing.assert_close(v.grad, torch.tensor([[8.0]], dtype=dtype))
    torch.testing.assert_close(dv.grad, torch.tensor([[4.0, 2.0]], dtype=dtype))


def test_format_table_is_markdown_without_pandas():
    text = format_table(
        [{"name": "relu", "score": 1.2345}],
        ["name", "score"],
        formats={"score": "{:.2f}"},
    )

    assert "| name | score |" in text
    assert "| relu | 1.23  |" in text
