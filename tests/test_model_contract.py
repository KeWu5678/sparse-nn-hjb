"""The signed model satisfies the PDAPModel contract the trainer depends on."""

import torch

from src.models.base import PDAPModel
from src.models.signed import SignedModel


def _atoms(n=4, d=2):
    W = torch.randn(n, d, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(n, dtype=torch.float64)
    c = torch.rand(n, dtype=torch.float64)
    return W, b, c


def test_signed_model_is_pdap_model():
    m = SignedModel(power=1.0, verbose=False)
    m.input_dim = 2
    m.set_atoms(*_atoms())
    assert isinstance(m, PDAPModel)


def test_signed_model_predicts_to_numpy():
    x = torch.randn(6, 2, dtype=torch.float64).numpy()
    model = SignedModel(power=1.0, verbose=False)
    model.input_dim = 2
    model.set_atoms(*_atoms())

    value, gradient = model.predict(x)

    assert value.shape == (6, 1)
    assert gradient.shape == (6, 2)
