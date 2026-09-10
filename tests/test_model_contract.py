"""The signed model satisfies the PDAPModel contract the trainer depends on."""

import pytest
import torch

from src.config.schema import ExperimentConfig
from src.models import build_model
from src.models.base import PDAPModel
from src.models.net import ShallowNetwork
from src.models.signed import SignedModel
from src.PDAP import PDAP


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


def test_network_initializes_in_float64():
    model = ShallowNetwork([2, 4, 1], torch.relu)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert model(torch.ones(3, 2, dtype=torch.float64)).dtype == torch.float64


def test_setting_and_pruning_atoms_preserves_float64_coefficients():
    model = SignedModel(power=1.0, verbose=False)
    W, b, _ = _atoms(n=2)
    c = torch.tensor([1.123456789012345, -1.0000000001], dtype=torch.float64)
    model.set_atoms(W, b, c)
    trainer = PDAP(ExperimentConfig())
    assert trainer._prune(model, amp_tol=1e-8) == 0
    for actual, expected in zip(model.get_atoms(), (W, b, c)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


@pytest.mark.parametrize("field,value", [("activation", "softplus"), ("power", 2.0)])
def test_fit_rejects_model_config_mismatch_before_mutating_state(field, value):
    trainer = PDAP(ExperimentConfig())
    model_cfg = ExperimentConfig()
    setattr(model_cfg.model, field, value)
    model = build_model(model_cfg, input_dim=2)
    model.set_atoms(*_atoms())
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    # No data can be consumed before the mismatch is rejected.
    with pytest.raises(ValueError, match=f"model {field}.*match"):
        trainer.fit(model, None, None, num_iterations=1, num_insertion=2, verbose=False)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
