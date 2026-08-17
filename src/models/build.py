"""Build the signed PDAP model from a config section."""

from __future__ import annotations

from ..config.activations import get_activation
from .signed import SignedModel


def build_model(cfg, input_dim: int):
    """Construct the model named by ``cfg.model`` with its input dimension set."""
    m = cfg.model
    if m.kind != "signed":
        raise ValueError(f"the active implementation only supports kind='signed'; got {m.kind!r}")
    activation = get_activation(m.activation)
    model = SignedModel(activation=activation, power=m.power, verbose=cfg.env.verbose)
    model.input_dim = input_dim
    return model
