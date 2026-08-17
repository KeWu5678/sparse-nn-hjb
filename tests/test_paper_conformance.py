"""The paper-conformance axes of ``PDAP`` (paper/paper_0805.tex, Sections 3 and 5).

Each axis defaults to the preserved comparator, so the checks here are about the
*opted-in* behaviour: that the normalized objective is the one the paper writes,
that the inserted coefficient delivers the decrease the theorem promises, that
the correction guard makes the loop monotone, and that the growth constants the
search radius is built from are actually upper bounds.

``tests/test_pdap_equivalence.py`` covers the other side: that the defaults still
reproduce the recorded behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.config.activations import ACTIVATIONS, GROWTH, get_growth
from src.config.schema import EnvConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import split_value_samples
from src.eval import data_loss_terms
from src.models import build_model
from src.PDAP import PDAP
from src.PDAP.history import objective_value
from src.PDAP.insertion import profile_threshold
from src.PDAP.moment import moment_weight
from src.PDAP.radius import certificate_radius, sample_extent
from src.SSN.penalty import _phi


def _data(n: int = 60, seed: int = 0) -> dict:
    """The same synthetic nonlinear target the golden characterization uses."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=torch.float64) * 3 - 1.5
    r2 = (x * x).sum(1, keepdim=True)
    return {"x": x.numpy(), "v": torch.log(1 + r2).numpy(), "dv": (2 * x / (1 + r2)).numpy()}


def _cfg(**over) -> ExperimentConfig:
    model_keys = {
        "kind", "insertion", "activation", "power", "alpha", "gamma", "th",
        "objective", "moment_beta", "moment_order", "loss_weights",
    }
    train_keys = {
        "insert_init", "insert_mode", "correction_guard", "loop_order", "radial_cap",
    }
    model = dict(kind="signed", insertion="profile", activation="softplus", power=1.0,
                 alpha=1e-4, gamma=1.0, th=0.5)
    train: dict = {}
    for key, value in over.items():
        if key in model_keys:
            model[key] = value
        elif key in train_keys:
            train[key] = value
        else:
            raise KeyError(key)
    return ExperimentConfig(
        model=ModelConfig(**model), training=TrainingConfig(**train),
        env=EnvConfig(verbose=False),
    )


def _fit(cfg: ExperimentConfig, iterations: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = _data()
    model = build_model(cfg, input_dim=2)
    train, valid = split_value_samples(data, cfg.data.train_fraction)
    history = PDAP(cfg).fit(
        model, train, valid, num_iterations=iterations, num_insertion=20, verbose=False
    )
    return model, history, train


# --------------------------------------------------------------------------- #
# Activation growth data: the declared constants must be genuine upper bounds,
# or the search radius built from them is not a bound at all.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(GROWTH))
def test_declared_growth_constants_are_upper_bounds(name: str) -> None:
    growth = GROWTH[name]
    fn = ACTIVATIONS[name][0]
    z = torch.linspace(-60.0, 60.0, 200_001, dtype=torch.float64).requires_grad_(True)
    y = fn(z)
    (grad,) = torch.autograd.grad(y.sum(), z)
    base = 1.0 + z.detach().abs()

    value_ratio = (y.detach().abs() / base.pow(growth.s0)).max().item()
    grad_ratio = (grad.abs() / base.pow(growth.s1 - 1.0)).max().item()
    assert value_ratio <= growth.C_rho + 1e-9, (
        f"{name}: |rho| needs C_rho >= {value_ratio:.6f}, declared {growth.C_rho}"
    )
    assert grad_ratio <= growth.C_rho + 1e-9, (
        f"{name}: |rho'| needs C_rho >= {grad_ratio:.6f}, declared {growth.C_rho}"
    )
    # The assumption's floor; s1 < 1 would break the radius exponent's derivation.
    assert growth.s1 >= max(growth.s0, 1.0)


# --------------------------------------------------------------------------- #
# The normalized objective and the u = w_p * c substitution
# --------------------------------------------------------------------------- #
def test_normalized_objective_is_phi_of_the_scaled_coefficient() -> None:
    """The recorded objective equals l^M + alpha * sum phi(w_p(omega_n)|c_n|)."""
    cfg = _cfg(objective="normalized_moment", moment_order=2.01)
    model, _, train = _fit(cfg)
    objective = PDAP(cfg).objective

    W, b, c = model.get_atoms()
    value, grad = model.predict_tensors(train[0])
    fidelity = float(data_loss_terms(value, grad, train[1], train[2], objective.loss_weights)[0])
    w_p = moment_weight(W, b, objective.moment_order).reshape(-1)
    penalty = float(objective.alpha * torch.sum(_phi(w_p * c.abs(), objective.th, objective.gamma)))

    assert objective_value(model, objective, train) == pytest.approx(fidelity + penalty, rel=1e-12)


def test_normalized_dictionary_reproduces_the_network() -> None:
    """K_p @ u == K @ c, which is what makes the substitution legitimate."""
    cfg = _cfg(objective="normalized_moment", moment_order=2.01)
    model, _, train = _fit(cfg)
    W, b, c = model.get_atoms()
    w_p = moment_weight(W, b, 2.01).reshape(-1)
    Phi = model.forward_network_matrix(train[0])
    assert torch.allclose(Phi @ c, (Phi / w_p) @ (w_p * c), atol=1e-12)


# --------------------------------------------------------------------------- #
# The theorem's insertion step
# --------------------------------------------------------------------------- #
def test_guaranteed_insertion_delivers_the_promised_decrease() -> None:
    """Inserting one accepted atom with c(omega) drops J by at least Delta^2/(2A).

    This is the conclusion of the quantitative insertion theorem, with the
    per-neuron curvature A = ||K_p(omega)||^2 in place of the uniform B_p^2 (the
    sharper of the two, and it implies the printed bound since A <= B_p^2).
    """
    cfg = _cfg(objective="normalized_moment", moment_order=2.01,
               insert_init="guaranteed", loop_order="insertion_first")
    model, _, train = _fit(cfg, iterations=3)
    objective = PDAP(cfg).objective
    trainer = PDAP(cfg)

    before = objective_value(model, objective, train)
    value, grad = model.predict_tensors(train[0])
    res_v, res_dv = (value - train[1]).detach(), (grad - train[2]).detach()

    W_old, b_old, c_old = model.get_atoms()
    W_new, b_new, c_new = profile_threshold(
        train[0], res_v, res_dv,
        activation=model.activation, power=model.power,
        loss_weights=objective.loss_weights, alpha=objective.alpha,
        sample_sphere=lambda n: trainer._sphere(2, n), N=20,
        max_insert=1, merge_tol=1e-2, two_sided=True, use_sphere=False,
        existing_atoms=(W_old, b_old), verbose=False,
        moment_order=2.01, normalized=True, insert_init="guaranteed",
    )
    if W_new.shape[0] == 0:
        pytest.skip("no candidate cleared the insertion threshold at this iterate")

    a = torch.as_tensor(W_new[0], dtype=torch.float64)
    bias = torch.as_tensor(b_new[0], dtype=torch.float64)
    w_p = float(moment_weight(a, bias, 2.01))

    # Delta and A in normalized units, recomputed independently of the insertion.
    x = train[0].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        neuron = model.activation(x @ a + bias).reshape(-1, 1) ** model.power
        (neuron_grad,) = torch.autograd.grad(neuron.sum(), x)
    neuron, neuron_grad = neuron.detach(), neuron_grad.detach()
    M = train[0].shape[0]
    w1, w2 = objective.loss_weights
    profile = float(
        (w1 / M) * neuron.reshape(-1).dot(res_v.reshape(-1))
        + (w2 / M) * neuron_grad.reshape(-1).dot(res_dv.reshape(-1))
    )
    curvature = float(
        (w1 / M) * neuron.reshape(-1).dot(neuron.reshape(-1))
        + (w2 / M) * neuron_grad.reshape(-1).dot(neuron_grad.reshape(-1))
    )
    delta = abs(profile) / w_p - objective.alpha
    a_p = curvature / w_p ** 2
    assert delta > 0

    model.set_atoms(
        torch.cat([W_old, a.reshape(1, -1)]),
        torch.cat([b_old, bias.reshape(1)]),
        torch.cat([c_old, torch.as_tensor(c_new[:1], dtype=torch.float64)]),
    )
    after = objective_value(model, objective, train)
    assert before - after >= delta ** 2 / (2 * a_p) - 1e-12


def test_correction_guard_keeps_the_objective_monotone() -> None:
    cfg = _cfg(objective="normalized_moment", moment_order=2.01,
               insert_init="guaranteed", correction_guard=True,
               loop_order="insertion_first")
    _, history, _ = _fit(cfg, iterations=6)
    losses = np.asarray(history.train_loss)
    assert np.all(np.diff(losses) <= 1e-12), f"objective increased: {losses}"


# --------------------------------------------------------------------------- #
# Loop order and insertion mode
# --------------------------------------------------------------------------- #
def test_insertion_first_ends_on_a_corrected_network() -> None:
    """final_neurons describes the same network as the last recorded metrics.

    The preserved order ends on an uncorrected insertion, so its final_neurons
    counts atoms that were never corrected or pruned.
    """
    cfg = _cfg(loop_order="insertion_first")
    _, history, _ = _fit(cfg, iterations=4)
    last_recorded = int(history.inner_weights[-1]["weight"].shape[0])
    assert history.final_neurons == last_recorded


def test_sequential_adds_at_most_one_atom_per_iteration() -> None:
    cfg = _cfg(objective="normalized_moment", moment_order=2.01,
               insert_init="guaranteed", insert_mode="sequential",
               loop_order="insertion_first")
    _, history, _ = _fit(cfg, iterations=6)
    widths = [int(w["weight"].shape[0]) for w in history.inner_weights]
    assert all(step <= 1 for step in np.diff(widths)), widths


# --------------------------------------------------------------------------- #
# The search radius
# --------------------------------------------------------------------------- #
def test_certificate_radius_never_exceeds_the_fixed_clamp() -> None:
    """The theorem radius only ever tightens the search (docs/adr/0006)."""
    growth = get_growth("softplus")
    for p in (2.01, 2.5, 3.0, 4.0):
        radius = certificate_radius(
            growth, extent=2.13, residual_norm=1.0, alpha=1e-5, moment_order=p
        )
        assert radius is not None and radius <= math.exp(5.0) + 1e-9


def test_certificate_radius_is_unavailable_without_hypotheses() -> None:
    extent = sample_extent(torch.as_tensor(_data()["x"]))
    # No declared growth data -> keep the fixed comparison bound.
    assert certificate_radius(get_growth("snake_b0_25"), extent=extent,
                              residual_norm=1.0, alpha=1e-5, moment_order=2.01) is None
    # p <= s1 fails the theorem's hypothesis, so there is no finite radius to claim.
    assert certificate_radius(get_growth("gelu_squared"), extent=extent,
                              residual_norm=1.0, alpha=1e-5, moment_order=2.0) is None


def test_theorem_radius_binds_only_at_large_moment_order() -> None:
    """Recorded expectation: at the sweep's alpha the clamp still binds below p=4."""
    growth = get_growth("softplus")
    loose = certificate_radius(growth, extent=2.13, residual_norm=1.0,
                               alpha=1e-5, moment_order=2.01)
    tight = certificate_radius(growth, extent=2.13, residual_norm=1.0,
                               alpha=1e-5, moment_order=4.0)
    assert loose == pytest.approx(math.exp(5.0))
    assert tight < math.exp(5.0)


# --------------------------------------------------------------------------- #
# Configuration guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over, message",
    [
        (dict(objective="normalized_moment", moment_beta=1e-5), "moment_beta == 0"),
        (dict(objective="normalized_moment", kind="semiconcave"), "signed model"),
        (dict(objective="normalized_moment", insertion="finite_step"), "Algorithm 1"),
        (dict(objective="normalized_moment", power=2.0), "power == 1"),
        (dict(insert_init="guaranteed", kind="semiconcave"), "signed theorem step"),
        (
            dict(insertion="finite_step", activation="relu", power=4.0, gamma=0.0),
            "powers 1, 2, or 3",
        ),
        (
            dict(insertion="finite_step", activation="relu", power=2.0, gamma=0.1),
            "gamma == 0",
        ),
        (
            dict(insertion="finite_step", activation="softplus", power=2.0, gamma=0.0),
            "sphere activation",
        ),
        (
            dict(insertion="finite_step", kind="semiconcave", activation="relu"),
            "signed model",
        ),
        (dict(objective="nonsense"), "model.objective must be one of"),
        (dict(insert_mode="nonsense"), "training.insert_mode must be one of"),
    ],
)
def test_incoherent_configurations_are_rejected(over: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PDAP(_cfg(**over))
