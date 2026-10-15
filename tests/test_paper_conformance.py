"""Paper-conformance checks for ``PDAP`` (paper Sections 3 and 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.config import make_penalty, resolve_model, validate_model_config, validate_optimizer_config
from src.config.activations import ACTIVATIONS, GROWTH, get_growth
from src.config.schema import EnvConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import make_dataloader, split_value_samples
from src.eval import evaluate_checkpoint
from src.models import SparseNetwork
from src.models.penalty import _phi
from src.PDAP import PDAP
from src.PDAP.insertion import profile_threshold
from src.PDAP.radius import certificate_radius, sample_extent
from src.training import run_training


def _data(n: int = 60, seed: int = 0) -> dict:
    """The same synthetic nonlinear target the golden characterization uses."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=torch.float64) * 3 - 1.5
    r2 = (x * x).sum(1, keepdim=True)
    return {"x": x.numpy(), "v": torch.log(1 + r2).numpy(), "dv": (2 * x / (1 + r2)).numpy()}


def _cfg(**over) -> ExperimentConfig:
    model_keys = {
        "kind", "insertion", "activation", "power", "alpha", "gamma",
        "moment_order", "loss_weights",
    }
    train_keys = {
        "insert_init", "insert_mode", "correction_guard", "loop_order", "radial_cap", "method",
    }
    model = dict(kind="signed", insertion="profile", activation="softplus", power=1.0,
                 alpha=1e-4, gamma=1.0, moment_order=2.0)
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
    model_config = resolve_model(cfg.model, input_dim=2)
    validate_model_config(model_config)
    network = SparseNetwork(model_config)
    train, valid = split_value_samples(data, cfg.data.train_fraction)
    cfg.training.num_of_initial_pts = 20
    history = run_training(
        PDAP(
            cfg.training, make_penalty(model_config), network, make_dataloader(train),
            insertion=model_config["insertion"],
        ),
        make_dataloader(valid),
        num_iterations=iterations,
    )
    return network, history, train


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
    cfg = _cfg(moment_order=2.01)
    network, _, train = _fit(cfg)

    c = network.c.detach()
    with torch.no_grad():
        value, grad = network(train[0])
    w1, w2 = cfg.model.loss_weights
    fidelity = float(
        (w1 * (value - train[1]).square().sum() + w2 * (grad - train[2]).square().sum())
        / (2 * train[0].shape[0])
    )
    w_p = network.moment_weight_of_atoms().reshape(-1)
    expected_penalty = float(cfg.model.alpha * torch.sum(_phi(w_p * c.abs(), cfg.model.gamma)))

    penalty = make_penalty(resolve_model(cfg.model, 2))
    assert evaluate_checkpoint(network, penalty, make_dataloader(train))["objective_train"] == pytest.approx(fidelity + expected_penalty, rel=1e-12)


def test_normalized_dictionary_reproduces_the_network() -> None:
    """K_p @ u == K @ c, which is what makes the substitution legitimate."""
    cfg = _cfg(moment_order=2.01)
    network, _, train = _fit(cfg)
    c = network.c.detach()
    w_p = network.moment_weight_of_atoms().reshape(-1)
    Phi, _ = network.kernel(train[0])
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
    cfg = _cfg(moment_order=2.01,
               insert_init="guaranteed", loop_order="insertion_first")
    network, _, train = _fit(cfg, iterations=3)
    train_loader = make_dataloader(train)
    trainer = PDAP(
        cfg.training, make_penalty(resolve_model(cfg.model, 2)), network, train_loader,
        insertion=cfg.model.insertion,
    )

    before = evaluate_checkpoint(network, trainer.penalty, train_loader)["objective_train"]
    with torch.no_grad():
        value, grad = network(train[0])
    res_v, res_dv = (value - train[1]).detach(), (grad - train[2]).detach()

    W_old, b_old, c_old = (network.inner_weight, network.inner_bias, network.c.detach())
    W_new, b_new, c_new = profile_threshold(
        network, trainer.penalty, (train[0], res_v, res_dv),
        initial_points=trainer._sphere(),
        max_insert=1, merge_tol=1e-2,
        insert_init="guaranteed",
    )
    if W_new.shape[0] == 0:
        pytest.skip("no candidate cleared the insertion threshold at this iterate")

    a = torch.as_tensor(W_new[0], dtype=torch.float64)
    bias = torch.as_tensor(b_new[0], dtype=torch.float64)
    w_p = float(network.moment_weight_of_atoms(a, bias))

    # Delta and A in normalized units, recomputed independently of the insertion.
    x = train[0].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        neuron = ACTIVATIONS[cfg.model.activation][0](x @ a + bias).reshape(-1, 1) ** cfg.model.power
        (neuron_grad,) = torch.autograd.grad(neuron.sum(), x)
    neuron, neuron_grad = neuron.detach(), neuron_grad.detach()
    M = train[0].shape[0]
    w1, w2 = cfg.model.loss_weights
    profile = float(
        (w1 / M) * neuron.reshape(-1).dot(res_v.reshape(-1))
        + (w2 / M) * neuron_grad.reshape(-1).dot(res_dv.reshape(-1))
    )
    curvature = float(
        (w1 / M) * neuron.reshape(-1).dot(neuron.reshape(-1))
        + (w2 / M) * neuron_grad.reshape(-1).dot(neuron_grad.reshape(-1))
    )
    delta = abs(profile) / w_p - cfg.model.alpha
    a_p = curvature / w_p ** 2
    assert delta > 0

    network.set_atoms(
        torch.cat([W_old, a.reshape(1, -1)]),
        torch.cat([b_old, bias.reshape(1)]),
        torch.cat([c_old, torch.as_tensor(c_new[:1], dtype=torch.float64)]),
    )
    after = evaluate_checkpoint(network, trainer.penalty, train_loader)["objective_train"]
    assert before - after >= delta ** 2 / (2 * a_p) - 1e-12


def test_correction_guard_keeps_the_objective_monotone() -> None:
    cfg = _cfg(moment_order=2.01,
               insert_init="guaranteed", correction_guard=True,
               loop_order="insertion_first")
    _, history, _ = _fit(cfg, iterations=6)
    losses = np.asarray([row["objective_train"] for row in history.rows])
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
    last_recorded = int(history.states[-1]["inner_weight"].shape[0])
    assert history.run["final_neurons"] == last_recorded


def test_sequential_adds_at_most_one_atom_per_iteration() -> None:
    cfg = _cfg(moment_order=2.01,
               insert_init="guaranteed", insert_mode="sequential",
               loop_order="insertion_first")
    _, history, _ = _fit(cfg, iterations=6)
    widths = [int(s["inner_weight"].shape[0]) for s in history.states]
    assert all(step <= 1 for step in np.diff(widths)), widths


# --------------------------------------------------------------------------- #
# The search radius
# --------------------------------------------------------------------------- #
def test_search_radius_never_exceeds_the_fixed_clamp() -> None:
    """The theorem radius only ever tightens the search (docs/adr/0006)."""
    for p in (2.01, 2.5, 3.0, 4.0):
        cfg = _cfg(moment_order=p, radial_cap="theorem")
        data = tuple(torch.as_tensor(_data()[key]) for key in ("x", "v", "dv"))
        model_config = resolve_model(cfg.model, 2)
        validate_model_config(model_config)
        network = SparseNetwork(model_config)
        trainer = PDAP(cfg.training, make_penalty(model_config), network, make_dataloader(data))
        radius = trainer._search_radius()
        assert radius is not None and radius <= math.exp(5.0) + 1e-9


def test_certificate_radius_is_one_for_zero_residual() -> None:
    radius = certificate_radius(
        get_growth("softplus"),
        extent=2.13,
        residual_norm=0.0,
        alpha=1e-5,
        moment_order=2.01,
    )

    assert radius == 1.0


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
    assert loose > math.exp(5.0)
    assert tight < math.exp(5.0)


def test_insertion_evaluates_the_network_once_per_stage() -> None:
    cfg = _cfg(
        alpha=1e-4,
        moment_order=2.01,
        insert_init="warm_start",
        insert_mode="sequential",
        loop_order="insertion_first",
        radial_cap="theorem",
    )
    samples = _data(n=8)
    data = tuple(
        torch.as_tensor(samples[key], dtype=torch.float64)
        for key in ("x", "v", "dv")
    )
    torch.manual_seed(0)
    np.random.seed(0)
    model_config = resolve_model(cfg.model, input_dim=2)
    validate_model_config(model_config)
    network = SparseNetwork(model_config)
    original_predict = network.forward
    prediction_calls = 0

    def counted_predict(x: torch.Tensor):
        nonlocal prediction_calls
        prediction_calls += 1
        return original_predict(x)

    network.forward = counted_predict
    cfg.training.num_of_initial_pts = cfg.training.max_insert = 1
    PDAP(
        cfg.training, make_penalty(model_config), network, make_dataloader(data),
        insertion=model_config["insertion"],
    ).insert()

    assert network.n_neurons == 1  # Exercise the warm-start branch with an accepted atom.
    # Candidate search, the theorem radius and the warm start each evaluate the network once.
    assert prediction_calls == 3


# --------------------------------------------------------------------------- #
# Configuration guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over, message",
    [
        (dict(activation="softplus", power=2.0), "Algorithm 1.*power == 1"),
        (dict(kind="semiconcave"), "only supports kind='signed'"),
        (
            dict(insertion="finite_step", activation="relu", power=4.0, gamma=0.0),
            "powers 1, 2, 3",
        ),
        (
            dict(insertion="profile", activation="relu", power=5.0, gamma=0.0),
            "powers 1, 2, 3",
        ),
        (
            dict(insertion="profile", activation="relu", power=2.0, gamma=0.1),
            "power penalty requires gamma == 0",
        ),
        (
            dict(insertion="profile", activation="relu", power=3.0, gamma=0.1),
            "power penalty requires gamma == 0",
        ),
        (
            dict(insertion="finite_step", activation="relu", power=2.0, gamma=0.1),
            "gamma == 0",
        ),
        (
            dict(insertion="finite_step", activation="softplus", power=1.0, gamma=0.0),
            "sphere activation",
        ),
        (dict(insertion="nonsense"), "insertion must be 'profile' or 'finite_step'"),
        (dict(insert_mode="nonsense"), "training.insert_mode must be one of"),
        (dict(method="nonsense"), "training.method must be one of"),
    ],
)
def test_incoherent_configurations_are_rejected(over: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cfg = _cfg(**over)
        validate_model_config(resolve_model(cfg.model, 2))
        validate_optimizer_config(cfg.training)
