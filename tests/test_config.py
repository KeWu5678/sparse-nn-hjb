"""Tests for the Hydra config system: composition, model groups, config→trainer."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.config import (
    get_activation,
    make_penalty,
    resolve_model,
    validate_model_config,
    validate_optimizer_config,
)
from src.config.schema import ExperimentConfig, ModelConfig, TrainingConfig
from src.data import DATA_DIR, load_value_samples
from src.models import SparseNetwork
from src.PDAP import PDAP
from src.SSN import SSN

# Must be registered before any compose() call below.
ConfigStore.instance().store(name="config_schema", node=ExperimentConfig)


def test_model_groups() -> None:
    with initialize(version_base=None, config_path="../conf"):
        fs = compose(config_name="config", overrides=["+model=finite_step"])
        alg1 = compose(config_name="config", overrides=[
            "+model=profile",
            "model.activation=softplus",
            "model.moment_order=2.0",
        ])
    # finite_step config group = signed + finite_step
    assert fs.model.kind == "signed"
    assert fs.model.insertion == "finite_step"
    model_config = resolve_model(alg1.model, 2)
    validate_model_config(model_config)
    assert SparseNetwork(model_config).normalized


def _compose_experiment(experiment: str, model: str, data: str):
    """Compose one point of an experiment's sweep.

    Experiments pin neither `model` nor `data`; both are sweep axes.
    """
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=[f"+experiment={experiment}", f"+model={model}", f"+data={data}"],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def test_curated_experiment_configs_compose() -> None:
    try:
        log_pen = _compose_experiment("log_penalty", "profile", "pendulum")
        assert log_pen.name == "pendulum_log_penalty"
        assert log_pen.model.power == 1.0
        assert log_pen.data.path.startswith("Pendulum")
        assert log_pen.hydra.runtime.choices["eval"] == "region_split"

        frac = _compose_experiment("frac_exp_penalty", "finite_step", "vdp")
        assert frac.name == "vdp_frac_exp_penalty"
        assert frac.model.insertion == "finite_step"
        assert frac.model.power == 2.0

        # The reference baseline is Algorithm 1's insertion with a convex L1
        # penalty (power=1 -> q=1, gamma=0 -> phi = identity), not Algorithm 2.
        baseline = _compose_experiment("relu_l1_baseline", "profile", "vdp")
        assert baseline.name == "vdp_relu_l1_baseline"
        assert baseline.model.insertion == "profile"
        assert baseline.model.power == 1.0
        assert baseline.model.gamma == 0.0
        assert baseline.training.insert_init == "warm_start"
    finally:
        HydraConfig.instance().cfg = None


def test_experiments_sweep_both_datasets() -> None:
    """Every curated experiment carries the dataset as a sweep axis."""
    with initialize(version_base=None, config_path="../conf"):
        for experiment in (
            "log_penalty",
            "frac_exp_penalty",
            "relu_l1_baseline",
        ):
            cfg = compose(
                config_name="config",
                overrides=[f"+experiment={experiment}", "+model=profile", "+data=vdp"],
                return_hydra_config=True,
            )
            params = cfg.hydra.sweeper.params
            assert params["+data"] == "vdp,pendulum", experiment
            assert "+model" in params, experiment


def test_config_builds_trainer_and_model() -> None:
    """The trainer and sole network read their respective configuration."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model.gamma=0.5", "env.verbose=false"])

    raw_model = OmegaConf.to_container(cfg.model, resolve=True)
    model_config = resolve_model(cfg.model, 2)
    validate_model_config(model_config)
    validate_optimizer_config(cfg.training)
    network = SparseNetwork(model_config)
    penalty = make_penalty(model_config)
    pdap = PDAP(cfg.training, penalty, network, None, insertion=model_config["insertion"])
    assert model_config["activation"] is torch.relu
    assert OmegaConf.to_container(cfg.model, resolve=True) == raw_model
    assert cfg.model.activation == "relu"
    assert pdap.penalty.alpha == cfg.model.alpha == 1e-5
    # Configured alpha and gamma determine the log penalty's normal-map scaling.
    optimizer = SSN(
        [torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], penalty,
        torch.eye(1, dtype=torch.float64),
    )
    assert optimizer.param_groups[0]["tau"] == pytest.approx(1 / 1.000005, rel=0, abs=1e-12)
    assert pdap.insertion_kind == "profile"
    assert pdap.solver["max_ls_iter"] == 500
    assert pdap.fit_outer_iterations == 20
    assert pdap.ins_merge_tol == 1e-2

    assert isinstance(network, torch.nn.Module)
    network.set_atoms(
        torch.tensor([[1., 0.]], dtype=torch.float64),
        torch.tensor([0.], dtype=torch.float64),
        torch.tensor([2.], dtype=torch.float64),
    )
    x = torch.tensor([[-1., 3.], [2., 3.]], dtype=torch.float64)
    value, _ = network(x)
    torch.testing.assert_close(value, x.new_tensor([[0.], [4.]]))
    assert network.input_dim == 2


def test_construct_network_from_loaded_dataset() -> None:
    """Loading the configured dataset and building the model gives input_dim=2."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["env.verbose=false"])
    if not (DATA_DIR / cfg.data.path).exists():
        pytest.skip(f"dataset not present (rawdata/ is gitignored): {cfg.data.path}")
    data = load_value_samples(cfg.data.path)
    model_config = resolve_model(cfg.model, input_dim=data.x.shape[1])
    validate_model_config(model_config)
    model = SparseNetwork(model_config)
    assert model.input_dim == 2


@pytest.mark.parametrize("specifications,error,message", [
    ({"activation": "unknown"}, ValueError, "unknown activation"),
    ({"activation": None}, TypeError, "activation must be a registry name or callable"),
    ({"activation": 3}, TypeError, "activation must be a registry name or callable"),
    *[({"input_dim": dim}, ValueError, "input_dim must be a positive integer")
      for dim in (0, -1, 2.0, True, None)],
    *[({"power": power}, ValueError, "power must be positive and finite")
      for power in (-1., 0., float("nan"), float("inf"), -float("inf"))],
])
def test_prediction_contracts_are_owned_by_config(specifications, error, message):
    arguments = dict(specifications)
    input_dim = arguments.pop("input_dim", 2)
    with pytest.raises(error, match=message):
        resolve_model(arguments, input_dim)


@pytest.mark.parametrize("name,use_sphere,order", [
    ("relu", True, None), ("relu2", True, None), ("softplus", False, 2.01),
])
def test_prediction_resolution_preserves_registered_identity_and_geometry(name, use_sphere, order):
    activation = get_activation(name)
    for selection in (name, activation):
        resolved = resolve_model(
            {"activation": selection, "power": 4., "moment_order": 2.01}, 2,
        )
        assert resolved["activation_name"] == name
        assert resolved["activation"] is activation
        assert resolved["use_sphere"] is use_sphere
        assert resolved["moment_order"] == order


def test_prediction_resolution_preserves_unregistered_callable():
    activation = lambda x: x.square()
    resolved = resolve_model({"activation": activation, "power": 1.5}, 2)
    assert resolved["activation_name"] is None and resolved["activation"] is activation
    assert resolved["use_sphere"] is False and resolved["moment_order"] is None


def test_model_kind_contract_is_owned_by_config():
    config = ModelConfig(kind="unsupported")
    with pytest.raises(ValueError, match="only supports kind='signed'"):
        resolve_model(config, 2)


def test_normalization_follows_the_activation_and_is_not_configurable():
    for activation, normalized in (("softplus", True), ("tanh", True), ("relu", False)):
        assert SparseNetwork(resolve_model({"activation": activation}, 2)).normalized is normalized
    with pytest.raises(TypeError, match="normalized"):
        resolve_model({"activation": "tanh", "normalized": False}, 1)


# Algorithm 1 settings that are valid unless a case overrides them.
ALGORITHM1 = {"activation": "softplus", "alpha": 1e-4, "gamma": 1.0, "moment_order": 2.0}


@pytest.mark.parametrize("model,training,message", [
    ({**ALGORITHM1, "power": 2.0}, {}, "Algorithm 1.*power == 1"),
    *[({"power": power}, {}, "supports activation powers 1, 2, 3")
      for power in (1.5, 4.0, 1.0 + 5e-13, 3.0 - 5e-13)],
    ({"insertion": "finite_step", "activation": "relu", "power": 4.0, "gamma": 0.0}, {},
     "powers 1, 2, 3"),
    *[({"activation": "relu", "power": power, "gamma": 0.1}, {},
       "power penalty requires gamma == 0") for power in (2.0, 3.0)],
    ({"insertion": "finite_step", "activation": "relu", "power": 2.0, "gamma": 0.1}, {},
     "gamma == 0"),
    ({"insertion": "finite_step", "activation": "softplus", "gamma": 0.0, "moment_order": 2.0}, {},
     "sphere activation"),
    ({**ALGORITHM1, "insertion": "nonsense"}, {}, "insertion must be 'profile' or 'finite_step'"),
    (ALGORITHM1, {"insert_mode": "nonsense"}, "training.insert_mode must be one of"),
    (ALGORITHM1, {"method": "nonsense"}, "training.method must be one of"),
])
def test_incoherent_configurations_are_rejected(model, training, message):
    with pytest.raises(ValueError, match=message):
        validate_model_config(resolve_model(model, 2))
        validate_optimizer_config(TrainingConfig(**training))


@pytest.mark.parametrize("order", [None, 0., -1., float("nan"), float("inf"), -float("inf")])
def test_nonhomogeneous_training_requires_an_explicit_positive_finite_order(order):
    config = ModelConfig(activation="softplus", moment_order=order)
    with pytest.raises(ValueError, match="moment_order"):
        validate_model_config(resolve_model(config, 2))


@pytest.mark.parametrize("activation,bound", [
    ("softplus", 1.), ("tanh", 1.), ("gaussian", 1.),
    ("gausscent_1", 1.), ("matern52", 1.), ("gelu_squared", 2.),
])
def test_training_order_exceeds_the_declared_h1_growth_bound(activation, bound):
    with pytest.raises(ValueError, match="moment_order must exceed"):
        validate_model_config(resolve_model(ModelConfig(activation=activation, moment_order=bound), 2))
    config = resolve_model(ModelConfig(activation=activation, moment_order=bound + .01), 2)
    validate_model_config(config)
    network = SparseNetwork(config)
    assert network.moment_order == bound + .01


def test_activation_without_growth_metadata_keeps_explicit_numerical_training():
    config = ModelConfig(activation="snake_b0_25", moment_order=.5)
    resolved = resolve_model(config, 2)
    validate_model_config(resolved)
    assert SparseNetwork(resolved).moment_order == .5
    config.moment_order = None
    with pytest.raises(ValueError, match="explicit moment_order"):
        validate_model_config(resolve_model(config, 2))


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_an_invalid_iteration_budget_is_rejected(iterations):
    with pytest.raises(ValueError, match="fit_outer_iterations must be a nonnegative integer"):
        validate_optimizer_config(TrainingConfig(fit_outer_iterations=iterations))


@pytest.mark.parametrize("activation,order", [("leaky_relu", None), ("gelu_squared", 2.01)])
def test_log_penalty_preset_supports_both_formulations(activation, order):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "+experiment=log_penalty", "+model=profile", f"model.activation={activation}",
        ])
    assert cfg.model.moment_order == 2.01
    model_config = resolve_model(cfg.model, 2)
    validate_model_config(model_config)
    assert SparseNetwork(model_config).moment_order == order
