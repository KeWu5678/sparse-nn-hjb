"""Tests for the Hydra config system: composition, model groups, config→trainer."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException

from src.config import get_activation, get_use_sphere
from src.config.schema import ExperimentConfig, ModelConfig
from src.data import DATA_DIR, load_value_samples
from src.PDAP import PDAP
from src.problems import SparseProblem

# Must be registered before any compose() call below.
ConfigStore.instance().store(name="config_schema", node=ExperimentConfig)


def test_compose_defaults() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
    assert cfg.model.kind == "signed"
    assert cfg.model.insertion == "profile"
    assert cfg.model.activation == "relu"
    assert cfg.model.power == 1.0
    assert cfg.model.alpha == 1e-5
    assert cfg.model.moment_order is None
    assert "c_init" not in cfg.model
    assert "th" not in cfg.model
    assert cfg.training.num_iterations == 10
    assert cfg.training.max_ls_iter == 500
    assert cfg.training.ins_merge_tol == 1e-2
    with initialize(version_base=None, config_path="../conf"):
        selected = compose(config_name="config", overrides=["+data=vdp"])
    assert cfg.data.path == selected.data.path
    assert cfg.data.path.endswith(".npz")
    assert cfg.data.normalize is True
    assert cfg.env.seed == 42


def test_log_mixture_is_not_configurable() -> None:
    with initialize(version_base=None, config_path="../conf"):
        with pytest.raises(ConfigCompositionException, match="model.th"):
            compose(config_name="config", overrides=["model.th=0.5"])


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
    assert SparseProblem.from_config(alg1.model, 2).normalized
    assert "objective" not in alg1.model
    assert "moment_beta" not in alg1.model


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
    """The trainer and sole network/problem read their respective configuration."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model.gamma=0.5", "env.verbose=false"])

    pdap = PDAP(cfg.training, insertion=cfg.model.insertion)
    problem = SparseProblem.from_config(cfg.model, 2)
    assert problem.alpha == cfg.model.alpha == 1e-5
    # Configured alpha and gamma determine the log penalty's normal-map scaling.
    assert problem.inverse_step(None) == pytest.approx(1.000005, rel=0, abs=1e-12)
    assert pdap.insertion_kind == "profile"
    assert pdap.solver["max_ls_iter"] == 500
    assert pdap.fit_outer_iterations == 20
    assert pdap.ins_merge_tol == 1e-2

    assert isinstance(problem, torch.nn.Module)
    problem.set_atoms(torch.tensor([[1., 0.]]), torch.tensor([0.]), torch.tensor([2.]))
    x = torch.tensor([[-1., 3.], [2., 3.]], dtype=torch.float64)
    value, _ = problem(x)
    torch.testing.assert_close(value, x.new_tensor([[0.], [4.]]))
    assert problem.input_dim == 2


def test_construct_problem_from_loaded_dataset() -> None:
    """Loading the configured dataset and building the model gives input_dim=2."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["env.verbose=false"])
    if not (DATA_DIR / cfg.data.path).exists():
        pytest.skip(f"dataset not present (rawdata/ is gitignored): {cfg.data.path}")
    data = load_value_samples(cfg.data.path)
    model = SparseProblem.from_config(cfg.model, input_dim=data["x"].shape[1])
    assert model.input_dim == 2


def test_activation_resolver() -> None:
    assert get_activation("relu") is torch.relu
    assert callable(get_activation("matern52"))


def test_use_sphere_bundled_with_activation() -> None:
    # use_sphere is co-located with the activation in the registry, not configured.
    assert get_use_sphere("relu") is True
    assert get_use_sphere("matern52") is False


def test_use_sphere_derives_from_activation() -> None:
    with initialize(version_base=None, config_path="../conf"):
        default = compose(config_name="config", overrides=["env.verbose=false"])
        smooth = compose(config_name="config",
                         overrides=["model.activation=matern52", "model.moment_order=2.0", "env.verbose=false"])
    # default activation is relu (homogeneous -> sphere); matern52 is not
    assert SparseProblem.from_config(default.model, 2).use_sphere is True
    assert SparseProblem.from_config(smooth.model, 2).use_sphere is False


@pytest.mark.parametrize("order", [None, 0., -1., float("nan"), float("inf"), -float("inf")])
def test_nonhomogeneous_training_requires_an_explicit_positive_finite_order(order):
    config = ModelConfig(activation="softplus", moment_order=order)
    with pytest.raises(ValueError, match="moment_order"):
        SparseProblem.from_config(config, 2)


@pytest.mark.parametrize("activation,bound", [
    ("softplus", 1.), ("tanh", 1.), ("gaussian", 1.),
    ("gausscent_1", 1.), ("matern52", 1.), ("gelu_squared", 2.),
])
def test_training_order_exceeds_the_declared_h1_growth_bound(activation, bound):
    with pytest.raises(ValueError, match="moment_order must exceed"):
        SparseProblem.from_config(ModelConfig(activation=activation, moment_order=bound), 2)
    problem = SparseProblem.from_config(ModelConfig(activation=activation, moment_order=bound + .01), 2)
    assert problem.moment_order == bound + .01


def test_activation_without_growth_metadata_keeps_explicit_numerical_training():
    config = ModelConfig(activation="snake_b0_25", moment_order=.5)
    assert SparseProblem.from_config(config, 2).moment_order == .5
    config.moment_order = None
    with pytest.raises(ValueError, match="explicit moment_order"):
        SparseProblem.from_config(config, 2)


@pytest.mark.parametrize("activation,order", [("leaky_relu", None), ("gelu_squared", 2.01)])
def test_log_penalty_preset_supports_both_formulations(activation, order):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "+experiment=log_penalty", "+model=profile", f"model.activation={activation}",
        ])
    assert cfg.model.moment_order == 2.01
    assert SparseProblem.from_config(cfg.model, 2).moment_order == order
