"""Tests for the Hydra config system: composition, model groups, config→trainer."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from src.config import get_activation, get_use_sphere
from src.config.schema import ExperimentConfig
from src.data import load_value_samples
from src.models import build_model
from src.paths import DATA_DIR
from src.PDAP import PDAP

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
    assert "c_init" not in cfg.model
    assert cfg.training.num_iterations == 10
    assert cfg.training.max_ls_iter == 500
    assert cfg.training.ins_merge_tol == 1e-2
    assert cfg.data.path.endswith("VDP_beta_0.1_grid_30x30.npy")
    assert cfg.data.normalize is True
    assert cfg.env.seed == 42


def test_model_groups() -> None:
    with initialize(version_base=None, config_path="../conf"):
        fs = compose(config_name="config", overrides=["+model=finite_step"])
        alg1 = compose(config_name="config", overrides=[
            "+model=profile",
            "model.activation=softplus",
        ])
    # finite_step config group = signed + finite_step
    assert fs.model.kind == "signed"
    assert fs.model.insertion == "finite_step"
    assert PDAP(alg1).objective.normalized
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
    """The trainer reads its config; the model is built separately by build_model."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model.gamma=0.5", "env.verbose=false"])

    pdap = PDAP(cfg)
    # objective + solver + insertion settings live on the trainer (config only)
    assert pdap.objective.alpha == cfg.model.alpha == 1e-5
    assert pdap.objective.gamma == 0.5
    assert pdap.insertion_kind == "profile"
    assert pdap.solver.max_ls_iter == 500
    assert pdap.fit_outer_iterations == 20
    assert pdap.ins_merge_tol == 1e-2

    model = build_model(cfg, input_dim=2)
    assert type(model).__name__ == "SignedModel"
    assert model.power == cfg.model.power
    assert model.activation is torch.relu
    assert model.input_dim == 2


def test_build_model_from_loaded_dataset() -> None:
    """Loading the configured dataset and building the model gives input_dim=2."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["env.verbose=false"])
    if not (DATA_DIR / cfg.data.path).exists():
        pytest.skip(f"dataset not present (rawdata/ is gitignored): {cfg.data.path}")
    data = load_value_samples(cfg.data.path)
    model = build_model(cfg, input_dim=data["x"].shape[1])
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
                         overrides=["model.activation=matern52", "env.verbose=false"])
    # default activation is relu (homogeneous -> sphere); matern52 is not
    assert PDAP(default)._use_sphere is True
    assert PDAP(smooth)._use_sphere is False


def test_algorithm2_provenance_describes_search_and_coefficient_solver() -> None:
    with initialize(version_base=None, config_path="../conf"):
        l1 = compose(
            config_name="config",
            overrides=["+model=finite_step", "model.power=1", "env.verbose=false"],
        )
        fractional = compose(
            config_name="config",
            overrides=["+model=finite_step", "model.power=2", "env.verbose=false"],
        )
        profile = compose(
            config_name="config",
            overrides=["+model=profile", "env.verbose=false"],
        )

    assert PDAP(l1).algorithm_provenance == {
        "candidate_starts": "random_sphere_multistart",
        "coefficient_solver": "soft_threshold",
    }
    assert PDAP(fractional).algorithm_provenance == {
        "candidate_starts": "random_sphere_multistart",
        "coefficient_solver": "global_prox_warmstart_scale",
        "existing_support_filter": "numerical_repeat_only",
        "existing_support_cosine_gap_tol": 1e-8,
        "rho": 0.5,
    }
    assert PDAP(profile).algorithm_provenance == {}
