"""Tests for the Hydra config system: composition, model groups, config→trainer."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize

import src.config.store  # noqa: F401  — registers `config_schema`
from src.config import get_activation, get_use_sphere
from src.data import load_value_samples
from src.models import build_model
from src.paths import DATA_DIR
from src.PDAP import PDAP


def test_compose_defaults() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
    assert cfg.model.kind == "signed"
    assert cfg.model.insertion == "profile"
    assert cfg.model.activation == "relu"
    assert cfg.model.power == 1.0
    assert cfg.model.alpha == 1e-5
    assert cfg.training.num_iterations == 10
    assert cfg.training.max_ls_iter == 500
    assert cfg.training.ins_merge_tol == 1e-2
    assert cfg.data.path.endswith("VDP_beta_0.1_grid_30x30.npy")
    assert cfg.data.normalize is True
    assert cfg.env.seed == 42


def test_model_groups() -> None:
    with initialize(version_base=None, config_path="../conf"):
        sc = compose(config_name="config", overrides=["model=semiconcave"])
        fs = compose(config_name="config", overrides=["model=frac_exp_penalty"])
    assert sc.model.kind == "semiconcave"
    assert sc.model.insertion == "profile"
    # frac_exp_penalty config group = signed + finite_step
    assert fs.model.kind == "signed"
    assert fs.model.insertion == "finite_step"


def test_curated_experiment_configs_compose() -> None:
    with initialize(version_base=None, config_path="../conf"):
        log_pen = compose(config_name="config", overrides=["+experiment=pendulum/log_penalty"])
        frac = compose(config_name="config", overrides=["+experiment=vdp/frac_exp_penalty"])

    assert log_pen.name == "pendulum_log_penalty"
    assert log_pen.model.power == 1.0
    assert log_pen.data.path.startswith("Pendulum")

    assert frac.name == "vdp_frac_exp_penalty"
    assert frac.model.insertion == "finite_step"
    assert frac.model.power == 2.0


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
