#!/usr/bin/env python3
"""Generic Hydra entry point: train a PDAP model on any value/gradient dataset.

A run = pick a registered model + a data source, override the rest::

    python scripts/train.py +model=profile +data=vdp model.activation=softplus model.moment_order=2.01
    python scripts/train.py -m +experiment=log_penalty

This entry is domain-agnostic — it loads a ``.npy``/``.npz`` with keys ``x, v, dv`` and
fits the PDAP model described by the config. With no group selected, the schema
defaults reproduce a single VDP signed-profile run. Every job is recorded as one
MLflow run, so ``MLFLOW_TRACKING_URI`` must name a reachable tracking store.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

# Compatibility shim: Python 3.14 added ArgumentParser._check_help, which rejects
# Hydra 1.3's --shell-completion help (it contains a literal '%'). The guard is a
# pure help-lint added in 3.14; neutralizing it restores pre-3.14 behavior so
# @hydra.main can build its CLI parser. Remove once Hydra ships a 3.14-safe release.
if hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda _self, _action: None

import hydra
import mlflow
import numpy as np
import structlog
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import make_penalty, resolve_model, validate_model_config, validate_optimizer_config
from src.config.schema import ExperimentConfig
from src.data import (
    DATA_DIR,
    file_sha256,
    load_arrays,
    load_value_samples,
    make_dataloaders,
    normalize_value_samples,
)
from src.eval import region_metrics
from src.logging import configure_logging
from src.models import SparseNetwork
from src.PDAP import PDAP, History
from src.tracking import git_tags, log_config, log_fit, require_tracking_store
from src.training import run_training

logger = structlog.get_logger(__name__)

# `conf/config.yaml` pulls in `config_schema` via its defaults list, so the typed
# base must be in the ConfigStore before @hydra.main composes below.
ConfigStore.instance().store(name="config_schema", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # This setup owns job logging; Hydra supplies each run's output directory.
    # `env.verbose` controls stderr, and `env.log_file` overrides the JSONL path.
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    log_file = cfg.env.log_file or (run_dir / "run.jsonl")
    configure_logging(console=cfg.env.verbose, level=cfg.env.log_level, log_file=log_file)
    run_name = run_name_from_config(cfg, hydra_cfg=hydra_cfg)
    with structlog.contextvars.bound_contextvars(
        run_name=run_name, experiment=cfg.name, seed=cfg.env.seed,
    ):
        logger.info("run_started")
        try:
            run_id = _tracked_experiment(cfg, hydra_cfg, run_name)
        except Exception:
            logger.exception("run_failed")
            raise
        logger.info("run_completed", run_id=run_id)


def _tracked_experiment(cfg: DictConfig, hydra_cfg, run_name: str) -> str:
    """Run the experiment inside one MLflow run and return its MLflow run ID.

    The run records configuration and provenance before training, so a run that
    raises is kept, marked FAILED, with what it was attempting.
    """
    require_tracking_store()
    # Pass the ID: when parallel jobs race to create the experiment, MLflow
    # returns it without activating it, and start_run would use a stale one.
    experiment = mlflow.set_experiment(cfg.name)
    tags = {**git_tags(REPO_ROOT), **hydra_tags(hydra_cfg)}
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name,
                          tags=tags) as run:
        log_config(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params(hydra_choice_params(hydra_cfg))
        _run_experiment(cfg)
    return run.info.run_id


def _run_experiment(cfg: DictConfig) -> None:
    """Train the configured model and record its results in the active MLflow run."""
    set_seed(cfg.env.seed)
    validate_optimizer_config(cfg.training)
    mlflow.set_tags(provenance_tags(cfg))

    # Data preprocessing is external to the network. The trainer holds only
    # configuration and returns a History.
    samples = load_value_samples(cfg.data.path)
    data, normalizer = normalize_value_samples(samples) if cfg.data.normalize else (samples, None)
    input_dim = data.x.shape[1]
    logger.info("data_loaded", path=cfg.data.path, input_dim=input_dim)
    model_config = resolve_model(cfg.model, input_dim)
    validate_model_config(model_config)

    history = _train(cfg, model_config, data, normalizer)
    summary = history.summary()
    logger.info(
        "evaluation_summary", best_iteration=summary["best_iteration"],
        neurons=summary["neurons"], rel_l2_val=summary["rel_l2_val"],
        rel_grad_val=summary["rel_grad_val"], rel_h1_val=summary["rel_h1_val"],
    )
    extra = _evaluate_regions(cfg, model_config, history, samples, normalizer)
    log_fit(history, normalizer=normalizer, extra=extra)


def _train(cfg: DictConfig, model_config: dict, data, normalizer) -> History:
    train_loader, valid_loader = make_dataloaders(
        data, cfg.data.train_fraction, seed=cfg.env.seed,
    )
    network = SparseNetwork(model_config)
    penalty = make_penalty(model_config)
    trainer = PDAP(cfg.training, penalty, network, train_loader, insertion=model_config["insertion"])
    return run_training(
        trainer, valid_loader,
        num_iterations=cfg.training.num_iterations,
        display_every=cfg.training.display_every,
        normalizer=normalizer,
    )


def _evaluate_regions(cfg: DictConfig, model_config: dict, history: History, samples, normalizer) -> dict:
    """Regional errors of the selected checkpoint, when ``eval.kind=region_split``.

    The default ``global`` evaluation adds nothing. ``region_split`` (pendulum,
    via conf/data/pendulum.yaml) scores the switching tube and the rest on the
    region-evaluation pool and, with a distance cache, the dataset's
    distance-binned error profile.
    """
    if cfg.eval.kind != "region_split":
        return {}
    if not cfg.eval.eval_pool:
        raise ValueError("eval.kind=region_split requires eval.eval_pool")
    network = history.restore(SparseNetwork(model_config))
    pool = load_arrays(cfg.eval.eval_pool, "x", "v", "dv", "distance")
    distance = (load_arrays(cfg.eval.distance_cache, "distance")["distance"]
                if cfg.eval.distance_cache else None)
    region = region_metrics(network, normalizer, pool, tube_radius=cfg.eval.tube_radius,
                            samples=samples, distance=distance)
    logger.info(
        "region_summary",
        rel_h1_switching=region["rel_h1_switching"] if region["count_switching"] else None,
        count_switching=int(region["count_switching"]),
        rel_h1_rest=region["rel_h1_rest"] if region["count_rest"] else None,
        count_rest=int(region["count_rest"]),
    )
    return region


def run_name_from_config(
    cfg: DictConfig,
    *,
    hydra_cfg=None,
    today: str | None = None,
    suffix: str | None = None,
) -> str:
    """Human-readable MLflow run name: experiment, data choice, date, random suffix."""
    choices = _select(hydra_cfg, "runtime.choices") if hydra_cfg is not None else {}
    data_choice = (choices or {}).get("data") or Path(str(cfg.data.path)).stem
    run_date = today or datetime.now().strftime("%Y%m%d")
    run_suffix = suffix or secrets.token_hex(2)
    return "_".join([_slug(cfg.name), _slug(data_choice), run_date, run_suffix])


def hydra_tags(hydra_cfg) -> dict[str, str]:
    """Where Hydra placed this job and which command-line overrides produced it."""
    tags = {"hydra.output_dir": str(_select(hydra_cfg, "runtime.output_dir"))}
    for key in ("name", "id", "num"):
        value = _select(hydra_cfg, f"job.{key}")
        if value is not None:
            tags[f"hydra.job.{key}"] = str(value)
    if overrides := _select(hydra_cfg, "overrides.task"):
        tags["hydra.overrides.task"] = json.dumps(overrides)
    return tags


def hydra_choice_params(hydra_cfg) -> dict[str, str]:
    """The config-group options Hydra composed, e.g. ``hydra.choice.data``."""
    choices = _select(hydra_cfg, "runtime.choices") or {}
    return {f"hydra.choice.{key}": str(value) for key, value in choices.items()}


def provenance_tags(cfg: DictConfig) -> dict[str, str]:
    """What a later reader needs to trust the stored results.

    Hashes of the files training and evaluation read; ``src.results`` refuses a
    dataset whose bytes changed.
    """
    tags = {"data_sha256": file_sha256(DATA_DIR / cfg.data.path)}
    if cfg.eval.kind == "region_split":
        for name in ("eval_pool", "distance_cache"):
            tags[f"{name}_sha256"] = file_sha256(DATA_DIR / cfg.eval[name])
    return tags


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", str(value).lower())
    return slug or "run"


def _select(hydra_cfg, path: str):
    return _to_container(OmegaConf.select(hydra_cfg, path))


def _to_container(value):
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


if __name__ == "__main__":
    main()
