"""Read saved fits and evaluate the original-scale value function.

Run Records own the activation, checkpoint artifact and data transform. This
module consolidates the reconstruction previously repeated by figure scripts;
it does not select experiments, change the objective, or write artifacts.
Pickles are trusted local Run Artifacts, as in the existing experiment loader.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config.activations import get_activation
from .data import TensorSamples, ValueSampleNormalizer, load_value_samples
from .eval import relative_errors
from .models.net import ShallowNetwork


def load_history(history: Any, run_index: int = 0) -> Any:
    """Read a fit pickle, or unwrap an existing History/legacy run container."""
    if isinstance(history, (str, os.PathLike)):
        with Path(history).open("rb") as stream:
            history = pickle.load(stream)
    if isinstance(history, (list, tuple)):
        history = history[run_index]
    return history


def _field(history: Any, name: str) -> Any:
    return getattr(history, name) if hasattr(history, name) else history[name]


def history_atoms(
    history: Any, iteration: int | None = None, run_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return recorded (W, b, c), defaulting to the saved best iteration.

    Shapes are (neurons, dimensions), (neurons,), (1, neurons). No pruning,
    moment weighting, or homogeneous rescaling is applied during restoration.
    """
    history = load_history(history, run_index)
    inner = _field(history, "inner_weights")
    i = int(_field(history, "best_iteration")) if iteration is None else int(iteration)
    if not 0 <= i < len(inner):
        raise IndexError(f"checkpoint {i} outside recorded history of length {len(inner)}")
    iw = inner[i]
    return (
        np.asarray(iw["weight"], dtype=np.float64),
        np.asarray(iw["bias"], dtype=np.float64).reshape(-1),
        np.asarray(_field(history, "outer_weights")[i], dtype=np.float64).reshape(1, -1),
    )


def model_from_history(
    history: Any, activation: Any = "relu", power: float = 1.0,
    *, iteration: int | None = None, run_index: int = 0,
) -> ShallowNetwork:
    """Compatibility reconstruction for callers holding a bare fit artifact.

    New reporting code should use load_run so configuration cannot be detached
    from the fitted weights. This retains the existing ShallowNetwork forward.
    """
    w, b, c = history_atoms(history, iteration, run_index)
    if isinstance(activation, str):
        activation = get_activation(activation)
    # Layer construction initializes parameters before replacing them. Reading
    # a fit must not perturb the caller's training/random-split RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = ShallowNetwork(
            [w.shape[1], w.shape[0], 1], activation, p=power,
            inner_weights=w, inner_bias=b, outer_weights=c,
        )
    model.eval()
    return model


def fit_history_path(record: dict[str, Any], record_path: str | Path) -> Path:
    """Prefer the run-adjacent fit, then resolve a declared artifact.

    Relocated run folders keep their local fit even when the record's old
    absolute artifact path still exists. Preflight uses this same policy.
    """
    record_path = Path(record_path)
    local = record_path.parent / f"result_{record.get('run_id', record_path.stem)}.pkl"
    if local.is_file():
        return local
    for artifact in record.get("artifacts", []):
        if artifact["name"] == "fit_history":
            path = Path(artifact["path"])
            candidates = [path] if path.is_absolute() else [record_path.parent / path, path]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f"fit_history for {record_path} not found (local fit: {local})")


@dataclass(frozen=True)
class SavedRun:
    record_path: Path
    record: dict[str, Any]
    history: Any
    model: ShallowNetwork
    normalizer: ValueSampleNormalizer | None
    iteration: int
    normalization_recovered: bool = False


def load_run(record_path: str | Path, *, recover_legacy_normalization: bool = False) -> SavedRun:
    """Restore one run's saved best checkpoint and recorded data transform.

    A missing transform is an error by default. The explicit legacy opt-in
    reconstructs the old max-absolute transform only when the field is absent,
    and requires reproducing the saved training-coordinate validation errors.
    It never repairs a contradictory/null transform or writes the Run Record.
    """
    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    return _restore_run(path, record, fit_history_path(record, path),
                        recover_legacy_normalization=recover_legacy_normalization)


def _restore_run(
    path: Path, record: dict[str, Any], artifact: Path, *, recover_legacy_normalization: bool,
) -> SavedRun:
    if record.get("status") != "completed":
        raise ValueError(f"expected a completed Run Record: {path}")
    cfg = record["config"]
    model_cfg = cfg["model"]
    if model_cfg["kind"] != "signed":
        raise ValueError(f"unsupported saved model kind: {model_cfg['kind']!r}")
    history = load_history(artifact)
    model = model_from_history(history, model_cfg["activation"], model_cfg["power"])
    transform = record.get("normalization")
    normalizer = None
    recovered = False
    if transform is not None:
        sx = np.asarray(transform["x_scale"], dtype=np.float64)
        sv = float(transform["v_scale"])
        if (sx.shape != (model.hidden.weight.shape[1],) or
                not np.all(np.isfinite(sx) & (sx > 0)) or not np.isfinite(sv) or sv <= 0):
            raise ValueError(f"invalid recorded normalization in {path}")
        if cfg["data"].get("normalize") is False:
            raise ValueError(f"normalization contradicts data.normalize=false in {path}")
        normalizer = ValueSampleNormalizer(sx, sv)
    elif cfg["data"].get("normalize", True):
        if not recover_legacy_normalization or "normalization" in record:
            raise ValueError(f"normalized run has no recorded normalization: {path}")
        normalizer = _recover_legacy_normalizer(record, model)
        recovered = True
    return SavedRun(path, record, history, model, normalizer,
                    int(_field(history, "best_iteration")), recovered)


def load_run_for_artifact(
    result_path: str | Path, *, recover_legacy_normalization: bool = False,
) -> SavedRun:
    """Compatibility bridge for the runner's result_<run_id>.pkl convention."""
    path = Path(result_path)
    if not path.stem.startswith("result_"):
        raise ValueError("a bare legacy artifact needs explicit config; use model_from_history")
    record_path = path.with_name(path.stem.removeprefix("result_") + ".json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    # A relocated record may still point at its old copy. Restore and validate
    # the explicitly supplied artifact without first opening that old copy.
    return _restore_run(record_path, record, path,
                        recover_legacy_normalization=recover_legacy_normalization)


def predict_model_physical(
    model: ShallowNetwork, x: Any, normalizer: ValueSampleNormalizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unclipped original-scale (V, dV), shapes (N,1) and (N,d).

    Compatibility seam for an already reconstructed model; reporting callers
    normally use predict_physical so the transform stays bound to its run.
    """
    x = torch.as_tensor(x, dtype=torch.float64, device=model.hidden.weight.device)
    if x.ndim != 2 or x.shape[1] != model.hidden.weight.shape[1]:
        raise ValueError("physical states must have shape (N, model input dimension)")
    if normalizer is not None:
        x = x / torch.as_tensor(normalizer.x_scale, dtype=x.dtype, device=x.device)
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v = model(x)
        dv = torch.autograd.grad(v.sum(), x)[0]
    v, dv = v.detach(), dv.detach()
    if normalizer is not None:
        v, dv = normalizer.denormalize_tensors(v, dv)
    return v, dv


def predict_physical(
    run: SavedRun, x: Any, iteration: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate V and its physical gradient without changing the saved model."""
    model = run.model
    if iteration is not None and iteration != run.iteration:
        cfg = run.record["config"]["model"]
        model = model_from_history(run.history, cfg["activation"], cfg["power"],
                                   iteration=iteration)
    return predict_model_physical(model, x, run.normalizer)


def validation_samples(run: SavedRun) -> TensorSamples:
    """Reproduce the runner's seeded holdout, in original coordinates.

    A local legacy RandomState matches np.random.seed/permutation used by the
    runner, without changing the caller's random state. No new split is drawn.
    """
    return _validation_samples(run.record)


def _validation_samples(record: dict[str, Any], samples=None) -> TensorSamples:
    cfg = record["config"]
    if samples is None:
        samples = load_value_samples(cfg["data"]["path"])
    n = len(samples["x"])
    split = int(n * float(cfg["data"]["train_fraction"]))
    if not 0 < split < n:
        raise ValueError("recorded train_fraction must leave nonempty train and validation sets")
    indices = np.random.RandomState(int(cfg["env"]["seed"])).permutation(n)[split:]
    return tuple(torch.as_tensor(samples[k][indices], dtype=torch.float64)
                 for k in ("x", "v", "dv"))


def _recover_legacy_normalizer(record: dict[str, Any], model) -> ValueSampleNormalizer:
    """Explicit adapter for records written before data transforms were saved."""
    if record.get("metric_coordinates") is not None:
        raise ValueError("normalization recovery is only supported for legacy training-coordinate metrics")
    samples = load_value_samples(record["config"]["data"]["path"])
    normalizer = ValueSampleNormalizer.fit(samples)
    x, v, dv = _validation_samples(record, samples)
    scale = torch.as_tensor(normalizer.x_scale, dtype=x.dtype)
    prediction = predict_model_physical(model, x / scale)
    reproduced = relative_errors(*prediction, v / normalizer.v_scale,
                                 dv * (scale / normalizer.v_scale))
    stored = record["metrics"][-1]["values"]
    expected = [float(stored[f"rel_{name}_val"]) for name in ("l2", "grad", "h1")]
    if not np.allclose(reproduced, expected, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "legacy normalization recovery did not reproduce saved validation errors: "
            f"{record.get('run_id')}: recovered={reproduced}, saved={expected}"
        )
    return normalizer


def value_surface_grid(
    run: SavedRun, *, x_range: tuple[float, float], y_range: tuple[float, float],
    grid_n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the existing 2D mesh with unclipped original-scale predictions."""
    if run.model.hidden.weight.shape[1] != 2:
        raise ValueError("value surfaces require a two-dimensional model")
    x0, x1 = np.meshgrid(np.linspace(*x_range, grid_n), np.linspace(*y_range, grid_n))
    value, _ = predict_physical(run, np.column_stack([x0.ravel(), x1.ravel()]))
    return x0, x1, value.cpu().numpy().reshape(x0.shape)


def validation_metrics(run: SavedRun) -> dict[str, float]:
    """Relative errors of the saved checkpoint's original-scale V and dV."""
    x, v, dv = validation_samples(run)
    values = relative_errors(*predict_physical(run, x), v, dv)
    return dict(zip(("rel_l2_val", "rel_grad_val", "rel_h1_val"), values))


def evaluate_history(run: SavedRun, samples: TensorSamples | None = None) -> dict[str, np.ndarray]:
    """Rescore checkpoints on physical samples (the original holdout by default).

    Rows stay in iteration order. Selecting/sorting a frontier belongs to the
    report, not to metric computation. Historical normalized errors are neither
    reused nor overwritten; the training objective and best checkpoint stay put.
    """
    x, v, dv = validation_samples(run) if samples is None else samples
    n = len(_field(run.history, "inner_weights"))
    errors = np.empty((n, 3), dtype=np.float64)
    neurons = np.empty(n, dtype=int)
    for i in range(n):
        errors[i] = relative_errors(*predict_physical(run, x, i), v, dv)
        neurons[i] = len(history_atoms(run.history, i)[0])
    return {"iteration": np.arange(n), "neurons": neurons,
            "rel_l2": errors[:, 0], "rel_grad": errors[:, 1], "rel_h1": errors[:, 2]}
