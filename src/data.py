"""Value samples: the dataset type, its file format, normalization, splitting, and batching.

Open-loop generators produce and PDAP training consumes one type,
:class:`ValueSamples`, holding ``n`` value samples ``(x, V(x), dV(x))`` as
read-only float64 arrays:

  * ``x``  has shape ``(n, d)``
  * ``v``  has shape ``(n,)``
  * ``dv`` has shape ``(n, d)``

On disk a dataset is an ``.npz`` file with arrays ``x``, ``v``, and ``dv``.
Tensor batches carry ``v`` with shape ``(N, 1)`` to match network outputs; that
reshape happens only in :func:`to_tensors`.

Normalization is a pre-training data transform, not optimizer behavior.  The
gradient is transformed by the chain rule so normalized samples remain
consistent: if ``x_norm = x / s_x`` and ``v_norm = v / s_v``, then
``dv_norm_i = dv_i * s_x_i / s_v``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

#: Dataset root. Datasets are referenced by bare filename everywhere else; this
#: is the one place that name becomes a path (see :func:`load_value_samples`).
DATA_DIR = Path(__file__).resolve().parents[1] / "rawdata" / "data"

TensorSamples = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ValueSamples:
    """Read-only float64 value samples; shapes are given in the module docstring."""

    x: np.ndarray
    v: np.ndarray
    dv: np.ndarray

    def __post_init__(self) -> None:
        # Copy before freezing so the caller's arrays stay writable.
        x = np.array(self.x, dtype=np.float64)
        v = np.array(self.v, dtype=np.float64).reshape(-1)
        dv = np.array(self.dv, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError(f"x must have shape (n, d) with d >= 1, got {x.shape}")
        if dv.shape != x.shape:
            raise ValueError(f"dv must have shape {x.shape}, got {dv.shape}")
        if v.shape != (x.shape[0],):
            raise ValueError(f"v must have {x.shape[0]} values, got {v.size}")
        for name, array in (("x", x), ("v", v), ("dv", dv)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def empty(cls, dimension: int) -> ValueSamples:
        return cls(np.empty((0, dimension)), np.empty(0), np.empty((0, dimension)))

    @classmethod
    def concatenate(cls, chunks: Sequence[ValueSamples]) -> ValueSamples:
        """Stack one or more chunks in order; prepend :meth:`empty` if there may be none."""
        if not chunks:
            raise ValueError("concatenate needs at least one chunk; use ValueSamples.empty(dimension)")
        return cls(
            x=np.concatenate([chunk.x for chunk in chunks], axis=0),
            v=np.concatenate([chunk.v for chunk in chunks], axis=0),
            dv=np.concatenate([chunk.dv for chunk in chunks], axis=0),
        )

    def save_npz(self, path: str | Path) -> Path:
        """Write the dataset file at an explicit path, creating its directory."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, x=self.x, v=self.v, dv=self.dv)
        return output_path


def load_value_samples(path: str | Path) -> ValueSamples:
    """Load a ``.npy``/``.npz`` value-sample file.

    ``path`` resolves under :data:`src.data.DATA_DIR`; an absolute ``path`` is
    used unchanged (``DATA_DIR / abs`` returns ``abs``). This is the single place
    data paths are resolved — callers pass a bare filename, not a built path.

    ``allow_pickle=True`` supports trusted legacy ``.npy`` files containing a
    dictionary. New project datasets use plain arrays in ``.npz`` files.
    """
    raw = np.load(DATA_DIR / path, allow_pickle=True)
    return _as_value_samples(raw)


def _as_value_samples(raw: Any) -> ValueSamples:
    if hasattr(raw, "files"):
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    elif isinstance(raw, np.ndarray) and raw.dtype.fields is not None:
        arrays = {name: np.asarray(raw[name]) for name in raw.dtype.fields}
    elif isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
        arrays = {name: np.asarray(value) for name, value in raw.item().items()}
    elif isinstance(raw, dict):
        arrays = {name: np.asarray(value) for name, value in raw.items()}
    else:
        raise TypeError("value-sample data must contain x, v, and dv arrays")

    missing = {"x", "v", "dv"} - set(arrays)
    if missing:
        raise KeyError(f"value-sample data is missing keys: {sorted(missing)}")
    return ValueSamples(arrays["x"], arrays["v"], arrays["dv"])


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, the identity recorded for datasets and generator sources."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize_value_samples(samples: ValueSamples) -> tuple[ValueSamples, ValueSampleNormalizer]:
    """Fit a normalizer and return normalized value samples plus the transform."""
    normalizer = ValueSampleNormalizer.fit(samples)
    return normalizer.normalize(samples), normalizer


@dataclass(frozen=True)
class ValueSampleNormalizer:
    """Reversible max-absolute scaling for value samples."""

    x_scale: np.ndarray
    v_scale: float

    @classmethod
    def fit(cls, samples: ValueSamples) -> "ValueSampleNormalizer":
        x_scale = np.maximum(np.max(np.abs(samples.x), axis=0), 1e-12)
        v_scale = float(np.maximum(np.max(np.abs(samples.v)), 1e-12))
        return cls(x_scale=np.asarray(x_scale, dtype=np.float64), v_scale=v_scale)

    def normalize(self, samples: ValueSamples) -> ValueSamples:
        return ValueSamples(
            x=samples.x / self.x_scale,
            v=samples.v / self.v_scale,
            dv=samples.dv * (self.x_scale / self.v_scale),
        )

    def denormalize_prediction(self, value: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value_phys = np.asarray(value, dtype=np.float64) * self.v_scale
        gradient_phys = np.asarray(gradient, dtype=np.float64) * (self.v_scale / self.x_scale)
        return value_phys, gradient_phys

    def denormalize_tensors(
        self, value: torch.Tensor, gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover original V and dV units, preserving dtype, device, and autograd."""
        gradient_scale = gradient.new_tensor(self.v_scale / self.x_scale)
        return value * self.v_scale, gradient * gradient_scale

    def to_dict(self) -> dict[str, Any]:
        return {"x_scale": self.x_scale.tolist(), "v_scale": self.v_scale}


def make_dataloaders(
    samples: ValueSamples,
    train_fraction: float = 0.9,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split prepared samples and construct independent train/validation loaders.

    The split uses the existing global NumPy RNG. ``seed`` controls only each
    loader's private PyTorch generator; normalization stays an explicit transform
    before this function is called.
    """
    train, valid = split_value_samples(samples, train_fraction)
    return (
        make_dataloader(train, batch_size=batch_size, num_workers=num_workers, seed=seed),
        make_dataloader(valid, batch_size=batch_size, num_workers=num_workers, seed=seed),
    )


def split_value_samples(
    samples: ValueSamples, train_fraction: float = 0.9,
) -> tuple[TensorSamples, TensorSamples]:
    """Split value samples into (train, valid) float64 tensor tuples (x, v, dv).

    A **random** floor(N * train_fraction) / remainder split. Samples are stored
    trajectory-by-trajectory, so a sequential slice would hand the whole validation
    set to the last few trajectories — a spatially clustered wedge, not a
    representative hold-out. The permutation draws from the global NumPy RNG, which
    the run script seeds (``env.seed``) before calling, so the split is
    reproducible per run. :func:`to_tensors` gives ``v`` its ``(N, 1)`` batch shape.
    """
    n = samples.size
    split = int(n * train_fraction)
    if not 0 < split < n:
        raise ValueError(
            f"train_fraction={train_fraction} gives a {split}/{n - split} split; "
            "both sides must be non-empty"
        )
    perm = np.random.permutation(n)
    return to_tensors(samples, perm[:split]), to_tensors(samples, perm[split:])


def to_tensors(samples: ValueSamples, rows: np.ndarray | slice = slice(None)) -> TensorSamples:
    """Float64 tensors ``(x, v, dv)`` of the selected rows, with ``v`` shaped ``(N, 1)``."""
    return (
        torch.tensor(samples.x[rows], dtype=torch.float64),
        torch.tensor(samples.v[rows], dtype=torch.float64).reshape(-1, 1),
        torch.tensor(samples.dv[rows], dtype=torch.float64),
    )


def make_dataloader(
    samples: TensorSamples,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Deliver aligned value samples in stable order, retaining every sample.

    By default the entire split is one batch. The dataset retains the supplied
    tensors without transforming or copying them. A private generator prevents
    iterator creation and worker seeding from consuming the optimizer's global
    PyTorch random stream.
    """
    x, v, dv = samples
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, d), got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("value samples must be non-empty")
    if v.shape != (x.shape[0], 1):
        raise ValueError(f"v must have shape ({x.shape[0]}, 1), got {v.shape}")
    if dv.shape != x.shape:
        raise ValueError(f"dv must have shape {x.shape}, got {dv.shape}")
    return DataLoader(
        TensorDataset(*samples),
        batch_size=x.shape[0] if batch_size is None else batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
    )


def full_batch(loader: DataLoader) -> TensorSamples:
    """Assemble one complete loader pass for full-data mathematical evaluation.

    This deliberately materializes the supplied samples. A single delivered
    batch is reused directly; smaller batches are concatenated in delivery order.
    """
    if loader.batch_sampler is None or loader.drop_last or getattr(loader.batch_sampler, "drop_last", False):
        raise ValueError("full-data evaluation requires batching without dropping samples")
    batches = list(loader)
    if not batches:
        raise ValueError("full-data evaluation requires nonempty batches")
    if len(batches) == 1:
        x, v, dv = batches[0]
    else:
        x, v, dv = (torch.cat(fields) for fields in zip(*batches, strict=True))
    return x, v, dv
