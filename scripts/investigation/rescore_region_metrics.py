#!/usr/bin/env python3
"""Rescore region metrics for existing pendulum runs on the region-eval pool.

The recorded per-run region metrics (percentile switching band over the emitted
samples, mostly seen data) are superseded by the consolidated definition:
**fixed switching tube** (distance to the ±2π-tiled switching curve ≤ radius)
on the **region-eval pool** (dense certified two-sided set, training rows
excluded — see ``build_region_eval_pool.py``). This script rebuilds each run's
fit from its result pickle and writes a sidecar
``region_rescored_<run id>.json`` next to the run record with:

  * ``switching_l1_{value,grad,h1}`` / ``rest_l1_{value,grad,h1}`` — region mean
    per-sample absolute L1, normalized by the pool's global mean ‖true‖
  * ``switching_h1`` / ``rest_h1`` — region-local relative H1 (well posed in
    the tube, |V| large there)
  * ``switching_count`` / ``rest_count``, ``tube_radius``, ``pool``

Analyses read the sidecars; run records are never modified.

    python scripts/investigation/rescore_region_metrics.py \
        rawdata/logs/multirun/pendulum/log_penalty \
        rawdata/logs/multirun/pendulum/frac_exp_penalty
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TUBE_RADIUS = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dirs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="recompute even if a sidecar exists")
    return parser.parse_args()


def _load_pool(data_rel: str, cache: dict):
    from src.paths import DATA_DIR
    if data_rel not in cache:
        data_path = DATA_DIR / data_rel
        pool_path = data_path.with_name(data_path.stem + "_region_eval_pool.npz")
        if not pool_path.exists():
            raise FileNotFoundError(
                f"{pool_path} missing — run build_region_eval_pool.py first"
            )
        with np.load(pool_path) as d:
            pool = {k: np.asarray(d[k], dtype=np.float64) for k in ("x", "v", "dv", "distance")}
        pool["switching"] = pool["distance"] <= TUBE_RADIUS
        pool["denom_v"] = float(np.mean(np.abs(pool["v"])))
        pool["denom_g"] = float(np.mean(np.linalg.norm(pool["dv"], axis=1)))
        cache[data_rel] = pool
    return cache[data_rel]


def _predict(net, x: np.ndarray, norm) -> tuple[np.ndarray, np.ndarray]:
    import torch
    v_out = np.empty(len(x))
    g_out = np.empty_like(x)
    for lo in range(0, len(x), 100_000):
        hi = min(lo + 100_000, len(x))
        xn = torch.tensor(x[lo:hi] / norm.x_scale, dtype=torch.float64,
                          requires_grad=True)
        val = net(xn)
        (grad,) = torch.autograd.grad(val.sum(), xn)
        v_out[lo:hi] = val.detach().numpy().reshape(-1) * norm.v_scale
        g_out[lo:hi] = grad.detach().numpy() * (norm.v_scale / norm.x_scale)
    return v_out, g_out


def rescore_run(record_path: Path, pool_cache: dict, norm_cache: dict,
                *, force: bool) -> str:
    from src.config.activations import get_activation
    from src.data import ValueSampleNormalizer, load_value_samples
    from src.models.net import ShallowNetwork
    from src.plots import _best_iteration_atoms

    record = json.loads(record_path.read_text(encoding="utf-8"))
    run_id = record.get("run_id", record_path.stem)
    out_path = record_path.parent / f"region_rescored_{run_id}.json"
    if out_path.exists() and not force:
        return "cached"
    model = record["config"]["model"]
    m = record["metrics"][0]["values"] if record.get("metrics") else {}
    if int(m.get("best_neurons", 0)) == 0:
        return "degenerate"
    data_rel = record["config"]["data"]["path"]
    if "pendulum" not in data_rel.lower():
        return "not-pendulum"
    pool = _load_pool(data_rel, pool_cache)
    if data_rel not in norm_cache:
        norm_cache[data_rel] = ValueSampleNormalizer.fit(load_value_samples(data_rel))
    norm = norm_cache[data_rel]

    pkl = record_path.parent / f"result_{run_id}.pkl"
    with open(pkl, "rb") as f:
        a, b, u = _best_iteration_atoms(pickle.load(f))
    net = ShallowNetwork(layer_sizes=[a.shape[1], a.shape[0], 1],
                         activation=get_activation(model["activation"]),
                         p=model["power"], inner_weights=a, inner_bias=b,
                         outer_weights=u)
    net.eval()
    v_pred, g_pred = _predict(net, pool["x"], norm)

    ev = np.abs(v_pred - pool["v"])
    eg = np.linalg.norm(g_pred - pool["dv"], axis=1)
    err_sq = (v_pred - pool["v"]) ** 2 + np.sum((g_pred - pool["dv"]) ** 2, axis=1)
    true_sq = pool["v"] ** 2 + np.sum(pool["dv"] ** 2, axis=1)

    metrics = {"tube_radius": TUBE_RADIUS, "pool": data_rel}
    for tag, sel in (("switching", pool["switching"]), ("rest", ~pool["switching"])):
        metrics[f"{tag}_count"] = int(sel.sum())
        metrics[f"{tag}_l1_value"] = float(ev[sel].mean() / pool["denom_v"])
        metrics[f"{tag}_l1_grad"] = float(eg[sel].mean() / pool["denom_g"])
        metrics[f"{tag}_l1_h1"] = float(
            0.5 * (ev[sel].mean() / pool["denom_v"] + eg[sel].mean() / pool["denom_g"])
        )
        metrics[f"{tag}_h1"] = float(
            np.sqrt(err_sq[sel].sum()) / np.sqrt(true_sq[sel].sum())
        )
    out_path.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return "rescored"


def main() -> int:
    args = parse_args()
    pool_cache: dict = {}
    norm_cache: dict = {}
    counts: dict[str, int] = {}
    for sweep_dir in args.sweep_dirs:
        for record_path in sorted(sweep_dir.glob("*/*.json")):
            if record_path.name.startswith("region_rescored_"):
                continue
            try:
                status = rescore_run(record_path, pool_cache, norm_cache, force=args.force)
            except Exception as exc:  # noqa: BLE001 — report and continue
                print(f"FAILED {record_path}: {exc}")
                status = "failed"
            counts[status] = counts.get(status, 0) + 1
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
