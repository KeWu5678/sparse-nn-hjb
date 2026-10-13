#!/usr/bin/env python3
"""Validate fresh experiment grids, their exact input bytes, and paper artifacts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config.activations import get_use_sphere
from src.config.schema import ModelConfig, TrainingConfig
from src.data import DATA_DIR

MULTIRUN = REPO_ROOT / "rawdata/logs/multirun"
MANIFEST = Path(__file__).with_name("current_run_manifest.json")
LEGACY_VDP_SHA = "d11e9f296c9e86125633939906d9a07b10391ab89c281a96d6e62adaa9d896a4"


def file_sha256(path):
    path = Path(path)
    stat = path.stat()
    return _hash_at_stat(path, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=None)
def _hash_at_stat(path, size, modified):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cell(model):
    order = None if get_use_sphere(model["activation"]) else float(model["moment_order"])
    return (model["activation"], float(model["power"]), float(model["alpha"]),
            float(model["gamma"]), tuple(model["loss_weights"]), order)


def validate_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    cfg = record["config"]
    choice = record["hydra"]["runtime"]["choices"]["data"]
    data_config = yaml.safe_load((REPO_ROOT / f"conf/data/{choice}.yaml").read_text())
    expected_path = data_config["data"]["path"]
    if cfg["data"]["path"] != expected_path:
        raise ValueError(f"obsolete dataset in {path}")
    actual_hash = file_sha256(DATA_DIR / expected_path)
    if actual_hash == LEGACY_VDP_SHA or record.get("data_sha256") != actual_hash:
        raise ValueError(f"missing or mismatched training-data hash in {path}")
    if record.get("status") != "completed" or record.get("metric_coordinates") != "physical":
        raise ValueError(f"incomplete run or nonphysical metrics in {path}")
    if not record.get("normalization") or not cfg["data"]["normalize"]:
        raise ValueError(f"missing normalization in {path}")
    if cfg["env"]["seed"] != 42 or cfg["data"]["train_fraction"] != 0.9:
        raise ValueError(f"unexpected split or seed in {path}")
    artifact = path.parent / f"result_{record['run_id']}.pkl"
    if not artifact.is_file():
        raise ValueError(f"missing run-adjacent fit: {artifact}")
    if cfg["eval"]["kind"] == "region_split":
        shared = yaml.safe_load((REPO_ROOT / "conf/eval/region_split.yaml").read_text())
        if cfg["eval"]["eval_pool"] != shared["eval_pool"]:
            raise ValueError(f"wrong evaluation pool in {path}")
        for key in ("eval_pool", "distance_cache"):
            actual = file_sha256(DATA_DIR / cfg["eval"][key])
            if record.get("evaluation_data_sha256", {}).get(key) != actual:
                raise ValueError(f"missing or mismatched {key} hash in {path}")
    return record


def validate_runs(problem=None):
    parser = OverridesParser.create()
    verified = {}
    for config_path in sorted((REPO_ROOT / "conf/experiment").glob("*.yaml")):
        experiment = config_path.stem
        config = yaml.safe_load(config_path.read_text())
        params = config["hydra"]["sweeper"]["params"]
        axes = []
        for key, value in params.items():
            override = parser.parse_override(f"{key}={value}")
            axes.append(list(override.sweep_iterator()) if override.is_sweep_override()
                        else [override.value()])
        expected = {}
        for values in itertools.product(*axes):
            setting = dict(zip(params, values))
            dataset = setting["+data"]
            if problem and not dataset.startswith(problem):
                continue
            model = asdict(ModelConfig()) | config.get("model", {})
            model.update({key[6:]: value for key, value in setting.items()
                          if key.startswith("model.")})
            expected.setdefault(dataset, set()).add(cell(model))
        for dataset, cells in expected.items():
            root = MULTIRUN / dataset / experiment
            found = set()
            for path in sorted(root.rglob("*.json")):
                record = validate_record(path)
                cfg = record["config"]
                key = cell(cfg["model"])
                if key in found:
                    raise ValueError(f"duplicate experiment cell: {path}")
                found.add(key)
                if "num_insertion" in cfg["training"]:
                    cfg["training"]["num_of_initial_pts"] = cfg["training"].pop("num_insertion")
                expected_training = asdict(TrainingConfig()) | config.get("training", {})
                for name, value in expected_training.items():
                    if cfg["training"][name] != value:
                        raise ValueError(f"unexpected training.{name} in {path}")
                expected_insertion = "finite_step" if "frac_exp" in experiment or experiment == "oversampling_relu2" else "profile"
                if cfg["model"]["insertion"] != expected_insertion:
                    raise ValueError(f"wrong insertion algorithm in {path}")
                artifact = path.parent / f"result_{record['run_id']}.pkl"
                for item in (path, artifact, DATA_DIR / cfg["data"]["path"]):
                    verified[str(item.relative_to(REPO_ROOT))] = file_sha256(item)
                for name in ("eval_pool", "distance_cache"):
                    if cfg["eval"].get(name):
                        item = DATA_DIR / cfg["eval"][name]
                        verified[str(item.relative_to(REPO_ROOT))] = file_sha256(item)
            if found != cells:
                raise ValueError(f"incomplete {dataset}/{experiment}: missing={cells-found}, unexpected={found-cells}")
            print(f"validated {dataset}/{experiment}: {len(found)} runs")
    return verified


def included_figure_paths(repo_root=REPO_ROOT):
    tex = repo_root / "paper/paper_0805.tex"
    source = re.sub(r"(?<!\\)%.*", "", tex.read_text())
    paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
    if not paths:
        raise ValueError(f"no included figures in {tex}")
    return {(tex.parent / path).resolve() for path in paths}


def validate_figure_files(repo_root=REPO_ROOT):
    expected = included_figure_paths(repo_root)
    actual = {p.resolve() for root in (repo_root / "experiments", repo_root / "paper/plot")
              for p in root.rglob("*.png")}
    missing = [str(p) for p in expected if not p.is_file()]
    extra = sorted(str(p) for p in actual - expected)
    if missing or extra:
        raise ValueError(f"manuscript figure mismatch: missing={missing}, unreferenced={extra}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "pendulum"))
    parser.add_argument("--require-figures", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    files = validate_runs(args.problem)
    if args.require_figures:
        for path in validate_figure_files():
            files[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    if args.write_manifest:
        if args.problem:
            raise ValueError("the publication manifest requires both benchmarks")
        # Bind the numerical sources, generated data, traces, and manuscript to
        # the same publication snapshot, beyond the per-run input-byte checks.
        paths = [REPO_ROOT / "paper/paper_0805.tex", REPO_ROOT / "paper/paper_0805.pdf"]
        for root, patterns in (("src", ("*.py",)), ("conf", ("*.yaml",)),
                               ("scripts", ("*.py",)), ("rawdata/data", ("*",)),
                               ("experiments", ("*.py", "*.npz", "*.csv", "*.json", "*.md"))):
            paths.extend(path for pattern in patterns
                         for path in (REPO_ROOT / root).rglob(pattern) if path.is_file())
        for path in paths:
            files[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
        run_count = sum(path.startswith("rawdata/logs/multirun/") and path.endswith(".json")
                        for path in files)
        MANIFEST.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                                        "run_count": run_count, "files": files}, indent=2) + "\n")
        print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
