#!/usr/bin/env python3
"""Validate every record consumed by the current-paper artifact pipeline.

The command is read-only.  It rejects incomplete grids, duplicate cells,
non-current configurations, missing fit histories, and (when requested)
missing physical-coordinate pendulum region sidecars.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
MULTIRUN = REPO_ROOT / "rawdata" / "logs" / "multirun"
DATA_DIR = REPO_ROOT / "rawdata" / "data"
PROVENANCE_MANIFEST = Path(__file__).with_name("algorithm1_record_manifest.json")

ACTIVATIONS = (
    "leaky_relu",
    "softplus",
    "tanh",
    "gaussian",
    "gausscent_1",
    "matern52",
    "gelu_squared",
)
P_ACTIVATIONS = ("softplus", "tanh", "gaussian", "gelu_squared")
ALPHAS = (1e-2, 1e-3, 1e-4, 1e-5)
GAMMAS = (0.0, 0.1, 1.0, 10.0)
LOSSES = ((1.0, 0.0), (1.0, 1.0))
P_ORDERS = (2.01, 2.5, 3.0, 4.0)
P_ALPHAS = (1e-4, 1e-5)
OVERSAMPLE_VARIANTS = ("base6k", "band40", "band60", "add2k")
ALGORITHM1_ROOTS = {
    "rawdata/logs/multirun/vdp/paper_log_penalty/batch",
    "rawdata/logs/multirun/vdp/paper_log_penalty/sequential",
    "rawdata/logs/multirun/vdp/paper_log_penalty/p_study",
    "rawdata/logs/multirun/vdp/paper_log_penalty/p_study_sequential",
    "rawdata/logs/multirun/vdp/paper_log_penalty/radius_ablation",
    "rawdata/logs/multirun/vdp/paper_log_penalty/radius_ablation_sequential",
    "rawdata/logs/multirun/pendulum/paper_log_penalty/batch",
    "rawdata/logs/multirun/pendulum/paper_log_penalty/sequential",
    "rawdata/logs/multirun/pendulum/paper_log_penalty/p_study",
    "rawdata/logs/multirun/pendulum/paper_log_penalty/p_study_sequential",
    "rawdata/logs/multirun/pendulum/paper_log_penalty/oversampling",
}
EXPECTED_DATA_PATHS = {
    "vdp": {"VDP_beta_0.1_grid_30x30.npy"},
    "pendulum": {
        "Pendulum_20260703_ada466c6182948469a197282906c3b6c/"
        "Pendulum_pmp_value_samples_2000_20260703.npz",
        *(f"Pendulum_2sided_oversample_20260704/{variant}.npz" for variant in OVERSAMPLE_VARIANTS),
    },
}
PRODUCTION_DATA = {
    "vdp": "VDP_beta_0.1_grid_30x30.npy",
    "pendulum": (
        "Pendulum_20260703_ada466c6182948469a197282906c3b6c/"
        "Pendulum_pmp_value_samples_2000_20260703.npz"
    ),
}
REGION_SCORING_VERSION = "fixed_tube_common_pool_v1"
ALGORITHM2_COEFFICIENT_SOLVER = "global_prox_warmstart_scale"
ALGORITHM2_PROX_RHO = 0.5


@dataclass(frozen=True)
class Record:
    path: Path
    payload: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return self.payload["config"]

    @property
    def model(self) -> dict[str, Any]:
        return self.config["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.config["training"]

    @property
    def provenance(self) -> dict[str, Any]:
        return self.payload.get("provenance", {})


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-14)


@functools.cache
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_digest(relative_root: str) -> str:
    root = REPO_ROOT / relative_root
    digest = hashlib.sha256()
    records = [
        path for path in sorted(root.glob("**/*.json"))
        if not path.name.startswith("region_rescored_")
    ]
    if not records:
        raise ValueError(f"no records available for provenance digest: {root}")
    artifacts: list[Path] = []
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = Record(path, payload)
        artifacts.extend((path, _fit_history(record)))
    for path in sorted(set(artifacts)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _manifest_payload() -> dict[str, Any]:
    return {
        "search_version": "joint_unconstrained_final_radius_filter_v1",
        "merge_tolerance": 1e-2,
        "run_date": "2026-08-13",
        "implementation_commit": "9660b15",
        "algorithm1_record_roots": sorted(ALGORITHM1_ROOTS),
        "record_digests": {
            root: _root_digest(root) for root in sorted(ALGORITHM1_ROOTS)
        },
    }


def _validate_provenance() -> None:
    if not PROVENANCE_MANIFEST.exists():
        raise ValueError(f"missing Algorithm 1 record provenance: {PROVENANCE_MANIFEST}")
    manifest = json.loads(PROVENANCE_MANIFEST.read_text(encoding="utf-8"))
    expected_scalars = {
        "search_version": "joint_unconstrained_final_radius_filter_v1",
        "merge_tolerance": 1e-2,
        "run_date": "2026-08-13",
        "implementation_commit": "9660b15",
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Algorithm 1 provenance mismatch for {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )
    roots = set(manifest.get("algorithm1_record_roots", []))
    if roots != ALGORITHM1_ROOTS:
        raise ValueError(
            "Algorithm 1 provenance root mismatch: "
            f"missing={sorted(ALGORITHM1_ROOTS - roots)}, "
            f"unexpected={sorted(roots - ALGORITHM1_ROOTS)}"
        )
    expected_digests = manifest.get("record_digests", {})
    for root in sorted(ALGORITHM1_ROOTS):
        actual = _root_digest(root)
        if expected_digests.get(root) != actual:
            raise ValueError(
                f"Algorithm 1 record digest mismatch under {root}: "
                "the record tree differs from the reviewed sweep"
            )


def _records(root: Path) -> list[Record]:
    if not root.is_dir():
        raise ValueError(f"missing record root: {root}")
    rows: list[Record] = []
    for path in sorted(root.glob("**/*.json")):
        if path.name.startswith("region_rescored_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON under record root: {path}: {exc}") from exc
        if "config" not in payload or "metrics" not in payload:
            raise ValueError(f"non-record JSON under record root: {path}")
        rows.append(Record(path, payload))
    if not rows:
        raise ValueError(f"no records under {root}")
    return rows


def _fit_history(record: Record) -> Path:
    run_id = str(record.payload.get("run_id", record.path.stem))
    local = record.path.parent / f"result_{run_id}.pkl"
    if local.exists():
        return local
    for artifact in record.payload.get("artifacts", []):
        if artifact.get("name") == "fit_history":
            candidate = Path(artifact["path"])
            if candidate.exists():
                return candidate
    raise ValueError(f"missing fit history for {record.path}")


def _validate_record(record: Record, *, problem: str, algorithm: int) -> None:
    payload, cfg = record.payload, record.config
    model, training = record.model, record.training
    if payload.get("status") != "completed":
        raise ValueError(f"record is not completed: {record.path}")
    if not payload.get("metrics") or not payload["metrics"][0].get("values"):
        raise ValueError(f"record has no metrics: {record.path}")
    if int(cfg["env"]["seed"]) != 42:
        raise ValueError(f"record seed is not 42: {record.path}")
    if problem not in str(cfg["name"]).lower():
        raise ValueError(f"record problem mismatch: {record.path}")
    if not bool(cfg["data"].get("normalize")):
        raise ValueError(f"record does not normalize data: {record.path}")
    data_rel = str(cfg["data"]["path"])
    if data_rel not in EXPECTED_DATA_PATHS[problem]:
        raise ValueError(f"record uses an unexpected {problem} dataset: {record.path}: {data_rel}")
    data_path = DATA_DIR / data_rel
    if not data_path.exists():
        raise ValueError(f"record dataset is missing: {record.path}: {data_path}")
    if model["kind"] != "signed":
        raise ValueError(f"record model is not signed: {record.path}")
    if training["loop_order"] != "insertion_first" or not training["correction_guard"]:
        raise ValueError(f"record does not use the paper loop and guard: {record.path}")
    if not _close(training["ins_merge_tol"], 1e-2):
        raise ValueError(f"record merge tolerance is not 1e-2: {record.path}")
    if (
        int(training["num_insertion"]) != 50
        or not _close(training["lbfgs_lr"], 1e-2)
        or int(training["lbfgs_steps"]) != 200
    ):
        raise ValueError(f"record candidate-search settings differ from the paper: {record.path}")
    if algorithm == 1:
        required = {
            "insertion": "profile",
            "objective": "normalized_moment",
            "power": 1.0,
        }
        if any(model[key] != value for key, value in required.items()):
            raise ValueError(f"record is not current Algorithm 1: {record.path}")
        if not _close(model.get("moment_beta", 0.0), 0.0):
            raise ValueError(f"record contains an additive moment term: {record.path}")
        if training["insert_init"] != "guaranteed":
            raise ValueError(f"record does not use the insertion coefficient: {record.path}")
    else:
        if model["insertion"] != "finite_step" or model["activation"] != "relu":
            raise ValueError(f"record is not current Algorithm 2: {record.path}")
        if not _close(model["gamma"], 0.0):
            raise ValueError(f"Algorithm 2 record has nonzero gamma: {record.path}")
        power = float(model["power"])
        if power == 1.0:
            valid_provenance = (
                record.provenance.get("coefficient_solver") == "soft_threshold"
                and "rho" not in record.provenance
            )
        elif power in (2.0, 3.0):
            valid_provenance = (
                record.provenance.get("coefficient_solver")
                == ALGORITHM2_COEFFICIENT_SOLVER
                and _close(record.provenance.get("rho", math.nan), ALGORITHM2_PROX_RHO)
            )
        else:
            valid_provenance = False
        if not valid_provenance:
            raise ValueError(
                f"Algorithm 2 coefficient solver provenance mismatch: {record.path}: "
                f"{record.provenance}"
            )
    _fit_history(record)


def _validate_grid(
    root: Path,
    *,
    problem: str,
    algorithm: int,
    key: Callable[[Record], tuple[Any, ...]],
    expected: Iterable[tuple[Any, ...]],
    expected_data: str | None = None,
    require_sidecars: bool = False,
) -> list[Record]:
    records = _records(root)
    for record in records:
        _validate_record(record, problem=problem, algorithm=algorithm)
        if expected_data is not None and record.config["data"]["path"] != expected_data:
            raise ValueError(
                f"record dataset mismatch under {root}: {record.path}: "
                f"expected {expected_data}, got {record.config['data']['path']}"
            )
    expected_set = set(expected)
    actual = [key(record) for record in records]
    actual_set = set(actual)
    duplicates = sorted({cell for cell in actual_set if actual.count(cell) > 1}, key=str)
    if actual_set != expected_set or duplicates or len(actual) != len(expected_set):
        raise ValueError(
            f"record grid mismatch under {root}: missing={sorted(expected_set - actual_set, key=str)}, "
            f"unexpected={sorted(actual_set - expected_set, key=str)}, "
            f"duplicates={duplicates}, records={len(actual)}, expected={len(expected_set)}"
        )
    if require_sidecars:
        required_keys = {
            "switching_l1_h1",
            "rest_l1_h1",
            "switching_h1",
            "rest_h1",
            "switching_count",
            "rest_count",
            "tube_radius",
            "pool",
            "scoring_version",
            "record_sha256",
            "fit_history_sha256",
            "pool_sha256",
        }
        for record in records:
            run_id = str(record.payload.get("run_id", record.path.stem))
            sidecar = record.path.parent / f"region_rescored_{run_id}.json"
            if not sidecar.exists():
                raise ValueError(f"missing region-rescore sidecar: {sidecar}")
            values = json.loads(sidecar.read_text(encoding="utf-8"))
            missing = required_keys - values.keys()
            if missing:
                raise ValueError(f"incomplete region-rescore sidecar {sidecar}: {sorted(missing)}")
            if values["scoring_version"] != REGION_SCORING_VERSION:
                raise ValueError(f"stale region scoring version in {sidecar}")
            if not _close(values["tube_radius"], 0.3):
                raise ValueError(f"wrong switching-tube radius in {sidecar}")
            if values["pool"] != record.config["data"]["path"]:
                raise ValueError(f"region sidecar pool mismatch in {sidecar}")
            if values["record_sha256"] != hashlib.sha256(record.path.read_bytes()).hexdigest():
                raise ValueError(f"region sidecar record digest mismatch in {sidecar}")
            if values["fit_history_sha256"] != _file_sha256(_fit_history(record)):
                raise ValueError(f"region sidecar fit-history digest mismatch in {sidecar}")
            data_path = DATA_DIR / values["pool"]
            pool_path = data_path.with_name(f"{data_path.stem}_region_eval_pool.npz")
            if values["pool_sha256"] != _file_sha256(pool_path):
                raise ValueError(f"region sidecar evaluation-pool digest mismatch in {sidecar}")
            numeric = required_keys - {
                "pool", "scoring_version", "record_sha256",
                "fit_history_sha256", "pool_sha256",
            }
            for key in numeric:
                if not math.isfinite(float(values[key])):
                    raise ValueError(f"non-finite {key} in {sidecar}")
    return records


def _mode_cell(record: Record) -> tuple[Any, ...]:
    model, training = record.model, record.training
    return (
        model["activation"],
        float(model["alpha"]),
        float(model["gamma"]),
        tuple(float(value) for value in model["loss_weights"]),
        float(model["moment_order"]),
        training["radial_cap"],
        training["insert_mode"],
        int(training["num_iterations"]),
    )


def _p_cell(record: Record) -> tuple[Any, ...]:
    model, training = record.model, record.training
    return (
        model["activation"],
        float(model["alpha"]),
        float(model["gamma"]),
        float(model["moment_order"]),
        tuple(float(value) for value in model["loss_weights"]),
        training["radial_cap"],
        training["insert_mode"],
        int(training["num_iterations"]),
    )


def _radius_cell(record: Record) -> tuple[Any, ...]:
    cell = _p_cell(record)
    return (*cell[:5], *cell[6:], record.training["radial_cap"])


def _algorithm2_cell(record: Record) -> tuple[Any, ...]:
    model, training = record.model, record.training
    return (
        float(model["power"]),
        float(model["alpha"]),
        tuple(float(value) for value in model["loss_weights"]),
        float(model["gamma"]),
        training["radial_cap"],
        training["insert_mode"],
        int(training["num_iterations"]),
    )


def _oversample_cell(record: Record) -> tuple[Any, ...]:
    model, training = record.model, record.training
    data_variant = Path(str(record.config["data"]["path"])).stem
    directory_variant = record.path.parent.parent.name
    return (
        directory_variant,
        data_variant,
        float(model["alpha"]),
        model["activation"],
        float(model["power"]),
        float(model["gamma"]),
        tuple(float(value) for value in model["loss_weights"]),
        float(model["moment_order"]),
        training["radial_cap"],
        training["insert_mode"],
        int(training["num_iterations"]),
    )


def validate_algorithm2(
    *, multirun_root: Path | None = None, require_sidecars: bool = False
) -> None:
    """Validate the complete current Algorithm 2 record tree."""
    multirun_root = MULTIRUN if multirun_root is None else multirun_root
    for problem in ("vdp", "pendulum"):
        root = multirun_root / problem / "paper_frac_exp_penalty"
        for mode, iterations in (("batch", 10), ("sequential", 150)):
            expected = itertools.product(
                (2.0, 3.0),
                (1e-3, 1e-4, 1e-5, 1e-6),
                LOSSES,
                (0.0,),
                ("fixed",),
                (mode,),
                (iterations,),
            )
            _validate_grid(
                root / mode,
                problem=problem,
                algorithm=2,
                key=_algorithm2_cell,
                expected=expected,
                expected_data=PRODUCTION_DATA[problem],
                require_sidecars=(
                    require_sidecars and problem == "pendulum" and mode == "sequential"
                ),
            )
        expected_l1 = itertools.product(
            (1.0,),
            (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
            ((1.0, 1.0),),
            (0.0,),
            ("fixed",),
            ("sequential",),
            (150,),
        )
        _validate_grid(
            root / "relu_l1",
            problem=problem,
            algorithm=2,
            key=_algorithm2_cell,
            expected=expected_l1,
            expected_data=PRODUCTION_DATA[problem],
        )

    pendulum_alg2 = (
        multirun_root / "pendulum" / "paper_frac_exp_penalty" / "oversampling"
    )
    expected_alg2 = (
        (variant, variant, alpha, "relu", 2.0, 0.0, (1.0, 1.0),
         2.0, "fixed", "sequential", 150)
        for variant in OVERSAMPLE_VARIANTS
        for alpha in (1e-4, 1e-5, 1e-6)
    )
    _validate_grid(
        pendulum_alg2,
        problem="pendulum",
        algorithm=2,
        key=_oversample_cell,
        expected=expected_alg2,
        require_sidecars=False,
    )


def validate_all(*, require_sidecars: bool, require_provenance: bool = True) -> None:
    if require_provenance:
        _validate_provenance()
    for problem in ("vdp", "pendulum"):
        root = MULTIRUN / problem / "paper_log_penalty"
        for mode, iterations in (("batch", 10), ("sequential", 150)):
            expected = itertools.product(
                ACTIVATIONS, ALPHAS, GAMMAS, LOSSES, (2.01,), ("theorem",),
                (mode,), (iterations,)
            )
            _validate_grid(
                root / mode,
                problem=problem,
                algorithm=1,
                key=_mode_cell,
                expected=expected,
                expected_data=PRODUCTION_DATA[problem],
                require_sidecars=require_sidecars and problem == "pendulum" and mode == "sequential",
            )
        for mode, iterations, dirname in (
            ("batch", 10, "p_study"),
            ("sequential", 150, "p_study_sequential"),
        ):
            expected = itertools.product(
                P_ACTIVATIONS,
                P_ALPHAS,
                (1.0,),
                P_ORDERS,
                ((1.0, 1.0),),
                ("theorem",),
                (mode,),
                (iterations,),
            )
            _validate_grid(
                root / dirname,
                problem=problem,
                algorithm=1,
                key=_p_cell,
                expected=expected,
                expected_data=PRODUCTION_DATA[problem],
            )

    vdp_root = MULTIRUN / "vdp" / "paper_log_penalty"
    for mode, iterations, dirname in (
        ("batch", 10, "radius_ablation"),
        ("sequential", 150, "radius_ablation_sequential"),
    ):
        expected = itertools.product(
            P_ACTIVATIONS,
            P_ALPHAS,
            (1.0,),
            P_ORDERS,
            ((1.0, 1.0),),
            (mode,),
            (iterations,),
            ("fixed", "theorem"),
        )
        _validate_grid(
            vdp_root / dirname,
            problem="vdp",
            algorithm=1,
            key=_radius_cell,
            expected=expected,
            expected_data=PRODUCTION_DATA["vdp"],
        )

    validate_algorithm2(require_sidecars=require_sidecars)

    pendulum_alg1 = MULTIRUN / "pendulum" / "paper_log_penalty" / "oversampling"
    expected_alg1 = (
        (variant, variant, alpha, "gaussian", 1.0, 0.0, (1.0, 1.0),
         2.01, "theorem", "sequential", 150)
        for variant in OVERSAMPLE_VARIANTS
        for alpha in (1e-3, 1e-4, 1e-5)
    )
    _validate_grid(
        pendulum_alg1,
        problem="pendulum",
        algorithm=1,
        key=_oversample_cell,
        expected=expected_alg1,
        require_sidecars=False,
    )
    # Oversampling fits are compared on one production out-of-sample pool,
    # not on per-variant pools (the variant directories contain no raw
    # trajectories from which such pools could be rebuilt).
    production = _records(
        MULTIRUN / "pendulum" / "paper_log_penalty" / "sequential"
    )
    data_paths = {str(record.config["data"]["path"]) for record in production}
    if len(data_paths) != 1:
        raise ValueError(f"pendulum production records use multiple datasets: {data_paths}")
    data_path = DATA_DIR / next(iter(data_paths))
    common_pool = data_path.with_name(f"{data_path.stem}_region_eval_pool.npz")
    if not common_pool.exists():
        raise ValueError(f"missing common pendulum region-eval pool: {common_pool}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-sidecars",
        action="store_true",
        help="also require complete physical-coordinate pendulum region sidecars",
    )
    parser.add_argument(
        "--write-provenance",
        action="store_true",
        help="write record-tree digests after a fresh, reviewed Algorithm 1 sweep",
    )
    parser.add_argument(
        "--algorithm2-root",
        type=Path,
        help="validate only an Algorithm 2 staging tree rooted like rawdata/logs/multirun",
    )
    args = parser.parse_args()
    if args.algorithm2_root is not None:
        validate_algorithm2(
            multirun_root=args.algorithm2_root,
            require_sidecars=args.require_sidecars,
        )
        print(f"validated current Algorithm 2 records under {args.algorithm2_root}")
        return 0
    if args.write_provenance:
        validate_all(require_sidecars=False, require_provenance=False)
        PROVENANCE_MANIFEST.write_text(
            json.dumps(_manifest_payload(), indent=2) + "\n", encoding="utf-8"
        )
    validate_all(require_sidecars=args.require_sidecars)
    suffix = " and region sidecars" if args.require_sidecars else ""
    print(f"validated all current-paper record grids{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
