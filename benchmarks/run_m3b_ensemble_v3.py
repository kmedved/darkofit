#!/usr/bin/env python3
"""Run the prospectively frozen private M3b B1/B2 attribution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from . import benchmark_adapters as adapters
    from . import paired_evidence_contract as paired
    from . import run_m3a_wave1 as m3a
except ImportError:  # direct script execution
    import benchmark_adapters as adapters
    import paired_evidence_contract as paired
    import run_m3a_wave1 as m3a


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "m3b_ensemble_v3_protocol.md"
CONTRACT_PATH = ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json"
ANALYZER_PATH = ROOT / "benchmarks" / "analyze_m3b_ensemble_v3.py"
FREEZER_PATH = ROOT / "benchmarks" / "freeze_m3b_ensemble_v3.py"
M3A_CONTRACT_PATH = ROOT / "benchmarks" / "m3a_wave1_contract.json"
DEFAULT_PANEL_CACHE = ROOT / ".cache" / "basketball-sports-panel-v2" / "panel.csv"

CONTRACT_NAME = "wave2_m3b_ensemble_v3_20260720"
SCHEMA_VERSION = 1
THREADS = paired.CONTRACT_THREADS
RANDOM_STATE = 4
SPLIT_SEED = 20_260_720
ITERATIONS = 600
PATIENCE = 30
MEMBERS = 8
WORKER_PREFIX = "M3B_ENSEMBLE_V3_RESULT="

SINGLE = "single_reference"
CONTROL = "control"
B1 = "b1_sampling"
B2 = "b2_member_policy"
COMBINED = "b1_b2_combined"
ARMS = (SINGLE, CONTROL, B1, B2, COMBINED)
CANDIDATE_ARMS = (B1, B2, COMBINED)

SPORTS_TARGETS = (
    "minutes_per_game",
    "game_score",
    "box_plus_minus",
)
SPORTS_SEASONS = (2014, 2015, 2016)
GENERAL_DATASETS = (
    "friedman_numeric",
    "categorical_reg",
    "numeric_binary",
    "categorical_multiclass",
)
BOUND_PATHS = {
    "protocol": "benchmarks/m3b_ensemble_v3_protocol.md",
    "runner": "benchmarks/run_m3b_ensemble_v3.py",
    "analyzer": "benchmarks/analyze_m3b_ensemble_v3.py",
    "freezer": "benchmarks/freeze_m3b_ensemble_v3.py",
    "harness_tests": "tests/test_m3b_ensemble_v3.py",
    "b0_contract": "benchmarks/b0_ensemble_v3_contract.md",
    "implementation": "darkofit/sklearn_api.py",
    "implementation_tests": "tests/test_private_ensemble_v3.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _is_hex_digest(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("M3b metadata contains a non-finite float")
    return value


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    return {
        "path": str(repo),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status": _git(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
    }


def case_specs() -> tuple[dict[str, Any], ...]:
    sports = tuple(
        {
            "case_id": f"sports_{season}_{target}",
            "domain": "sports",
            "task": "regression",
            "season": season,
            "target": target,
            "sampling_unit": "groups",
            "weight_mode": "none",
        }
        for season in SPORTS_SEASONS
        for target in SPORTS_TARGETS
    )
    general = tuple(
        {
            "case_id": f"general_{dataset}",
            "domain": "general",
            "task": adapters.DATASETS[dataset].task,
            "dataset": dataset,
            "size": "medium",
            "seed": 0,
            "split_seed": SPLIT_SEED,
            "test_fraction": 0.25,
            "sampling_unit": "rows",
            "weight_mode": "stress",
        }
        for dataset in GENERAL_DATASETS
    )
    return sports + general


def quality_orders() -> dict[str, list[str]]:
    orders = {}
    for index, spec in enumerate(case_specs()):
        offset = index % len(ARMS)
        orders[spec["case_id"]] = list(ARMS[offset:] + ARMS[:offset])
    return orders


def timing_order(case_id: str, repeat: int, arms: tuple[str, ...]) -> list[str]:
    case_ids = [spec["case_id"] for spec in case_specs()]
    index = case_ids.index(case_id)
    offset = (index + int(repeat)) % len(arms)
    return list(arms[offset:] + arms[:offset])


def _named_hash(data: Mapping[str, Any], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        paired._update_array_hash(digest, name, data.get(name))
    return digest.hexdigest()


def case_fingerprints(
    spec: Mapping[str, Any], data: Mapping[str, Any]
) -> dict[str, str]:
    dataset_names = (
        "X_fit",
        "X_test",
        "y_fit",
        "y_test",
        "groups_fit",
        "cold_test_mask",
    )
    weight_names = ("w_fit", "w_test")
    return {
        "case_sha256": _json_sha256(dict(spec)),
        "dataset_sha256": _named_hash(data, dataset_names),
        "split_sha256": _named_hash(
            data,
            (*dataset_names, *weight_names),
        ),
        "weight_sha256": _named_hash(data, weight_names),
    }


def _m3a_contract() -> dict[str, Any]:
    return json.loads(M3A_CONTRACT_PATH.read_text(encoding="utf-8"))


def build_case(
    spec: Mapping[str, Any], panel_cache: Path = DEFAULT_PANEL_CACHE
) -> dict[str, Any]:
    if spec["domain"] == "sports":
        panel, manifest = m3a.load_panel(panel_cache, _m3a_contract())
        primary, holdout, seen, _folds = m3a._season_views(
            panel,
            manifest,
            int(spec["season"]),
        )
        feature_columns = manifest["transformation"]["feature_columns"]
        target = str(spec["target"])
        data = {
            "X_fit": primary.loc[:, feature_columns].to_numpy(dtype=np.float64),
            "X_test": holdout.loc[:, feature_columns].to_numpy(dtype=np.float64),
            "y_fit": primary[target].to_numpy(dtype=np.float64),
            "y_test": holdout[target].to_numpy(dtype=np.float64),
            "w_fit": None,
            "w_test": None,
            "groups_fit": primary["bref_id"].astype(str).to_numpy(),
            "cold_test_mask": np.asarray(~seen, dtype=np.bool_),
            "cat_features": None,
        }
    else:
        dataset, X, y, cat_features = adapters.build_dataset(
            str(spec["dataset"]),
            str(spec["size"]),
            int(spec["seed"]),
        )
        weights = adapters.make_sample_weight(
            y,
            dataset.task,
            str(spec["weight_mode"]),
        )
        indices = np.arange(len(y), dtype=np.int64)
        stratify = y if dataset.task != "regression" else None
        train, test = train_test_split(
            indices,
            test_size=float(spec["test_fraction"]),
            random_state=int(spec["split_seed"]),
            stratify=stratify,
        )
        data = {
            "X_fit": X[train],
            "X_test": X[test],
            "y_fit": np.asarray(y)[train],
            "y_test": np.asarray(y)[test],
            "w_fit": None if weights is None else weights[train],
            "w_test": None if weights is None else weights[test],
            "groups_fit": None,
            "cold_test_mask": None,
            "cat_features": cat_features,
        }
    for name in ("X_fit", "X_test", "y_fit", "y_test"):
        if len(data[name]) < 1:
            raise RuntimeError(f"M3b case {spec['case_id']} has empty {name}")
    cold = data["cold_test_mask"]
    if cold is not None and (
        np.asarray(cold).shape != (len(data["y_test"]),) or not np.any(cold)
    ):
        raise RuntimeError(f"M3b case {spec['case_id']} has no cold test rows")
    return data


def expected_case_fingerprints(
    panel_cache: Path = DEFAULT_PANEL_CACHE,
) -> dict[str, dict[str, str]]:
    return {
        case_id: record["fingerprints"]
        for case_id, record in expected_case_manifests(panel_cache).items()
    }


def expected_case_manifests(
    panel_cache: Path = DEFAULT_PANEL_CACHE,
) -> dict[str, dict[str, Any]]:
    manifests = {}
    for spec in case_specs():
        data = build_case(spec, panel_cache)
        cold = data["cold_test_mask"]
        manifests[spec["case_id"]] = {
            "fingerprints": case_fingerprints(spec, data),
            "fit_rows": int(len(data["y_fit"])),
            "test_rows": int(len(data["y_test"])),
            "primary_rows": int(
                len(data["y_test"])
                if cold is None
                else np.sum(np.asarray(cold, dtype=np.bool_))
            ),
            "feature_count": int(np.asarray(data["X_fit"]).shape[1]),
            "class_count": (
                None
                if spec["task"] == "regression"
                else int(np.unique(data["y_fit"]).size)
            ),
        }
    return manifests


def arm_config(arm: str) -> dict[str, Any]:
    if arm == SINGLE:
        return {"kind": "single"}
    if arm not in CANDIDATE_ARMS + (CONTROL,):
        raise ValueError(f"unknown M3b arm: {arm!r}")
    return {
        "kind": "private_ensemble_v3",
        "sampling": ("without_replacement" if arm in {B1, COMBINED} else "bootstrap"),
        "sample_fraction": 0.8 if arm in {B1, COMBINED} else None,
        "member_policy": ("donor_balanced_v1" if arm in {B2, COMBINED} else "none"),
    }


def _bound_file_ok(record: Mapping[str, Any]) -> bool:
    try:
        path = ROOT / str(record["path"])
        expected_bytes = record["bytes"]
        expected_sha256 = record["sha256"]
        return (
            not isinstance(expected_bytes, bool)
            and isinstance(expected_bytes, int)
            and expected_bytes >= 0
            and _is_hex_digest(expected_sha256, 64)
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == expected_bytes
            and _sha256(path) == expected_sha256
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("M3b contract must be a regular file")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise RuntimeError("M3b contract must be a JSON object")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("name") != CONTRACT_NAME
        or contract.get("contract_frozen") is not True
        or contract.get("outcomes_opened") is not False
        or contract.get("paired_execution_contract") != paired.CONTRACT_VERSION
        or contract.get("threads") != THREADS
    ):
        raise RuntimeError("M3b contract is not the frozen pre-outcome contract")
    bound_files = contract.get("bound_files")
    if not isinstance(bound_files, Mapping) or set(bound_files) != set(BOUND_PATHS):
        raise RuntimeError("M3b contract bound-file set changed")
    for name, expected_path in BOUND_PATHS.items():
        record = bound_files[name]
        if (
            not isinstance(record, Mapping)
            or record.get("path") != expected_path
            or not _bound_file_ok(record)
        ):
            raise RuntimeError(f"M3b bound file changed: {expected_path}")
    if contract.get("cases") != list(case_specs()):
        raise RuntimeError("M3b case grid changed")
    if contract.get("arms") != {arm: arm_config(arm) for arm in ARMS}:
        raise RuntimeError("M3b arm grid changed")
    if contract.get("quality_orders") != quality_orders():
        raise RuntimeError("M3b quality order changed")
    if contract.get("decision_rules") != decision_rules():
        raise RuntimeError("M3b decision rules changed")
    if contract.get("execution") != execution_contract():
        raise RuntimeError("M3b execution contract changed")
    if contract.get("claims") != claim_contract():
        raise RuntimeError("M3b claim boundary changed")
    case_manifests = contract.get("case_manifests")
    case_fingerprints_value = contract.get("case_fingerprints")
    expected_case_ids = {spec["case_id"] for spec in case_specs()}
    if (
        not isinstance(case_manifests, Mapping)
        or set(case_manifests) != expected_case_ids
        or not isinstance(case_fingerprints_value, Mapping)
        or set(case_fingerprints_value) != expected_case_ids
        or any(
            case_manifests[case_id].get("fingerprints")
            != case_fingerprints_value[case_id]
            for case_id in expected_case_ids
        )
    ):
        raise RuntimeError("M3b case manifest is invalid")
    panel_cache = contract.get("panel_cache")
    source_head = contract.get("sources", {}).get("darkofit")
    if (
        not isinstance(panel_cache, Mapping)
        or not isinstance(panel_cache.get("bytes"), int)
        or panel_cache["bytes"] < 1
        or not isinstance(panel_cache.get("sha256"), str)
        or not _is_hex_digest(panel_cache["sha256"], 64)
        or panel_cache.get("contract_path")
        != str(DEFAULT_PANEL_CACHE.relative_to(ROOT))
        or not _is_hex_digest(source_head, 40)
    ):
        raise RuntimeError("M3b source or panel binding is invalid")
    return contract


def decision_rules() -> dict[str, Any]:
    return {
        "quality": {
            "all_primary_geomean_at_most": 1.005,
            "general_primary_geomean_at_most": 1.005,
            "sports_cold_geomean_at_most": 1.005,
            "sports_held_geomean_at_most": 1.010,
            "worst_primary_at_most": 1.030,
        },
        "common_final": {
            "predict_ratio_at_most": 1.10,
            "archive_to_control_at_most": 1.05,
            "rss_to_control_at_most": 1.10,
            "median_archive_to_single_at_most": 4.0,
            "median_rss_to_single_at_most": 2.0,
        },
        "value": {
            B1: {
                "all_primary_at_most": 1.002,
                "fit_ratio_at_most": 0.90,
            },
            B2: {
                "all_primary_at_most": 0.995,
                "fit_ratio_at_most": 1.10,
            },
            COMBINED: {
                "quality_route_at_most": 0.995,
                "pareto_primary_at_most": 1.002,
                "pareto_fit_at_most": 0.90,
            },
        },
        "timing_repeats": [1, 2],
    }


def execution_contract() -> dict[str, Any]:
    return {
        "quality_first": True,
        "fresh_worker_per_case_arm_repeat": True,
        "same_arm_warmup_outside_measurement": True,
        "failed_attempts_terminal": True,
        "selective_timing_only_after_frozen_gate": True,
        "ensemble_members": MEMBERS,
        "iterations": ITERATIONS,
        "early_stopping_rounds": PATIENCE,
        "random_state": RANDOM_STATE,
    }


def claim_contract() -> dict[str, Any]:
    return {
        "tier": "E",
        "spent_private_development_evidence": True,
        "public_or_default_change_authorized": False,
        "b3_authorized": False,
        "fresh_confirmation_authorized": False,
        "tabarena_authorized": False,
        "lockbox_access_authorized": False,
    }


def _activate_source(source: Path, expected_head: str) -> str:
    source = source.expanduser().resolve()
    state = git_state(source)
    if state["head"] != expected_head or state["status"]:
        raise RuntimeError("M3b DarkoFit source is not the exact clean pin")
    sys.path.insert(0, str(source))
    return str(source)


def _build_estimator(spec: Mapping[str, Any], arm: str):
    from darkofit import DarkoClassifier, DarkoRegressor

    estimator_class = (
        DarkoRegressor if spec["task"] == "regression" else DarkoClassifier
    )
    params = {
        "iterations": ITERATIONS,
        "early_stopping_rounds": PATIENCE,
        "early_stopping": True,
        "use_best_model": True,
        "refit": False,
        "validation_fraction": 0.15,
        "validation_strategy": ("group" if spec["domain"] == "sports" else "random"),
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
        "diagnostic_warnings": "never",
        "ensemble_shared_preprocessing": True,
        "n_ensembles": 1 if arm == SINGLE else MEMBERS,
    }
    return estimator_class(**params)


def _fit_model(model, spec, arm, data):
    fit_kwargs = {
        "cat_features": data["cat_features"],
        "groups": data["groups_fit"],
        "sample_weight": data["w_fit"],
    }
    config = arm_config(arm)
    if config["kind"] == "single":
        return model.fit(data["X_fit"], data["y_fit"], **fit_kwargs)
    from darkofit.sklearn_api import _fit_private_ensemble_v3

    return _fit_private_ensemble_v3(
        model,
        data["X_fit"],
        data["y_fit"],
        sampling=config["sampling"],
        sampling_unit=spec["sampling_unit"],
        sample_fraction=config["sample_fraction"],
        member_policy=config["member_policy"],
        **fit_kwargs,
    )


def _warmup(spec: Mapping[str, Any], arm: str) -> None:
    rng = np.random.default_rng(91)
    n_rows = 100
    X = rng.normal(size=(n_rows, 5))
    if spec["task"] == "regression":
        y = X[:, 0] - 0.5 * X[:, 1]
    elif spec["task"] == "binary":
        y = np.resize(np.array([0, 1]), n_rows)
    else:
        y = np.resize(np.array([0, 1, 2]), n_rows)
    warm_spec = dict(spec)
    warm_spec["domain"] = "sports" if spec["sampling_unit"] == "groups" else "general"
    model = _build_estimator(warm_spec, arm)
    model.set_params(iterations=2, n_ensembles=1 if arm == SINGLE else 2)
    warm_data = {
        "X_fit": X,
        "y_fit": y,
        "w_fit": np.ones(n_rows, dtype=np.float64),
        "groups_fit": (
            np.repeat(np.arange(20), 5) if spec["sampling_unit"] == "groups" else None
        ),
        "cat_features": None,
    }
    _fit_model(model, warm_spec, arm, warm_data)
    gc.collect()


def _prediction_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    paired._update_array_hash(digest, "prediction", value)
    return digest.hexdigest()


def _metrics(model, spec, data, prediction, probability):
    y_test = np.asarray(data["y_test"])
    w_test = data["w_test"]
    if spec["domain"] == "sports":
        cold = np.asarray(data["cold_test_mask"], dtype=np.bool_)
        primary = math.sqrt(mean_squared_error(y_test[cold], prediction[cold]))
        secondary = math.sqrt(mean_squared_error(y_test, prediction))
        return {
            "primary_metric": "cold_player_rmse",
            "primary_loss": float(primary),
            "secondary_metric": "held_team_rmse",
            "secondary_loss": float(secondary),
            "primary_rows": int(np.sum(cold)),
            "test_rows": int(len(y_test)),
        }
    if spec["task"] == "regression":
        return {
            "primary_metric": "weighted_rmse",
            "primary_loss": float(
                math.sqrt(
                    mean_squared_error(
                        y_test,
                        prediction,
                        sample_weight=w_test,
                    )
                )
            ),
            "secondary_metric": "rmse",
            "secondary_loss": float(math.sqrt(mean_squared_error(y_test, prediction))),
            "primary_rows": int(len(y_test)),
            "test_rows": int(len(y_test)),
        }
    labels = np.asarray(model.classes_)
    return {
        "primary_metric": "weighted_log_loss",
        "primary_loss": float(
            log_loss(y_test, probability, labels=labels, sample_weight=w_test)
        ),
        "secondary_metric": "log_loss",
        "secondary_loss": float(log_loss(y_test, probability, labels=labels)),
        "primary_rows": int(len(y_test)),
        "test_rows": int(len(y_test)),
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    runtime_before = paired.assert_worker_contract(THREADS)
    contract = load_contract(Path(args.contract))
    source = Path(args.source).expanduser().resolve()
    _activate_source(source, contract["sources"]["darkofit"])
    spec = next(item for item in case_specs() if item["case_id"] == args.case_id)
    if args.arm not in ARMS:
        raise ValueError(f"unknown M3b arm: {args.arm}")
    data = build_case(spec, Path(args.panel_cache))
    fingerprints = case_fingerprints(spec, data)
    if fingerprints != contract["case_fingerprints"][args.case_id]:
        raise RuntimeError(f"M3b case fingerprint drifted: {args.case_id}")

    _warmup(spec, args.arm)
    model = _build_estimator(spec, args.arm)
    warning_records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with m3a.AggregateRSSSampler() as rss:
            fit_start = time.perf_counter()
            _fit_model(model, spec, args.arm, data)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = np.asarray(model.predict(data["X_test"]))
            probability = (
                None
                if spec["task"] == "regression"
                else np.asarray(model.predict_proba(data["X_test"]))
            )
            predict_seconds = time.perf_counter() - predict_start
            with tempfile.TemporaryDirectory(
                prefix="darkofit-m3b-archive-",
                dir=os.environ["NUMBA_CACHE_DIR"],
            ) as directory:
                archive = Path(directory) / "model.npz"
                model.save_model(archive)
                archive_bytes = archive.stat().st_size
                restored = model.__class__.load_model(archive)
                restored_prediction = np.asarray(restored.predict(data["X_test"]))
                restored_probability = (
                    None
                    if probability is None
                    else np.asarray(restored.predict_proba(data["X_test"]))
                )
        warning_records = [
            {
                "category": item.category.__name__,
                "message": str(item.message),
            }
            for item in caught
        ]
    if prediction.shape != (len(data["y_test"]),):
        raise RuntimeError("M3b prediction shape is invalid")
    if not np.array_equal(prediction, restored_prediction):
        raise RuntimeError("M3b safe-load prediction parity failed")
    if probability is not None:
        if (
            probability.ndim != 2
            or probability.shape != (len(data["y_test"]), len(model.classes_))
            or not np.isfinite(probability).all()
            or np.min(probability) < -paired.PROBABILITY_VALUE_TOLERANCE
            or np.max(probability) > 1.0 + paired.PROBABILITY_VALUE_TOLERANCE
            or np.max(np.abs(probability.sum(axis=1) - 1.0))
            > paired.PROBABILITY_SUM_TOLERANCE
            or not np.array_equal(probability, restored_probability)
        ):
            raise RuntimeError("M3b probability or safe-load parity is invalid")
    runtime_after = paired.assert_worker_contract(THREADS)
    implementation_path = str(Path(inspect.getfile(model.__class__)).resolve())
    if not Path(implementation_path).is_relative_to(source):
        raise RuntimeError("M3b estimator imported outside the pinned source")
    fitted = paired.fitted_model_metadata(model)
    if fitted["resolved_thread_counts"] != [THREADS]:
        raise RuntimeError("M3b fitted member thread counts drifted")
    oob_member_scores = (
        None
        if args.arm == SINGLE
        else [float(member.best_score_) for member in model.estimators_]
    )
    if oob_member_scores is not None and (
        len(oob_member_scores) != MEMBERS or not np.isfinite(oob_member_scores).all()
    ):
        raise RuntimeError("M3b fitted OOB scores are invalid")
    metrics = _metrics(model, spec, data, prediction, probability)
    return _to_builtin(
        {
            "phase": args.phase,
            "repeat": int(args.repeat),
            "case_id": args.case_id,
            "domain": spec["domain"],
            "task": spec["task"],
            "arm": args.arm,
            "arm_config": arm_config(args.arm),
            **fingerprints,
            **metrics,
            "fit_rows": int(len(data["y_fit"])),
            "feature_count": int(np.asarray(data["X_fit"]).shape[1]),
            "class_count": (
                None
                if spec["task"] == "regression"
                else int(np.unique(data["y_fit"]).size)
            ),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "peak_rss_bytes": int(rss.peak_bytes),
            "rss_samples": int(rss.samples),
            "rss_errors": list(rss.errors),
            "archive_bytes": int(archive_bytes),
            "prediction_sha256": _prediction_sha256(prediction),
            "probability_sha256": (
                None if probability is None else _prediction_sha256(probability)
            ),
            "safe_roundtrip_exact": True,
            "implementation_path": implementation_path,
            "fitted_model_metadata": fitted,
            "ensemble_metadata": getattr(model, "ensemble_metadata_", None),
            "oob_member_scores": oob_member_scores,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
            "warnings": warning_records,
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
    )


def _parse_worker(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("M3b worker did not emit exactly one result")
    return json.loads(matches[0])


def _run_one_worker(
    *,
    contract_path: Path,
    source: Path,
    panel_cache: Path,
    cache_dir: Path,
    phase: str,
    repeat: int,
    case_id: str,
    arm: str,
) -> dict[str, Any]:
    environment = paired.fixed_worker_environment(cache_dir)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--contract",
        str(contract_path),
        "--source",
        str(source),
        "--panel-cache",
        str(panel_cache),
        "--phase",
        phase,
        "--repeat",
        str(repeat),
        "--case-id",
        case_id,
        "--arm",
        arm,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"M3b worker failed for {case_id}/{arm}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return _parse_worker(completed.stdout)


def _load_gate(path: Path, contract_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("M3b timing gate must be a regular file")
    gate = json.loads(path.read_text(encoding="utf-8"))
    eligible = gate.get("eligible_candidates")
    if (
        gate.get("schema_version") != 1
        or gate.get("name") != CONTRACT_NAME
        or gate.get("contract_sha256") != contract_sha256
        or not isinstance(eligible, list)
        or any(arm not in CANDIDATE_ARMS for arm in eligible)
        or len(eligible) != len(set(eligible))
        or eligible != [arm for arm in CANDIDATE_ARMS if arm in eligible]
        or gate.get("timing_required") != bool(eligible)
        or not _is_hex_digest(gate.get("quality_artifact_sha256"), 64)
    ):
        raise RuntimeError("M3b timing gate is invalid")
    return gate


def terminal_failure_artifact(
    *,
    phase: str,
    contract_path: Path,
    contract_sha256: str,
    source_before: Mapping[str, Any],
    source_after: Mapping[str, Any],
    harness_before: Mapping[str, Any],
    harness_after: Mapping[str, Any],
    case_fingerprints_value: Mapping[str, Any],
    completed_rows: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": CONTRACT_NAME,
        "phase": phase,
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "source_state_before": dict(source_before),
        "source_state_after": dict(source_after),
        "harness_state_before": dict(harness_before),
        "harness_state_after": dict(harness_after),
        "case_fingerprints": dict(case_fingerprints_value),
        "completed_rows_discarded": int(completed_rows),
        "rows": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    paired.write_create_only(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _state_or_error(repo: Path) -> dict[str, Any]:
    try:
        return git_state(repo)
    except BaseException as exc:  # preserve the original formal failure
        return {
            "path": str(repo.expanduser().resolve()),
            "state_error": f"{type(exc).__name__}: {exc}",
        }


def _publish_terminal_failure(
    *,
    output: Path,
    phase: str,
    contract_path: Path,
    contract_sha256: str,
    source: Path,
    source_before: Mapping[str, Any],
    harness_before: Mapping[str, Any],
    case_fingerprints_value: Mapping[str, Any],
    completed_rows: int,
    error: BaseException,
) -> NoReturn:
    failure = terminal_failure_artifact(
        phase=phase,
        contract_path=contract_path,
        contract_sha256=contract_sha256,
        source_before=source_before,
        source_after=_state_or_error(source),
        harness_before=harness_before,
        harness_after=_state_or_error(ROOT),
        case_fingerprints_value=case_fingerprints_value,
        completed_rows=completed_rows,
        error=error,
    )
    failure_sha256 = _write_json_create_only(output, failure)
    raise RuntimeError(
        "M3b formal attempt failed terminally; failure artifact "
        f"{output} has SHA-256 {failure_sha256}"
    ) from error


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = Path(args.contract).expanduser().resolve()
    contract = load_contract(contract_path)
    contract_sha256 = _sha256(contract_path)
    source = Path(args.source).expanduser().resolve()
    panel_cache = Path(args.panel_cache).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing M3b artifact: {output}")
    root_before = git_state(ROOT)
    source_before = git_state(source)
    if root_before["status"]:
        raise RuntimeError("M3b harness tree must be clean")
    if (
        source_before["status"]
        or source_before["head"] != contract["sources"]["darkofit"]
    ):
        raise RuntimeError("M3b source tree must be the exact clean pin")
    observed_manifests = expected_case_manifests(panel_cache)
    observed_fingerprints = {
        case_id: record["fingerprints"]
        for case_id, record in observed_manifests.items()
    }
    if (
        observed_fingerprints != contract["case_fingerprints"]
        or observed_manifests != contract["case_manifests"]
    ):
        raise RuntimeError("M3b preflight case fingerprints drifted")

    rows = []
    try:
        if args.phase == "quality":
            for spec in case_specs():
                for arm in contract["quality_orders"][spec["case_id"]]:
                    rows.append(
                        _run_one_worker(
                            contract_path=contract_path,
                            source=source,
                            panel_cache=panel_cache,
                            cache_dir=cache_dir,
                            phase="quality",
                            repeat=0,
                            case_id=spec["case_id"],
                            arm=arm,
                        )
                    )
            gate_sha256 = None
        elif args.phase == "timing":
            gate_path = Path(args.gate).expanduser().resolve()
            gate = _load_gate(gate_path, contract_sha256)
            if gate.get("quality_artifact_sha256") != args.quality_sha256:
                raise RuntimeError("M3b gate/quality artifact binding changed")
            eligible = tuple(gate["eligible_candidates"])
            if not eligible:
                raise RuntimeError(
                    "M3b repeat timing must be skipped when no candidate is eligible"
                )
            timing_arms = (SINGLE, CONTROL, *eligible)
            for repeat in contract["decision_rules"]["timing_repeats"]:
                for spec in case_specs():
                    for arm in timing_order(spec["case_id"], repeat, timing_arms):
                        rows.append(
                            _run_one_worker(
                                contract_path=contract_path,
                                source=source,
                                panel_cache=panel_cache,
                                cache_dir=cache_dir,
                                phase="timing",
                                repeat=repeat,
                                case_id=spec["case_id"],
                                arm=arm,
                            )
                        )
            gate_sha256 = _sha256(gate_path)
        else:
            raise ValueError("M3b phase must be 'quality' or 'timing'")
    except BaseException as exc:
        _publish_terminal_failure(
            output=output,
            phase=args.phase,
            contract_path=contract_path,
            contract_sha256=contract_sha256,
            source=source,
            source_before=source_before,
            harness_before=root_before,
            case_fingerprints_value=observed_fingerprints,
            completed_rows=len(rows),
            error=exc,
        )
    try:
        root_after = git_state(ROOT)
        source_after = git_state(source)
        if root_after != root_before or source_after != source_before:
            raise RuntimeError("M3b source or harness tree changed during execution")
        artifact = {
            "schema_version": 1,
            "name": CONTRACT_NAME,
            "phase": args.phase,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha256,
            "quality_artifact_sha256": (
                None if args.phase == "quality" else args.quality_sha256
            ),
            "gate_sha256": gate_sha256,
            "source_state": source_before,
            "harness_state": root_before,
            "panel_cache": {
                "path": str(panel_cache),
                "bytes": panel_cache.stat().st_size,
                "sha256": _sha256(panel_cache),
            },
            "case_fingerprints": observed_fingerprints,
            "rows": rows,
        }
        artifact_sha256 = _write_json_create_only(output, artifact)
    except BaseException as exc:
        _publish_terminal_failure(
            output=output,
            phase=args.phase,
            contract_path=contract_path,
            contract_sha256=contract_sha256,
            source=source,
            source_before=source_before,
            harness_before=root_before,
            case_fingerprints_value=observed_fingerprints,
            completed_rows=len(rows),
            error=exc,
        )
    return {
        "output": str(output),
        "sha256": artifact_sha256,
        "rows": len(rows),
        "phase": args.phase,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(CONTRACT_PATH))
    parser.add_argument("--source", required=True)
    parser.add_argument("--panel-cache", default=str(DEFAULT_PANEL_CACHE))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=("quality", "timing"), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--case-id")
    parser.add_argument("--arm")
    parser.add_argument("--output")
    parser.add_argument("--cache-dir")
    parser.add_argument("--gate")
    parser.add_argument("--quality-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if not args.case_id or not args.arm:
            raise RuntimeError("M3b worker requires --case-id and --arm")
        result = run_worker(args)
        print(WORKER_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if not args.output or not args.cache_dir:
        raise RuntimeError("M3b parent requires --output and --cache-dir")
    if args.phase == "timing" and (not args.gate or not args.quality_sha256):
        raise RuntimeError("M3b timing requires --gate and --quality-sha256")
    print(json.dumps(run_parent(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
