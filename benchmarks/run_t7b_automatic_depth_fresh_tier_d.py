#!/usr/bin/env python3
"""Preflight and execute the automatic-depth fresh Tier-D one-shot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_contamination_registry.json")
)
CONTRACT = (
    ROOT / "benchmarks" / ("t7b_automatic_depth_fresh_tier_d_execution_contract.json")
)
ANALYZER = ROOT / "benchmarks" / ("analyze_t7b_automatic_depth_fresh_tier_d.py")
FRESH_REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
CTR_SNAPSHOT = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
CTR_DECLARATIONS = ROOT / "benchmarks" / "ctr23_contamination_sources.json"
CONTRACT_ID = "t7b-automatic-depth-fresh-tier-d-execution-v1-20260723"
CONTROL_HEAD = "e23d2b164f10374b1c0e02521c33fc96d48980da"
CANDIDATE_HEAD = "41e948f0c53b1d124e16071a7fa66eba47d084d3"
THREADS = 14
ITERATIONS = 600
RANDOM_STATE = 20260723
PREDICTION_ROWS = 50_000
CAP_ROWS_PER_FEATURE = 3_250
WEIGHT_VALUES = (1.0, 1.25)


class EligibilityError(RuntimeError):
    """A value-free registry eligibility failure."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise RuntimeError("unsupported execution contract")
    if contract.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("execution contract identity changed")
    candidate = contract["candidate"]
    if (
        candidate["source_commit"] != CANDIDATE_HEAD
        or candidate["control_commit"] != CONTROL_HEAD
        or candidate["candidate_must_remain_byte_identical"] is not True
    ):
        raise RuntimeError("candidate/control contract changed")
    execution = contract["execution"]
    expected_execution = {
        "logical_cpu_count": THREADS,
        "threads_per_worker": THREADS,
        "iterations": ITERATIONS,
        "early_stopping_rounds": 30,
        "validation_fraction": 0.15,
        "use_best_model": True,
        "refit": False,
        "random_state": RANDOM_STATE,
        "depth_input": None,
        "prediction_rows_per_repeat": PREDICTION_ROWS,
        "prediction_repeats": 3,
        "same_source_warmup_iterations": 2,
        "high_density_cap_rows_per_input_feature": CAP_ROWS_PER_FEATURE,
        "coordinate_folds": [0, 1, 2],
        "ordinary_folds": [0, 2],
        "nonuniform_weight_fold": 1,
        "nonuniform_weight_values": list(WEIGHT_VALUES),
    }
    for key, value in expected_execution.items():
        if execution.get(key) != value:
            raise RuntimeError(f"execution contract field changed: {key}")
    power = _load_json(ROOT / contract["power_design"]["contract"]["path"])
    if contract["quality_gates"] != power["quality_gates"]:
        raise RuntimeError("execution quality gates differ from power design")
    bindings = {
        "protocol": ROOT
        / "benchmarks"
        / "t7b_automatic_depth_fresh_tier_d_execution_protocol.md",
        "registry_declarations": ROOT
        / "benchmarks"
        / "t7b_automatic_depth_fresh_tier_d_registry_declarations.json",
        "registry_builder": ROOT
        / "benchmarks"
        / "build_t7b_automatic_depth_fresh_tier_d_registry.py",
        "runner": Path(__file__),
        "analyzer": ANALYZER,
    }
    for name, path in bindings.items():
        if contract["source_hashes"][name] != file_sha256(path):
            raise RuntimeError(f"execution source hash changed: {name}")
    if contract["registry"]["file_sha256"] != file_sha256(REGISTRY):
        raise RuntimeError("execution registry file hash changed")
    registry = _load_json(REGISTRY)
    if contract["registry"]["registry_sha256"] != registry["registry_sha256"]:
        raise RuntimeError("execution registry identity changed")
    if contract["authorization"] != {
        "registry_and_contamination_review": True,
        "target_preflight_after_publish": True,
        "one_shot_fresh_confirmation_after_clean_preflight": True,
        "go_default_promotion_for_v0_12": True,
        "no_go_closes_default_candidate": True,
        "candidate_modification": False,
        "gate_relaxation": False,
        "second_attempt": False,
        "partial_read": False,
        "tabarena": False,
        "ctr23": False,
        "lockbox": False,
        "v0_12_release_publication": False,
    }:
        raise RuntimeError("execution authorization boundary changed")


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _git(repository: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _hash_mod(modulus: int, *parts: Any) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return _seed(*parts) % modulus


def _effective_sample_size(weights: np.ndarray | None, rows: int) -> float:
    if weights is None:
        return float(rows)
    total = float(np.sum(weights))
    squared = float(np.dot(weights, weights))
    return total * total / squared


def _target_attestation(target: Any, *, expected_rows: int) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or values.shape[0] != expected_rows:
        raise EligibilityError("target must be a bound one-dimensional vector")
    if np.iscomplexobj(values):
        raise EligibilityError("target must not contain complex values")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EligibilityError("target is not numeric float64") from exc
    if not np.isfinite(numeric).all():
        raise EligibilityError("target contains non-finite values")
    return numeric


def _normalize_features(X):
    import pandas as pd

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.copy()
    X.columns = [str(column) for column in X.columns]
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            X[column] = series.astype("string")
        elif pd.api.types.is_timedelta64_dtype(series.dtype):
            X[column] = series.astype("string")
    categorical = [
        column
        for column in X.columns
        if (
            isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(X[column].dtype)
            or pd.api.types.is_string_dtype(X[column].dtype)
            or pd.api.types.is_bool_dtype(X[column].dtype)
        )
    ]
    return X, categorical


def _load_openml_lineage(lineage: Mapping[str, Any]):
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError("openml is required in darko311") from exc

    task = openml.tasks.get_task(
        int(lineage["task_id"]),
        download_splits=False,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    if int(task.dataset_id) != int(lineage["dataset_id"]):
        raise EligibilityError("OpenML task dataset ID drifted")
    if str(task.target_name) != str(lineage["target_name"]):
        raise EligibilityError("OpenML task target drifted")
    task_type = getattr(task.task_type_id, "value", task.task_type_id)
    if int(task_type) != 2:
        raise EligibilityError("OpenML task is not supervised regression")
    dataset = openml.datasets.get_dataset(
        int(lineage["dataset_id"]),
        download_data=False,
        download_qualities=False,
    )
    if str(dataset.name) != str(lineage["dataset_name"]):
        raise EligibilityError("OpenML dataset name drifted")
    X, target, _categorical, _names = dataset.get_data(
        target=task.target_name,
        include_row_id=False,
        include_ignore_attribute=False,
        dataset_format="dataframe",
    )
    X, categorical = _normalize_features(X)
    numeric_target = _target_attestation(target, expected_rows=len(X))
    if len(X.columns) < 1:
        raise EligibilityError("dataset has no usable input features")
    if len(set(X.columns)) != len(X.columns):
        raise EligibilityError("dataset feature names are not unique")
    group_column = lineage.get("group_column")
    if group_column is not None:
        if str(group_column) not in X.columns:
            raise EligibilityError("declared group column is absent")
        groups = X[str(group_column)].astype("string")
        if groups.isna().any() or groups.nunique(dropna=False) < 3:
            raise EligibilityError("declared group column is unusable")
    else:
        groups = None
    return task, dataset, X, numeric_target, categorical, groups


def _cap_indices(lineage_id: str, rows: int, features: int) -> np.ndarray:
    cap = min(rows, CAP_ROWS_PER_FEATURE * features)
    if cap == rows:
        return np.arange(rows, dtype=np.int64)
    rng = np.random.default_rng(_seed(lineage_id, 20260725))
    return np.sort(rng.choice(rows, size=cap, replace=False).astype(np.int64))


def _split_view(
    lineage: Mapping[str, Any],
    X,
    target: np.ndarray,
    groups,
    coordinate: int,
) -> dict[str, Any]:
    import pandas as pd

    lineage_id = str(lineage["lineage_id"])
    original_rows = np.arange(len(X), dtype=np.int64)
    if str(lineage["branch"]) == "depth_8":
        selected = _cap_indices(lineage_id, len(X), len(X.columns))
    else:
        selected = original_rows
    X_selected = X.iloc[selected].reset_index(drop=True)
    y_selected = target[selected]
    identities = original_rows[selected]
    selected_groups = (
        groups.iloc[selected].reset_index(drop=True) if groups is not None else None
    )

    if lineage["split_kind"] == "group_hash_3fold":
        if selected_groups is None:
            raise EligibilityError("group-safe split has no groups")
        fold = np.asarray(
            [
                _hash_mod(
                    3,
                    lineage_id,
                    str(value),
                    20260724,
                )
                for value in selected_groups
            ],
            dtype=np.int8,
        )
    else:
        fold = np.asarray(
            [
                _hash_mod(5, lineage_id, int(identity), 20260724)
                for identity in identities
            ],
            dtype=np.int8,
        )
    test_mask = fold == int(coordinate)
    train_mask = ~test_mask
    if np.count_nonzero(test_mask) == 0 or np.count_nonzero(train_mask) == 0:
        raise EligibilityError("frozen split produced an empty partition")
    if selected_groups is not None:
        train_group_values = set(selected_groups[train_mask].astype(str))
        test_group_values = set(selected_groups[test_mask].astype(str))
        if train_group_values & test_group_values:
            raise EligibilityError("group-safe outer split leaks a group")

    weight_mode = "nonuniform" if int(coordinate) == 1 else "ordinary"
    if weight_mode == "nonuniform":
        weight_all = np.asarray(
            [
                WEIGHT_VALUES[_hash_mod(2, lineage_id, int(identity), 20260723)]
                for identity in identities
            ],
            dtype=np.float64,
        )
        train_weight = weight_all[train_mask]
        test_weight = weight_all[test_mask]
    else:
        train_weight = None
        test_weight = None
    train_rows = int(np.count_nonzero(train_mask))
    n_eff = _effective_sample_size(train_weight, train_rows)
    density = n_eff / float(len(X.columns))
    expected = str(lineage["branch"])
    observed = (
        "depth_4" if density < 100.0 else "depth_8" if density >= 2500.0 else "depth_6"
    )
    if observed != expected:
        raise EligibilityError(
            f"realized density resolves {observed}, expected {expected}"
        )

    split_binding = {
        "lineage_id": lineage_id,
        "coordinate": int(coordinate),
        "selected_original_rows_sha256": hashlib.sha256(
            identities.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "train_original_rows_sha256": hashlib.sha256(
            identities[train_mask].astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "test_original_rows_sha256": hashlib.sha256(
            identities[test_mask].astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "weight_mode": weight_mode,
        "train_weight_sha256": (
            None
            if train_weight is None
            else hashlib.sha256(
                train_weight.astype("<f8", copy=False).tobytes()
            ).hexdigest()
        ),
    }
    return {
        "X_train": X_selected.loc[train_mask].reset_index(drop=True),
        "y_train": y_selected[train_mask],
        "X_test": X_selected.loc[test_mask].reset_index(drop=True),
        "y_test": y_selected[test_mask],
        "groups_train": (
            selected_groups[train_mask].reset_index(drop=True)
            if selected_groups is not None
            else None
        ),
        "train_weight": train_weight,
        "test_weight": test_weight,
        "weight_mode": weight_mode,
        "train_rows": train_rows,
        "test_rows": int(np.count_nonzero(test_mask)),
        "input_features": len(X.columns),
        "effective_train_rows": n_eff,
        "effective_rows_per_feature": density,
        "split_binding": split_binding,
        "split_sha256": json_sha256(split_binding),
    }


def _known_fingerprints() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    fresh = _load_json(FRESH_REGISTRY)
    for row in fresh["tasks"]:
        record = row.get("task_record", {})
        if isinstance(record.get("fingerprint"), dict):
            sources.append(
                {
                    "source": f"fresh_confirmation:{row['task_id']}",
                    "fingerprint": record["fingerprint"],
                }
            )
    ctr = _load_json(CTR_SNAPSHOT)
    for row in ctr["ctr23_tasks"] + ctr["spent_source_tasks"]:
        if isinstance(row.get("fingerprint"), dict):
            sources.append(
                {
                    "source": f"ctr23_or_spent:{row['openml_task_id']}",
                    "fingerprint": row["fingerprint"],
                }
            )
    thresholds = _load_json(CTR_DECLARATIONS)["near_match_thresholds"]
    return sources, thresholds


def _preflight_lineage(
    lineage: Mapping[str, Any],
    known: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    from benchmarks import build_ctr23_contamination_registry as ctr
    from benchmarks import panel3_data_contract

    task, dataset, X, target, categorical, groups = _load_openml_lineage(lineage)
    if lineage["feature_family"] == "numeric" and categorical:
        raise EligibilityError("numeric stratum contains categorical input features")
    if (
        lineage["feature_family"] == "categorical_or_grouped"
        and not categorical
        and groups is None
    ):
        raise EligibilityError("categorical-or-grouped stratum has neither route")
    fingerprint = ctr.dataset_fingerprint(X, target)
    if fingerprint["canonicalization_ambiguous"]:
        raise EligibilityError("dataset fingerprint canonicalization is ambiguous")
    near = []
    for source in known:
        evidence = ctr.near_match_evidence(
            fingerprint, source["fingerprint"], **thresholds
        )
        if evidence["ambiguous"]:
            near.append({"source": source["source"], **evidence})
    if near:
        raise EligibilityError("dataset has an exact/near-lineage contamination alarm")
    coordinates = []
    for coordinate in (0, 1, 2):
        view = _split_view(lineage, X, target, groups, coordinate)
        coordinates.append(
            {
                "coordinate": coordinate,
                "weight_mode": view["weight_mode"],
                "train_rows": view["train_rows"],
                "test_rows": view["test_rows"],
                "input_features": view["input_features"],
                "effective_train_rows": view["effective_train_rows"],
                "effective_rows_per_feature": (view["effective_rows_per_feature"]),
                "split_sha256": view["split_sha256"],
            }
        )
    selected = (
        _cap_indices(str(lineage["lineage_id"]), len(X), len(X.columns))
        if lineage["branch"] == "depth_8"
        else np.arange(len(X), dtype=np.int64)
    )
    selected_view_sha256 = panel3_data_contract.ordered_task_view_sha256(
        X.iloc[selected].reset_index(drop=True), target[selected]
    )
    return {
        **lineage,
        "status": "eligible",
        "openml_binding": {
            "task_id": int(lineage["task_id"]),
            "dataset_id": int(lineage["dataset_id"]),
            "dataset_version": int(dataset.version),
            "dataset_name": str(dataset.name),
            "target_name": str(task.target_name),
            "default_target_attribute": str(dataset.default_target_attribute),
            "declared_md5": str(getattr(dataset, "md5_checksum", "")),
            "url": str(getattr(dataset, "url", "")),
        },
        "full_dataset_rows": len(X),
        "selected_rows": len(selected),
        "input_features": len(X.columns),
        "categorical_features": categorical,
        "dataset_fingerprint_sha256": ctr.sha256_json(fingerprint),
        "selected_ordered_view_sha256": selected_view_sha256,
        "near_lineage_matches": [],
        "target_attestation": {
            "numeric_float64_all_finite": True,
            "target_statistics_computed": False,
            "target_values_persisted": False,
        },
        "coordinates": coordinates,
    }


def build_preflight() -> dict[str, Any]:
    validate_contract(_load_json(CONTRACT))
    registry = _load_json(REGISTRY)
    if registry["contract_id"] != CONTRACT_ID:
        raise RuntimeError("registry contract identity changed")
    if registry["registry_sha256"] != json_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    ):
        raise RuntimeError("registry self-hash is invalid")
    known, thresholds = _known_fingerprints()
    primaries = sorted(
        (row for row in registry["lineages"] if int(row["priority"]) == 0),
        key=lambda row: str(row["slot"]),
    )
    reserves: dict[str, list[dict[str, Any]]] = {}
    for row in registry["lineages"]:
        if int(row["priority"]) == 1:
            reserves.setdefault(str(row["stratum"]), []).append(row)
    for values in reserves.values():
        values.sort(key=lambda row: str(row["slot"]))
    selected = []
    rejected = []
    used_reserves: set[str] = set()
    for primary in primaries:
        slot = str(primary["slot"])
        candidates = [primary] + [
            row
            for row in reserves[str(primary["stratum"])]
            if str(row["lineage_id"]) not in used_reserves
        ]
        chosen = None
        for lineage in candidates:
            if int(lineage["priority"]) == 1:
                used_reserves.add(str(lineage["lineage_id"]))
            try:
                chosen = _preflight_lineage(lineage, known, thresholds)
            except EligibilityError as exc:
                rejected.append(
                    {
                        "slot": slot,
                        "priority": int(lineage["priority"]),
                        "lineage_id": lineage["lineage_id"],
                        "reason": str(exc),
                        "value_free": True,
                    }
                )
                continue
            if int(lineage["priority"]) == 1:
                chosen["registry_identity_slot"] = chosen["slot"]
                chosen["slot"] = slot
            break
        if chosen is None:
            raise RuntimeError(f"no eligible frozen identity for slot {slot}")
        selected.append(chosen)
    if len(selected) != 32:
        raise RuntimeError("preflight did not resolve exactly 32 lineages")
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    if sorted(counts.values()) != [8, 8, 8, 8]:
        raise RuntimeError("preflight stratum composition changed")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "registry_file_sha256": file_sha256(REGISTRY),
        "registry_sha256": registry["registry_sha256"],
        "status": "preflight_passed",
        "active_lineage_count": len(selected),
        "active_stratum_counts": dict(sorted(counts.items())),
        "active_lineages": selected,
        "rejected_frozen_candidates": rejected,
        "attestations": {
            "no_model_fit": True,
            "no_target_statistics_computed": True,
            "no_target_values_persisted": True,
            "no_quality_outcomes_inspected": True,
            "fixed_reserve_order_used": True,
            "exact_and_near_lineage_fingerprints_checked": True,
            "all_realized_branches_match_frozen_roles": True,
            "all_group_splits_disjoint": True,
            "lockbox_data_used": False,
        },
    }


class _PeakRSS:
    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tree_rss(self) -> int:
        total = 0
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (self._psutil.AccessDenied, self._psutil.NoSuchProcess):
            pass
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except (self._psutil.AccessDenied, self._psutil.NoSuchProcess):
                pass
        return total

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self._peak = max(self._peak, self._tree_rss())

    def __enter__(self):
        self._peak = self._tree_rss()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._peak = max(self._peak, self._tree_rss())
        self._stop.set()
        self._thread.join()

    @property
    def peak(self) -> int:
        return int(self._peak)


def _rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - truth
    if weights is None:
        return float(np.sqrt(np.mean(residual * residual)))
    return float(np.sqrt(np.average(residual * residual, weights=weights)))


def _model_params() -> dict[str, Any]:
    return {
        "iterations": ITERATIONS,
        "early_stopping": True,
        "early_stopping_rounds": 30,
        "validation_fraction": 0.15,
        "use_best_model": True,
        "refit": False,
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
        "diagnostic_warnings": "never",
    }


def _warmup(DarkoRegressor) -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(96, 4))
    y = X[:, 0] - 0.5 * X[:, 1]
    DarkoRegressor(
        iterations=2,
        early_stopping=False,
        ordered_boosting=False,
        random_state=123,
        thread_count=THREADS,
        diagnostic_warnings="never",
    ).fit(X, y)


def run_worker(
    lineage: Mapping[str, Any],
    *,
    coordinate: int,
    arm: str,
    source: Path,
) -> dict[str, Any]:
    import numba

    source = source.expanduser().resolve()
    expected = CONTROL_HEAD if arm == "control" else CANDIDATE_HEAD
    state = source_state(source)
    if not state["clean"] or state["head"] != expected:
        raise RuntimeError(f"{arm} source state changed")
    sys.path.insert(0, str(source))
    from darkofit import DarkoRegressor
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("worker imported DarkoFit from the wrong source")
    _warmup(DarkoRegressor)
    _task, dataset, X, target, categorical, groups = _load_openml_lineage(lineage)
    binding = lineage["openml_binding"]
    if int(dataset.version) != int(binding["dataset_version"]) or str(
        getattr(dataset, "md5_checksum", "")
    ) != str(binding["declared_md5"]):
        raise RuntimeError("worker OpenML dataset binding differs from preflight")
    from benchmarks import panel3_data_contract

    selected = (
        _cap_indices(str(lineage["lineage_id"]), len(X), len(X.columns))
        if lineage["branch"] == "depth_8"
        else np.arange(len(X), dtype=np.int64)
    )
    selected_view_sha256 = panel3_data_contract.ordered_task_view_sha256(
        X.iloc[selected].reset_index(drop=True), target[selected]
    )
    if selected_view_sha256 != lineage["selected_ordered_view_sha256"]:
        raise RuntimeError("worker dataset view differs from preflight")
    view = _split_view(lineage, X, target, groups, coordinate)
    expected_coordinate = next(
        row
        for row in lineage["coordinates"]
        if int(row["coordinate"]) == int(coordinate)
    )
    if view["split_sha256"] != expected_coordinate["split_sha256"]:
        raise RuntimeError("worker split differs from preflight")
    cat_features = [column for column in categorical if column in X.columns]
    model = DarkoRegressor(**_model_params())
    ambient = int(numba.get_num_threads())
    with _PeakRSS() as rss:
        fit_start = time.perf_counter()
        model.fit(
            view["X_train"],
            view["y_train"],
            cat_features=cat_features or None,
            groups=view["groups_train"],
            sample_weight=view["train_weight"],
        )
        fit_seconds = time.perf_counter() - fit_start
        if int(numba.get_num_threads()) != ambient:
            raise RuntimeError("fit leaked the ambient Numba thread mask")
        prediction = model.predict(view["X_test"])
        if int(numba.get_num_threads()) != ambient:
            raise RuntimeError("predict leaked the ambient Numba thread mask")
        rmse = _rmse(view["y_test"], prediction, view["test_weight"])
        tile = np.resize(
            np.arange(len(view["X_test"]), dtype=np.int64),
            PREDICTION_ROWS,
        )
        X_prediction = view["X_test"].iloc[tile].reset_index(drop=True)
        predict_seconds = []
        for _ in range(3):
            start = time.perf_counter()
            repeated_prediction = model.predict(X_prediction)
            predict_seconds.append(time.perf_counter() - start)
        with tempfile.TemporaryDirectory(prefix="t7b-fresh-worker-") as temp:
            archive = Path(temp) / "model.npz"
            model.save_model(archive)
            archive_bytes = archive.stat().st_size
            restored = DarkoRegressor.load_model(archive)
            check_rows = min(256, len(view["X_test"]))
            exact = np.array_equal(
                restored.predict(view["X_test"].iloc[:check_rows]),
                prediction[:check_rows],
            )
    fitted_depth = int(model.model_.depth)
    expected_depth = 4 if lineage["branch"] == "depth_4" else 8
    if arm == "candidate":
        structure = model.model_.auto_params_["auto_structure"]
        policy = structure["candidates"]["depth"]
        policy_ok = (
            fitted_depth == expected_depth
            and policy["branch"]
            == ("low_density" if expected_depth == 4 else "high_density")
            and policy["rule"] == "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8"
        )
    else:
        policy = None
        policy_ok = fitted_depth == 6
    integrity = bool(
        exact
        and policy_ok
        and np.isfinite(rmse)
        and rmse > 0.0
        and fit_seconds > 0.0
        and all(value > 0.0 for value in predict_seconds)
        and int(numba.get_num_threads()) == ambient
    )
    return {
        "status": "ok" if integrity else "integrity_failed",
        "contract_id": CONTRACT_ID,
        "arm": arm,
        "source": state,
        "slot": lineage["slot"],
        "lineage_id": lineage["lineage_id"],
        "stratum": lineage["stratum"],
        "branch": lineage["branch"],
        "task_id": int(lineage["task_id"]),
        "dataset_id": int(lineage["dataset_id"]),
        "coordinate": int(coordinate),
        "weight_mode": view["weight_mode"],
        "split_sha256": view["split_sha256"],
        "train_rows": view["train_rows"],
        "test_rows": view["test_rows"],
        "input_features": view["input_features"],
        "effective_train_rows": view["effective_train_rows"],
        "effective_rows_per_feature": view["effective_rows_per_feature"],
        "rmse": rmse,
        "fit_seconds": fit_seconds,
        "predict_seconds_repeats": predict_seconds,
        "prediction_rows_per_repeat": PREDICTION_ROWS,
        "peak_process_tree_rss_bytes": rss.peak,
        "archive_bytes": archive_bytes,
        "fitted_depth": fitted_depth,
        "automatic_depth_policy": policy,
        "safe_npz_exact": bool(exact),
        "ambient_thread_restored": int(numba.get_num_threads()) == ambient,
        "integrity_passes": integrity,
        "best_iteration": (
            None
            if getattr(model, "best_iteration_", None) is None
            else int(model.best_iteration_)
        ),
        "resolved_threads": (
            None
            if getattr(model, "n_threads_", None) is None
            else int(model.n_threads_)
        ),
    }


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_t7b_automatic_depth_fresh_tier_d",
        "run_panel3_confirmation",
        "run_tabarena",
        "run_m2",
        "run_m3",
        "run_b3",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
    }


def _environment() -> dict[str, Any]:
    import numba
    import openml
    import pandas
    import psutil
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": int(psutil.virtual_memory().total),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "openml": openml.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _worker_command(
    lineage_path: Path,
    coordinate: int,
    arm: str,
    source: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--lineage",
        str(lineage_path),
        "--coordinate",
        str(coordinate),
        "--arm",
        arm,
        "--source",
        str(source),
    ]


def _worker_env(source: Path, cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(THREADS),
            "OMP_THREAD_LIMIT": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
            "NUMBA_NUM_THREADS": str(THREADS),
            "VECLIB_MAXIMUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "DARKOFIT_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache),
        }
    )
    return environment


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("one-shot outputs must be outside the source tree")
    return {
        "launch": Path(str(prefix) + "_launch_manifest.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
        "terminal": Path(str(prefix) + "_terminal_attestation.json"),
    }


def execute(
    *,
    preflight_path: Path,
    control: Path,
    candidate: Path,
    prefix: Path,
) -> dict[str, Any]:
    contract = _load_json(CONTRACT)
    validate_contract(contract)
    preflight = _load_json(preflight_path)
    if preflight["status"] != "preflight_passed":
        raise RuntimeError("fresh preflight has not passed")
    if preflight["contract_id"] != CONTRACT_ID:
        raise RuntimeError("preflight contract identity changed")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"one-shot output collision: {collisions}")
    harness = source_state(ROOT)
    control_state = source_state(control)
    candidate_state = source_state(candidate)
    if not all(state["clean"] for state in (harness, control_state, candidate_state)):
        raise RuntimeError("one-shot requires clean source trees")
    if control_state["head"] != CONTROL_HEAD:
        raise RuntimeError("control source pin changed")
    if candidate_state["head"] != CANDIDATE_HEAD:
        raise RuntimeError("candidate source pin changed")
    if file_sha256(REGISTRY) != preflight["registry_file_sha256"]:
        raise RuntimeError("registry changed after preflight")
    if contract["source_hashes"]["runner"] != file_sha256(Path(__file__)):
        raise RuntimeError("runner hash differs from execution contract")
    if contract["source_hashes"]["analyzer"] != file_sha256(ANALYZER):
        raise RuntimeError("analyzer hash differs from execution contract")
    audit = _exclusive_machine_audit()
    environment = _environment()
    if environment["logical_cpu_count"] != THREADS:
        raise RuntimeError(f"execution requires exactly {THREADS} logical CPUs")
    published = sorted(
        line.strip()
        for line in _git(
            ROOT, "branch", "-r", "--contains", harness["head"]
        ).splitlines()
        if line.strip()
    )
    if not published:
        raise RuntimeError("execution contract commit is not published")

    launch = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sole_inspection_spent": True,
        "sources": {
            "harness": harness,
            "control": control_state,
            "candidate": candidate_state,
            "published_harness_refs": published,
        },
        "source_hashes": {
            "contract": file_sha256(CONTRACT),
            "registry": file_sha256(REGISTRY),
            "preflight": file_sha256(preflight_path),
            "runner": file_sha256(Path(__file__)),
            "analyzer": file_sha256(ANALYZER),
        },
        "environment": environment,
        "exclusive_machine_audit": audit,
        "output_paths": {key: str(value) for key, value in paths.items()},
        "planned_arm_rows": 192,
        "active_lineages": [
            {
                "slot": row["slot"],
                "lineage_id": row["lineage_id"],
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
            }
            for row in preflight["active_lineages"]
        ],
        "no_rerun": True,
        "partial_reads_forbidden": True,
    }
    _write_create_only(paths["launch"], launch)

    rows: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="t7b-fresh-one-shot-") as temp:
            temp_path = Path(temp)
            caches = {
                "control": temp_path / "numba-control",
                "candidate": temp_path / "numba-candidate",
            }
            for cache in caches.values():
                cache.mkdir()
            for lineage_index, lineage in enumerate(preflight["active_lineages"]):
                lineage_path = temp_path / (f"lineage-{lineage_index:02d}.json")
                lineage_path.write_bytes(canonical_json_bytes(lineage))
                for coordinate in (0, 1, 2):
                    arms = (
                        ("control", "candidate")
                        if (lineage_index + coordinate) % 2 == 0
                        else ("candidate", "control")
                    )
                    for arm in arms:
                        source = control if arm == "control" else candidate
                        completed = subprocess.run(
                            _worker_command(lineage_path, coordinate, arm, source),
                            cwd=ROOT,
                            env=_worker_env(source, caches[arm]),
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        if completed.returncode:
                            raise RuntimeError(
                                f"worker failed for {lineage['lineage_id']}/"
                                f"{coordinate}/{arm}: "
                                f"{completed.stderr[-4000:]}"
                            )
                        output_lines = [
                            line
                            for line in completed.stdout.splitlines()
                            if line.strip()
                        ]
                        if not output_lines:
                            raise RuntimeError("worker returned no JSON row")
                        row = json.loads(output_lines[-1])
                        if row.get("status") != "ok":
                            raise RuntimeError(f"worker integrity failed: {row}")
                        rows.append(row)
        if len(rows) != 192:
            raise RuntimeError("one-shot row census changed")
        raw = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "complete": True,
            "launch_manifest_sha256": file_sha256(paths["launch"]),
            "preflight_sha256": file_sha256(preflight_path),
            "environment": environment,
            "rows": rows,
        }
        _write_create_only(paths["raw"], raw)
        analysis_module_path = str(ANALYZER)
        completed = subprocess.run(
            [
                sys.executable,
                analysis_module_path,
                "--raw",
                str(paths["raw"]),
                "--output",
                str(paths["result"]),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError("terminal analyzer failed: " + completed.stderr[-4000:])
        result = _load_json(paths["result"])
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "terminal_complete",
            "disposition": result["disposition"],
            "go": result["go"],
            "artifact_hashes": {
                "launch": file_sha256(paths["launch"]),
                "raw": file_sha256(paths["raw"]),
                "result": file_sha256(paths["result"]),
            },
            "rerun_authorized": False,
        }
        _write_create_only(paths["terminal"], terminal)
        return result
    except BaseException as exc:
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "terminal_failed_after_launch",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_rows_unpublished_and_unread": len(rows),
            "rerun_authorized": False,
            "launch_sha256": file_sha256(paths["launch"]),
        }
        if not paths["terminal"].exists():
            _write_create_only(paths["terminal"], terminal)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--lineage", type=Path, required=True)
    worker.add_argument("--coordinate", type=int, choices=(0, 1, 2), required=True)
    worker.add_argument("--arm", choices=("control", "candidate"), required=True)
    worker.add_argument("--source", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--preflight", type=Path, required=True)
    execute_parser.add_argument("--control", type=Path, required=True)
    execute_parser.add_argument("--candidate", type=Path, required=True)
    execute_parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        artifact = build_preflight()
        _write_create_only(args.output, artifact)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": artifact["status"],
                    "active_lineages": artifact["active_lineage_count"],
                    "rejected": len(artifact["rejected_frozen_candidates"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        lineage = _load_json(args.lineage)
        row = run_worker(
            lineage,
            coordinate=args.coordinate,
            arm=args.arm,
            source=args.source,
        )
        print(json.dumps(row, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        preflight_path=args.preflight,
        control=args.control,
        candidate=args.candidate,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "disposition": result["disposition"],
                "go": result["go"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
