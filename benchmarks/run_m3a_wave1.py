#!/usr/bin/env python3
"""Run the frozen Wave-1 M3a shipped-ensemble campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "m3a_wave1_protocol.md"
CONTRACT_PATH = ROOT / "benchmarks" / "m3a_wave1_contract.json"
ANALYZER_PATH = ROOT / "benchmarks" / "analyze_m3a_wave1.py"
DEFAULT_CACHE = ROOT / ".cache" / "basketball-sports-panel-v2" / "panel.csv"
DEFAULT_DARKO_SOURCE = Path("/private/tmp/darkofit-wave1-source-726e5d8")
DEFAULT_CHIMERA_SOURCE = ROOT.parent / "chimeraboost"

DARKO_SOURCE_HEAD = "726e5d8e6131c580bce948db833a5007d0692dca"
CHIMERA_SOURCE_HEAD = "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"

DARKO_SINGLE = "darkofit_single"
DARKO_GROUP8 = "darkofit_group_ensemble8"
CHIMERA_SINGLE = "chimeraboost_single"
CHIMERA_ENSEMBLE8 = "chimeraboost_ensemble8"
DARKO_ROW5 = "darkofit_row_ensemble5"
DARKO_ROW8 = "darkofit_row_ensemble8"
DARKO_GROUP5 = "darkofit_group_ensemble5"
CHIMERA_FLOAT_SINGLE = "chimeraboost_float_single"
CHIMERA_FLOAT_ENSEMBLE8 = "chimeraboost_float_ensemble8"

PRIMARY_ARMS = (
    DARKO_SINGLE,
    DARKO_GROUP8,
    CHIMERA_SINGLE,
    CHIMERA_ENSEMBLE8,
)
DIAGNOSTIC_ARMS = (
    DARKO_ROW5,
    DARKO_ROW8,
    DARKO_GROUP5,
    CHIMERA_FLOAT_SINGLE,
    CHIMERA_FLOAT_ENSEMBLE8,
)
ALL_ARMS = PRIMARY_ARMS + DIAGNOSTIC_ARMS
GENERAL_ARMS = (
    DARKO_SINGLE,
    DARKO_ROW8,
    CHIMERA_SINGLE,
    CHIMERA_ENSEMBLE8,
)

PHASES = ("primary-quality", "diagnostics", "primary-repeats")
THREADS = 14
RANDOM_STATE = 4
CREATOR_FOLD_SEED_BASE = 20_260_720
WORKER_RESULT_PREFIX = "M3A_WAVE1_RESULT="

_THREAD_LIMIT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_EXECUTION_PREFIXES = (
    "NUMBA_",
    "JOBLIB_",
    "LOKY_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "BLIS_",
    "VECLIB_",
    "NUMEXPR_",
    "TBB_",
    "KMP_",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_create(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
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
        raise RuntimeError("M3a metadata contains a non-finite float")
    return value


def _git_output(repo: Path, *args: str) -> str:
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
    status = _git_output(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    return {
        "path": str(repo),
        "head": _git_output(repo, "rev-parse", "HEAD"),
        "branch": _git_output(repo, "branch", "--show-current"),
        "clean": not bool(status),
        "status": status.splitlines(),
        "describe": _git_output(repo, "describe", "--tags", "--always"),
    }


def source_states(
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> dict[str, dict[str, Any]]:
    states = {
        "harness": git_state(ROOT),
        "darkofit": git_state(darkofit_source),
        "chimeraboost": git_state(chimeraboost_source),
    }
    if not all(state["clean"] for state in states.values()):
        raise RuntimeError("M3a requires clean harness and package-source trees")
    if states["darkofit"]["head"] != DARKO_SOURCE_HEAD:
        raise RuntimeError("M3a DarkoFit source head changed")
    if states["chimeraboost"]["head"] != CHIMERA_SOURCE_HEAD:
        raise RuntimeError("M3a ChimeraBoost source head changed")
    return states


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != 1
        or contract.get("name") != "wave1_m3a_20260720"
        or contract.get("contract_frozen") is not True
        or contract.get("outcomes_opened") is not False
    ):
        raise RuntimeError("M3a contract is not the frozen pre-outcome contract")
    for key in (
        "protocol",
        "runner",
        "analyzer",
        "freezer",
        "sports_manifest",
        "m6_adapter",
        "m5_baseline",
    ):
        record = contract["bound_files"][key]
        file_path = ROOT / record["path"]
        if not file_path.is_file() or _sha256(file_path) != record["sha256"]:
            raise RuntimeError(f"M3a bound file changed: {record['path']}")
    if contract["sources"] != {
        "darkofit": DARKO_SOURCE_HEAD,
        "chimeraboost": CHIMERA_SOURCE_HEAD,
    }:
        raise RuntimeError("M3a package-source pins changed")
    if contract["threads"] != THREADS:
        raise RuntimeError("M3a thread budget changed")
    if tuple(contract["arms"]["primary"]) != PRIMARY_ARMS:
        raise RuntimeError("M3a primary arms changed")
    if tuple(contract["arms"]["diagnostic"]) != DIAGNOSTIC_ARMS:
        raise RuntimeError("M3a diagnostic arms changed")
    return contract


def load_panel(
    cache_path: Path,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sports_record = contract["bound_files"]["sports_manifest"]
    manifest_path = ROOT / sports_record["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["processed_panel"]
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.is_file() or cache_path.is_symlink():
        raise RuntimeError(f"M3a sports-panel cache is unavailable: {cache_path}")
    if (
        cache_path.stat().st_size != expected["bytes"]
        or _sha256(cache_path) != expected["sha256"]
    ):
        raise RuntimeError("M3a sports-panel cache differs from its manifest")
    frame = pd.read_csv(cache_path)
    transformation = manifest["transformation"]
    columns = [
        *transformation["identity_columns"],
        *transformation["feature_columns"],
        *transformation["target_columns"],
    ]
    if frame.columns.tolist() != columns or len(frame) != expected["rows"]:
        raise RuntimeError("M3a sports-panel shape differs from its manifest")
    identities = frame.loc[
        :, transformation["identity_columns"]
    ].values.tolist()
    if _json_sha256(identities) != expected["identities_sha256"]:
        raise RuntimeError("M3a sports-panel identities differ from its manifest")
    numeric = frame.loc[
        :,
        [
            *transformation["feature_columns"],
            *transformation["target_columns"],
        ],
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("M3a sports panel contains non-finite values")
    return frame, manifest


def creator_fold_records(n_rows: int, season: int) -> list[dict[str, Any]]:
    records = []
    splitter = KFold(
        n_splits=10,
        shuffle=True,
        random_state=CREATOR_FOLD_SEED_BASE + int(season),
    )
    for fold, (train, test) in enumerate(splitter.split(np.arange(n_rows))):
        records.append(
            {
                "fold": int(fold),
                "train_indices": [int(value) for value in train],
                "test_indices": [int(value) for value in test],
            }
        )
    return records


def _season_views(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    seasonal = panel.loc[panel["year"] == int(season)].reset_index(drop=True)
    split = manifest["split"]["seasons"][str(season)]
    held = frozenset(split["held_teams"])
    primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
    holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
    primary_players = frozenset(primary["bref_id"].astype(str))
    seen = holdout["bref_id"].astype(str).isin(primary_players).to_numpy()
    if (
        len(primary) != split["primary_rows"]
        or len(holdout) != split["held_team_rows"]
        or int(np.sum(seen)) != split["seen_player_rows"]
    ):
        raise RuntimeError("M3a sports split differs from its manifest")
    player_values = primary["bref_id"].astype(str).to_numpy()
    for frozen in split["folds"]:
        train = np.asarray(frozen["train_indices"], dtype=np.int64)
        test = np.asarray(frozen["test_indices"], dtype=np.int64)
        if not set(player_values[train]).isdisjoint(player_values[test]):
            raise RuntimeError("M3a player-disjoint fold leaked a player")
    return primary, holdout, seen, split["folds"]


class AggregateRSSSampler:
    """Sample aggregate RSS for this worker and recursive child processes."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes = 0
        self.samples = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        seen: set[int] = set()
        total = 0
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_bytes = max(self.peak_bytes, total)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:  # pragma: no cover - platform telemetry
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()
        if self.peak_bytes <= 0 or self.samples < 2:
            raise RuntimeError("M3a aggregate RSS sampler did not engage")
        return False


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        return os.path.commonpath((path.resolve(), root.resolve())) == str(
            root.resolve()
        )


def _activate_sources(
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> None:
    for source in (darkofit_source, chimeraboost_source):
        value = str(source.expanduser().resolve())
        sys.path[:] = [
            entry
            for entry in sys.path
            if str(Path(entry or ".").resolve()) != value
        ]
        sys.path.insert(0, value)


def _implementation(model: Any, expected_source: Path) -> dict[str, Any]:
    package = model.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package)
    module_file = Path(module.__file__).resolve()
    if not _path_is_under(module_file, expected_source):
        raise RuntimeError(
            f"M3a imported {package} outside {expected_source}: {module_file}"
        )
    return {
        "package": package,
        "module_version": getattr(module, "__version__", None),
        "module_file": str(module_file),
        "estimator_class": (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        ),
    }


def _build_estimator(
    arm: str,
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> tuple[Any, Path]:
    _activate_sources(darkofit_source, chimeraboost_source)
    if arm.startswith("darkofit_"):
        from darkofit import DarkoRegressor

        params: dict[str, Any] = {
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
            "diagnostic_warnings": "never",
            "ensemble_shared_preprocessing": True,
        }
        if arm == DARKO_SINGLE:
            params["n_ensembles"] = 1
        elif arm == DARKO_ROW5:
            params.update(n_ensembles=5, ensemble_bootstrap="rows")
        elif arm == DARKO_ROW8:
            params.update(n_ensembles=8, ensemble_bootstrap="rows")
        elif arm == DARKO_GROUP5:
            params.update(n_ensembles=5, ensemble_bootstrap="groups")
        elif arm == DARKO_GROUP8:
            params.update(n_ensembles=8, ensemble_bootstrap="groups")
        else:  # pragma: no cover - parent validates arms
            raise ValueError(f"unknown DarkoFit M3a arm: {arm}")
        return DarkoRegressor(**params), darkofit_source

    from chimeraboost import ChimeraBoostRegressor

    quantized = arm not in {
        CHIMERA_FLOAT_SINGLE,
        CHIMERA_FLOAT_ENSEMBLE8,
    }
    ensemble = arm in {
        CHIMERA_ENSEMBLE8,
        CHIMERA_FLOAT_ENSEMBLE8,
    }
    if arm not in {
        CHIMERA_SINGLE,
        CHIMERA_ENSEMBLE8,
        CHIMERA_FLOAT_SINGLE,
        CHIMERA_FLOAT_ENSEMBLE8,
    }:
        raise ValueError(f"unknown ChimeraBoost M3a arm: {arm}")
    return (
        ChimeraBoostRegressor(
            random_state=RANDOM_STATE,
            thread_count=THREADS,
            n_ensembles=8 if ensemble else None,
            ensemble_n_jobs=-1,
            quantize_gradients=quantized,
        ),
        chimeraboost_source,
    )


def _uses_groups(arm: str) -> bool:
    return arm not in {DARKO_ROW5, DARKO_ROW8}


def _core_summary(model: Any) -> dict[str, Any]:
    core = model.model_
    trees = getattr(core, "trees_", ())
    tree_count = int(getattr(model, "n_estimators_", len(trees)))
    best = int(
        getattr(
            model,
            "best_n_estimators_",
            getattr(model, "best_iteration_", tree_count),
        )
    )
    learning_rate = getattr(
        model,
        "learning_rate_",
        getattr(core, "lr_", None),
    )
    return {
        "tree_count": tree_count,
        "best_iteration": best,
        "resolved_thread_count": int(getattr(core, "n_threads_", 0)),
        "resolved_learning_rate": (
            None if learning_rate is None else float(learning_rate)
        ),
        "selected_tree_mode": getattr(core, "tree_mode_", None),
        "stop_reason": getattr(core, "stop_reason_", None),
    }


def _fit_metadata(model: Any, arm: str) -> dict[str, Any]:
    members = getattr(model, "estimators_", None)
    if members:
        summaries = [_core_summary(member) for member in members]
    else:
        summaries = [_core_summary(model)]
    metadata: dict[str, Any] = {
        "kind": "ensemble" if members else "single",
        "member_count": len(summaries),
        "members": summaries,
        "total_tree_count": int(
            sum(item["tree_count"] for item in summaries)
        ),
        "member_thread_counts": [
            item["resolved_thread_count"] for item in summaries
        ],
    }
    if arm.startswith("darkofit_") and members:
        ensemble = dict(getattr(model, "ensemble_metadata_", {}) or {})
        metadata["ensemble"] = ensemble
        metadata["oob_members"] = [
            {
                key: row.get(key)
                for key in (
                    "seed",
                    "sampled_rows",
                    "sampled_unique_rows",
                    "oob_rows",
                    "sampled_groups",
                    "sampled_unique_groups",
                    "oob_groups",
                )
                if key in row
            }
            for row in ensemble.get("members", [])
        ]
    elif arm.startswith("chimeraboost_") and members:
        metadata["ensemble"] = {
            "sampling": "rows_without_replacement",
            "max_samples": float(model.max_samples),
            "ensemble_n_jobs": int(model.ensemble_n_jobs),
            "member_params": dict(getattr(model, "member_params_", {}) or {}),
            "oob_member_indices_exposed": False,
            "player_overlap_exposed": True,
        }
    return _to_builtin(metadata)


def _fit_predict(
    arm: str,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    *,
    darkofit_source: Path,
    chimeraboost_source: Path,
    groups: Any = None,
    cat_features: list[int] | None = None,
) -> dict[str, Any]:
    model, expected_source = _build_estimator(
        arm, darkofit_source, chimeraboost_source
    )
    implementation = _implementation(model, expected_source)
    fit_kwargs: dict[str, Any] = {}
    if groups is not None:
        fit_kwargs["groups"] = np.asarray(groups)
    if cat_features:
        fit_kwargs["cat_features"] = list(cat_features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter_ns()
        model.fit(X_train, y_train, **fit_kwargs)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        prediction = np.asarray(model.predict(X_test), dtype=np.float64)
        predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(X_test),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"M3a arm {arm} returned invalid predictions")
    model_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    return {
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "model_bytes": int(model_bytes),
        "fit_metadata": _fit_metadata(model, arm),
        "implementation": implementation,
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in caught
        ],
    }


def _score(y_true: Any, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(y_true, dtype=np.float64)
    if values.shape != prediction.shape or values.size < 2:
        raise RuntimeError("M3a score view has an invalid shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("M3a score target is non-finite")
    return {
        "rows": int(values.size),
        "rmse": float(mean_squared_error(values, prediction) ** 0.5),
        "r2": float(r2_score(values, prediction)),
        "target_sha256": _array_sha256(values),
        "prediction_sha256": _array_sha256(prediction),
    }


def _new_costs() -> dict[str, dict[str, Any]]:
    return {
        scope: {
            "fits": 0,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "model_bytes": [],
        }
        for scope in ("player_disjoint", "creator", "held_team", "general")
    }


def _add_cost(
    costs: dict[str, dict[str, Any]],
    scope: str,
    fitted: dict[str, Any],
) -> None:
    record = costs[scope]
    record["fits"] += 1
    record["fit_seconds"] += float(fitted["fit_seconds"])
    record["predict_seconds"] += float(fitted["predict_seconds"])
    record["model_bytes"].append(int(fitted["model_bytes"]))


def _finalize_costs(costs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scope, record in costs.items():
        sizes = record["model_bytes"]
        result[scope] = {
            "fits": int(record["fits"]),
            "fit_seconds": float(record["fit_seconds"]),
            "predict_seconds": float(record["predict_seconds"]),
            "median_model_bytes": (
                None if not sizes else float(median(sizes))
            ),
            "max_model_bytes": None if not sizes else int(max(sizes)),
        }
    result["player_plus_held"] = {
        "fits": (
            result["player_disjoint"]["fits"] + result["held_team"]["fits"]
        ),
        "fit_seconds": (
            result["player_disjoint"]["fit_seconds"]
            + result["held_team"]["fit_seconds"]
        ),
        "predict_seconds": (
            result["player_disjoint"]["predict_seconds"]
            + result["held_team"]["predict_seconds"]
        ),
        "held_median_model_bytes": result["held_team"]["median_model_bytes"],
    }
    return result


def _fold_view(
    arm: str,
    X: pd.DataFrame,
    y: pd.Series,
    players: np.ndarray,
    folds: list[dict[str, Any]],
    *,
    scope: str,
    costs: dict[str, dict[str, Any]],
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    oof = np.empty(len(y), dtype=np.float64)
    records = []
    for frozen in folds:
        train = np.asarray(frozen["train_indices"], dtype=np.int64)
        test = np.asarray(frozen["test_indices"], dtype=np.int64)
        fitted = _fit_predict(
            arm,
            X.iloc[train],
            y.iloc[train],
            X.iloc[test],
            darkofit_source=darkofit_source,
            chimeraboost_source=chimeraboost_source,
            groups=players[train] if _uses_groups(arm) else None,
        )
        prediction = fitted.pop("prediction")
        oof[test] = prediction
        _add_cost(costs, scope, fitted)
        records.append(
            {
                "fold": int(frozen["fold"]),
                "train_rows": int(train.size),
                "test_rows": int(test.size),
                "test_indices_sha256": _json_sha256(
                    [int(value) for value in test]
                ),
                "score": _score(y.iloc[test], prediction),
                "fit": fitted,
            }
        )
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"M3a {scope} folds did not fill every row")
    return _score(y, oof), records


def _load_adapter(contract: dict[str, Any]):
    record = contract["bound_files"]["m6_adapter"]
    return _load_module("m3a_frozen_benchmark_adapters", ROOT / record["path"])


def _run_general_cells(
    arm: str,
    contract: dict[str, Any],
    costs: dict[str, dict[str, Any]],
    *,
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> list[dict[str, Any]]:
    if arm not in GENERAL_ARMS:
        return []
    adapter = _load_adapter(contract)
    records = []
    for dataset in contract["general"]["datasets"]:
        for seed in contract["general"]["seeds"]:
            spec, X, y, cat_features = adapter.build_dataset(
                dataset, contract["general"]["size"], int(seed)
            )
            split = adapter.split_case(X, y, spec.task, int(seed), None)
            X_train = np.concatenate(
                (split["X_fit"], split["X_val"]), axis=0
            )
            y_train = np.concatenate(
                (split["y_fit"], split["y_val"]), axis=0
            )
            fitted = _fit_predict(
                arm,
                X_train,
                y_train,
                split["X_test"],
                darkofit_source=darkofit_source,
                chimeraboost_source=chimeraboost_source,
                cat_features=list(cat_features or []),
            )
            prediction = fitted.pop("prediction")
            _add_cost(costs, "general", fitted)
            records.append(
                {
                    "dataset": dataset,
                    "size": contract["general"]["size"],
                    "seed": int(seed),
                    "task": spec.task,
                    "train_rows": int(len(y_train)),
                    "test_rows": int(len(split["y_test"])),
                    "features": int(split["n_features"]),
                    "cat_features": list(cat_features or []),
                    "score": _score(split["y_test"], prediction),
                    "fit": fitted,
                }
            )
    return records


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    def fit_payload(fit: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_bytes": fit["model_bytes"],
            "fit_metadata": fit["fit_metadata"],
            "implementation": fit["implementation"],
            "warnings": fit["warnings"],
        }

    return {
        "arm": result["arm"],
        "sports_cells": [
            {
                "season": cell["season"],
                "target": cell["target"],
                "player_disjoint": cell["player_disjoint"],
                "creator": cell["creator"],
                "held_team": cell["held_team"],
                "seen_player": cell["seen_player"],
                "cold_player": cell["cold_player"],
                "player_folds": [
                    {
                        "fold": row["fold"],
                        "score": row["score"],
                        "fit": fit_payload(row["fit"]),
                    }
                    for row in cell["player_folds"]
                ],
                "creator_folds": [
                    {
                        "fold": row["fold"],
                        "score": row["score"],
                        "fit": fit_payload(row["fit"]),
                    }
                    for row in cell["creator_folds"]
                ],
                "held_fit": fit_payload(cell["held_fit"]),
            }
            for cell in result["sports_cells"]
        ],
        "general_cells": [
            {
                "dataset": cell["dataset"],
                "size": cell["size"],
                "seed": cell["seed"],
                "score": cell["score"],
                "fit": fit_payload(cell["fit"]),
            }
            for cell in result["general_cells"]
        ],
    }


def run_worker(
    arm: str,
    contract_path: Path,
    cache_path: Path,
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown M3a arm: {arm}")
    panel, sports_manifest = load_panel(cache_path, contract)
    features = sports_manifest["transformation"]["feature_columns"]
    targets = sports_manifest["transformation"]["target_columns"]
    seasons = sports_manifest["transformation"]["seasons"]
    costs = _new_costs()

    first, _, _, first_folds = _season_views(
        panel, sports_manifest, int(seasons[0])
    )
    first_fold = first_folds[0]
    first_train = np.asarray(first_fold["train_indices"], dtype=np.int64)
    first_test = np.asarray(first_fold["test_indices"], dtype=np.int64)
    first_players = first["bref_id"].astype(str).to_numpy()
    X_first = first.loc[:, features]

    with AggregateRSSSampler() as rss:
        warm_started = time.perf_counter_ns()
        warm = _fit_predict(
            arm,
            X_first.iloc[first_train],
            first[targets[0]].iloc[first_train],
            X_first.iloc[first_test],
            darkofit_source=darkofit_source,
            chimeraboost_source=chimeraboost_source,
            groups=(
                first_players[first_train] if _uses_groups(arm) else None
            ),
        )
        warmup_seconds = (time.perf_counter_ns() - warm_started) / 1e9
        warmup_fingerprint = _array_sha256(warm.pop("prediction"))
        del warm
        gc.collect()

        cells = []
        for season in seasons:
            primary, holdout, seen, player_folds = _season_views(
                panel, sports_manifest, int(season)
            )
            creator_folds = creator_fold_records(len(primary), int(season))
            expected_creator = contract["creator_folds"]["seasons"][str(season)]
            if _json_sha256(creator_folds) != expected_creator["sha256"]:
                raise RuntimeError("M3a creator fold plan changed")
            X_primary = primary.loc[:, features]
            X_holdout = holdout.loc[:, features]
            players = primary["bref_id"].astype(str).to_numpy()
            for target in targets:
                player_score, player_records = _fold_view(
                    arm,
                    X_primary,
                    primary[target],
                    players,
                    player_folds,
                    scope="player_disjoint",
                    costs=costs,
                    darkofit_source=darkofit_source,
                    chimeraboost_source=chimeraboost_source,
                )
                creator_score, creator_records = _fold_view(
                    arm,
                    X_primary,
                    primary[target],
                    players,
                    creator_folds,
                    scope="creator",
                    costs=costs,
                    darkofit_source=darkofit_source,
                    chimeraboost_source=chimeraboost_source,
                )
                held_fit = _fit_predict(
                    arm,
                    X_primary,
                    primary[target],
                    X_holdout,
                    darkofit_source=darkofit_source,
                    chimeraboost_source=chimeraboost_source,
                    groups=players if _uses_groups(arm) else None,
                )
                held_prediction = held_fit.pop("prediction")
                _add_cost(costs, "held_team", held_fit)
                cells.append(
                    {
                        "season": int(season),
                        "target": target,
                        "primary_rows": int(len(primary)),
                        "held_rows": int(len(holdout)),
                        "seen_rows": int(np.sum(seen)),
                        "cold_rows": int(np.sum(~seen)),
                        "player_disjoint": player_score,
                        "creator": creator_score,
                        "held_team": _score(
                            holdout[target], held_prediction
                        ),
                        "seen_player": _score(
                            holdout.loc[seen, target], held_prediction[seen]
                        ),
                        "cold_player": _score(
                            holdout.loc[~seen, target],
                            held_prediction[~seen],
                        ),
                        "player_folds": player_records,
                        "creator_folds": creator_records,
                        "held_fit": held_fit,
                    }
                )
        general_cells = _run_general_cells(
            arm,
            contract,
            costs,
            darkofit_source=darkofit_source,
            chimeraboost_source=chimeraboost_source,
        )

    first_model, expected_source = _build_estimator(
        arm, darkofit_source, chimeraboost_source
    )
    result = {
        "arm": arm,
        "implementation": _implementation(first_model, expected_source),
        "overlap_disclosure": contract["arms"]["disclosures"][arm],
        "sports_cells": cells,
        "general_cells": general_cells,
        "costs": _finalize_costs(costs),
        "warmup_seconds_outside_timing": float(warmup_seconds),
        "warmup_prediction_sha256": warmup_fingerprint,
        "aggregate_peak_rss_bytes": int(rss.peak_bytes),
        "rss_sampling": {
            "interval_seconds": rss.interval_seconds,
            "samples": int(rss.samples),
            "errors": rss.errors,
            "scope": "worker plus recursive child-process RSS sum",
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                *_THREAD_LIMIT_KEYS,
                "LOKY_MAX_CPU_COUNT",
                "NUMBA_CACHE_DIR",
                "JOBLIB_TEMP_FOLDER",
            )
        },
    }
    result["behavior_fingerprint_sha256"] = _json_sha256(
        _behavior_payload(result)
    )
    return _to_builtin(result)


def _worker_environment(
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(_EXECUTION_PREFIXES):
            environment.pop(key)
    for key in _THREAD_LIMIT_KEYS:
        environment[key] = str(THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(darkofit_source.resolve()),
                    str(chimeraboost_source.resolve()),
                    str(ROOT),
                )
            ),
            "NUMBA_CACHE_DIR": "/private/tmp/darkofit-m3a-numba-cache",
            "JOBLIB_TEMP_FOLDER": "/private/tmp/darkofit-m3a-joblib",
            "MPLCONFIGDIR": "/private/tmp/darkofit-m3a-mpl",
            "LOKY_MAX_CPU_COUNT": str(THREADS),
        }
    )
    return environment


def _run_worker_process(
    arm: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--contract",
        str(args.contract),
        "--data-cache",
        str(args.data_cache),
        "--darkofit-source",
        str(args.darkofit_source),
        "--chimeraboost-source",
        str(args.chimeraboost_source),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(
            args.darkofit_source, args.chimeraboost_source
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"M3a worker {arm} failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_RESULT_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _phase_orders(
    phase: str,
    contract: dict[str, Any],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(order) for order in contract["execution"]["orders"][phase]
    )


def _assert_states_unchanged(
    expected: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    observed = source_states(
        args.darkofit_source, args.chimeraboost_source
    )
    if observed != expected:
        raise RuntimeError("M3a source or harness state changed during execution")


def _repeat_phase_authorized(
    primary_artifact: Path,
    contract_path: Path,
) -> dict[str, Any]:
    analyzer = _load_module("m3a_frozen_analyzer", ANALYZER_PATH)
    decision = analyzer.primary_decision(
        primary_artifact, contract_path=contract_path
    )
    if decision["survives"] is not True:
        raise RuntimeError(
            "M3a primary repeats are forbidden because group-ensemble8 "
            "did not survive the frozen quality-first rule"
        )
    return decision


def _machine_details() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing M3a output: {args.output}")
    contract = load_contract(args.contract)
    orders = _phase_orders(args.phase, contract)
    if args.phase == "primary-repeats":
        if args.primary_artifact is None:
            raise RuntimeError("primary repeats require --primary-artifact")
        repeat_authorization = _repeat_phase_authorized(
            args.primary_artifact, args.contract
        )
    else:
        repeat_authorization = None
    expected_states = source_states(
        args.darkofit_source, args.chimeraboost_source
    )
    results = []
    for block, order in enumerate(orders):
        for position, arm in enumerate(order):
            _assert_states_unchanged(expected_states, args)
            print(
                f"M3a phase={args.phase} block={block + 1}/{len(orders)} "
                f"position={position + 1}/{len(order)} arm={arm}",
                flush=True,
            )
            result = _run_worker_process(arm, args)
            result["block"] = int(block)
            result["position"] = int(position)
            results.append(result)
    _assert_states_unchanged(expected_states, args)
    artifact = {
        "schema_version": 1,
        "name": "wave1_m3a_phase",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": _sha256(args.contract),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "analyzer_sha256": _sha256(ANALYZER_PATH),
        },
        "sources": expected_states,
        "sports_cache": {
            "path": str(args.data_cache.resolve()),
            "sha256": _sha256(args.data_cache),
            "bytes": args.data_cache.stat().st_size,
        },
        "orders": [list(order) for order in orders],
        "results": results,
        "repeat_authorization": repeat_authorization,
        "environment": {
            "machine": _machine_details(),
            "threads": THREADS,
        },
        "default_change_authorized": False,
    }
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_create(args.output, payload)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "workers": len(results),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--darkofit-source", type=Path, default=DEFAULT_DARKO_SOURCE
    )
    parser.add_argument(
        "--chimeraboost-source", type=Path, default=DEFAULT_CHIMERA_SOURCE
    )
    parser.add_argument("--primary-artifact", type=Path)
    parser.add_argument("--worker-arm", choices=ALL_ARMS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_arm is not None:
        result = run_worker(
            args.worker_arm,
            args.contract,
            args.data_cache,
            args.darkofit_source,
            args.chimeraboost_source,
        )
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    if args.phase is None or args.output is None:
        raise RuntimeError("parent execution requires --phase and --output")
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
