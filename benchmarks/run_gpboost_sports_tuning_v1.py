#!/usr/bin/env python3
"""Tune regularized GPBoost on fresh basketball development seasons and confirm once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
_ROOT_TEXT = str(ROOT)
if _ROOT_TEXT in sys.path:
    sys.path.remove(_ROOT_TEXT)
sys.path.insert(0, _ROOT_TEXT)

from benchmarks import build_basketball_sports_panel as basketball  # noqa: E402


PROTOCOL = ROOT / "benchmarks" / "gpboost_sports_tuning_v1_protocol.md"
REGISTRY = ROOT / "benchmarks" / "gpboost_sports_tuning_v1_registry_20260723.json"
DEVELOPMENT = ROOT / "benchmarks" / "gpboost_sports_tuning_v1_development_20260723.json"
CONFIRMATION = ROOT / "benchmarks" / "gpboost_sports_tuning_v1_confirmation_20260723.json"
CACHE_DIR = ROOT / ".cache" / "gpboost-sports-tuning-v1"
DEV_CACHE = CACHE_DIR / "development.csv"
CONFIRM_CACHE = CACHE_DIR / "confirmation.csv"
SOURCE = Path(
    "/Users/konstantinmedvedovsky/Library/CloudStorage/Dropbox/github/darko/"
    "calculated_data/temp/bbr_advanced_game_logs.csv"
)
SOURCE_SHA256 = "96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826"
SOURCE_BYTES = 214_366_516
DEV_SEASONS = (2008, 2009, 2010, 2011, 2012, 2013)
CONFIRM_SEASON = 2020
THREADS = 4
SEED = 20_260_723
TRIALS = 48
EARLY_STOPPING_ROUNDS = 50
WORKER_PREFIX = "GPBOOST_SPORTS_TUNING_RESULT="
ARMS = ("darkofit_default", "gpboost_default", "gpboost_tuned")
BLOCK_ORDERS = (
    ("darkofit_default", "gpboost_default", "gpboost_tuned"),
    ("gpboost_tuned", "gpboost_default", "darkofit_default"),
    ("gpboost_default", "darkofit_default", "gpboost_tuned"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _prediction_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def _atomic_create(path: Path, value: bytes) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_create(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _write_panel(path: Path, panel: pd.DataFrame) -> None:
    _atomic_create(path, basketball.panel_csv_bytes(panel))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS unavailable")
    return value


def _score(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.shape != prediction.shape or not len(y_true):
        raise RuntimeError("invalid scoring vectors")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("model produced non-finite predictions")
    r2 = None if len(y_true) < 2 else float(r2_score(y_true, prediction))
    return {
        "rows": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - prediction)))),
        "r2": r2,
        "target_sha256": _prediction_sha256(y_true),
        "prediction_sha256": _prediction_sha256(prediction),
    }


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise RuntimeError("invalid geometric mean inputs")
    return float(np.exp(np.mean(np.log(array))))


def _source_metadata(source: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"basketball source unavailable: {source}")
    if source.stat().st_size != SOURCE_BYTES or _sha256(source) != SOURCE_SHA256:
        raise RuntimeError("basketball source differs from attested export")
    return {
        "path": str(source),
        "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256,
        "encoding": "latin1",
    }


def _panel_metadata(panel: pd.DataFrame) -> dict[str, Any]:
    panel_bytes = basketball.panel_csv_bytes(panel)
    identities = panel.loc[:, list(basketball.IDENTITY_COLUMNS)].values.tolist()
    targets = {
        target: _json_sha256(panel[target].to_numpy(dtype=np.float64).tolist())
        for target in basketball.TARGET_COLUMNS
    }
    return {
        "rows": int(len(panel)),
        "columns": panel.columns.tolist(),
        "bytes": len(panel_bytes),
        "sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "identities_sha256": _json_sha256(identities),
        "target_sha256": targets,
        "seasons": sorted(int(value) for value in panel["year"].unique()),
    }


def _group_fold(groups: np.ndarray, key: str, splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups, dtype=str)
    if len(np.unique(groups)) < splits:
        raise RuntimeError("not enough player groups for requested split")
    plans = list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(plans)
    train, test = plans[index]
    if not set(groups[train]).isdisjoint(groups[test]):
        raise RuntimeError("player split leaked a group")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _development_splits(panel: pd.DataFrame) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for season in DEV_SEASONS:
        seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
        groups = seasonal["bref_id"].astype(str).to_numpy()
        for target in basketball.TARGET_COLUMNS:
            train, validation = _group_fold(groups, f"development:{season}:{target}")
            records[f"{season}:{target}"] = {
                "season": season,
                "target": target,
                "train_indices": train.tolist(),
                "validation_indices": validation.tolist(),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_groups": int(len(set(groups[train]))),
                "validation_groups": int(len(set(groups[validation]))),
                "validation_identities_sha256": _json_sha256(
                    seasonal.loc[validation, list(basketball.IDENTITY_COLUMNS)].values.tolist()
                ),
            }
    return records


def _confirmation_splits(panel: pd.DataFrame) -> dict[str, Any]:
    teams = tuple(sorted(panel["Tm"].astype(str).unique().tolist()))
    if len(teams) != 30:
        raise RuntimeError("confirmation season does not contain 30 teams")
    held = frozenset(teams[10:20])
    primary = panel.loc[~panel["Tm"].isin(held)].reset_index(drop=True)
    holdout = panel.loc[panel["Tm"].isin(held)].reset_index(drop=True)
    primary_groups = primary["bref_id"].astype(str).to_numpy()
    seen = holdout["bref_id"].astype(str).isin(set(primary_groups)).to_numpy()
    folds = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=10).split(primary, groups=primary_groups)
    ):
        if not set(primary_groups[train]).isdisjoint(primary_groups[test]):
            raise RuntimeError("confirmation player split leaked a group")
        folds.append(
            {
                "fold": int(fold),
                "train_indices": [int(value) for value in train],
                "test_indices": [int(value) for value in test],
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_identities_sha256": _json_sha256(
                    primary.loc[test, list(basketball.IDENTITY_COLUMNS)].values.tolist()
                ),
            }
        )
    return {
        "held_teams": sorted(held),
        "primary_rows": int(len(primary)),
        "held_rows": int(len(holdout)),
        "seen_player_rows": int(np.sum(seen)),
        "cold_player_rows": int(np.sum(~seen)),
        "primary_identities_sha256": _json_sha256(
            primary.loc[:, list(basketball.IDENTITY_COLUMNS)].values.tolist()
        ),
        "held_identities_sha256": _json_sha256(
            holdout.loc[:, list(basketball.IDENTITY_COLUMNS)].values.tolist()
        ),
        "folds": folds,
        "folds_sha256": _json_sha256(folds),
    }


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    if args.registry.exists() or args.registry.is_symlink():
        raise FileExistsError(f"refusing to replace registry: {args.registry}")
    source = _source_metadata(args.source)
    frame, loaded = basketball.load_raw_source(args.source)
    if loaded["sha256"] != SOURCE_SHA256:
        raise RuntimeError("basketball builder source attestation changed")
    development = basketball.prepare_panel(frame, seasons=DEV_SEASONS)
    confirmation = basketball.prepare_panel(frame, seasons=(CONFIRM_SEASON,))
    _write_panel(args.dev_cache, development)
    _write_panel(args.confirm_cache, confirmation)
    dev_metadata = _panel_metadata(development)
    confirm_metadata = _panel_metadata(confirmation)
    registry = {
        "schema_version": 1,
        "name": "gpboost_sports_tuning_v1_registry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source": source,
        "development": {
            "cache": str(args.dev_cache),
            "panel": dev_metadata,
            "splits": _development_splits(development),
        },
        "confirmation": {
            "cache": str(args.confirm_cache),
            "panel": confirm_metadata,
            "splits": _confirmation_splits(confirmation),
            "scored": False,
        },
    }
    _write_json(args.registry, registry)
    return registry


def _load_registry(args: argparse.Namespace) -> dict[str, Any]:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("name") != "gpboost_sports_tuning_v1_registry":
        raise RuntimeError("unexpected GPBoost tuning registry")
    registry_protocol = args.registry_protocol or args.protocol
    if registry["protocol"]["sha256"] != _sha256(registry_protocol):
        raise RuntimeError("tuning protocol changed after registry creation")
    if registry["source"] != _source_metadata(args.source):
        raise RuntimeError("tuning source differs from registry")
    return registry


def _load_panel(path: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != metadata["bytes"]
        or _sha256(path) != metadata["sha256"]
    ):
        raise RuntimeError(f"panel cache differs from registry: {path}")
    panel = pd.read_csv(path)
    if panel.columns.tolist() != metadata["columns"] or len(panel) != metadata["rows"]:
        raise RuntimeError("panel cache schema differs from registry")
    if _panel_metadata(panel)["identities_sha256"] != metadata["identities_sha256"]:
        raise RuntimeError("panel identities differ from registry")
    return panel


def _gpboost_default(threads: int):
    from gpboost import GPBoostRegressor

    return GPBoostRegressor(random_state=4, n_jobs=int(threads))


def _trial_params(trial: Any, threads: int) -> dict[str, Any]:
    return {
        "random_state": 4,
        "n_jobs": int(threads),
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
    }


def _fit_gpboost(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> Any:
    from gpboost import GPBoostRegressor

    model = GPBoostRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if X_validation is None:
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_validation, y_validation)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
    return model


def _development_baseline(
    panel: pd.DataFrame, registry: dict[str, Any], threads: int
) -> dict[str, Any]:
    rows = {}
    for key, split in registry["development"]["splits"].items():
        seasonal = panel.loc[panel["year"] == split["season"]].reset_index(drop=True)
        X = seasonal.loc[:, list(basketball.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        y = seasonal[split["target"]].to_numpy(dtype=np.float64)
        train = np.asarray(split["train_indices"], dtype=np.int64)
        validation = np.asarray(split["validation_indices"], dtype=np.int64)
        model = _gpboost_default(threads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model.fit(X[train], y[train], verbose=False)
        prediction = np.asarray(model.predict(X[validation]), dtype=np.float64)
        rows[key] = {
            "rmse": _score(y[validation], prediction)["rmse"],
            "fitted_tree_count": int(model.booster_.num_trees()),
        }
    return rows


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    if args.development_output.exists() or args.development_output.is_symlink():
        raise FileExistsError("refusing to replace development output")
    registry = _load_registry(args)
    panel = _load_panel(args.dev_cache, registry["development"]["panel"])
    baseline = _development_baseline(panel, registry, args.threads)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    def objective(trial: Any) -> float:
        params = _trial_params(trial, args.threads)
        rows = []
        for key, split in registry["development"]["splits"].items():
            seasonal = panel.loc[panel["year"] == split["season"]].reset_index(drop=True)
            X = seasonal.loc[:, list(basketball.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            y = seasonal[split["target"]].to_numpy(dtype=np.float64)
            train = np.asarray(split["train_indices"], dtype=np.int64)
            validation = np.asarray(split["validation_indices"], dtype=np.int64)
            model = _fit_gpboost(params, X[train], y[train], X[validation], y[validation])
            prediction = np.asarray(model.predict(X[validation]), dtype=np.float64)
            score = _score(y[validation], prediction)
            ratio = score["rmse"] / baseline[key]["rmse"]
            rows.append(
                {
                    "lineage": key,
                    "rmse": score["rmse"],
                    "default_rmse": baseline[key]["rmse"],
                    "ratio_to_default": ratio,
                    "fitted_tree_count": int(model.booster_.num_trees()),
                }
            )
        objective_value = _geomean([row["ratio_to_default"] for row in rows])
        trial.set_user_attr("lineages", rows)
        trial.set_user_attr("worst_ratio", max(row["ratio_to_default"] for row in rows))
        return objective_value

    study.optimize(objective, n_trials=args.trials, n_jobs=1, show_progress_bar=False)
    best = study.best_trial
    selected = _trial_params(best, args.threads)
    trials = [
        {
            "number": int(trial.number),
            "objective": float(trial.value),
            "params": trial.params,
            "worst_ratio": float(trial.user_attrs["worst_ratio"]),
            "lineages": trial.user_attrs["lineages"],
        }
        for trial in study.trials
        if trial.value is not None
    ]
    output = {
        "schema_version": 1,
        "name": "gpboost_sports_tuning_v1_development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "fresh_development_tuning_not_confirmation",
        "protocol": {"path": str(args.protocol.relative_to(ROOT)), "sha256": _sha256(args.protocol)},
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "registry": {"path": str(args.registry.relative_to(ROOT)), "sha256": _sha256(args.registry)},
        "execution": {
            "python": sys.version,
            "platform": platform.platform(),
            "threads": args.threads,
            "trials": args.trials,
            "optuna_version": importlib.metadata.version("optuna"),
            "gpboost_version": importlib.metadata.version("gpboost"),
            "seed": SEED,
        },
        "baseline": baseline,
        "selected_trial": int(best.number),
        "selected_objective": float(best.value),
        "selected_worst_ratio": float(best.user_attrs["worst_ratio"]),
        "selected_params": selected,
        "selected_lineages": best.user_attrs["lineages"],
        "trials": trials,
        "confirmation_unscored": True,
    }
    _write_json(args.development_output, output)
    return output


def _import_darkofit(source: Path) -> Any:
    source = source.resolve()
    if not (source / "darkofit" / "__init__.py").is_file():
        raise RuntimeError("invalid DarkoFit source snapshot")
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    module = importlib.import_module("darkofit")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("DarkoFit imported outside requested snapshot")
    return module


def _tuned_refit(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    context: str,
) -> Any:
    inner_train, inner_validation = _group_fold(groups, f"confirmation:{context}")
    selected = _fit_gpboost(
        params,
        X[inner_train],
        y[inner_train],
        X[inner_validation],
        y[inner_validation],
    )
    best_iteration = selected.best_iteration_
    if best_iteration is None or int(best_iteration) < 1:
        best_iteration = max(1, int(selected.booster_.current_iteration()))
    refit_params = dict(params)
    refit_params["n_estimators"] = int(best_iteration)
    return _fit_gpboost(refit_params, X, y), int(best_iteration)


def _confirmation_model(
    arm: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    context: str,
    darkofit_source: Path,
    threads: int,
) -> tuple[Any, dict[str, Any]]:
    if arm == "darkofit_default":
        module = _import_darkofit(darkofit_source)
        model = module.DarkoRegressor(
            random_state=4, thread_count=threads, diagnostic_warnings="never"
        )
        model.fit(X, y)
        return model, {"fitted_tree_count": int(model.n_estimators_)}
    if arm == "gpboost_default":
        model = _gpboost_default(threads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model.fit(X, y, verbose=False)
        return model, {"fitted_tree_count": int(model.booster_.num_trees())}
    if arm == "gpboost_tuned":
        model, selected_iteration = _tuned_refit(params, X, y, groups, context)
        return model, {
            "fitted_tree_count": int(model.booster_.num_trees()),
            "early_stopping_selected_iteration": selected_iteration,
        }
    raise ValueError(f"unknown confirmation arm: {arm}")


def _fit_predict_confirmation(
    arm: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    context: str,
    darkofit_source: Path,
    threads: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    model, metadata = _confirmation_model(
        arm, params, X_train, y_train, groups, context, darkofit_source, threads
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return {
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "metadata": metadata,
    }


def _confirmation_views(
    panel: pd.DataFrame, registry: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    split = registry["confirmation"]["splits"]
    held = frozenset(split["held_teams"])
    primary = panel.loc[~panel["Tm"].isin(held)].reset_index(drop=True)
    holdout = panel.loc[panel["Tm"].isin(held)].reset_index(drop=True)
    groups = primary["bref_id"].astype(str).to_numpy()
    seen = holdout["bref_id"].astype(str).isin(set(groups)).to_numpy()
    if (
        len(primary) != split["primary_rows"]
        or len(holdout) != split["held_rows"]
        or _json_sha256(primary.loc[:, list(basketball.IDENTITY_COLUMNS)].values.tolist())
        != split["primary_identities_sha256"]
        or _json_sha256(holdout.loc[:, list(basketball.IDENTITY_COLUMNS)].values.tolist())
        != split["held_identities_sha256"]
    ):
        raise RuntimeError("confirmation views differ from frozen registry")
    return primary, holdout, seen, list(split["folds"])


def run_confirmation_worker(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_registry(args)
    development = json.loads(args.development_output.read_text(encoding="utf-8"))
    if (
        development.get("confirmation_unscored") is not True
        or development["registry"]["sha256"] != _sha256(args.registry)
        or development["protocol"]["sha256"] != registry["protocol"]["sha256"]
    ):
        raise RuntimeError("development artifact does not bind confirmation")
    panel = _load_panel(args.confirm_cache, registry["confirmation"]["panel"])
    primary, holdout, seen, folds = _confirmation_views(panel, registry)
    X_primary = primary.loc[:, list(basketball.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    X_holdout = holdout.loc[:, list(basketball.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    groups = primary["bref_id"].astype(str).to_numpy()
    warmup = _fit_predict_confirmation(
        args.worker_arm,
        development["selected_params"],
        X_primary[np.asarray(folds[0]["train_indices"], dtype=np.int64)],
        primary[basketball.TARGET_COLUMNS[0]].to_numpy(dtype=np.float64)[
            np.asarray(folds[0]["train_indices"], dtype=np.int64)
        ],
        groups[np.asarray(folds[0]["train_indices"], dtype=np.int64)],
        X_primary[np.asarray(folds[0]["test_indices"], dtype=np.int64)],
        "warmup",
        args.darkofit_source,
        args.threads,
    )
    del warmup
    gc.collect()
    cells = []
    fit_total = predict_total = 0.0
    started = time.perf_counter_ns()
    for target in basketball.TARGET_COLUMNS:
        y_primary = primary[target].to_numpy(dtype=np.float64)
        y_holdout = holdout[target].to_numpy(dtype=np.float64)
        oof = np.empty(len(primary), dtype=np.float64)
        fold_rows = []
        for frozen in folds:
            train = np.asarray(frozen["train_indices"], dtype=np.int64)
            test = np.asarray(frozen["test_indices"], dtype=np.int64)
            fitted = _fit_predict_confirmation(
                args.worker_arm,
                development["selected_params"],
                X_primary[train],
                y_primary[train],
                groups[train],
                X_primary[test],
                f"{target}:fold:{frozen['fold']}",
                args.darkofit_source,
                args.threads,
            )
            prediction = fitted.pop("prediction")
            oof[test] = prediction
            fit_total += fitted["fit_seconds"]
            predict_total += fitted["predict_seconds"]
            fold_rows.append({"fold": frozen["fold"], "score": _score(y_primary[test], prediction), **fitted})
        guardrail = _fit_predict_confirmation(
            args.worker_arm,
            development["selected_params"],
            X_primary,
            y_primary,
            groups,
            X_holdout,
            f"{target}:held",
            args.darkofit_source,
            args.threads,
        )
        prediction = guardrail.pop("prediction")
        fit_total += guardrail["fit_seconds"]
        predict_total += guardrail["predict_seconds"]
        cells.append(
            {
                "season": CONFIRM_SEASON,
                "target": target,
                "primary": _score(y_primary, oof),
                "folds": fold_rows,
                "guardrail": {
                    "held_team": _score(y_holdout, prediction),
                    "seen_player": _score(y_holdout[seen], prediction[seen]),
                    "cold_player": _score(y_holdout[~seen], prediction[~seen]),
                    **guardrail,
                },
            }
        )
    return {
        "arm": args.worker_arm,
        "cells": cells,
        "total_fit_seconds": float(fit_total),
        "total_predict_seconds": float(predict_total),
        "steady_wall_seconds": float((time.perf_counter_ns() - started) / 1e9),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _worker_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    threads = str(args.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "NUMBA_NUM_THREADS": threads,
            "LOKY_MAX_CPU_COUNT": threads,
            "PYTHONHASHSEED": "0",
            "DARKOFIT_WARMUP": "0",
            "MPLCONFIGDIR": str(args.worker_cache),
            "PYTHONPATH": os.pathsep.join((str(args.darkofit_source), str(ROOT))),
        }
    )
    return environment


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "confirm-worker", "--worker-arm", arm,
        "--registry", str(args.registry), "--development-output", str(args.development_output),
        "--confirm-cache", str(args.confirm_cache), "--source", str(args.source),
        "--protocol", str(args.protocol), "--darkofit-source", str(args.darkofit_source),
        "--darkofit-revision", args.darkofit_revision,
        "--threads", str(args.threads), "--worker-cache", str(args.worker_cache),
    ]
    if args.registry_protocol is not None:
        command.extend(("--registry-protocol", str(args.registry_protocol)))
    completed = subprocess.run(command, cwd=ROOT, env=_worker_environment(args), capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if line.startswith(WORKER_PREFIX)]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(f"confirmation worker {arm} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(lines[0][len(WORKER_PREFIX):])


def _view(cell: dict[str, Any], name: str) -> dict[str, Any]:
    return cell["primary"] if name == "primary" else cell["guardrail"][name]


def _ratio_summary(left: dict[str, Any], right: dict[str, Any], view: str) -> dict[str, Any]:
    left_cells = {cell["target"]: cell for cell in left["cells"]}
    right_cells = {cell["target"]: cell for cell in right["cells"]}
    rows = []
    for target in basketball.TARGET_COLUMNS:
        left_score, right_score = _view(left_cells[target], view), _view(right_cells[target], view)
        if left_score["target_sha256"] != right_score["target_sha256"]:
            raise RuntimeError("target changed between confirmation arms")
        rows.append({"target": target, "ratio": right_score["rmse"] / left_score["rmse"]})
    return {"geometric_mean_ratio": _geomean([row["ratio"] for row in rows]), "rows": rows}


def _confirmation_summary(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"orientation": "right_arm / left_arm; lower favors right arm"}
    for left, right, name in (
        ("darkofit_default", "gpboost_default", "gpboost_default_over_darkofit"),
        ("darkofit_default", "gpboost_tuned", "gpboost_tuned_over_darkofit"),
        ("gpboost_default", "gpboost_tuned", "gpboost_tuned_over_default"),
    ):
        quality = {}
        for view in ("primary", "held_team", "cold_player"):
            values = []
            for block in range(len(BLOCK_ORDERS)):
                arms = {row["arm"]: row["result"] for row in repeats if row["block"] == block}
                values.append(_ratio_summary(arms[left], arms[right], view))
            ratios = [value["geometric_mean_ratio"] for value in values]
            quality[view] = {"by_block": values, "median": float(np.median(ratios)), "min": float(np.min(ratios)), "max": float(np.max(ratios))}
        costs = {}
        for metric in ("total_fit_seconds", "total_predict_seconds", "steady_wall_seconds", "peak_rss_bytes"):
            ratios = []
            for block in range(len(BLOCK_ORDERS)):
                arms = {row["arm"]: row["result"] for row in repeats if row["block"] == block}
                ratios.append(arms[right][metric] / arms[left][metric])
            costs[metric] = {"by_block": ratios, "median": float(np.median(ratios))}
        summary[name] = {"quality": quality, "cost": costs}
    return summary


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation_output.exists() or args.confirmation_output.is_symlink():
        raise FileExistsError("refusing to replace confirmation output")
    _load_registry(args)
    development = json.loads(args.development_output.read_text(encoding="utf-8"))
    if development.get("confirmation_unscored") is not True:
        raise RuntimeError("development artifact cannot authorize confirmation")
    repeats = []
    args.worker_cache.mkdir(parents=True, exist_ok=True)
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            print(f"[gpboost-tuning] block={block} position={position} arm={arm}", flush=True)
            repeats.append({"block": block, "position": position, "order": list(order), "arm": arm, "result": _run_worker_process(args, arm)})
    output = {
        "schema_version": 1,
        "name": f"{args.protocol.stem.removesuffix('_protocol')}_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "fresh_sports_external_comparator_characterization",
        "protocol": {"path": str(args.protocol.relative_to(ROOT)), "sha256": _sha256(args.protocol)},
        "registry_protocol": {
            "path": str((args.registry_protocol or args.protocol).relative_to(ROOT)),
            "sha256": _sha256(args.registry_protocol or args.protocol),
        },
        "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())},
        "registry": {"path": str(args.registry.relative_to(ROOT)), "sha256": _sha256(args.registry)},
        "development": {"path": str(args.development_output.relative_to(ROOT)), "sha256": _sha256(args.development_output), "selected_params": development["selected_params"]},
        "darkofit": {
            "archive_path": str(args.darkofit_source),
            "revision": args.darkofit_revision,
            "package_init_sha256": _sha256(args.darkofit_source / "darkofit" / "__init__.py"),
        },
        "execution": {"python": sys.version, "platform": platform.platform(), "threads": args.threads, "gpboost_version": importlib.metadata.version("gpboost"), "block_orders": [list(order) for order in BLOCK_ORDERS]},
        "repeats": repeats,
        "summary": _confirmation_summary(repeats),
        "non_claims": ["No GPModel/group-effect evaluation.", "No DarkoFit product or release claim.", "Three 2020 lineages are external-comparator context, not broad Pareto evidence."],
    }
    _write_json(args.confirmation_output, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "tune", "confirm", "confirm-worker"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--source", type=Path, default=SOURCE)
        subparser.add_argument("--protocol", type=Path, default=PROTOCOL)
        subparser.add_argument("--registry-protocol", type=Path)
        subparser.add_argument("--registry", type=Path, default=REGISTRY)
        subparser.add_argument("--dev-cache", type=Path, default=DEV_CACHE)
        subparser.add_argument("--confirm-cache", type=Path, default=CONFIRM_CACHE)
        subparser.add_argument("--development-output", type=Path, default=DEVELOPMENT)
        subparser.add_argument("--confirmation-output", type=Path, default=CONFIRMATION)
        subparser.add_argument("--threads", type=int, default=THREADS)
        subparser.add_argument("--trials", type=int, default=TRIALS)
        subparser.add_argument("--darkofit-source", type=Path)
        subparser.add_argument("--darkofit-revision")
        subparser.add_argument("--worker-cache", type=Path, default=Path("/private/tmp/gpboost-sports-tuning-mpl"))
        if name == "confirm-worker":
            subparser.add_argument("--worker-arm", choices=ARMS, required=True)
    args = parser.parse_args()
    for key in ("source", "protocol", "registry", "dev_cache", "confirm_cache", "development_output", "confirmation_output", "worker_cache"):
        setattr(args, key, getattr(args, key).resolve())
    if args.registry_protocol is not None:
        args.registry_protocol = args.registry_protocol.resolve()
    if args.darkofit_source is not None:
        args.darkofit_source = args.darkofit_source.resolve()
    if args.threads < 1 or args.trials < 1:
        raise ValueError("threads and trials must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "build":
        result = build_registry(args)
        print(json.dumps({"registry": str(args.registry), "sha256": _sha256(args.registry), "development_rows": result["development"]["panel"]["rows"], "confirmation_rows": result["confirmation"]["panel"]["rows"]}, sort_keys=True))
        return 0
    if args.command == "tune":
        result = run_tuning(args)
        print(json.dumps({"development": str(args.development_output), "sha256": _sha256(args.development_output), "selected_objective": result["selected_objective"]}, sort_keys=True))
        return 0
    if args.command == "confirm-worker":
        if args.darkofit_source is None or not args.darkofit_revision:
            raise ValueError("confirmation worker needs DarkoFit source and revision")
        print(WORKER_PREFIX + json.dumps(run_confirmation_worker(args), sort_keys=True, allow_nan=False))
        return 0
    if args.command == "confirm":
        if args.darkofit_source is None or not args.darkofit_revision:
            raise ValueError("confirmation needs DarkoFit source and revision")
        result = run_confirmation(args)
        print(json.dumps({"confirmation": str(args.confirmation_output), "sha256": _sha256(args.confirmation_output), "workers": len(result["repeats"])}, sort_keys=True))
        return 0
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
