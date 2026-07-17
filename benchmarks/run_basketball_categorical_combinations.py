#!/usr/bin/env python3
"""Run the frozen basketball categorical-combinations donor screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
CHIMERABOOST_REPO = REPO_ROOT.parent / "chimeraboost"
for import_path in (str(REPO_ROOT), str(CHIMERABOOST_REPO)):
    if import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from benchmarks import basketball_guardrails as guardrails  # noqa: E402
from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_categorical_combinations_protocol.md"
)
PROTOCOL_COMMIT = "03ca0e73d595a112162f20e16a039c544378ab1f"
EXPECTED_PROTOCOL_SHA256 = (
    "f4922578cc0eea7feee6c0bdc5f250972d4ae772cc5db210ac7e4be8a14a1123"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = "68d187ef89df971abc4b1559dc2c1fc8b7edf9bd7ee906ce60dfcddf6f6ec1cb"
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
}
EXPECTED_CHIMERABOOST_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_CHIMERABOOST_TREE = "112078745db7f58b6d4399ecbb4ecebafe860256"
EXPECTED_CHIMERABOOST_SHA256 = {
    "chimeraboost/preprocessing.py": (
        "c3f062058a40df17b35b7a3c932d16173ba45c5294ca450b08ef447f563bcecb"
    ),
    "chimeraboost/sklearn_api.py": (
        "d354d360fea762be46a92e6cfaf9bc244c60690b7ecd0eebb8753aabc4b78c15"
    ),
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
    },
}
EXPECTED_DATA = {
    "raw_sha256": "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2",
    "numeric_x_sha256": (
        "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
    ),
    "y_sha256": "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf",
    "fold_sha256": "7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea",
    "cold_mask_sha256": (
        "e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19"
    ),
    "categorical_train_sha256": (
        "8f201e2c36b4addc6a223fb58b91912a5f2a0a6e732bea6558b0230c000cec17"
    ),
    "categorical_holdout_sha256": (
        "ca708478f883aae7b2ebb1c01eea0ba6566af328505fd0810af152a3ac2ca18d"
    ),
    "train_players_sha256": (
        "f59ca6aefdbdafb0ac6be4e9073bd5cbf5e5b0b8413004c30e776f7cae19c22d"
    ),
    "holdout_players_sha256": (
        "4f161a9233d4bfe5017c2f13e9b52d511c728e2b7c345479375b5acd3a8e995e"
    ),
}

DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_categorical_combinations.json"
)
NUMBA_CACHE_ROOT = (
    REPO_ROOT / ".cache" / "basketball-categorical-combinations-numba"
)
WORKER_PREFIX = "BASKETBALL_CAT_COMBINATIONS_RESULT="
CATEGORICAL_COLUMNS = ("Pos", "Age", "Tm", "starter")
CATEGORICAL_FEATURES = (0, 1, 2, 3)
EXPECTED_COMBO_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
CONTROL = "control"
CANDIDATE = "candidate"
NUMERIC_AUTO = "numeric_auto"
NUMERIC_OFF = "numeric_off"
FOLD_MODE = "fold"
FULL_MODE = "full"
NUMERIC_MODE = "numeric"
TIMING_BLOCKS = 5
HELD_PREDICTION_CALLS = 100
COLD_PREDICTION_CALLS = 200
EXPECTED_THREADS = 18
MAX_TIMING_IQR_FRACTION = 0.20
MIB = 1024 * 1024

COMMON_MODEL_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": None,
    "depth": None,
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "cat_smoothing": 1.0,
    "cat_n_permutations": 4,
    "early_stopping_rounds": None,
    "loss": "RMSE",
    "min_child_weight": 1.0,
    "thread_count": EXPECTED_THREADS,
    "random_state": creator.RANDOM_STATE,
    "ordered_boosting": False,
    "leaf_estimation_iterations": 1,
    "linear_leaves": False,
    "cross_features": False,
    "selection_rounds": 100,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "n_ensembles": None,
}


@dataclass(frozen=True)
class BasketballCategoricalView:
    dataset: harness.BasketballDataset
    X_train: np.ndarray
    X_holdout: np.ndarray
    y_train: np.ndarray
    y_holdout: np.ndarray
    train_players: np.ndarray
    holdout_players: np.ndarray
    cold_mask: np.ndarray
    seen_mask: np.ndarray
    metadata: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return _sha256_bytes(array.tobytes())


def _canonical_string_sha256(values: Any) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=object).reshape(-1):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _normalized_runner_sha256() -> str:
    payload = Path(__file__).resolve().read_bytes()
    pattern = (
        rb'EXPECTED_NORMALIZED_RUNNER_SHA256 = "[0-9a-f]{64}"'
    )
    replacement = (
        b'EXPECTED_NORMALIZED_RUNNER_SHA256 = "' + (b"0" * 64) + b'"'
    )
    payload, count = re.subn(pattern, replacement, payload)
    if count != 1:
        raise RuntimeError("runner normalization field changed")
    return _sha256_bytes(payload)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_state(repository: Path) -> dict[str, Any]:
    remotes = {}
    for remote_ref in ("origin/main", "upstream/main"):
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", remote_ref],
            cwd=repository,
            capture_output=True,
            text=True,
        )
        remotes[remote_ref] = (
            completed.stdout.strip() if completed.returncode == 0 else None
        )
    return {
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "status": _git(
            repository, "status", "--porcelain=v1", "--untracked-files=all"
        ),
        "remotes": remotes,
    }


def _runtime_record() -> dict[str, Any]:
    dependencies = {
        name: importlib.metadata.version(name)
        for name in EXPECTED_RUNTIME["dependencies"]
    }
    cpu_brand = creator._machine_details()["cpu_brand"]
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu_brand,
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependencies,
    }


def _validate_parent_binding(output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing output: {output}")
    if _sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("categorical-combinations protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("categorical-combinations runner changed")
    for relative, expected in EXPECTED_SUPPORT_SHA256.items():
        if _sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"support file changed: {relative}")

    darkofit = _repository_state(REPO_ROOT)
    if darkofit["branch"] != "main" or darkofit["status"]:
        raise RuntimeError("DarkoFit must be clean on main")
    if darkofit["remotes"]["origin/main"] != darkofit["head"]:
        raise RuntimeError("DarkoFit main is not pushed to origin")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    ).returncode:
        raise RuntimeError("frozen protocol commit is not an ancestor")

    chimera = _repository_state(CHIMERABOOST_REPO)
    if chimera["branch"] != "main" or chimera["status"]:
        raise RuntimeError("ChimeraBoost must be clean on main")
    if chimera["head"] != EXPECTED_CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost head changed")
    for remote_ref, remote_head in chimera["remotes"].items():
        if remote_head != EXPECTED_CHIMERABOOST_HEAD:
            raise RuntimeError(f"ChimeraBoost {remote_ref} changed")
    if (
        _git(CHIMERABOOST_REPO, "rev-parse", "HEAD:chimeraboost")
        != EXPECTED_CHIMERABOOST_TREE
    ):
        raise RuntimeError("ChimeraBoost package tree changed")
    for relative, expected in EXPECTED_CHIMERABOOST_SHA256.items():
        if _sha256_file(CHIMERABOOST_REPO / relative) != expected:
            raise RuntimeError(f"ChimeraBoost source changed: {relative}")

    runtime = _runtime_record()
    expected_python = str(
        REPO_ROOT / ".cache" / "basketball-py312" / "bin" / "python"
    )
    if runtime["python_executable"] != expected_python:
        raise RuntimeError("runtime python executable changed")
    for field in ("python", "platform", "machine", "logical_cpu_count"):
        if runtime[field] != EXPECTED_RUNTIME[field]:
            raise RuntimeError(f"runtime {field} changed")
    if runtime["dependencies"] != EXPECTED_RUNTIME["dependencies"]:
        raise RuntimeError("runtime dependency stack changed")
    return {
        "darkofit": darkofit,
        "chimeraboost": chimera,
        "runtime": runtime,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_normalized_sha256": EXPECTED_NORMALIZED_RUNNER_SHA256,
        "support_sha256": EXPECTED_SUPPORT_SHA256,
        "chimeraboost_source_sha256": EXPECTED_CHIMERABOOST_SHA256,
    }


def _assert_source_unchanged(binding: dict[str, Any]) -> None:
    if _repository_state(REPO_ROOT) != binding["darkofit"]:
        raise RuntimeError("DarkoFit source state changed during the campaign")
    if _repository_state(CHIMERABOOST_REPO) != binding["chimeraboost"]:
        raise RuntimeError("ChimeraBoost source state changed during the campaign")


def _prepare_categorical_view(
    cache_path: Path = harness.DEFAULT_CACHE,
) -> BasketballCategoricalView:
    dataset = harness.load_basketball_dataset(cache_path)
    frame = dataset.frame.copy()
    frame["starter"] = np.where(
        frame["GS"] / frame["G"] >= 0.5, 1, 0
    )
    train_index = dataset.player_guardrail.X_train.index
    holdout_index = dataset.player_guardrail.X_holdout.index
    X_train_frame = frame.loc[train_index, list(CATEGORICAL_COLUMNS)]
    X_holdout_frame = frame.loc[holdout_index, list(CATEGORICAL_COLUMNS)]
    train_players = frame.loc[train_index, "Player"].astype(str).to_numpy()
    holdout_players = frame.loc[holdout_index, "Player"].astype(str).to_numpy()
    y_train = dataset.y.to_numpy(dtype=np.float64)
    y_holdout = dataset.player_guardrail.y_holdout.to_numpy(dtype=np.float64)
    cold_mask = np.asarray(
        dataset.player_guardrail.cold_player_mask, dtype=bool
    )
    seen_mask = ~cold_mask

    metadata = {
        "columns": list(CATEGORICAL_COLUMNS),
        "train_rows": len(X_train_frame),
        "holdout_rows": len(X_holdout_frame),
        "cold_rows": int(cold_mask.sum()),
        "seen_rows": int(seen_mask.sum()),
        "train_levels": {
            column: int(X_train_frame[column].nunique(dropna=False))
            for column in CATEGORICAL_COLUMNS
        },
        "holdout_levels": {
            column: int(X_holdout_frame[column].nunique(dropna=False))
            for column in CATEGORICAL_COLUMNS
        },
        "categorical_train_sha256": _canonical_string_sha256(X_train_frame),
        "categorical_holdout_sha256": _canonical_string_sha256(X_holdout_frame),
        "train_players_sha256": _canonical_string_sha256(train_players),
        "holdout_players_sha256": _canonical_string_sha256(holdout_players),
        "raw_sha256": dataset.raw_metadata["sha256"],
        "numeric_x_sha256": dataset.processed_metadata["x_train_sha256"],
        "y_sha256": dataset.processed_metadata["y_train_sha256"],
        "fold_sha256": dataset.fold_fingerprint_sha256,
        "fold_test_sizes": dataset.fold_test_sizes,
        "cold_mask_sha256": dataset.player_guardrail.metadata[
            "cold_player_mask_sha256"
        ],
        "test_teams": dataset.player_guardrail.metadata["test_teams"],
    }
    expected = {
        **EXPECTED_DATA,
        "train_rows": 5241,
        "holdout_rows": 2409,
        "cold_rows": 585,
        "seen_rows": 1824,
        "train_levels": {"Pos": 5, "Age": 23, "Tm": 25, "starter": 2},
        "holdout_levels": {"Pos": 5, "Age": 24, "Tm": 12, "starter": 2},
        "fold_test_sizes": [525] + [524] * 9,
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise RuntimeError(
                f"categorical basketball binding changed for {key}: "
                f"{metadata[key]!r}"
            )
    if not np.array_equal(
        y_train, dataset.player_guardrail.y_train.to_numpy(dtype=np.float64)
    ):
        raise RuntimeError("categorical view changed the training target")
    return BasketballCategoricalView(
        dataset=dataset,
        X_train=X_train_frame.to_numpy(dtype=object),
        X_holdout=X_holdout_frame.to_numpy(dtype=object),
        y_train=y_train,
        y_holdout=y_holdout,
        train_players=train_players,
        holdout_players=holdout_players,
        cold_mask=cold_mask,
        seen_mask=seen_mask,
        metadata=metadata,
    )


def _build_estimator(arm: str):
    from chimeraboost import ChimeraBoostRegressor

    if arm == CONTROL or arm == NUMERIC_OFF:
        cat_combinations = False
    elif arm == CANDIDATE:
        cat_combinations = True
    elif arm == NUMERIC_AUTO:
        cat_combinations = None
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return ChimeraBoostRegressor(
        **COMMON_MODEL_PARAMS,
        cat_combinations=cat_combinations,
    )


def _assert_chimera_import(model: Any) -> dict[str, Any]:
    import chimeraboost

    module_path = Path(chimeraboost.__file__).resolve()
    if not module_path.is_relative_to(CHIMERABOOST_REPO.resolve()):
        raise RuntimeError(f"ChimeraBoost imported outside source tree: {module_path}")
    if chimeraboost.__version__ != "0.15.0":
        raise RuntimeError("ChimeraBoost version changed")
    if model.__class__.__module__ != "chimeraboost.sklearn_api":
        raise RuntimeError("unexpected ChimeraBoost estimator class")
    try:
        distribution_version = importlib.metadata.version("chimeraboost")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "module_file": str(module_path),
        "module_version": chimeraboost.__version__,
        "distribution_version": distribution_version,
        "estimator_class": (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        ),
    }


def _model_metadata(
    model: Any,
    *,
    expected_core_rows: int,
) -> dict[str, Any]:
    core = model.model_
    prep = core.prep_
    pairs = [list(pair) for pair in prep.combo_pairs_]
    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    if importance.shape != (model.n_features_in_,) or not np.all(
        np.isfinite(importance)
    ):
        raise RuntimeError("invalid ChimeraBoost feature importance")
    metadata = {
        "best_iteration": int(core.best_iteration_),
        "fitted_tree_count": len(core.trees_),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "stop_reason": (
            "iteration_limit"
            if len(core.trees_) == int(model.n_estimators)
            else "validation_best_prefix"
        ),
        "combo_pairs": pairs,
        "combo_map_sizes": [len(mapping) for mapping in prep.combo_maps_],
        "base_cat_map_sizes": [len(mapping) for mapping in prep.cat_maps_],
        "preprocessed_columns": int(len(prep.n_bins_)),
        "expected_core_fit_rows": int(expected_core_rows),
        "combination_code_cells": int(expected_core_rows * len(pairs)),
        "feature_importance": importance.tolist(),
        "feature_importance_sha256": _sha256_array(importance),
        "common_model_params": COMMON_MODEL_PARAMS,
    }
    if metadata["resolved_thread_count"] != EXPECTED_THREADS:
        raise RuntimeError("ChimeraBoost did not resolve 18 threads")
    if metadata["resolved_learning_rate"] != 0.1:
        raise RuntimeError("ChimeraBoost learning-rate resolution changed")
    return metadata


def _expected_internal_split_rows(X: Any, y: Any, groups: Any) -> int:
    from chimeraboost.sklearn_api import _make_eval_split

    split = _make_eval_split(
        X,
        y,
        COMMON_MODEL_PARAMS["validation_fraction"],
        COMMON_MODEL_PARAMS["random_state"],
        groups=groups,
        stratify=None,
    )
    if split is None:
        raise RuntimeError("group-aware internal split unexpectedly disabled")
    train_index, validation_index = split
    if set(np.asarray(groups)[train_index]) & set(
        np.asarray(groups)[validation_index]
    ):
        raise RuntimeError("internal validation split leaked a player group")
    return int(len(train_index))


def _fit_model(
    arm: str,
    X_train: Any,
    y_train: Any,
    groups: Any,
    *,
    categorical: bool,
) -> tuple[Any, dict[str, Any]]:
    model = _build_estimator(arm)
    implementation = _assert_chimera_import(model)
    expected_core_rows = _expected_internal_split_rows(
        X_train, y_train, groups
    )
    started = time.perf_counter_ns()
    model.fit(
        X_train,
        y_train,
        cat_features=list(CATEGORICAL_FEATURES) if categorical else None,
        groups=groups,
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    return model, {
        "fit_seconds": float(fit_seconds),
        "metadata": _model_metadata(
            model, expected_core_rows=expected_core_rows
        ),
        "implementation": implementation,
    }


def _predict_once(model: Any, X: Any) -> np.ndarray:
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    if prediction.ndim != 1 or len(prediction) != len(X):
        raise RuntimeError("invalid ChimeraBoost prediction shape")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("non-finite ChimeraBoost prediction")
    return prediction


def _timed_predict(model: Any, X: Any, calls: int) -> tuple[np.ndarray, float]:
    _predict_once(model, X)
    prediction = None
    started = time.perf_counter_ns()
    for _ in range(int(calls)):
        prediction = _predict_once(model, X)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    assert prediction is not None
    return prediction, float(elapsed / calls)


def _fold_splits(view: BasketballCategoricalView):
    return list(creator.creator_cv().split(view.X_train, view.y_train))


def _worker_fold(
    view: BasketballCategoricalView, arm: str, fold: int
) -> dict[str, Any]:
    train_index, test_index = _fold_splits(view)[fold]
    model, fit = _fit_model(
        arm,
        view.X_train[train_index],
        view.y_train[train_index],
        view.train_players[train_index],
        categorical=True,
    )
    prediction = _predict_once(model, view.X_train[test_index])
    return {
        "mode": FOLD_MODE,
        "arm": arm,
        "fold": int(fold),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "test_index": test_index.tolist(),
        **fit,
        "prediction": prediction.tolist(),
        "prediction_sha256": _sha256_array(prediction),
        "r2": float(r2_score(view.y_train[test_index], prediction)),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def _worker_full(
    view: BasketballCategoricalView, arm: str
) -> dict[str, Any]:
    model, fit = _fit_model(
        arm,
        view.X_train,
        view.y_train,
        view.train_players,
        categorical=True,
    )
    held, held_seconds = _timed_predict(
        model, view.X_holdout, HELD_PREDICTION_CALLS
    )
    cold, cold_seconds = _timed_predict(
        model, view.X_holdout[view.cold_mask], COLD_PREDICTION_CALLS
    )
    if not np.array_equal(cold, held[view.cold_mask]):
        raise RuntimeError("cold-player prediction route changed values")
    scores = guardrails.score_player_guardrails(
        view.y_holdout, held, view.cold_mask
    )
    return {
        "mode": FULL_MODE,
        "arm": arm,
        **fit,
        "held_prediction": held.tolist(),
        "held_prediction_sha256": _sha256_array(held),
        "cold_prediction": cold.tolist(),
        "cold_prediction_sha256": _sha256_array(cold),
        "guardrail_scores": scores,
        "held_predict_seconds_per_call": held_seconds,
        "held_prediction_calls": HELD_PREDICTION_CALLS,
        "cold_predict_seconds_per_call": cold_seconds,
        "cold_prediction_calls": COLD_PREDICTION_CALLS,
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def _worker_numeric(
    view: BasketballCategoricalView, arm: str
) -> dict[str, Any]:
    X_train = view.dataset.X.to_numpy(dtype=np.float64)
    X_holdout = view.dataset.player_guardrail.X_holdout.to_numpy(
        dtype=np.float64
    )
    model, fit = _fit_model(
        arm,
        X_train,
        view.y_train,
        view.train_players,
        categorical=False,
    )
    train_prediction = _predict_once(model, X_train)
    held_prediction = _predict_once(model, X_holdout)
    return {
        "mode": NUMERIC_MODE,
        "arm": arm,
        **fit,
        "train_prediction": train_prediction.tolist(),
        "train_prediction_sha256": _sha256_array(train_prediction),
        "held_prediction": held_prediction.tolist(),
        "held_prediction_sha256": _sha256_array(held_prediction),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def _worker_payload(mode: str, arm: str, fold: int | None) -> dict[str, Any]:
    view = _prepare_categorical_view()
    if mode == FOLD_MODE:
        if fold is None or fold < 0 or fold >= creator.N_SPLITS:
            raise ValueError("fold worker requires a valid fold")
        payload = _worker_fold(view, arm, fold)
    elif mode == FULL_MODE:
        payload = _worker_full(view, arm)
    elif mode == NUMERIC_MODE:
        payload = _worker_numeric(view, arm)
    else:
        raise ValueError(f"unknown worker mode {mode!r}")
    if not _numeric_finite(payload):
        raise RuntimeError("worker payload contains non-finite values")
    return payload


def _worker_environment() -> dict[str, str]:
    if NUMBA_CACHE_ROOT.is_symlink():
        raise RuntimeError(f"refusing symlink Numba cache: {NUMBA_CACHE_ROOT}")
    NUMBA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not NUMBA_CACHE_ROOT.is_dir():
        raise RuntimeError(f"Numba cache is not a directory: {NUMBA_CACHE_ROOT}")
    environment = harness.worker_environment(EXPECTED_THREADS)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(CHIMERABOOST_REPO.resolve()), str(REPO_ROOT.resolve()))
    )
    environment["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _run_fresh_worker(
    binding: dict[str, Any],
    *,
    mode: str,
    arm: str,
    fold: int | None = None,
) -> dict[str, Any]:
    _assert_source_unchanged(binding)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--arm",
        arm,
    ]
    if fold is not None:
        command.extend(("--fold", str(fold)))
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    process_seconds = (time.perf_counter_ns() - started) / 1e9
    _assert_source_unchanged(binding)
    if completed.returncode:
        raise RuntimeError(
            f"{mode}/{arm} worker failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    lines = [
        line[len(WORKER_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"{mode}/{arm} worker emitted {len(lines)} result records"
        )
    payload = json.loads(lines[0])
    payload["process_seconds"] = float(process_seconds)
    payload["worker_stderr"] = completed.stderr
    return payload


def _series_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (TIMING_BLOCKS,) or not np.all(np.isfinite(array)):
        raise RuntimeError("timing series shape or finiteness changed")
    if np.any(array <= 0.0):
        raise RuntimeError("timing series must be positive")
    median = float(np.median(array))
    q25, q75 = np.percentile(array, [25.0, 75.0])
    iqr = float(q75 - q25)
    return {
        "values": array.tolist(),
        "minimum": float(array.min()),
        "median": median,
        "maximum": float(array.max()),
        "p25": float(q25),
        "p75": float(q75),
        "iqr": iqr,
        "iqr_fraction": float(iqr / median),
    }


def _array_exact(records: list[dict[str, Any]], key: str) -> bool:
    reference = np.asarray(records[0][key], dtype=np.float64)
    return all(
        np.array_equal(reference, np.asarray(record[key], dtype=np.float64))
        for record in records[1:]
    )


def _metadata_exact(records: list[dict[str, Any]], key: str) -> bool:
    reference = records[0]["metadata"][key]
    return all(record["metadata"][key] == reference for record in records[1:])


def _summarize_folds(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for fold in range(creator.N_SPLITS):
        control = results[CONTROL][fold]
        candidate = results[CANDIDATE][fold]
        if control["test_index"] != candidate["test_index"]:
            raise RuntimeError("candidate/control outer fold changed")
        delta = float(candidate["r2"] - control["r2"])
        rows.append(
            {
                "fold": fold,
                "test_rows": control["test_rows"],
                "control_r2": control["r2"],
                "candidate_r2": candidate["r2"],
                "delta": delta,
                "control_prediction_sha256": control["prediction_sha256"],
                "candidate_prediction_sha256": candidate["prediction_sha256"],
            }
        )
    control_mean = float(statistics.fmean(row["control_r2"] for row in rows))
    candidate_mean = float(
        statistics.fmean(row["candidate_r2"] for row in rows)
    )
    deltas = [row["delta"] for row in rows]
    return {
        "rows": rows,
        "control_mean_r2": control_mean,
        "candidate_mean_r2": candidate_mean,
        "mean_delta": float(candidate_mean - control_mean),
        "wins": sum(delta > 0.0 for delta in deltas),
        "losses": sum(delta < 0.0 for delta in deltas),
        "ties": sum(delta == 0.0 for delta in deltas),
        "worst_delta": float(min(deltas)),
    }


def _summarize_full(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    arms = {}
    for arm in (CONTROL, CANDIDATE):
        records = results[arm]
        summaries = {
            "fit": _series_summary(
                [record["fit_seconds"] for record in records]
            ),
            "held_prediction": _series_summary(
                [
                    record["held_predict_seconds_per_call"]
                    for record in records
                ]
            ),
            "cold_prediction": _series_summary(
                [
                    record["cold_predict_seconds_per_call"]
                    for record in records
                ]
            ),
            "peak_rss_bytes": _series_summary(
                [float(record["peak_rss_bytes"]) for record in records]
            ),
        }
        arms[arm] = {
            "timing": summaries,
            "held_predictions_exact": _array_exact(
                records, "held_prediction"
            ),
            "cold_predictions_exact": _array_exact(
                records, "cold_prediction"
            ),
            "tree_counts_exact": _metadata_exact(
                records, "fitted_tree_count"
            ),
            "learning_rates_exact": _metadata_exact(
                records, "resolved_learning_rate"
            ),
            "combo_pairs_exact": _metadata_exact(records, "combo_pairs"),
            "representative_guardrail_scores": records[0][
                "guardrail_scores"
            ],
            "representative_metadata": records[0]["metadata"],
        }
    control_timing = arms[CONTROL]["timing"]
    candidate_timing = arms[CANDIDATE]["timing"]
    ratios = {
        name: float(
            candidate_timing[name]["median"]
            / control_timing[name]["median"]
        )
        for name in (
            "fit",
            "held_prediction",
            "cold_prediction",
            "peak_rss_bytes",
        )
    }
    control_scores = arms[CONTROL]["representative_guardrail_scores"]
    candidate_scores = arms[CANDIDATE]["representative_guardrail_scores"]
    score_deltas = {
        view: float(candidate_scores[view]["r2"] - control_scores[view]["r2"])
        for view in (
            "overlap_exposed_team_holdout",
            "cold_player_subset",
            "seen_player_subset",
        )
    }
    return {
        "arms": arms,
        "ratios": ratios,
        "score_deltas": score_deltas,
        "peak_rss_median_delta_bytes": float(
            candidate_timing["peak_rss_bytes"]["median"]
            - control_timing["peak_rss_bytes"]["median"]
        ),
    }


def _summarize_numeric(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    auto = results[NUMERIC_AUTO]
    off = results[NUMERIC_OFF]
    return {
        "train_predictions_exact": np.array_equal(
            np.asarray(auto["train_prediction"], dtype=np.float64),
            np.asarray(off["train_prediction"], dtype=np.float64),
        ),
        "held_predictions_exact": np.array_equal(
            np.asarray(auto["held_prediction"], dtype=np.float64),
            np.asarray(off["held_prediction"], dtype=np.float64),
        ),
        "feature_importance_exact": np.array_equal(
            np.asarray(auto["metadata"]["feature_importance"], dtype=np.float64),
            np.asarray(off["metadata"]["feature_importance"], dtype=np.float64),
        ),
        "tree_count_exact": (
            auto["metadata"]["fitted_tree_count"]
            == off["metadata"]["fitted_tree_count"]
        ),
        "auto_combo_pairs": auto["metadata"]["combo_pairs"],
        "off_combo_pairs": off["metadata"]["combo_pairs"],
        "auto_train_prediction_sha256": auto["train_prediction_sha256"],
        "off_train_prediction_sha256": off["train_prediction_sha256"],
        "auto_held_prediction_sha256": auto["held_prediction_sha256"],
        "off_held_prediction_sha256": off["held_prediction_sha256"],
    }


def _route_summary(
    fold_results: dict[str, list[dict[str, Any]]],
    full_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    control_records = fold_results[CONTROL] + full_results[CONTROL]
    candidate_records = fold_results[CANDIDATE] + full_results[CANDIDATE]
    expected_pairs = [list(pair) for pair in EXPECTED_COMBO_PAIRS]
    return {
        "control_zero_pairs": all(
            record["metadata"]["combo_pairs"] == []
            for record in control_records
        ),
        "candidate_six_expected_pairs": all(
            record["metadata"]["combo_pairs"] == expected_pairs
            for record in candidate_records
        ),
        "all_threads_18": all(
            record["metadata"]["resolved_thread_count"] == EXPECTED_THREADS
            for record in control_records + candidate_records
        ),
        "all_learning_rates_point_1": all(
            record["metadata"]["resolved_learning_rate"] == 0.1
            for record in control_records + candidate_records
        ),
        "candidate_six_combination_columns": all(
            len(record["metadata"]["combo_map_sizes"]) == 6
            and record["metadata"]["preprocessed_columns"] == 10
            and record["metadata"]["combination_code_cells"]
            == 6 * record["metadata"]["expected_core_fit_rows"]
            for record in candidate_records
        ),
    }


def _decision(
    folds: dict[str, Any],
    full: dict[str, Any],
    numeric: dict[str, Any],
    routes: dict[str, Any],
) -> dict[str, Any]:
    quality = {
        "mean_delta_at_least_0002": folds["mean_delta"] >= 0.002,
        "wins_at_least_6": folds["wins"] >= 6,
        "worst_fold_delta_at_least_minus_0010": (
            folds["worst_delta"] >= -0.010
        ),
        "held_team_delta_at_least_minus_0001": (
            full["score_deltas"]["overlap_exposed_team_holdout"] >= -0.001
        ),
        "cold_player_delta_at_least_minus_0001": (
            full["score_deltas"]["cold_player_subset"] >= -0.001
        ),
        "seen_player_delta_at_least_minus_0005": (
            full["score_deltas"]["seen_player_subset"] >= -0.005
        ),
    }
    repeat_exactness = all(
        full["arms"][arm][key]
        for arm in (CONTROL, CANDIDATE)
        for key in (
            "held_predictions_exact",
            "cold_predictions_exact",
            "tree_counts_exact",
            "learning_rates_exact",
            "combo_pairs_exact",
        )
    )
    numeric_exactness = all(
        numeric[key]
        for key in (
            "train_predictions_exact",
            "held_predictions_exact",
            "feature_importance_exact",
            "tree_count_exact",
        )
    ) and not numeric["auto_combo_pairs"] and not numeric["off_combo_pairs"]
    behavior = {
        "route_exact": all(routes.values()),
        "full_repeat_exact": repeat_exactness,
        "numeric_non_engagement_exact": numeric_exactness,
    }
    timing_stable = all(
        full["arms"][arm]["timing"][series]["iqr_fraction"]
        <= MAX_TIMING_IQR_FRACTION
        for arm in (CONTROL, CANDIDATE)
        for series in ("fit", "held_prediction", "cold_prediction")
    )
    resources = {
        "fit_ratio_at_most_150": full["ratios"]["fit"] <= 1.50,
        "held_prediction_ratio_at_most_110": (
            full["ratios"]["held_prediction"] <= 1.10
        ),
        "cold_prediction_ratio_at_most_110": (
            full["ratios"]["cold_prediction"] <= 1.10
        ),
        "timing_stable": timing_stable,
        "rss_ratio_at_most_150": (
            full["ratios"]["peak_rss_bytes"] <= 1.50
        ),
        "rss_delta_at_most_256_mib": (
            full["peak_rss_median_delta_bytes"] <= 256 * MIB
        ),
    }
    quality_passes = all(quality.values())
    behavior_passes = all(behavior.values())
    resource_passes = all(resources.values())
    passes = quality_passes and behavior_passes and resource_passes
    if passes:
        recommendation = "authorize_explicit_default_off_darkofit_port"
    elif not quality_passes:
        recommendation = "close_categorical_combinations_without_port"
    else:
        recommendation = "stop_donor_port_on_behavior_or_resource_gate"
    return {
        "quality_gates": quality,
        "behavior_gates": behavior,
        "resource_gates": resources,
        "quality_passes": quality_passes,
        "behavior_passes": behavior_passes,
        "resource_passes": resource_passes,
        "passes_all_gates": passes,
        "recommendation": recommendation,
        "default_policy_change_authorized": False,
        "darkofit_implementation_authorized": passes,
        "ctr23_or_lockbox_used": False,
    }


def _numeric_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def run(output: Path) -> dict[str, Any]:
    binding = _validate_parent_binding(output)
    output = output.resolve()
    view = _prepare_categorical_view()

    warmups = []
    for arm in (CONTROL, CANDIDATE):
        warmup = _run_fresh_worker(
            binding, mode=FOLD_MODE, arm=arm, fold=0
        )
        warmups.append(
            {
                "arm": arm,
                "process_seconds": warmup["process_seconds"],
                "prediction_sha256": warmup["prediction_sha256"],
                "fitted_tree_count": warmup["metadata"]["fitted_tree_count"],
            }
        )

    fold_results = {CONTROL: [], CANDIDATE: []}
    fold_schedule = []
    for fold in range(creator.N_SPLITS):
        order = (
            (CONTROL, CANDIDATE)
            if fold % 2 == 0
            else (CANDIDATE, CONTROL)
        )
        fold_schedule.append({"fold": fold, "order": list(order)})
        by_arm = {}
        for arm in order:
            by_arm[arm] = _run_fresh_worker(
                binding, mode=FOLD_MODE, arm=arm, fold=fold
            )
        for arm in (CONTROL, CANDIDATE):
            fold_results[arm].append(by_arm[arm])

    full_results = {CONTROL: [], CANDIDATE: []}
    full_schedule = harness.reciprocal_schedule(
        CONTROL, CANDIDATE, TIMING_BLOCKS
    )
    for order in full_schedule:
        for arm in order:
            full_results[arm].append(
                _run_fresh_worker(binding, mode=FULL_MODE, arm=arm)
            )

    numeric_results = {}
    for arm in (NUMERIC_AUTO, NUMERIC_OFF):
        numeric_results[arm] = _run_fresh_worker(
            binding, mode=NUMERIC_MODE, arm=arm
        )

    folds = _summarize_folds(fold_results)
    full = _summarize_full(full_results)
    numeric = _summarize_numeric(numeric_results)
    routes = _route_summary(fold_results, full_results)
    decision = _decision(folds, full, numeric, routes)
    _assert_source_unchanged(binding)

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "commit": PROTOCOL_COMMIT,
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "outcome_observed_before_freeze": False,
        },
        "binding": binding,
        "data": view.metadata,
        "model": {
            "common_params": COMMON_MODEL_PARAMS,
            "categorical_features": list(CATEGORICAL_COLUMNS),
            "categorical_feature_indices": list(CATEGORICAL_FEATURES),
            "control_cat_combinations": False,
            "candidate_cat_combinations": True,
            "player_identifier_used_as_model_feature": False,
            "player_identifier_used_as_internal_validation_group": True,
        },
        "execution": {
            "threads_per_worker": EXPECTED_THREADS,
            "workers_concurrent": 1,
            "fresh_worker_per_fit": True,
            "numba_cache_dir": str(NUMBA_CACHE_ROOT),
            "warmups": warmups,
            "fold_schedule": fold_schedule,
            "full_schedule": [list(order) for order in full_schedule],
            "full_timing_blocks": TIMING_BLOCKS,
            "held_prediction_calls": HELD_PREDICTION_CALLS,
            "cold_prediction_calls": COLD_PREDICTION_CALLS,
        },
        "raw": {
            "folds": fold_results,
            "full": full_results,
            "numeric_non_engagement": numeric_results,
        },
        "summary": {
            "folds": folds,
            "full": full,
            "numeric_non_engagement": numeric,
            "routes": routes,
        },
        "decision": decision,
        "source_final_prepublication": {
            "darkofit": _repository_state(REPO_ROOT),
            "chimeraboost": _repository_state(CHIMERABOOST_REPO),
        },
    }
    if not _numeric_finite(artifact):
        raise RuntimeError("artifact contains non-finite numeric values")
    _write_create_only(output, artifact)
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=(FOLD_MODE, FULL_MODE, NUMERIC_MODE),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--arm",
        choices=(CONTROL, CANDIDATE, NUMERIC_AUTO, NUMERIC_OFF),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        if args.mode is None or args.arm is None:
            raise SystemExit("--worker requires --mode and --arm")
        payload = _worker_payload(args.mode, args.arm, args.fold)
        print(
            WORKER_PREFIX
            + json.dumps(payload, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    artifact = run(args.output)
    print(
        f"{artifact['decision']['recommendation']}: "
        f"{artifact['decision']['passes_all_gates']}"
    )
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
