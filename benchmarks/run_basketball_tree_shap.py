#!/usr/bin/env python3
"""Run the frozen basketball exact-TreeSHAP confirmation campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


EXPECTED_THREADS = 18
CONFIRMATION_FOLD = 1
EXPECTED_TREES = 1000
EXPLAINED_ROWS = 8
BACKGROUND_ROWS = 32
TIMING_BLOCKS = 11
TIMING_INNER_CALLS = 5
MAX_IQR_FRACTION = 0.30
MAX_RUNTIME_RATIO = 1.50
ATTRIBUTION_ATOL = 1e-12
EFFICIENCY_ATOL = 1e-9
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_DARKOFIT_PACKAGE_MANIFEST = (
    "4fe4830c9c36de36ce29e626743c83fa0f20e60db8f15e2c7ccd5c96d3226068"
)
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
}
EXPECTED_DARKOFIT_PARAMS_SHA256 = (
    "4410482d0e7c74b8dca299733850f98880111648dbd85d00338da3efc58c6089"
)
EXPECTED_CHIMERA_PARAMS_SHA256 = (
    "181539d6aa97ae810beaaa4a667353713a4b4e728efe24c769d90cb4420bab32"
)
EXPECTED_PROTOCOL_SHA256 = (
    "8462fa78107a23a4c184f17866898bbd3d06e7b405d34a4010fec3a9cb5ffc12"
)
PROTOCOL_PATH = ROOT / "benchmarks/basketball_tree_shap_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks/basketball_tree_shap.json"
DEFAULT_CHIMERA_REPO = ROOT.parent / "chimeraboost"

DARKOFIT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "subsample": 1.0,
    "colsample": 1.0,
    "min_child_weight": 1.0,
    "min_child_samples": 1,
    "ordered_boosting": False,
    "early_stopping": False,
    "tree_mode": "catboost",
    "linear_leaves": False,
    "thread_count": EXPECTED_THREADS,
    "random_state": 4,
    "diagnostic_warnings": "never",
}
CHIMERABOOST_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "subsample": 1.0,
    "colsample": 1.0,
    "min_child_weight": 1.0,
    "ordered_boosting": False,
    "early_stopping": False,
    "linear_leaves": False,
    "cross_features": False,
    "cat_combinations": False,
    "thread_count": EXPECTED_THREADS,
    "random_state": 4,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _array_sha256(value: Any, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return _sha256_bytes(array.tobytes())


def _params_sha256(params: dict[str, Any]) -> str:
    payload = json.dumps(
        params, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tracked_content_manifest(repo: Path, prefix: str) -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", prefix]
    )
    paths = sorted(item.decode() for item in raw.split(b"\0") if item)
    digest = hashlib.sha256()
    for relative in paths:
        name = relative.encode()
        content = (repo / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _source_state(repo: Path) -> dict[str, str]:
    return {
        "repository": str(repo.resolve()),
        "head": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "status_porcelain": _git(
            repo, "status", "--porcelain", "--untracked-files=all"
        ),
    }


def _require_clean_sources(chimera_repo: Path) -> dict[str, Any]:
    states = {
        "darkofit": _source_state(ROOT),
        "chimeraboost": _source_state(chimera_repo),
    }
    for name, state in states.items():
        if state["status_porcelain"]:
            raise RuntimeError(f"{name} source is not clean")
        if state["branch"] != "main":
            raise RuntimeError(f"{name} must be on main")
    if _git(ROOT, "rev-parse", "origin/main") != states["darkofit"]["head"]:
        raise RuntimeError("DarkoFit main must be pushed before the formal run")
    package_manifest = _tracked_content_manifest(ROOT, "darkofit")
    if package_manifest != EXPECTED_DARKOFIT_PACKAGE_MANIFEST:
        raise RuntimeError(
            f"DarkoFit package manifest changed: {package_manifest}"
        )
    support = {
        name: _sha256_file(ROOT / name) for name in EXPECTED_SUPPORT_SHA256
    }
    if support != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError(f"basketball support files changed: {support}")
    chimera_head = states["chimeraboost"]["head"]
    if chimera_head != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError(f"unexpected ChimeraBoost head: {chimera_head}")
    for remote in ("origin/main", "upstream/main"):
        try:
            remote_head = _git(chimera_repo, "rev-parse", remote)
        except subprocess.CalledProcessError:
            continue
        if remote_head != EXPECTED_CHIMERA_HEAD:
            raise RuntimeError(f"ChimeraBoost {remote} is not pinned")
    states["darkofit"]["package_manifest_sha256"] = package_manifest
    states["darkofit"]["support_sha256"] = support
    return states


def _prepend_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _assert_model_source(model: Any, repository: Path) -> None:
    package = model.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package)
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(repository.resolve()):
        raise RuntimeError(f"{package} imported from unexpected path {source}")


def _model_metadata(model: Any, arm: str) -> dict[str, Any]:
    core = model.model_
    trees = list(core.trees_)
    depths = np.asarray([tree.depth for tree in trees], dtype=np.int64)
    constant = all(
        (
            tree.linear_coefficients is None
            if arm == "darkofit"
            else tree.lin_coef is None
        )
        for tree in trees
    )
    return {
        "arm": arm,
        "requested_params": model.get_params(deep=False),
        "requested_params_sha256": _params_sha256(
            model.get_params(deep=False)
        ),
        "fitted_tree_count": int(len(trees)),
        "tree_depths_unique": sorted(int(value) for value in set(depths)),
        "tree_depths_sha256": _array_sha256(depths, "<i8"),
        "all_constant_leaves": bool(constant),
        "resolved_learning_rate": float(core.lr_),
        "resolved_thread_count": int(core.n_threads_),
    }


def _metadata_is_frozen(metadata: dict[str, Any]) -> bool:
    expected_hash = (
        EXPECTED_DARKOFIT_PARAMS_SHA256
        if metadata["arm"] == "darkofit"
        else EXPECTED_CHIMERA_PARAMS_SHA256
    )
    return bool(
        metadata["requested_params_sha256"] == expected_hash
        and metadata["fitted_tree_count"] == EXPECTED_TREES
        and metadata["tree_depths_unique"] == [6]
        and metadata["all_constant_leaves"]
        and metadata["resolved_learning_rate"] == 0.1
        and metadata["resolved_thread_count"] == EXPECTED_THREADS
    )


def _fit_models(dataset: harness.BasketballDataset, chimera_repo: Path):
    train, test = list(
        creator.creator_cv().split(dataset.X, dataset.y)
    )[CONFIRMATION_FOLD]
    X_train = dataset.X.iloc[train]
    y_train = dataset.y.iloc[train]
    from darkofit import DarkoRegressor

    _prepend_import_path(chimera_repo)
    from chimeraboost import ChimeraBoostRegressor

    specifications = {
        "darkofit": (DarkoRegressor, DARKOFIT_PARAMS, ROOT),
        "chimeraboost": (
            ChimeraBoostRegressor,
            CHIMERABOOST_PARAMS,
            chimera_repo,
        ),
    }
    models = {}
    fit_seconds = {}
    metadata = {}
    for arm, (estimator, params, repository) in specifications.items():
        model = estimator(**params)
        _assert_model_source(model, repository)
        expected_hash = (
            EXPECTED_DARKOFIT_PARAMS_SHA256
            if arm == "darkofit"
            else EXPECTED_CHIMERA_PARAMS_SHA256
        )
        if _params_sha256(model.get_params(deep=False)) != expected_hash:
            raise RuntimeError(f"{arm} requested parameters changed")
        started = time.perf_counter_ns()
        model.fit(X_train, y_train)
        fit_seconds[arm] = (time.perf_counter_ns() - started) / 1e9
        metadata[arm] = _model_metadata(model, arm)
        if not _metadata_is_frozen(metadata[arm]):
            raise RuntimeError(f"{arm} fitted metadata is not frozen")
        models[arm] = model
    return models, metadata, fit_seconds, np.asarray(train), np.asarray(test)


def _timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    iqr = float(np.subtract(*np.percentile(array, [75, 25])))
    return {
        "seconds_per_call": [float(value) for value in array],
        "median_seconds": median,
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median,
    }


def _time_reciprocal(
    functions: dict[str, Callable[[], np.ndarray]],
) -> dict[str, dict[str, Any]]:
    values = {name: [] for name in functions}
    orders = list(itertools.permutations(functions))
    for block in range(TIMING_BLOCKS):
        for name in orders[block % len(orders)]:
            was_enabled = gc.isenabled()
            gc.disable()
            started = time.perf_counter_ns()
            try:
                for _ in range(TIMING_INNER_CALLS):
                    functions[name]()
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e9
                if was_enabled:
                    gc.enable()
            values[name].append(elapsed / TIMING_INNER_CALLS)
    return {name: _timing_summary(series) for name, series in values.items()}


def _efficiency_error(
    prediction: np.ndarray, contributions: np.ndarray, expected_value: float
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(contributions).sum(axis=1)
                + expected_value
                - np.asarray(prediction)
            )
        )
    )


def _explanation_case(
    name: str,
    X,
    background,
    models: dict[str, Any],
) -> dict[str, Any]:
    outputs = {}
    for arm, model in models.items():
        contributions = model.shap_values(
            X, X_background=background
        )
        outputs[arm] = {
            "contributions": contributions,
            "expected_value": float(model.expected_value_),
            "prediction": np.asarray(model.predict(X), dtype=np.float64),
            "background_prediction": np.asarray(
                model.predict(background), dtype=np.float64
            ),
        }
    darko = outputs["darkofit"]
    chimera = outputs["chimeraboost"]
    repeated = models["darkofit"].shap_values(
        X, X_background=background
    )
    difference = np.abs(
        darko["contributions"] - chimera["contributions"]
    )
    result = {
        "name": name,
        "rows": int(len(X)),
        "background_rows": int(len(background)),
        "columns": int(X.shape[1]),
        "input_sha256": _array_sha256(X),
        "background_sha256": _array_sha256(background),
        "attribution_sha256": {
            arm: _array_sha256(output["contributions"])
            for arm, output in outputs.items()
        },
        "prediction_sha256": {
            arm: _array_sha256(output["prediction"])
            for arm, output in outputs.items()
        },
        "expected_value": {
            arm: output["expected_value"] for arm, output in outputs.items()
        },
        "efficiency_error": {
            arm: _efficiency_error(
                output["prediction"],
                output["contributions"],
                output["expected_value"],
            )
            for arm, output in outputs.items()
        },
        "background_baseline_error": {
            arm: abs(
                output["expected_value"]
                - float(output["background_prediction"].mean())
            )
            for arm, output in outputs.items()
        },
        "predictions_array_exact": bool(
            np.array_equal(darko["prediction"], chimera["prediction"])
        ),
        "attributions_close": bool(
            np.allclose(
                darko["contributions"],
                chimera["contributions"],
                rtol=0.0,
                atol=ATTRIBUTION_ATOL,
            )
        ),
        "maximum_attribution_difference": float(difference.max()),
        "expected_value_difference": abs(
            darko["expected_value"] - chimera["expected_value"]
        ),
        "darkofit_repeat_array_exact": bool(
            np.array_equal(darko["contributions"], repeated)
        ),
    }
    return result


def _full_prediction_exactness(dataset, test, models) -> dict[str, Any]:
    guardrail = dataset.player_guardrail
    cases = {
        "confirmation_fold": dataset.X.iloc[test],
        "team_holdout": guardrail.X_holdout,
        "cold_player": guardrail.X_holdout.iloc[
            np.flatnonzero(guardrail.cold_player_mask)
        ],
    }
    result = {}
    for name, X in cases.items():
        predictions = {
            arm: np.asarray(model.predict(X), dtype=np.float64)
            for arm, model in models.items()
        }
        result[name] = {
            "rows": int(len(X)),
            "array_exact": bool(
                np.array_equal(
                    predictions["darkofit"], predictions["chimeraboost"]
                )
            ),
            "sha256": {
                arm: _array_sha256(value)
                for arm, value in predictions.items()
            },
        }
    return result


def _analyze(
    cases: dict[str, dict[str, Any]],
    timing: dict[str, dict[str, Any]],
    prediction_exactness: dict[str, Any],
    default_background: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    darko_time = timing["darkofit"]["median_seconds"]
    chimera_time = timing["chimeraboost"]["median_seconds"]
    gates = {
        "basketball_is_primary_gate": True,
        "frozen_model_metadata": all(
            _metadata_is_frozen(item) for item in metadata.values()
        ),
        "fold_and_guardrail_predictions_array_exact": all(
            item["array_exact"] for item in prediction_exactness.values()
        ),
        "fold_and_cold_attributions_close": all(
            item["attributions_close"] for item in cases.values()
        ),
        "expected_values_close": all(
            item["expected_value_difference"] <= ATTRIBUTION_ATOL
            for item in cases.values()
        ),
        "darkofit_efficiency": all(
            item["efficiency_error"]["darkofit"] <= EFFICIENCY_ATOL
            and item["background_baseline_error"]["darkofit"]
            <= EFFICIENCY_ATOL
            for item in cases.values()
        ),
        "darkofit_repeat_array_exact": all(
            item["darkofit_repeat_array_exact"] for item in cases.values()
        ),
        "runtime_at_chimera_parity": (
            darko_time / chimera_time <= MAX_RUNTIME_RATIO
        ),
        "timing_stability": all(
            item["iqr_fraction"] <= MAX_IQR_FRACTION
            for item in timing.values()
        ),
        "default_background_efficiency": (
            default_background["efficiency_error"] <= EFFICIENCY_ATOL
        ),
        "default_background_storage_bounded": bool(
            default_background["storage_bounded"]
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "runtime_ratio_darkofit_over_chimeraboost": darko_time / chimera_time,
        "passed": passed,
        "recommendation": (
            "promote_exact_tree_shap_api"
            if passed
            else "reject_tree_shap_without_gate_changes"
        ),
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def run(args) -> dict[str, Any]:
    if args.threads != EXPECTED_THREADS:
        raise ValueError(f"this protocol requires {EXPECTED_THREADS} threads")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol hash changed: {protocol_hash}")
    source_before = _require_clean_sources(args.chimeraboost_repo)
    numba.set_num_threads(args.threads)
    if numba.get_num_threads() != EXPECTED_THREADS:
        raise RuntimeError("Numba did not resolve the frozen thread count")

    dataset = harness.load_basketball_dataset(args.cache_path)
    models, metadata, fit_seconds, train, test = _fit_models(
        dataset, args.chimeraboost_repo
    )
    fold_X = dataset.X.iloc[test]
    background = dataset.X.iloc[train[:BACKGROUND_ROWS]]
    cold_rows = np.flatnonzero(dataset.player_guardrail.cold_player_mask)
    cold_X = dataset.player_guardrail.X_holdout.iloc[
        cold_rows[:EXPLAINED_ROWS]
    ]
    explained_fold = fold_X.iloc[:EXPLAINED_ROWS]

    # Compile both kernels outside the formal timer with the frozen small call.
    for model in models.values():
        model.shap_values(
            explained_fold.iloc[:1], X_background=background.iloc[:2]
        )

    cases = {
        "confirmation_fold": _explanation_case(
            "confirmation_fold", explained_fold, background, models
        ),
        "cold_player": _explanation_case(
            "cold_player", cold_X, background, models
        ),
    }
    timing_functions = {
        arm: (
            lambda model=model: model.shap_values(
                explained_fold, X_background=background
            )
        )
        for arm, model in models.items()
    }
    timing = _time_reciprocal(timing_functions)
    prediction_exactness = _full_prediction_exactness(
        dataset, test, models
    )

    darko_core = models["darkofit"].model_
    stored = np.asarray(darko_core._shap_background_)
    default_phi, default_base = darko_core.shap_values(
        explained_fold, max_background=BACKGROUND_ROWS, random_state=0
    )
    default_prediction = darko_core.predict_raw(explained_fold)
    default_background = {
        "stored_shape": [int(value) for value in stored.shape],
        "stored_dtype": str(stored.dtype),
        "stored_nbytes": int(stored.nbytes),
        "maximum_allowed_values": int(
            200 * darko_core.prep_.feature_map_.shape[0]
        ),
        "storage_bounded": bool(
            stored.ndim == 2
            and stored.shape[0] <= 200
            and stored.shape[1] == darko_core.prep_.feature_map_.shape[0]
        ),
        "attribution_sha256": _array_sha256(default_phi),
        "expected_value": float(default_base),
        "efficiency_error": _efficiency_error(
            default_prediction, default_phi, default_base
        ),
    }
    analysis = _analyze(
        cases,
        timing,
        prediction_exactness,
        default_background,
        metadata,
    )
    source_after = _require_clean_sources(args.chimeraboost_repo)
    if source_after != source_before:
        raise RuntimeError("source state changed during the campaign")

    payload = {
        "schema_version": 1,
        "campaign": "basketball_tree_shap",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": protocol_hash,
            "primary_dataset": "basketball",
            "confirmation_fold": CONFIRMATION_FOLD,
            "explained_rows": EXPLAINED_ROWS,
            "background_rows": BACKGROUND_ROWS,
            "cold_player_guardrail": True,
            "ctr23_used": False,
        },
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "source": source_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "threads": numba.get_num_threads(),
        },
        "data": {
            "raw": dataset.raw_metadata,
            "processed": dataset.processed_metadata,
            "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
            "fold_test_sizes": dataset.fold_test_sizes,
            "confirmation_train_indices_sha256": _array_sha256(train, "<i8"),
            "confirmation_test_indices_sha256": _array_sha256(test, "<i8"),
            "guardrail": dataset.player_guardrail.metadata,
            "cold_request_rows_sha256": _array_sha256(cold_rows[:8], "<i8"),
        },
        "fit_seconds": fit_seconds,
        "metadata": metadata,
        "prediction_exactness": prediction_exactness,
        "cases": cases,
        "timing": {
            "blocks": TIMING_BLOCKS,
            "inner_calls": TIMING_INNER_CALLS,
            "arms": timing,
        },
        "default_background": default_background,
        "analysis": analysis,
    }
    _write_create_only(args.output, payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument(
        "--chimeraboost-repo", type=Path, default=DEFAULT_CHIMERA_REPO
    )
    parser.add_argument(
        "--cache-path", type=Path, default=harness.DEFAULT_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
