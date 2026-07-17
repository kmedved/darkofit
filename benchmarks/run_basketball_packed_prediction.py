#!/usr/bin/env python3
"""Confirm DarkoFit's forest-work-aware packed predictor on basketball."""

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
from darkofit.flat_model import (  # noqa: E402
    FlatObliviousEnsemble,
    _PARALLEL_MIN_ROWS,
    _flat_oblivious_add,
    _flat_oblivious_add_parallel,
)


EXPECTED_THREADS = 18
CONFIRMATION_FOLD = 1
EXPECTED_TREES = 1000
EXPECTED_CUTOFF_ROWS = 132
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_DARKOFIT_PACKAGE_MANIFEST = (
    "6e80c24202ef503d43f6655ea66e866d7cb52ff670df8054fbf962483b8e9846"
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
    "c9acc371eb734fd818899474961a18358eda9f1e9c5090dd396e77fd89320152"
)
PROTOCOL_PATH = ROOT / "benchmarks/basketball_packed_prediction_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks/basketball_packed_prediction.json"
DEFAULT_CHIMERA_REPO = ROOT.parent / "chimeraboost"
TIMING_REPEATS = 11
MAX_IQR_FRACTION = 0.30

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


def _prediction_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
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
    paths = sorted(
        item.decode("utf-8") for item in raw.split(b"\0") if item
    )
    digest = hashlib.sha256()
    for relative in paths:
        name = relative.encode("utf-8")
        content = (repo / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _support_hashes() -> dict[str, str]:
    return {
        relative: _sha256_file(ROOT / relative)
        for relative in EXPECTED_SUPPORT_SHA256
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _source_state(repo: Path) -> dict[str, Any]:
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
            raise RuntimeError(f"{name} must be on main, got {state['branch']!r}")
    package_manifest = _tracked_content_manifest(ROOT, "darkofit")
    if package_manifest != EXPECTED_DARKOFIT_PACKAGE_MANIFEST:
        raise RuntimeError(
            "DarkoFit package manifest is "
            f"{package_manifest}, expected {EXPECTED_DARKOFIT_PACKAGE_MANIFEST}"
        )
    support_hashes = _support_hashes()
    if support_hashes != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError(
            f"basketball support hashes changed: {support_hashes}"
        )
    if states["chimeraboost"]["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError(
            "unexpected ChimeraBoost head: "
            f"{states['chimeraboost']['head']}"
        )
    for remote_ref in ("origin/main", "upstream/main"):
        try:
            remote_head = _git(chimera_repo, "rev-parse", remote_ref)
        except subprocess.CalledProcessError:
            continue
        if remote_head != EXPECTED_CHIMERA_HEAD:
            raise RuntimeError(
                f"ChimeraBoost {remote_ref} is {remote_head}, expected "
                f"{EXPECTED_CHIMERA_HEAD}"
            )
    states["darkofit"]["package_manifest_sha256"] = package_manifest
    states["darkofit"]["support_sha256"] = support_hashes
    return states


def _timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    iqr = float(np.subtract(*np.percentile(array, [75, 25])))
    return {
        "seconds": [float(value) for value in array],
        "median_seconds": median,
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median if median > 0.0 else float("inf"),
    }


def _time_alternating(
    functions: dict[str, Callable[[], np.ndarray]],
    *,
    inner_calls: int,
) -> dict[str, dict[str, Any]]:
    values = {name: [] for name in functions}
    names = list(functions)
    orders = list(itertools.permutations(names))
    for repeat in range(TIMING_REPEATS):
        order = orders[repeat % len(orders)]
        for name in order:
            gc.disable()
            started = time.perf_counter_ns()
            try:
                for _ in range(inner_calls):
                    functions[name]()
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e9
                gc.enable()
            values[name].append(elapsed / inner_calls)
    return {name: _timing_summary(series) for name, series in values.items()}


def _repeat_rows(frame, rows: int):
    indices = np.arange(rows, dtype=np.int64) % len(frame)
    return frame.iloc[indices].reset_index(drop=True)


def _packed_nbytes(flat: FlatObliviousEnsemble) -> int:
    return int(
        flat.depths.nbytes
        + flat.feats.nbytes
        + flat.thrs.nbytes
        + flat.value_offsets.nbytes
        + flat.values.nbytes
    )


def _candidate_parallel_min_rows(n_trees: int) -> int:
    """Reproduce the rejected candidate cutoff for the pinned campaign."""
    if n_trees <= 0:
        return _PARALLEL_MIN_ROWS
    return min(8192, max(128, (131_072 + n_trees - 1) // n_trees))


def _candidate_output(flat, X_binned, initial: float) -> np.ndarray:
    output = np.full(X_binned.shape[0], initial, dtype=np.float64)
    flat.add_predict(X_binned, output)
    return output


def _legacy_output(flat, X_binned, initial: float) -> np.ndarray:
    output = np.full(X_binned.shape[0], initial, dtype=np.float64)
    arguments = (
        X_binned,
        flat.depths,
        flat.feats,
        flat.thrs,
        flat.value_offsets,
        flat.values,
        output,
    )
    if numba.get_num_threads() > 1 and X_binned.shape[0] >= _PARALLEL_MIN_ROWS:
        _flat_oblivious_add_parallel(*arguments)
    else:
        _flat_oblivious_add(*arguments)
    return output


def _chimera_output(core, X_binned, predict_forest) -> np.ndarray:
    feats, thresholds, depths, values, offsets = core._forest_
    return predict_forest(
        X_binned,
        feats,
        thresholds,
        depths,
        values,
        offsets,
        core.init_,
    )


def _prepend_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _assert_model_source(model, repository: Path) -> None:
    package = model.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package)
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(repository.resolve()):
        raise RuntimeError(
            f"{package} imported from {module_path}, outside {repository}"
        )


def _model_metadata(model, arm: str) -> dict[str, Any]:
    core = model.model_
    trees = list(core.trees_)
    depths = np.asarray([tree.depth for tree in trees], dtype=np.int64)
    params = model.get_params(deep=False)
    if arm == "darkofit":
        constant_leaves = all(
            tree.linear_coefficients is None for tree in trees
        )
        resolved_tree_mode = str(core.tree_mode_)
        resolved_ordered = bool(core.ordered_boosting_)
        min_child_samples = int(core.min_child_samples)
        cross_features = False
        cat_combinations = False
    else:
        constant_leaves = all(tree.lin_coef is None for tree in trees)
        resolved_tree_mode = "oblivious"
        resolved_ordered = bool(core.ordered_boosting)
        min_child_samples = None
        cross_features = bool(model.cross_features)
        cat_combinations = bool(model.cat_combinations)
    return {
        "arm": arm,
        "requested_params": params,
        "requested_params_sha256": _params_sha256(params),
        "fitted_tree_count": int(len(trees)),
        "tree_depths_unique": sorted(int(value) for value in set(depths)),
        "tree_depths_sha256": _sha256_bytes(
            np.ascontiguousarray(depths, dtype="<i8").tobytes()
        ),
        "all_constant_leaves": bool(constant_leaves),
        "resolved": {
            "best_iteration": int(core.best_iteration_),
            "learning_rate": float(core.lr_),
            "depth": int(core.depth),
            "l2_leaf_reg": float(core.l2_leaf_reg),
            "max_bins": int(core.max_bins),
            "subsample": float(core.subsample),
            "colsample": float(core.colsample),
            "min_child_weight": float(core.min_child_weight),
            "min_child_samples": min_child_samples,
            "ordered_boosting": resolved_ordered,
            "tree_mode": resolved_tree_mode,
            "thread_count": int(core.n_threads_),
            "random_state": int(core.random_state),
            "early_stopping": bool(model.early_stopping),
            "linear_leaves": bool(model.linear_leaves),
            "cross_features": cross_features,
            "cat_combinations": cat_combinations,
        },
    }


def _metadata_is_frozen(metadata: dict[str, Any]) -> bool:
    arm = metadata["arm"]
    expected_params = (
        EXPECTED_DARKOFIT_PARAMS_SHA256
        if arm == "darkofit"
        else EXPECTED_CHIMERA_PARAMS_SHA256
    )
    expected_resolved = {
        "best_iteration": EXPECTED_TREES,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "subsample": 1.0,
        "colsample": 1.0,
        "min_child_weight": 1.0,
        "min_child_samples": 1 if arm == "darkofit" else None,
        "ordered_boosting": False,
        "tree_mode": "catboost" if arm == "darkofit" else "oblivious",
        "thread_count": EXPECTED_THREADS,
        "random_state": 4,
        "early_stopping": False,
        "linear_leaves": False,
        "cross_features": False,
        "cat_combinations": False,
    }
    return bool(
        metadata["requested_params_sha256"] == expected_params
        and metadata["fitted_tree_count"] == EXPECTED_TREES
        and metadata["tree_depths_unique"] == [6]
        and metadata["all_constant_leaves"]
        and metadata["resolved"] == expected_resolved
    )


def _observe_candidate_route(flat, X_binned, initial: float):
    import darkofit.flat_model as flat_module

    serial_kernel = flat_module._flat_oblivious_add
    parallel_kernel = flat_module._flat_oblivious_add_parallel
    calls: list[str] = []

    def observed_serial(*arguments):
        calls.append("serial")
        return serial_kernel(*arguments)

    def observed_parallel(*arguments):
        calls.append("parallel")
        return parallel_kernel(*arguments)

    flat_module._flat_oblivious_add = observed_serial
    flat_module._flat_oblivious_add_parallel = observed_parallel
    try:
        output = _candidate_output(flat, X_binned, initial)
    finally:
        flat_module._flat_oblivious_add = serial_kernel
        flat_module._flat_oblivious_add_parallel = parallel_kernel
    if len(calls) != 1:
        raise RuntimeError(f"candidate dispatch invoked kernels {calls!r}")
    return calls[0], output


def _fit_models(dataset, chimera_repo: Path) -> tuple[Any, Any, dict[str, Any]]:
    train_indices, test_indices = list(
        creator.creator_cv().split(dataset.X, dataset.y)
    )[CONFIRMATION_FOLD]
    X_train = dataset.X.iloc[train_indices]
    y_train = dataset.y.iloc[train_indices]
    if creator.RANDOM_STATE != 4:
        raise RuntimeError("basketball creator random state changed")
    from darkofit import DarkoRegressor

    _prepend_import_path(chimera_repo)
    from chimeraboost import ChimeraBoostRegressor

    fitted = {}
    fit_seconds = {}
    specifications = {
        "darkofit": (DarkoRegressor, DARKOFIT_PARAMS, ROOT),
        "chimeraboost": (
            ChimeraBoostRegressor,
            CHIMERABOOST_PARAMS,
            chimera_repo,
        ),
    }
    for arm, (estimator, params, repository) in specifications.items():
        model = estimator(**params)
        _assert_model_source(model, repository)
        expected_params_hash = (
            EXPECTED_DARKOFIT_PARAMS_SHA256
            if arm == "darkofit"
            else EXPECTED_CHIMERA_PARAMS_SHA256
        )
        actual_params_hash = _params_sha256(model.get_params(deep=False))
        if actual_params_hash != expected_params_hash:
            raise RuntimeError(
                f"{arm} requested parameters changed: {actual_params_hash}"
            )
        started = time.perf_counter_ns()
        model.fit(X_train, y_train)
        fit_seconds[arm] = (time.perf_counter_ns() - started) / 1e9
        fitted[arm] = model
    return (
        fitted["darkofit"],
        fitted["chimeraboost"],
        {
            "train_indices": np.asarray(train_indices, dtype=np.int64),
            "test_indices": np.asarray(test_indices, dtype=np.int64),
            "fit_seconds": fit_seconds,
        },
    )


def _case_result(
    name: str,
    X,
    darko_model,
    chimera_model,
    *,
    inner_calls: int,
) -> dict[str, Any]:
    darko_core = darko_model.model_
    chimera_core = chimera_model.model_
    flat = darko_core._flat_ensemble()
    X_darko = darko_core.prep_.transform(darko_core._prepare_predict_X(X))
    X_chimera = chimera_core.prep_.transform(
        np.asarray(X, dtype=np.float64)
    )
    if not np.array_equal(X_darko, X_chimera):
        raise RuntimeError(f"{name}: binned matrices differ")

    from chimeraboost.tree import _predict_forest_rm as chimera_predict_forest

    observed_route, observed_output = _observe_candidate_route(
        flat, X_darko, darko_core.init_
    )

    core_functions = {
        "darkofit_candidate": lambda: _candidate_output(
            flat, X_darko, darko_core.init_
        ),
        "darkofit_legacy": lambda: _legacy_output(
            flat, X_darko, darko_core.init_
        ),
        "chimeraboost": lambda: _chimera_output(
            chimera_core, X_chimera, chimera_predict_forest
        ),
    }
    public_functions = {
        "darkofit": lambda: np.asarray(
            darko_model.predict(X), dtype=np.float64
        ),
        "chimeraboost": lambda: np.asarray(
            chimera_model.predict(X), dtype=np.float64
        ),
    }

    # Compile/load all measured paths and materialize both lazy packed caches.
    core_outputs = {key: function() for key, function in core_functions.items()}
    public_outputs = {
        key: function() for key, function in public_functions.items()
    }
    reference = core_outputs["chimeraboost"]
    exactness = {
        "observed_candidate_dispatch": bool(
            np.array_equal(observed_output, reference)
        ),
        **{
            f"core_{key}": bool(np.array_equal(value, reference))
            for key, value in core_outputs.items()
        },
        **{
            f"public_{key}": bool(np.array_equal(value, reference))
            for key, value in public_outputs.items()
        },
    }
    if not all(exactness.values()):
        raise RuntimeError(f"{name}: prediction exactness failed")

    core_timing = _time_alternating(core_functions, inner_calls=inner_calls)
    public_timing = _time_alternating(public_functions, inner_calls=inner_calls)
    candidate_median = core_timing["darkofit_candidate"]["median_seconds"]
    legacy_median = core_timing["darkofit_legacy"]["median_seconds"]
    chimera_median = core_timing["chimeraboost"]["median_seconds"]
    darko_public_median = public_timing["darkofit"]["median_seconds"]
    chimera_public_median = public_timing["chimeraboost"]["median_seconds"]
    cutoff = _candidate_parallel_min_rows(flat.depths.shape[0])
    return {
        "name": name,
        "rows": int(len(X)),
        "columns": int(X.shape[1]),
        "inner_calls": int(inner_calls),
        "candidate_route": observed_route,
        "legacy_route": (
            "parallel"
            if numba.get_num_threads() > 1 and len(X) >= _PARALLEL_MIN_ROWS
            else "serial"
        ),
        "candidate_parallel_cutoff_rows": int(cutoff),
        "binned_sha256": _sha256_bytes(
            np.ascontiguousarray(X_darko).tobytes()
        ),
        "prediction_sha256": _prediction_sha256(reference),
        "exactness": exactness,
        "core_timing": core_timing,
        "public_timing": public_timing,
        "ratios": {
            "legacy_over_candidate_speedup": legacy_median / candidate_median,
            "candidate_over_chimeraboost": candidate_median / chimera_median,
            "candidate_over_legacy": candidate_median / legacy_median,
            "darkofit_public_over_chimeraboost": (
                darko_public_median / chimera_public_median
            ),
        },
        "output": {
            "shape": [int(value) for value in reference.shape],
            "dtype": str(reference.dtype),
            "nbytes": int(reference.nbytes),
        },
    }


def _analyze(cases: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> dict:
    real_names = ("confirmation_fold", "cold_player")
    large_names = ("repeated_8192", "repeated_100000")
    timing_names = (*real_names, "held_team", *large_names)

    exact = all(
        all(case["exactness"].values()) for case in cases.values()
    )
    routes = {
        "tiny_stays_serial": cases["tiny_127"]["candidate_route"] == "serial",
        "real_batches_parallel": all(
            cases[name]["candidate_route"] == "parallel"
            for name in ("confirmation_fold", "cold_player", "held_team")
        ),
        "large_batches_parallel": all(
            cases[name]["candidate_route"] == "parallel" for name in large_names
        ),
    }
    timings_stable = all(
        summary["iqr_fraction"] <= MAX_IQR_FRACTION
        for name in timing_names
        for group in ("core_timing", "public_timing")
        for summary in cases[name][group].values()
    )
    gates = {
        "predictions_array_exact": exact,
        "frozen_fit_metadata": bool(metadata["frozen_fit_metadata"]),
        **routes,
        "real_core_speedup_at_least_2x": all(
            cases[name]["ratios"]["legacy_over_candidate_speedup"] >= 2.0
            for name in real_names
        ),
        "real_core_at_chimera_parity": all(
            cases[name]["ratios"]["candidate_over_chimeraboost"] <= 1.15
            for name in real_names
        ),
        "real_public_at_chimera_parity": all(
            cases[name]["ratios"][
                "darkofit_public_over_chimeraboost"
            ] <= 1.20
            for name in real_names
        ),
        "large_no_legacy_regression": all(
            cases[name]["ratios"]["candidate_over_legacy"] <= 1.10
            for name in large_names
        ),
        "large_at_chimera_parity": all(
            cases[name]["ratios"]["candidate_over_chimeraboost"] <= 1.20
            for name in large_names
        ),
        "timing_stability": timings_stable,
        "same_packed_storage": bool(metadata["same_packed_storage"]),
        "same_output_contract": all(
            case["output"]["dtype"] == "float64"
            and case["output"]["shape"] == [case["rows"]]
            and case["output"]["nbytes"] == case["rows"] * 8
            for case in cases.values()
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "passed": passed,
        "recommendation": (
            "promote_constant_leaf_oblivious_work_router"
            if passed
            else "reject_work_router_without_threshold_retuning"
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
        raise ValueError(f"this protocol requires exactly {EXPECTED_THREADS} threads")
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol hash is {protocol_hash}, expected {EXPECTED_PROTOCOL_SHA256}"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source_before = _require_clean_sources(args.chimeraboost_repo)
    numba.set_num_threads(args.threads)
    if numba.get_num_threads() != EXPECTED_THREADS:
        raise RuntimeError("Numba did not resolve the frozen thread count")

    dataset = harness.load_basketball_dataset(args.cache_path)
    darko_model, chimera_model, split = _fit_models(
        dataset, args.chimeraboost_repo
    )
    darko_core = darko_model.model_
    chimera_core = chimera_model.model_
    flat = darko_core._flat_ensemble()
    if not isinstance(flat, FlatObliviousEnsemble):
        raise RuntimeError(f"unexpected DarkoFit packed type {type(flat).__name__}")

    # Build the comparator's lazy packed forest before measured calls.
    fold_X = dataset.X.iloc[split["test_indices"]]
    chimera_model.predict(fold_X)
    cutoff = _candidate_parallel_min_rows(len(darko_core.trees_))
    if cutoff != EXPECTED_CUTOFF_ROWS:
        raise RuntimeError(f"candidate cutoff is {cutoff}, expected 132")

    darko_metadata = _model_metadata(darko_model, "darkofit")
    chimera_metadata = _model_metadata(chimera_model, "chimeraboost")
    frozen_fit_metadata = all(
        _metadata_is_frozen(item)
        for item in (darko_metadata, chimera_metadata)
    )

    guardrail = dataset.player_guardrail
    cold_X = guardrail.X_holdout.iloc[
        np.flatnonzero(guardrail.cold_player_mask)
    ]
    case_inputs = {
        "tiny_127": (_repeat_rows(fold_X, 127), 40),
        "confirmation_fold": (fold_X, 20),
        "cold_player": (cold_X, 20),
        "held_team": (guardrail.X_holdout, 8),
        "repeated_8192": (_repeat_rows(fold_X, 8192), 2),
        "repeated_100000": (_repeat_rows(fold_X, 100_000), 1),
    }
    cases = {
        name: _case_result(
            name,
            X,
            darko_model,
            chimera_model,
            inner_calls=inner,
        )
        for name, (X, inner) in case_inputs.items()
    }

    packed_bytes_before = _packed_nbytes(flat)
    packed_bytes_after = _packed_nbytes(darko_core._flat_ensemble())
    metadata = {
        "darkofit": darko_metadata,
        "chimeraboost": chimera_metadata,
        "fit_seconds": split["fit_seconds"],
        "frozen_fit_metadata": frozen_fit_metadata,
        "candidate_parallel_cutoff_rows": int(cutoff),
        "legacy_parallel_cutoff_rows": int(_PARALLEL_MIN_ROWS),
        "packed_nbytes_before": packed_bytes_before,
        "packed_nbytes_after": packed_bytes_after,
        "same_packed_storage": bool(
            darko_core._flat_ensemble() is flat
            and packed_bytes_before == packed_bytes_after
        ),
    }
    analysis = _analyze(cases, metadata)
    source_after = _require_clean_sources(args.chimeraboost_repo)
    if source_after != source_before:
        raise RuntimeError("source state changed during the campaign")

    payload = {
        "schema_version": 1,
        "campaign": "basketball_packed_prediction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": protocol_hash,
            "confirmation_fold": CONFIRMATION_FOLD,
            "exploratory_fold": 0,
        },
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "source": source_before,
        "darkofit_package_tree": _git(ROOT, "rev-parse", "HEAD:darkofit"),
        "darkofit_package_manifest_sha256": (
            EXPECTED_DARKOFIT_PACKAGE_MANIFEST
        ),
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
            "confirmation_train_indices_sha256": _sha256_bytes(
                np.ascontiguousarray(split["train_indices"], dtype="<i8").tobytes()
            ),
            "confirmation_test_indices_sha256": _sha256_bytes(
                np.ascontiguousarray(split["test_indices"], dtype="<i8").tobytes()
            ),
            "guardrail": guardrail.metadata,
        },
        "metadata": metadata,
        "cases": cases,
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
