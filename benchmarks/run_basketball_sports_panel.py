#!/usr/bin/env python3
"""Run the frozen S4 multi-season basketball confirmation campaign."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import build_basketball_sports_panel as panel_builder  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CONTROL = "darkofit_control"
CANDIDATE = "darkofit_random_strength_0_5"
CHIMERABOOST = "chimeraboost_0_15_0"
CATBOOST = "catboost_1_2_10"
ARM_ORDER = (CONTROL, CANDIDATE, CHIMERABOOST, CATBOOST)
BLOCK_ORDERS = (
    (CONTROL, CANDIDATE, CHIMERABOOST, CATBOOST),
    (CATBOOST, CHIMERABOOST, CANDIDATE, CONTROL),
    (CANDIDATE, CONTROL, CATBOOST, CHIMERABOOST),
)

EXPECTED_THREADS = 18
EXPECTED_CHIMERABOOST_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_CATBOOST_VERSION = "1.2.10"
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "basketball_sports_panel_protocol.md"
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "basketball_sports_panel_manifest.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "basketball-sports-panel-v1" / "panel.csv"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_sports_panel_raw.json"
DEFAULT_CHIMERABOOST_REPO = REPO_ROOT.parent / "chimeraboost"
WORKER_RESULT_PREFIX = "BASKETBALL_SPORTS_PANEL_RESULT="


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_create(path: Path, value: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
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


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def load_panel(cache_path: Path, manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["processed_panel"]
    if not cache_path.is_file() or cache_path.is_symlink():
        raise RuntimeError(f"sports panel cache is unavailable: {cache_path}")
    if (
        cache_path.stat().st_size != expected["bytes"]
        or _sha256(cache_path) != expected["sha256"]
    ):
        raise RuntimeError("sports panel cache differs from its frozen manifest")
    frame = pd.read_csv(cache_path)
    expected_columns = [
        *panel_builder.IDENTITY_COLUMNS,
        *panel_builder.FEATURE_COLUMNS,
        *panel_builder.TARGET_COLUMNS,
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("sports panel columns differ from the frozen manifest")
    if len(frame) != expected["rows"]:
        raise RuntimeError("sports panel row count differs from the frozen manifest")
    identities = frame.loc[:, list(panel_builder.IDENTITY_COLUMNS)].values.tolist()
    if _json_sha256(identities) != expected["identities_sha256"]:
        raise RuntimeError("sports panel identities differ from the frozen manifest")
    numeric = frame.loc[
        :, [*panel_builder.FEATURE_COLUMNS, *panel_builder.TARGET_COLUMNS]
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("sports panel contains non-finite values")
    return frame


def _prepend_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def build_estimator(arm: str, chimeraboost_repo: Path):
    if arm == CONTROL or arm == CANDIDATE:
        from darkofit import DarkoRegressor

        return DarkoRegressor(
            random_state=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
            random_strength=0.5 if arm == CANDIDATE else 0.0,
            diagnostic_warnings="never",
        )
    if arm == CHIMERABOOST:
        _prepend_import_path(chimeraboost_repo)
        from chimeraboost import ChimeraBoostRegressor

        return ChimeraBoostRegressor(
            random_state=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
        )
    if arm == CATBOOST:
        from catboost import CatBoostRegressor

        if importlib.metadata.version("catboost") != EXPECTED_CATBOOST_VERSION:
            raise RuntimeError("CatBoost version differs from the frozen runner")
        return CatBoostRegressor(
            random_seed=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown basketball sports arm: {arm}")


def _implementation(model: Any) -> dict[str, Any]:
    package = model.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package)
    try:
        distribution_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "package": package,
        "module_version": getattr(module, "__version__", None),
        "distribution_version": distribution_version,
        "module_file": str(Path(module.__file__).resolve()),
        "estimator_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    }


def _assert_implementation_source(
    model: Any, arm: str, chimeraboost_repo: Path
) -> None:
    if arm == CATBOOST:
        return
    expected = REPO_ROOT if arm in (CONTROL, CANDIDATE) else chimeraboost_repo
    package = model.__class__.__module__.split(".", 1)[0]
    module_path = Path(importlib.import_module(package).__file__).resolve()
    if not module_path.is_relative_to(expected.resolve()):
        raise RuntimeError(f"{arm} imported outside the frozen source checkout")


def _chimera_metadata(model: Any) -> dict[str, Any]:
    core = model.model_
    trees = len(core.trees_)
    return {
        "best_iteration": int(core.best_iteration_),
        "fitted_tree_count": int(trees),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "selected_lane": (
            "linear_leaves"
            if bool(getattr(model, "linear_leaves_selected_", False))
            else "constant_leaves"
        ),
        "linear_leaves_selected": getattr(model, "linear_leaves_selected_", None),
        "cross_features_selected": getattr(model, "cross_features_selected_", None),
        "cross_pairs": getattr(model, "cross_pairs_", None),
        "selection_rounds": int(model.selection_rounds),
        "early_stopping": bool(model.early_stopping),
        "early_stopping_rounds": core.early_stopping_rounds,
        "stop_reason": (
            "iteration_limit"
            if trees == int(model.n_estimators)
            else "validation_best_prefix"
        ),
    }


def _catboost_metadata(model: Any) -> dict[str, Any]:
    params = model.get_all_params()
    reported_best = model.get_best_iteration()
    best_iteration = (
        int(model.tree_count_) if reported_best is None else int(reported_best)
    )
    return {
        "best_iteration": best_iteration,
        "fitted_tree_count": int(model.tree_count_),
        "resolved_learning_rate": float(params["learning_rate"]),
        "resolved_depth": int(params["depth"]),
        "resolved_thread_count": int(model.get_param("thread_count")),
        "selected_lane": "symmetric_tree_constant_leaves",
        "loss_function": str(params["loss_function"]),
        "bootstrap_type": str(params["bootstrap_type"]),
        "stop_reason": (
            "iteration_limit"
            if int(model.tree_count_) == int(params["iterations"])
            else "validation_best_prefix"
        ),
    }


def extract_metadata(model: Any, arm: str) -> dict[str, Any]:
    if arm in (CONTROL, CANDIDATE):
        return harness.extract_fit_metadata(model)
    if arm == CHIMERABOOST:
        return _chimera_metadata(model)
    return _catboost_metadata(model)


def _prediction_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _fit_predict(
    arm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    model = build_estimator(arm, chimeraboost_repo)
    _assert_implementation_source(model, arm, chimeraboost_repo)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(X_test),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"{arm} returned invalid sports-panel predictions")
    metadata = extract_metadata(model, arm)
    if int(metadata["resolved_thread_count"]) != EXPECTED_THREADS:
        raise RuntimeError(f"{arm} did not resolve {EXPECTED_THREADS} threads")
    if arm in (CONTROL, CANDIDATE) and metadata["fitted_tree_count"] != 1000:
        raise RuntimeError(f"{arm} changed DarkoFit's fixed default horizon")
    return {
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": _prediction_sha256(prediction),
        "fit_metadata": metadata,
    }


def _score_view(y_true: pd.Series, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(y_true, dtype=np.float64)
    if values.shape != prediction.shape or len(values) < 2:
        raise RuntimeError("sports guardrail view has an invalid shape")
    return {
        "rows": int(len(values)),
        "r2": float(r2_score(values, prediction)),
        "target_sha256": _prediction_sha256(values),
        "prediction_sha256": _prediction_sha256(prediction),
    }


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": result["arm"],
        "implementation": result["implementation"],
        "cells": [
            {
                "season": cell["season"],
                "target": cell["target"],
                "primary_mean_r2": cell["primary_mean_r2"],
                "folds": [
                    {
                        "fold": row["fold"],
                        "test_indices": row["test_indices"],
                        "r2": row["r2"],
                        "prediction_sha256": row["prediction_sha256"],
                        "fit_metadata": row["fit_metadata"],
                    }
                    for row in cell["folds"]
                ],
                "guardrail": {
                    "scores": cell["guardrail"]["scores"],
                    "prediction_sha256": cell["guardrail"]["prediction_sha256"],
                    "fit_metadata": cell["guardrail"]["fit_metadata"],
                },
            }
            for cell in result["cells"]
        ],
    }


def run_worker(
    arm: str,
    cache_path: Path,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    panel = load_panel(cache_path)
    held = frozenset(panel_builder.held_teams(panel))
    first = panel.loc[panel["year"] == panel_builder.SEASONS[0]].reset_index(drop=True)
    first = first.loc[~first["Tm"].isin(held)].reset_index(drop=True)
    first_train, first_test = next(
        KFold(n_splits=panel_builder.N_SPLITS, shuffle=False).split(first)
    )
    X_first = first.loc[:, list(panel_builder.FEATURE_COLUMNS)]
    warmup_started = time.perf_counter_ns()
    warmup = _fit_predict(
        arm,
        X_first.iloc[first_train],
        first[panel_builder.TARGET_COLUMNS[0]].iloc[first_train],
        X_first.iloc[first_test],
        chimeraboost_repo,
    )
    warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9
    del warmup
    gc.collect()

    cells = []
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    started = time.perf_counter_ns()
    for season in panel_builder.SEASONS:
        seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
        primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
        holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
        train_players = frozenset(primary["bref_id"].astype(str))
        seen_mask = holdout["bref_id"].astype(str).isin(train_players).to_numpy()
        X_primary = primary.loc[:, list(panel_builder.FEATURE_COLUMNS)]
        X_holdout = holdout.loc[:, list(panel_builder.FEATURE_COLUMNS)]
        for target in panel_builder.TARGET_COLUMNS:
            folds = []
            splitter = KFold(n_splits=panel_builder.N_SPLITS, shuffle=False)
            for fold, (train, test) in enumerate(splitter.split(primary)):
                fitted = _fit_predict(
                    arm,
                    X_primary.iloc[train],
                    primary[target].iloc[train],
                    X_primary.iloc[test],
                    chimeraboost_repo,
                )
                prediction = fitted.pop("prediction")
                total_fit_seconds += fitted["fit_seconds"]
                total_predict_seconds += fitted["predict_seconds"]
                folds.append(
                    {
                        "fold": int(fold),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "test_indices": [int(value) for value in test],
                        "r2": float(r2_score(primary[target].iloc[test], prediction)),
                        "predictions": prediction.tolist(),
                        **fitted,
                    }
                )
            guardrail_fit = _fit_predict(
                arm,
                X_primary,
                primary[target],
                X_holdout,
                chimeraboost_repo,
            )
            holdout_prediction = guardrail_fit.pop("prediction")
            total_fit_seconds += guardrail_fit["fit_seconds"]
            total_predict_seconds += guardrail_fit["predict_seconds"]
            scores = {
                "overlap_exposed_team_holdout": _score_view(
                    holdout[target], holdout_prediction
                ),
                "seen_player_subset": _score_view(
                    holdout.loc[seen_mask, target], holdout_prediction[seen_mask]
                ),
                "cold_player_subset": _score_view(
                    holdout.loc[~seen_mask, target], holdout_prediction[~seen_mask]
                ),
            }
            cells.append(
                {
                    "season": int(season),
                    "target": target,
                    "primary_rows": int(len(primary)),
                    "held_team_rows": int(len(holdout)),
                    "seen_player_rows": int(np.sum(seen_mask)),
                    "cold_player_rows": int(np.sum(~seen_mask)),
                    "primary_mean_r2": float(np.mean([row["r2"] for row in folds])),
                    "folds": folds,
                    "guardrail": {
                        "scores": scores,
                        "predictions": holdout_prediction.tolist(),
                        **guardrail_fit,
                    },
                }
            )
    wall_seconds = (time.perf_counter_ns() - started) / 1e9

    implementation_model = build_estimator(arm, chimeraboost_repo)
    result = {
        "arm": arm,
        "implementation": _implementation(implementation_model),
        "cells": cells,
        "equal_cell_mean_r2": float(
            np.mean([cell["primary_mean_r2"] for cell in cells])
        ),
        "total_fit_seconds": float(total_fit_seconds),
        "total_predict_seconds": float(total_predict_seconds),
        "steady_wall_seconds": float(wall_seconds),
        "warmup_seconds_outside_timing": float(warmup_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _source_states(chimeraboost_repo: Path) -> dict[str, Any]:
    darkofit = creator.git_state(REPO_ROOT)
    chimeraboost = creator.git_state(chimeraboost_repo)
    if not darkofit["clean"] or not chimeraboost["clean"]:
        raise RuntimeError("sports confirmation requires both repositories clean")
    if darkofit["branch"] != "main":
        raise RuntimeError("sports confirmation requires DarkoFit main")
    if darkofit["tracked_main_refs"].get("origin/main") != darkofit["head"]:
        raise RuntimeError("DarkoFit main is not published to origin")
    if chimeraboost["head"] != EXPECTED_CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost differs from frozen 0.15.0")
    for ref in ("origin/main", "upstream/main"):
        value = chimeraboost["tracked_main_refs"].get(ref)
        if value is not None and value != EXPECTED_CHIMERABOOST_HEAD:
            raise RuntimeError(f"ChimeraBoost {ref} differs from frozen 0.15.0")
    return {"darkofit": darkofit, "chimeraboost": chimeraboost}


def _assert_sources_unchanged(
    expected: dict[str, Any],
    observed: dict[str, Any],
    boundary: str,
) -> None:
    for repository in expected:
        for field in ("path", "head", "branch", "clean", "status"):
            if expected[repository][field] != observed[repository][field]:
                raise RuntimeError(f"{repository} source changed {boundary}: {field}")


def _validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError("sports confirmation requires exactly 18 threads")
    if args.output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("sports confirmation output path is not exact")
    if args.data_cache.resolve() != DEFAULT_CACHE.resolve():
        raise RuntimeError("sports confirmation cache path is not exact")
    if args.manifest.resolve() != MANIFEST_PATH.resolve():
        raise RuntimeError("sports confirmation manifest path is not exact")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["candidate_data_scored"] or manifest["comparators_scored"]:
        raise RuntimeError("sports panel manifest is not pre-score")
    if manifest["panel_spent"]:
        raise RuntimeError("sports panel manifest was already spent")
    if _sha256(PROTOCOL_PATH) != manifest["protocol"]["sha256"]:
        raise RuntimeError("sports panel protocol changed after the manifest")
    if _sha256(Path(panel_builder.__file__).resolve()) != manifest["builder"]["sha256"]:
        raise RuntimeError("sports panel builder changed after the manifest")
    if not manifest["power_analysis"]["passes"]:
        raise RuntimeError("sports panel lacks preregistered power")
    load_panel(args.data_cache, args.manifest)
    return manifest


def _worker_environment(threads: int, chimeraboost_repo: Path) -> dict[str, str]:
    environment = harness.worker_environment(threads)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(chimeraboost_repo.resolve()), str(REPO_ROOT))
    )
    environment["CHIMERABOOST_WARMUP"] = "0"
    environment["DARKOFIT_WARMUP"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--threads",
        str(args.threads),
        "--data-cache",
        str(args.data_cache),
        "--manifest",
        str(args.manifest),
        "--chimeraboost-repo",
        str(args.chimeraboost_repo),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(args.threads, args.chimeraboost_repo),
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
            f"worker {arm} failed with exit code {completed.returncode}"
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
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


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to replace sports result: {args.output}")
    manifest = _validate_frozen_inputs(args)
    source_states = _source_states(args.chimeraboost_repo)
    repeats = []
    fingerprints = {arm: set() for arm in ARM_ORDER}
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            _assert_sources_unchanged(
                source_states,
                _source_states(args.chimeraboost_repo),
                f"before block {block} position {position}",
            )
            print(
                f"[sports-panel] block={block} position={position} arm={arm}",
                flush=True,
            )
            result = _run_worker_process(args, arm)
            fingerprints[arm].add(result["behavior_fingerprint_sha256"])
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "order": list(order),
                    "arm": arm,
                    "result": result,
                }
            )
    _assert_sources_unchanged(
        source_states,
        _source_states(args.chimeraboost_repo),
        "after final worker",
    )
    changed = {
        arm: sorted(values) for arm, values in fingerprints.items() if len(values) != 1
    }
    if changed:
        raise RuntimeError(f"sports-panel behavior did not reproduce: {changed}")

    artifact = {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "panel_manifest": {
            "path": str(args.manifest.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(args.manifest),
            "processed_panel_sha256": manifest["processed_panel"]["sha256"],
            "split_manifest_sha256": manifest["split"]["split_manifest_sha256"],
            "power_pass_probability": manifest["power_analysis"]["pass_probability"],
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source": source_states,
        "execution": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threads": args.threads,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "worker_count": len(repeats),
            "candidate_or_comparator_outcomes_previously_scored": False,
        },
        "behavior_fingerprints": {
            arm: next(iter(values)) for arm, values in fingerprints.items()
        },
        "repeats": repeats,
        "panel_spent_by_this_run": True,
    }
    _atomic_create(
        args.output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-arm", choices=ARM_ORDER)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--chimeraboost-repo", type=Path, default=DEFAULT_CHIMERABOOST_REPO
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_arm is not None:
        result = run_worker(
            args.worker_arm,
            args.data_cache,
            args.chimeraboost_repo,
        )
        print(
            WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    artifact = run_parent(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "workers": len(artifact["repeats"]),
                "panel_spent": artifact["panel_spent_by_this_run"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
