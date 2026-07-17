#!/usr/bin/env python3
"""Run the frozen DarkoFit/ChimeraBoost 0.15 basketball characterization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_basketball_fused_oblivious as engine_base  # noqa: E402


DARKOFIT_ARM = "darkofit"
CHIMERABOOST_ARM = "chimeraboost"
ARM_ORDER = (DARKOFIT_ARM, CHIMERABOOST_ARM)
PRODUCT_LANE = "product_defaults"
MATCHED_LANE = "matched_engine"
LANE_ORDER = (PRODUCT_LANE, MATCHED_LANE)
EXPECTED_THREADS = 18
EXPECTED_CHIMERABOOST_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_PROTOCOL_SHA256 = (
    "0c21ede5e90cb64d394fa8abc29448bd56ce7bbf04133f94228995f0583d33f7"
)
EXPECTED_REPOSITORY_MANIFEST_SHA256 = "0db9e0a3292c3312e6c0ea2d307e8cb75bc0ac53b21c6d637f5f9cc85276a456"
PROTOCOL_PATH = REPO_ROOT / "benchmarks/basketball_chimera_v015_protocol.md"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/basketball_chimera_v015.json"
DEFAULT_CHIMERABOOST_REPO = REPO_ROOT.parent / "chimeraboost"
WORKER_RESULT_PREFIX = "BASKETBALL_CHIMERA_V015_RESULT="
MAX_ENGINE_RATIO = 1.10
MAX_RSS_RATIO = 1.10
MAX_PRODUCT_MEAN_R2_GAP = 0.002
MAX_PRODUCT_GUARDRAIL_R2_GAP = 0.01


def _prepend_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def build_estimator(lane: str, arm: str, chimeraboost_repo: Path):
    if lane not in LANE_ORDER or arm not in ARM_ORDER:
        raise ValueError(f"unknown basketball configuration: {lane}/{arm}")
    if arm == DARKOFIT_ARM:
        from darkofit import DarkoRegressor

        if lane == PRODUCT_LANE:
            return DarkoRegressor(
                random_state=creator.RANDOM_STATE,
                thread_count=EXPECTED_THREADS,
                diagnostic_warnings="never",
            )
        return DarkoRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=1.0,
            max_bins=128,
            subsample=1.0,
            colsample=1.0,
            min_child_weight=1.0,
            min_child_samples=1,
            ordered_boosting=False,
            early_stopping=False,
            tree_mode="catboost",
            linear_leaves=False,
            thread_count=EXPECTED_THREADS,
            random_state=creator.RANDOM_STATE,
            diagnostic_warnings="never",
        )

    _prepend_import_path(chimeraboost_repo)
    from chimeraboost import ChimeraBoostRegressor

    if lane == PRODUCT_LANE:
        return ChimeraBoostRegressor(
            random_state=creator.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
        )
    return ChimeraBoostRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        subsample=1.0,
        colsample=1.0,
        min_child_weight=1.0,
        ordered_boosting=False,
        early_stopping=False,
        linear_leaves=False,
        cross_features=False,
        cat_combinations=False,
        thread_count=EXPECTED_THREADS,
        random_state=creator.RANDOM_STATE,
    )


def _implementation(model) -> dict[str, Any]:
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


def _assert_implementation_source(model, arm: str, chimeraboost_repo: Path) -> None:
    expected = REPO_ROOT if arm == DARKOFIT_ARM else chimeraboost_repo
    module = importlib.import_module(model.__class__.__module__.split(".", 1)[0])
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(expected.resolve()):
        raise RuntimeError(
            f"{arm} imported from {module_path}, outside {expected.resolve()}"
        )


def _chimera_metadata(model) -> dict[str, Any]:
    core = model.model_
    tree_count = len(core.trees_)
    if bool(getattr(model, "linear_leaves_selected_", False)):
        selected_lane = "linear_leaves"
    else:
        selected_lane = "constant_leaves"
    return {
        "best_iteration": int(core.best_iteration_),
        "fitted_tree_count": int(tree_count),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "selected_lane": selected_lane,
        "linear_leaves_selected": getattr(model, "linear_leaves_selected_", None),
        "cross_features_selected": getattr(model, "cross_features_selected_", None),
        "cross_pairs": getattr(model, "cross_pairs_", None),
        "selection_rounds": model.selection_rounds,
        "early_stopping": bool(model.early_stopping),
        "early_stopping_rounds": core.early_stopping_rounds,
        "stop_reason": (
            "iteration_limit" if tree_count == int(model.n_estimators)
            else "validation_best_prefix"
        ),
    }


def extract_metadata(model, arm: str) -> dict[str, Any]:
    if arm == DARKOFIT_ARM:
        return harness.extract_fit_metadata(model)
    return _chimera_metadata(model)


def _fit_one(lane: str, arm: str, X_train, y_train, X_test, chimeraboost_repo):
    model = build_estimator(lane, arm, chimeraboost_repo)
    _assert_implementation_source(model, arm, chimeraboost_repo)
    fit_started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - fit_started) / 1e9
    predict_started = time.perf_counter_ns()
    prediction = harness.validate_prediction(model.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - predict_started) / 1e9
    metadata = extract_metadata(model, arm)
    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    if importance.shape != (X_train.shape[1],) or not np.all(np.isfinite(importance)):
        raise RuntimeError(f"{arm} produced invalid feature importances")
    return {
        "model": model,
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": harness.prediction_sha256(prediction),
        "feature_importance_sha256": harness.prediction_sha256(importance),
        "fit_metadata": metadata,
    }


def _validate_fit(lane: str, arm: str, fitted: dict[str, Any]) -> None:
    metadata = fitted["fit_metadata"]
    if int(metadata["resolved_thread_count"]) != EXPECTED_THREADS:
        raise RuntimeError(f"{lane}/{arm} did not resolve 18 threads")
    trees = int(metadata["fitted_tree_count"])
    if lane == MATCHED_LANE and trees != 1000:
        raise RuntimeError(f"{lane}/{arm} retained {trees}, expected 1000 trees")
    if lane == PRODUCT_LANE and arm == DARKOFIT_ARM and trees != 1000:
        raise RuntimeError(
            f"{lane}/{arm} retained {trees}, expected fixed 1000 trees"
        )
    if (
        lane == PRODUCT_LANE
        and arm == CHIMERABOOST_ARM
        and not 1 <= trees <= 2000
    ):
        raise RuntimeError(f"{lane}/{arm} retained invalid tree count {trees}")


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    fold_fields = (
        "fold",
        "test_indices",
        "r2",
        "prediction_sha256",
        "feature_importance_sha256",
        "fit_metadata",
    )
    holdout_fields = (
        "scores",
        "prediction_sha256",
        "feature_importance_sha256",
        "fit_metadata",
    )
    return {
        "lane": result["lane"],
        "arm": result["arm"],
        "mean_r2": result["mean_r2"],
        "folds": [
            {field: row[field] for field in fold_fields} for row in result["folds"]
        ],
        "holdout": {
            field: result["holdout"][field] for field in holdout_fields
        },
    }


def run_worker(
    lane: str,
    arm: str,
    cache_path: Path,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    first_train, first_test = next(creator.creator_cv().split(dataset.X, dataset.y))
    warmup_started = time.perf_counter_ns()
    warmup = _fit_one(
        lane,
        arm,
        dataset.X.iloc[first_train],
        dataset.y.iloc[first_train],
        dataset.X.iloc[first_test],
        chimeraboost_repo,
    )
    _validate_fit(lane, arm, warmup)
    warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9
    del warmup
    gc.collect()

    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        fitted = _fit_one(
            lane,
            arm,
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            dataset.X.iloc[test],
            chimeraboost_repo,
        )
        _validate_fit(lane, arm, fitted)
        model = fitted.pop("model")
        prediction = fitted.pop("prediction")
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_indices": [int(value) for value in test],
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "predictions": [float(value) for value in prediction],
                **fitted,
            }
        )
        del model
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    guardrail = dataset.player_guardrail
    holdout = _fit_one(
        lane,
        arm,
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
        chimeraboost_repo,
    )
    _validate_fit(lane, arm, holdout)
    holdout.pop("model")
    holdout_prediction = holdout.pop("prediction")
    holdout.update(
        {
            "predictions": [float(value) for value in holdout_prediction],
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                holdout_prediction,
                guardrail.cold_player_mask,
            ),
        }
    )
    fold_scores = [float(row["r2"]) for row in folds]
    result = {
        "lane": lane,
        "arm": arm,
        "implementation": _implementation(
            build_estimator(lane, arm, chimeraboost_repo)
        ),
        "estimator_params": creator._jsonable(
            build_estimator(lane, arm, chimeraboost_repo).get_params(deep=False)
        ),
        "mean_r2": float(np.mean(fold_scores)),
        "std_r2": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "folds": folds,
        "holdout": holdout,
        "guardrail": guardrail.metadata,
        "steady_wall_seconds": float(steady_seconds),
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "warmup_seconds_outside_timing": float(warmup_seconds),
        "peak_rss_bytes": engine_base._peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _source_states(args: argparse.Namespace) -> dict[str, Any]:
    darkofit = creator.git_state(REPO_ROOT)
    chimeraboost = creator.git_state(args.chimeraboost_repo)
    if not darkofit["clean"] or not chimeraboost["clean"]:
        raise RuntimeError("basketball comparison requires both clean repositories")
    if chimeraboost["head"] != EXPECTED_CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost head changed from frozen 0.15.0")
    for ref in ("origin/main", "upstream/main"):
        value = chimeraboost["tracked_main_refs"].get(ref)
        if value is not None and value != EXPECTED_CHIMERABOOST_HEAD:
            raise RuntimeError(f"ChimeraBoost {ref} differs from frozen head")
    return {"darkofit": darkofit, "chimeraboost": chimeraboost}


def _assert_sources_unchanged(expected, observed, boundary: str) -> None:
    fields = ("path", "head", "branch", "clean", "status")
    for repository in expected:
        changed = [
            field
            for field in fields
            if expected[repository][field] != observed[repository][field]
        ]
        if changed:
            raise RuntimeError(
                f"{repository} source changed {boundary}: {', '.join(changed)}"
            )


def _repository_manifest_sha256() -> str:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("could not enumerate tracked repository files")
    paths = [
        Path(value.decode("utf-8"))
        for value in completed.stdout.split(b"\0")
        if value
    ]
    runner_path = Path(__file__).resolve().relative_to(REPO_ROOT)
    expected = (
        "EXPECTED_REPOSITORY_MANIFEST_SHA256 = "
        f'"{EXPECTED_REPOSITORY_MANIFEST_SHA256}"'
    ).encode("utf-8")
    normalized = b'EXPECTED_REPOSITORY_MANIFEST_SHA256 = "<FROZEN>"'
    digest = hashlib.sha256()
    for relative in paths:
        payload = (REPO_ROOT / relative).read_bytes()
        if relative == runner_path:
            if payload.count(expected) != 1:
                raise RuntimeError("could not normalize repository manifest")
            payload = payload.replace(expected, normalized, 1)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _validate_frozen_execution(args: argparse.Namespace) -> None:
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError("basketball comparison requires exactly 18 threads")
    if args.output != DEFAULT_OUTPUT:
        raise RuntimeError("basketball comparison output path is not exact")
    if hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() != (
        EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("basketball comparison protocol changed")
    if _repository_manifest_sha256() != EXPECTED_REPOSITORY_MANIFEST_SHA256:
        raise RuntimeError("basketball comparison repository manifest changed")


def _worker_environment(args: argparse.Namespace, arm: str) -> dict[str, str]:
    environment = harness.worker_environment(args.threads)
    paths = [str(REPO_ROOT)]
    if arm == CHIMERABOOST_ARM:
        paths.insert(0, str(args.chimeraboost_repo))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["CHIMERABOOST_WARMUP"] = "0"
    return environment


def _run_worker_process(args: argparse.Namespace, lane: str, arm: str):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-lane",
        lane,
        "--worker-arm",
        arm,
        "--threads",
        str(args.threads),
        "--data-cache",
        str(args.data_cache),
        "--chimeraboost-repo",
        str(args.chimeraboost_repo),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(args, arm),
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
            f"worker {lane}/{arm} failed with exit code {completed.returncode}"
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


def _quality_analysis(product: dict[str, dict[str, Any]]) -> dict[str, Any]:
    darko = product[DARKOFIT_ARM]
    chimera = product[CHIMERABOOST_ARM]
    mean_gap = float(darko["mean_r2"] - chimera["mean_r2"])
    fold_gaps = [
        float(left - right)
        for left, right in zip(darko["fold_scores"], chimera["fold_scores"])
    ]
    guardrail_gaps = {}
    for name in ("overlap_exposed_team_holdout", "seen_player_subset", "cold_player_subset"):
        guardrail_gaps[name] = float(
            darko["holdout"]["scores"][name]["r2"]
            - chimera["holdout"]["scores"][name]["r2"]
        )
    gates = {
        "mean_within_descriptive_band": abs(mean_gap) <= MAX_PRODUCT_MEAN_R2_GAP,
        "held_team_within_descriptive_band": abs(
            guardrail_gaps["overlap_exposed_team_holdout"]
        )
        <= MAX_PRODUCT_GUARDRAIL_R2_GAP,
        "cold_player_within_descriptive_band": abs(
            guardrail_gaps["cold_player_subset"]
        )
        <= MAX_PRODUCT_GUARDRAIL_R2_GAP,
    }
    return {
        "darkofit_minus_chimeraboost_mean_r2": mean_gap,
        "darkofit_minus_chimeraboost_fold_r2": fold_gaps,
        "darkofit_fold_wins": sum(gap > 0.0 for gap in fold_gaps),
        "ties": sum(gap == 0.0 for gap in fold_gaps),
        "chimeraboost_fold_wins": sum(gap < 0.0 for gap in fold_gaps),
        "guardrail_r2_gaps": guardrail_gaps,
        "descriptive_quality_gates": gates,
        "broadly_comparable": all(gates.values()),
    }


def _matched_exactness(matched: dict[str, dict[str, Any]]) -> dict[str, Any]:
    darko = matched[DARKOFIT_ARM]
    chimera = matched[CHIMERABOOST_ARM]
    fold_exact = len(darko["folds"]) == len(chimera["folds"]) and all(
        left["test_indices"] == right["test_indices"]
        and left["r2"] == right["r2"]
        and left["prediction_sha256"] == right["prediction_sha256"]
        for left, right in zip(darko["folds"], chimera["folds"])
    )
    holdout_exact = (
        darko["holdout"]["scores"] == chimera["holdout"]["scores"]
        and darko["holdout"]["prediction_sha256"]
        == chimera["holdout"]["prediction_sha256"]
    )
    tree_counts_exact = all(
        int(row["fit_metadata"]["fitted_tree_count"]) == 1000
        for arm in (darko, chimera)
        for row in [*arm["folds"], arm["holdout"]]
    )
    gates = {
        "mean_r2_exact": darko["mean_r2"] == chimera["mean_r2"],
        "fold_predictions_exact": fold_exact,
        "guardrail_predictions_exact": holdout_exact,
        "tree_counts_exact": tree_counts_exact,
    }
    return {"matched_exactness_gates": gates, "passes_exactness": all(gates.values())}


def _timing_analysis(values, rss_values, matched_exactness) -> dict[str, Any]:
    summaries = {}
    for lane in LANE_ORDER:
        summaries[lane] = {}
        for metric in ("wall", "fit", "predict"):
            summaries[lane][metric] = {
                arm: harness.timing_summary(values[lane][metric][arm])
                for arm in ARM_ORDER
            }
        darko_wall = summaries[lane]["wall"][DARKOFIT_ARM]["median_seconds"]
        chimera_wall = summaries[lane]["wall"][CHIMERABOOST_ARM]["median_seconds"]
        darko_fit = summaries[lane]["fit"][DARKOFIT_ARM]["median_seconds"]
        chimera_fit = summaries[lane]["fit"][CHIMERABOOST_ARM]["median_seconds"]
        darko_predict = summaries[lane]["predict"][DARKOFIT_ARM]["median_seconds"]
        chimera_predict = summaries[lane]["predict"][CHIMERABOOST_ARM]["median_seconds"]
        rss_ratio = float(
            statistics.median(rss_values[lane][DARKOFIT_ARM])
            / statistics.median(rss_values[lane][CHIMERABOOST_ARM])
        )
        summaries[lane]["ratios"] = {
            "darkofit_over_chimeraboost_wall": float(darko_wall / chimera_wall),
            "darkofit_over_chimeraboost_fit": float(darko_fit / chimera_fit),
            "darkofit_over_chimeraboost_predict": float(
                darko_predict / chimera_predict
            ),
            "darkofit_over_chimeraboost_peak_rss": rss_ratio,
        }
    matched = summaries[MATCHED_LANE]
    gates = {
        "darkofit_timing_stable": matched["wall"][DARKOFIT_ARM]["stable"],
        "chimeraboost_timing_stable": matched["wall"][CHIMERABOOST_ARM]["stable"],
        "fit_engine_parity": matched["ratios"][
            "darkofit_over_chimeraboost_fit"
        ]
        <= MAX_ENGINE_RATIO,
        "wall_engine_parity": matched["ratios"][
            "darkofit_over_chimeraboost_wall"
        ]
        <= MAX_ENGINE_RATIO,
        "rss_within_budget": matched["ratios"][
            "darkofit_over_chimeraboost_peak_rss"
        ]
        <= MAX_RSS_RATIO,
        "matched_predictions_exact": matched_exactness["passes_exactness"],
    }
    return {
        "lane_summaries": summaries,
        "engine_parity_gates": gates,
        "passes_engine_parity": all(gates.values()),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink output: {args.output}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    _validate_frozen_execution(args)
    sources = _source_states(args)
    dataset = harness.load_basketball_dataset(args.data_cache)
    schedule = harness.reciprocal_schedule(DARKOFIT_ARM, CHIMERABOOST_ARM)
    canonical = {lane: {} for lane in LANE_ORDER}
    fingerprints = {
        lane: {arm: set() for arm in ARM_ORDER} for lane in LANE_ORDER
    }
    values = {
        lane: {
            metric: {arm: [] for arm in ARM_ORDER}
            for metric in ("wall", "fit", "predict")
        }
        for lane in LANE_ORDER
    }
    rss_values = {
        lane: {arm: [] for arm in ARM_ORDER} for lane in LANE_ORDER
    }
    repeats = []
    for lane in LANE_ORDER:
        for block, order in enumerate(schedule):
            for position, arm in enumerate(order):
                _assert_sources_unchanged(
                    sources,
                    _source_states(args),
                    f"before {lane} block {block} {arm}",
                )
                print(
                    f"running {lane} block {block + 1}/{len(schedule)} "
                    f"position {position + 1}: {arm}...",
                    flush=True,
                )
                result = _run_worker_process(args, lane, arm)
                fingerprints[lane][arm].add(result["behavior_fingerprint_sha256"])
                values[lane]["wall"][arm].append(result["steady_wall_seconds"])
                values[lane]["fit"][arm].append(result["summed_fit_seconds"])
                values[lane]["predict"][arm].append(
                    result["summed_predict_seconds"]
                )
                rss_values[lane][arm].append(result["peak_rss_bytes"])
                repeats.append(
                    {
                        "lane": lane,
                        "block": int(block),
                        "position": int(position),
                        "arm": arm,
                        "steady_wall_seconds": result["steady_wall_seconds"],
                        "summed_fit_seconds": result["summed_fit_seconds"],
                        "summed_predict_seconds": result["summed_predict_seconds"],
                        "warmup_seconds_outside_timing": result[
                            "warmup_seconds_outside_timing"
                        ],
                        "peak_rss_bytes": result["peak_rss_bytes"],
                        "behavior_fingerprint_sha256": result[
                            "behavior_fingerprint_sha256"
                        ],
                        "worker_stdout": result["worker_stdout"],
                        "worker_stderr": result["worker_stderr"],
                    }
                )
                canonical[lane].setdefault(arm, result)
                print(
                    f"  mean R2={result['mean_r2']:.12f}, "
                    f"steady={result['steady_wall_seconds']:.2f}s",
                    flush=True,
                )
                _assert_sources_unchanged(
                    sources,
                    _source_states(args),
                    f"during {lane} block {block} {arm}",
                )

    repeat_behavior_exact = all(
        len(fingerprints[lane][arm]) == 1
        for lane in LANE_ORDER
        for arm in ARM_ORDER
    )
    product_quality = _quality_analysis(canonical[PRODUCT_LANE])
    matched_exactness = _matched_exactness(canonical[MATCHED_LANE])
    timing = _timing_analysis(values, rss_values, matched_exactness)
    passes_engine_parity = timing["passes_engine_parity"] and repeat_behavior_exact
    recommendation = (
        "stop_low_level_default_tree_optimization"
        if passes_engine_parity
        else "profile_matched_engine_gap"
    )
    decision = {
        "kind": "characterization_only",
        "default_change_authorized": False,
        "repeat_behavior_exact": repeat_behavior_exact,
        "product_quality": product_quality,
        "matched_exactness": matched_exactness,
        "timing": timing,
        "passes_engine_parity": passes_engine_parity,
        "recommendation": recommendation,
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_darkofit_vs_chimeraboost_0_15_0",
            "threads": args.threads,
            "random_state": creator.RANDOM_STATE,
            "lanes": list(LANE_ORDER),
            "arms": list(ARM_ORDER),
            "timing_schedule": [list(order) for order in schedule],
            "warmup": "one complete first-fold fit and prediction per worker",
            "guardrail_outside_steady_timer": True,
            "weights_used": False,
            "lockbox_data_used": False,
            "matched_engine_ratio_limit": MAX_ENGINE_RATIO,
            "maximum_timing_spread_ratio": harness.MAX_TIMING_SPREAD_RATIO,
            "maximum_rss_ratio": MAX_RSS_RATIO,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
        },
        "sources": sources,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": [
            canonical[lane][arm] for lane in LANE_ORDER for arm in ARM_ORDER
        ],
        "timing_repeats": repeats,
        "timing_values": values,
        "peak_rss_values": rss_values,
        "decision": decision,
    }
    engine_base._atomic_write_new_bytes(
        args.output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(f"decision: {recommendation}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument(
        "--chimeraboost-repo", type=Path, default=DEFAULT_CHIMERABOOST_REPO
    )
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--worker-lane", choices=LANE_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--worker-arm", choices=ARM_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    args.chimeraboost_repo = args.chimeraboost_repo.expanduser().resolve()
    if args.threads != EXPECTED_THREADS:
        parser.error("--threads must be exactly 18")
    if not args.worker_arm and args.output != DEFAULT_OUTPUT:
        parser.error("--output must be the frozen artifact path")
    if bool(args.worker_lane) != bool(args.worker_arm):
        parser.error("worker lane and arm must be supplied together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_arm:
        result = run_worker(
            args.worker_lane,
            args.worker_arm,
            args.data_cache,
            args.chimeraboost_repo,
        )
        print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
