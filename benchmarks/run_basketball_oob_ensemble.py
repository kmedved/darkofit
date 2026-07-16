#!/usr/bin/env python3
"""Run the frozen basketball default-versus-OOB-ensemble screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


DEFAULT_CONFIG = "default"
CANDIDATE_CONFIG = "oob_ensemble5"
CONFIG_ORDER = (DEFAULT_CONFIG, CANDIDATE_CONFIG)
N_MEMBERS = 5
MIN_FOLD_WINS = 6
MAX_TOTAL_RUNTIME_RATIO = 4.0
MAX_PREDICT_RUNTIME_RATIO = 6.0
EXPECTED_DEFAULT_MEAN_R2 = 0.5267495183883605
WORKER_RESULT_PREFIX = "BASKETBALL_OOB_ENSEMBLE_RESULT="
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_oob_ensemble.json"


def member_seeds(
    random_state: int = creator.RANDOM_STATE,
    n_members: int = N_MEMBERS,
) -> tuple[int, ...]:
    if int(n_members) < 2:
        raise ValueError("OOB ensemble requires at least two members")
    values = np.random.default_rng(int(random_state)).integers(
        0, 2**31 - 1, size=int(n_members)
    )
    return tuple(int(value) for value in values)


def bootstrap_plan(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(n_rows)
    if n_rows < 2:
        raise ValueError("OOB bootstrap requires at least two rows")
    sampled = np.random.default_rng(int(seed)).integers(
        0, n_rows, size=n_rows, dtype=np.int64
    )
    oob_mask = np.ones(n_rows, dtype=np.bool_)
    oob_mask[sampled] = False
    oob = np.flatnonzero(oob_mask).astype(np.int64, copy=False)
    if len(oob) == 0:
        raise RuntimeError("basketball bootstrap produced no OOB rows")
    return sampled, oob


def _index_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validation_metadata(model: Any) -> dict[str, Any]:
    result = dict(model.model_.auto_params_.get("validation_split", {}))
    if result.get("source") != "explicit_eval_set":
        raise RuntimeError("OOB member did not use its explicit eval set")
    return result


def _fit_single(
    X_train, y_train, X_test
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    from darkofit import DarkoRegressor

    model = DarkoRegressor(
        random_state=creator.RANDOM_STATE,
        verbose_timing=True,
    )
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(model.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return prediction, float(fit_seconds), float(predict_seconds), {
        "kind": "single",
        "fit_metadata": harness.extract_fit_metadata(model),
    }


def _fit_oob_ensemble(
    X_train, y_train, X_test
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    from darkofit import DarkoRegressor

    predictions = []
    members = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    seeds = member_seeds()
    for member_index, seed in enumerate(seeds):
        sampled, oob = bootstrap_plan(len(X_train), seed)
        model = DarkoRegressor(
            random_state=seed,
            early_stopping=True,
            early_stopping_rounds=None,
            use_best_model=True,
            refit=False,
            verbose_timing=True,
        )
        started = time.perf_counter_ns()
        model.fit(
            X_train.iloc[sampled],
            y_train.iloc[sampled],
            eval_set=(X_train.iloc[oob], y_train.iloc[oob]),
        )
        member_fit_seconds = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        prediction = harness.validate_prediction(
            model.predict(X_test), len(X_test)
        )
        member_predict_seconds = (time.perf_counter_ns() - started) / 1e9
        fit_seconds += member_fit_seconds
        predict_seconds += member_predict_seconds
        fitted = harness.extract_fit_metadata(model)
        validation = _validation_metadata(model)
        if int(validation.get("eval_n_samples", -1)) != len(oob):
            raise RuntimeError("OOB validation metadata changed its row count")
        predictions.append(prediction)
        members.append(
            {
                "member": int(member_index),
                "seed": int(seed),
                "bootstrap_rows": int(len(sampled)),
                "bootstrap_unique_rows": int(len(np.unique(sampled))),
                "bootstrap_indices_sha256": _index_sha256(sampled),
                "oob_rows": int(len(oob)),
                "oob_indices_sha256": _index_sha256(oob),
                "prediction_sha256": harness.prediction_sha256(prediction),
                "fit_seconds": float(member_fit_seconds),
                "predict_seconds": float(member_predict_seconds),
                "validation": validation,
                "fit_metadata": fitted,
            }
        )
    averaged = harness.validate_prediction(
        np.mean(np.vstack(predictions), axis=0), len(X_test)
    )
    return averaged, float(fit_seconds), float(predict_seconds), {
        "kind": "oob_ensemble",
        "member_count": len(members),
        "member_seeds": list(seeds),
        "members": members,
    }


def fit_and_predict(
    config_name: str, X_train, y_train, X_test
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    if config_name not in CONFIG_ORDER:
        raise ValueError(f"unknown basketball config: {config_name}")
    if config_name == DEFAULT_CONFIG:
        return _fit_single(X_train, y_train, X_test)
    return _fit_oob_ensemble(X_train, y_train, X_test)


def _sum_phase_times(metadata: dict[str, Any]) -> dict[str, float]:
    if metadata["kind"] == "single":
        records = [{"fit_metadata": metadata["fit_metadata"]}]
    else:
        records = [
            {"fit_metadata": member["fit_metadata"]}
            for member in metadata["members"]
        ]
    return harness.sum_phase_times(records, "final_fit")


def validate_fitted_metadata(config_name: str, metadata: dict[str, Any]) -> None:
    if config_name == DEFAULT_CONFIG:
        if metadata.get("kind") != "single":
            raise RuntimeError("default fit unexpectedly became an ensemble")
        fits = [metadata["fit_metadata"]]
    else:
        if metadata.get("kind") != "oob_ensemble":
            raise RuntimeError("candidate did not report an OOB ensemble")
        if int(metadata.get("member_count", 0)) != N_MEMBERS:
            raise RuntimeError("candidate fitted the wrong member count")
        if tuple(metadata.get("member_seeds", ())) != member_seeds():
            raise RuntimeError("candidate member seeds changed")
        fits = [member["fit_metadata"] for member in metadata["members"]]
        for member in metadata["members"]:
            if int(member["oob_rows"]) < 1:
                raise RuntimeError("candidate member has no OOB rows")
            if member["validation"].get("source") != "explicit_eval_set":
                raise RuntimeError("candidate member changed validation source")
            if int(member["validation"].get("eval_n_samples", -1)) != int(
                member["oob_rows"]
            ):
                raise RuntimeError("candidate member validation is not its OOB set")
    for fitted in fits:
        if fitted["selected_lane"] != "boosting":
            raise RuntimeError("basketball fit unexpectedly selected another lane")
        if fitted["selected_tree_mode"] != "catboost":
            raise RuntimeError("basketball fit unexpectedly changed tree mode")
        if int(fitted["resolved_thread_count"]) < 1:
            raise RuntimeError("basketball fit resolved no worker threads")
        if not math.isfinite(float(fitted["resolved_learning_rate"])):
            raise RuntimeError("basketball fit resolved an invalid learning rate")
        if int(fitted["fitted_tree_count"]) != int(fitted["best_iteration"]):
            raise RuntimeError("basketball fit retained the wrong tree count")
        if config_name == CANDIDATE_CONFIG:
            if fitted["refit"] or fitted["selection_fit"] is not None:
                raise RuntimeError("OOB member unexpectedly refit")
            if fitted["final_early_stopping_rounds"] is None:
                raise RuntimeError("OOB member did not resolve early stopping")
            if fitted["final_fit"]["stop_reason"] != "early_stopping":
                raise RuntimeError("OOB member did not stop on OOB validation")


def _warmup(config_name: str, dataset: harness.BasketballDataset) -> float:
    train_indices, test_indices = next(
        creator.creator_cv().split(dataset.X, dataset.y)
    )
    started = time.perf_counter_ns()
    fit_and_predict(
        config_name,
        dataset.X.iloc[train_indices],
        dataset.y.iloc[train_indices],
        dataset.X.iloc[test_indices],
    )
    return float((time.perf_counter_ns() - started) / 1e9)


def run_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    warmup_seconds = _warmup(config_name, dataset)
    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train_indices, test_indices) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        prediction, fit_seconds, predict_seconds, metadata = fit_and_predict(
            config_name,
            dataset.X.iloc[train_indices],
            dataset.y.iloc[train_indices],
            dataset.X.iloc[test_indices],
        )
        validate_fitted_metadata(config_name, metadata)
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "test_indices": [int(value) for value in test_indices],
                "r2": float(r2_score(dataset.y.iloc[test_indices], prediction)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_sha256": harness.prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                "fit_metadata": metadata,
                "summed_final_phase_seconds": _sum_phase_times(metadata),
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    guardrail = dataset.player_guardrail
    held_prediction, held_fit, held_predict, held_metadata = fit_and_predict(
        config_name,
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    validate_fitted_metadata(config_name, held_metadata)
    scores = np.asarray([fold["r2"] for fold in folds], dtype=np.float64)
    result = {
        "config": config_name,
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "warmup_seconds_outside_timing": warmup_seconds,
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "holdout": {
            "fit_seconds": held_fit,
            "predict_seconds": held_predict,
            "prediction_sha256": harness.prediction_sha256(held_prediction),
            "predictions": [float(value) for value in held_prediction],
            "fit_metadata": held_metadata,
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                held_prediction,
                guardrail.cold_player_mask,
            ),
        },
        "guardrail": guardrail.metadata,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(result)
    return result


def _run_worker_process(args: argparse.Namespace, config_name: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        config_name,
        "--data-cache",
        str(args.data_cache),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=harness.worker_environment(args.threads),
        check=False,
        capture_output=True,
        text=True,
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(result_lines) != 1:
        raise RuntimeError(
            f"basketball worker {config_name!r} failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(result_lines[0][len(WORKER_RESULT_PREFIX) :])
    result["worker_stdout"] = "\n".join(
        line
        for line in completed.stdout.splitlines()
        if not line.startswith(WORKER_RESULT_PREFIX)
    ).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def analyze_results(
    canonical: dict[str, dict[str, Any]],
    wall_timing: dict[str, dict[str, Any]],
    predict_timing: dict[str, dict[str, Any]],
    *,
    source_clean: bool = True,
) -> dict[str, Any]:
    default = canonical[DEFAULT_CONFIG]
    candidate = canonical[CANDIDATE_CONFIG]
    default_scores = np.asarray(default["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    if default_scores.shape != (creator.N_SPLITS,) or candidate_scores.shape != (
        creator.N_SPLITS,
    ):
        raise RuntimeError("basketball result does not contain ten folds")
    if not math.isclose(
        float(default["mean_r2"]),
        EXPECTED_DEFAULT_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("basketball default score no longer reproduces")
    deltas = candidate_scores - default_scores
    jackknife = [
        float(np.mean(np.delete(deltas, fold)))
        for fold in range(creator.N_SPLITS)
    ]
    default_guardrail = default["holdout"]["scores"]
    candidate_guardrail = candidate["holdout"]["scores"]
    team_delta = float(
        candidate_guardrail["overlap_exposed_team_holdout"]["r2"]
        - default_guardrail["overlap_exposed_team_holdout"]["r2"]
    )
    cold_delta = float(
        candidate_guardrail["cold_player_subset"]["r2"]
        - default_guardrail["cold_player_subset"]["r2"]
    )
    wall_ratio = float(
        wall_timing[CANDIDATE_CONFIG]["median_seconds"]
        / wall_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    predict_ratio = float(
        predict_timing[CANDIDATE_CONFIG]["median_seconds"]
        / predict_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    quality_gates = {
        "mean_r2_no_regression": float(np.mean(deltas)) >= 0.0,
        "fold_breadth": int(np.count_nonzero(deltas > 0.0)) >= MIN_FOLD_WINS,
        "leave_one_fold_out_no_regression": min(jackknife) >= 0.0,
        "overlap_exposed_team_no_regression": team_delta >= 0.0,
        "cold_player_no_regression": cold_delta >= 0.0,
    }
    runtime_gates = {
        "default_timing_stable": wall_timing[DEFAULT_CONFIG]["stable"],
        "candidate_timing_stable": wall_timing[CANDIDATE_CONFIG]["stable"],
        "beats_naive_fivefold_wall_scaling": (
            wall_ratio <= MAX_TOTAL_RUNTIME_RATIO
        ),
        "prediction_cost_within_budget": (
            predict_ratio <= MAX_PREDICT_RUNTIME_RATIO
        ),
    }
    passes_quality = all(quality_gates.values())
    passes_runtime = all(runtime_gates.values())
    evidence_gates = {"committed_clean_source": bool(source_clean)}
    evidence_eligible = all(evidence_gates.values())
    passes_all = passes_quality and passes_runtime and evidence_eligible
    return {
        "candidate": CANDIDATE_CONFIG,
        "candidate_scope": "opt_in_only",
        "default_promotion_authorized": False,
        "mean_r2_delta": float(np.mean(deltas)),
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "fold_losses": int(np.count_nonzero(deltas < 0.0)),
        "fold_ties": int(np.count_nonzero(deltas == 0.0)),
        "fold_r2_deltas": [float(value) for value in deltas],
        "leave_one_fold_out_mean_deltas": jackknife,
        "overlap_exposed_team_r2_delta": team_delta,
        "cold_player_r2_delta": cold_delta,
        "candidate_over_default_median_wall_time": wall_ratio,
        "candidate_over_default_median_predict_time": predict_ratio,
        "quality_gates": quality_gates,
        "runtime_gates": runtime_gates,
        "evidence_gates": evidence_gates,
        "passes_quality_gates": passes_quality,
        "passes_runtime_gates": passes_runtime,
        "evidence_eligible": evidence_eligible,
        "passes_all_gates": passes_all,
        "recommendation": (
            "advance_to_public_api_implementation"
            if passes_all
            else (
                "development_only_dirty_source"
                if not evidence_eligible
                else "advance_none"
            )
        ),
    }


def _source_state(allow_dirty: bool) -> dict[str, Any]:
    state = creator.git_state(REPO_ROOT)
    if not allow_dirty and not state["clean"]:
        raise RuntimeError("refusing to benchmark a dirty DarkoFit source tree")
    return state


def _assert_source_unchanged(expected, observed, boundary: str) -> None:
    fields = ("path", "head", "branch", "clean", "status")
    changed = [field for field in fields if expected[field] != observed[field]]
    if changed:
        raise RuntimeError(
            f"DarkoFit source changed {boundary}: " + ", ".join(changed)
        )


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    source = _source_state(args.allow_dirty_source)
    dataset = harness.load_basketball_dataset(args.data_cache)
    schedule = harness.reciprocal_schedule(DEFAULT_CONFIG, CANDIDATE_CONFIG)
    canonical: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {name: set() for name in CONFIG_ORDER}
    wall_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    predict_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    repeats = []
    for block, order in enumerate(schedule):
        for position, config_name in enumerate(order):
            _assert_source_unchanged(
                source,
                _source_state(args.allow_dirty_source),
                boundary=f"before block {block} {config_name}",
            )
            print(
                f"running block {block + 1}/{len(schedule)} "
                f"position {position + 1}: {config_name}...",
                flush=True,
            )
            result = _run_worker_process(args, config_name)
            fingerprint = result["behavior_fingerprint_sha256"]
            fingerprints[config_name].add(fingerprint)
            wall_values[config_name].append(result["steady_wall_seconds"])
            predict_values[config_name].append(result["summed_predict_seconds"])
            repeats.append(
                {
                    "block": int(block),
                    "position": int(position),
                    "config": config_name,
                    "steady_wall_seconds": result["steady_wall_seconds"],
                    "warmup_seconds_outside_timing": result[
                        "warmup_seconds_outside_timing"
                    ],
                    "summed_fit_seconds": result["summed_fit_seconds"],
                    "summed_predict_seconds": result["summed_predict_seconds"],
                    "holdout_fit_seconds": result["holdout"]["fit_seconds"],
                    "holdout_predict_seconds": result["holdout"][
                        "predict_seconds"
                    ],
                    "behavior_fingerprint_sha256": fingerprint,
                    "worker_stdout": result["worker_stdout"],
                    "worker_stderr": result["worker_stderr"],
                }
            )
            canonical.setdefault(config_name, result)
            cold = result["holdout"]["scores"]["cold_player_subset"]["r2"]
            print(
                f"  mean R2={result['mean_r2']:.12f}, cold={cold:.12f}, "
                f"steady={result['steady_wall_seconds']:.2f}s",
                flush=True,
            )
    for config_name, values in fingerprints.items():
        if len(values) != 1:
            raise RuntimeError(f"{config_name} behavior changed across repeats")
    _assert_source_unchanged(
        source,
        _source_state(args.allow_dirty_source),
        boundary="during the experiment",
    )
    wall_timing = {
        name: harness.timing_summary(values)
        for name, values in wall_values.items()
    }
    predict_timing = {
        name: harness.timing_summary(values)
        for name, values in predict_values.items()
    }
    decision = analyze_results(
        canonical,
        wall_timing,
        predict_timing,
        source_clean=bool(source["clean"]),
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_oob_ensemble5_screen",
            "diagnostic_only": True,
            "candidate_scope": "opt_in_only",
            "default_promotion_authorized": False,
            "creator_benchmark_changed": False,
            "n_members": N_MEMBERS,
            "member_params": {
                "early_stopping": True,
                "early_stopping_rounds": None,
                "use_best_model": True,
                "refit": False,
            },
            "timing_schedule": [list(order) for order in schedule],
            "timing_repetitions_per_config": len(schedule),
            "maximum_timing_spread_ratio": harness.MAX_TIMING_SPREAD_RATIO,
            "maximum_total_runtime_ratio": MAX_TOTAL_RUNTIME_RATIO,
            "maximum_predict_runtime_ratio": MAX_PREDICT_RUNTIME_RATIO,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "random_state": creator.RANDOM_STATE,
            "threads_per_fit": args.threads,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
            "warmup": "one complete first-fold fit and prediction per worker",
            "weights_used": False,
            "lockbox_data_used": False,
        },
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "source": source,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": [canonical[name] for name in CONFIG_ORDER],
        "timing_repeats": repeats,
        "wall_timing_summary": wall_timing,
        "predict_timing_summary": predict_timing,
        "decision": decision,
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        result = run_worker(args.worker_config, args.data_cache)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
