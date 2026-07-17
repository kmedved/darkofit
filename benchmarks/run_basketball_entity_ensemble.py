#!/usr/bin/env python3
"""Run the preregistered basketball entity-aware ensemble screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_campaign_harness as campaign  # noqa: E402
from benchmarks import basketball_guardrails as guardrails  # noqa: E402
from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CONTROL = "control"
CANDIDATE = "entity_ensemble5"
CONFIGS = (CONTROL, CANDIDATE)
BLOCK_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)
N_MEMBERS = 5
MIN_MEAN_R2_GAIN = 0.004
MAX_WALL_RATIO = 3.0
MAX_RSS_RATIO = 3.0
EXPECTED_CONTROL_MEAN_R2 = 0.5267495183883605
PROTOCOL = ROOT / "benchmarks" / "basketball_entity_ensemble_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "basketball_entity_ensemble.json"
WORKER_RESULT_PREFIX = "BASKBALL_ENTITY_ENSEMBLE_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value, dtype="<i8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _strings_sha256(values) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def member_seeds(
    random_state: int = creator.RANDOM_STATE,
    n_members: int = N_MEMBERS,
) -> tuple[int, ...]:
    if int(n_members) < 2:
        raise ValueError("entity ensemble requires at least two members")
    values = np.random.default_rng(int(random_state)).integers(
        0, 2**31 - 1, size=int(n_members)
    )
    return tuple(int(value) for value in values)


def group_bootstrap_plan(
    groups, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("group bootstrap requires a one-dimensional group vector")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("group bootstrap requires at least two unique groups")
    sampled = np.random.default_rng(int(seed)).choice(
        unique, size=len(unique), replace=True
    )
    sampled_rows = np.concatenate(
        [np.flatnonzero(groups == group) for group in sampled]
    ).astype(np.int64, copy=False)
    selected = set(sampled.tolist())
    oob_rows = np.flatnonzero(
        np.asarray([group not in selected for group in groups], dtype=np.bool_)
    ).astype(np.int64, copy=False)
    if len(oob_rows) == 0:
        raise RuntimeError("entity bootstrap produced no group-disjoint OOB rows")
    if selected.intersection(set(groups[oob_rows].tolist())):
        raise RuntimeError("entity bootstrap leaked a selected group into OOB")
    return sampled_rows, oob_rows, tuple(str(value) for value in sampled)


def _player_series(dataset: harness.BasketballDataset):
    players = dataset.frame.loc[
        dataset.X.index, guardrails.PLAYER_COLUMN
    ].astype(str)
    if len(players) != len(dataset.X) or players.isna().any():
        raise RuntimeError("player metadata is not aligned to creator rows")
    return players.reset_index(drop=True)


def _fit_control(X_train, y_train, X_test):
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


def _shared_numeric_preprocessor(X_train, y_train, seed: int):
    from darkofit.booster import GradientBoosting

    X_array = np.asarray(X_train, dtype=np.float64)
    y_array = np.asarray(y_train, dtype=np.float64)
    core = GradientBoosting(loss="RMSE", random_state=int(seed))
    prep = core._new_preprocessor()
    binned = prep.fit_transform(X_array, [y_array], cat_features=None)
    if prep.cat_features_ or len(prep.num_features_) != X_array.shape[1]:
        raise RuntimeError("entity ensemble shared preprocessor is not numeric-only")
    state = {
        "train_rows": int(len(X_array)),
        "feature_count": int(X_array.shape[1]),
        "binned_dtype": str(binned.dtype),
        "binned_sha256": _array_sha256(binned, dtype=binned.dtype),
        "n_bins_sha256": _array_sha256(prep.n_bins_),
        "feature_map_sha256": _array_sha256(prep.feature_map_),
    }
    return prep, np.asarray(binned), state


def _fit_entity_ensemble(X_train, y_train, groups, X_test):
    from darkofit import DarkoRegressor
    from darkofit.booster import GradientBoosting

    seeds = member_seeds()
    started = time.perf_counter_ns()
    prep, full_binned, shared_state = _shared_numeric_preprocessor(
        X_train, y_train, seeds[0]
    )
    shared_preprocess_seconds = (time.perf_counter_ns() - started) / 1e9
    shared_state["fit_seconds"] = float(shared_preprocess_seconds)
    predictions = []
    members = []
    total_fit = float(shared_preprocess_seconds)
    total_predict = 0.0
    groups = np.asarray(groups, dtype=str)
    for member, seed in enumerate(seeds):
        sampled, oob, sampled_groups = group_bootstrap_plan(groups, seed)
        original = GradientBoosting._fit_transform_preprocessor

        def use_shared(
            booster,
            X,
            encode_targets,
            cat_features,
            sample_weight,
            *,
            _sampled=sampled,
        ):
            if cat_features is not None and len(cat_features):
                raise RuntimeError("shared basketball lane received categoricals")
            if len(X) != len(_sampled):
                raise RuntimeError("shared basketball row selection changed")
            booster.prep_ = copy.deepcopy(prep)
            return np.asarray(full_binned[_sampled]).copy()

        GradientBoosting._fit_transform_preprocessor = use_shared
        try:
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
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
        finally:
            GradientBoosting._fit_transform_preprocessor = original
        started = time.perf_counter_ns()
        prediction = harness.validate_prediction(
            model.predict(X_test), len(X_test)
        )
        predict_seconds = (time.perf_counter_ns() - started) / 1e9
        metadata = harness.extract_fit_metadata(model)
        validation = dict(model.model_.auto_params_.get("validation_split", {}))
        if validation.get("source") != "explicit_eval_set":
            raise RuntimeError("entity member did not use explicit group OOB")
        if int(validation.get("eval_n_samples", -1)) != len(oob):
            raise RuntimeError("entity member OOB row count changed")
        if metadata["final_fit"]["stop_reason"] != "early_stopping":
            raise RuntimeError("entity member did not stop on group OOB")
        predictions.append(prediction)
        total_fit += fit_seconds
        total_predict += predict_seconds
        selected_groups = set(sampled_groups)
        oob_groups = set(groups[oob].tolist())
        members.append(
            {
                "member": int(member),
                "seed": int(seed),
                "sampled_group_draws": int(len(sampled_groups)),
                "sampled_unique_groups": int(len(selected_groups)),
                "sampled_groups_sha256": _strings_sha256(sampled_groups),
                "sampled_rows": int(len(sampled)),
                "sampled_rows_sha256": _array_sha256(sampled),
                "oob_groups": int(len(oob_groups)),
                "oob_groups_sha256": _strings_sha256(sorted(oob_groups)),
                "oob_rows": int(len(oob)),
                "oob_rows_sha256": _array_sha256(oob),
                "group_disjoint": not bool(selected_groups & oob_groups),
                "prediction_sha256": harness.prediction_sha256(prediction),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "validation": validation,
                "fit_metadata": metadata,
            }
        )
    averaged = harness.validate_prediction(
        np.mean(np.vstack(predictions), axis=0), len(X_test)
    )
    return averaged, float(total_fit), float(total_predict), {
        "kind": "entity_ensemble",
        "member_count": len(members),
        "member_seeds": list(seeds),
        "shared_preprocessor": shared_state,
        "members": members,
    }


def fit_and_predict(config, X_train, y_train, groups, X_test):
    if config == CONTROL:
        return _fit_control(X_train, y_train, X_test)
    if config == CANDIDATE:
        return _fit_entity_ensemble(
            X_train, y_train, groups, X_test
        )
    raise ValueError(f"unknown entity ensemble config: {config}")


def _warmup(config, dataset, players) -> float:
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    started = time.perf_counter_ns()
    fit_and_predict(
        config,
        dataset.X.iloc[train],
        dataset.y.iloc[train],
        players.iloc[train],
        dataset.X.iloc[test],
    )
    return float((time.perf_counter_ns() - started) / 1e9)


def run_worker(config: str, cache_path: Path) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    players = _player_series(dataset)
    warmup_seconds = _warmup(config, dataset, players)
    folds = []
    started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        prediction, fit_seconds, predict_seconds, metadata = fit_and_predict(
            config,
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            players.iloc[train],
            dataset.X.iloc[test],
        )
        folds.append(
            {
                "fold": int(fold),
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "prediction_sha256": harness.prediction_sha256(prediction),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "metadata": metadata,
            }
        )
    steady_seconds = (time.perf_counter_ns() - started) / 1e9
    guardrail = dataset.player_guardrail
    prediction, fit_seconds, predict_seconds, metadata = fit_and_predict(
        config,
        guardrail.X_train,
        guardrail.y_train,
        players,
        guardrail.X_holdout,
    )
    fold_scores = [float(row["r2"]) for row in folds]
    result = {
        "config": config,
        "mean_r2": float(np.mean(fold_scores)),
        "fold_scores": fold_scores,
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "warmup_seconds_outside_timing": warmup_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "holdout": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "prediction_sha256": harness.prediction_sha256(prediction),
            "scores": guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
            "metadata": metadata,
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        result
    )
    return result


def analyze_quality(control, candidate) -> dict[str, Any]:
    control_scores = np.asarray(control["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    if control_scores.shape != (10,) or candidate_scores.shape != (10,):
        raise RuntimeError("entity ensemble requires ten creator folds")
    if not math.isclose(
        float(control["mean_r2"]),
        EXPECTED_CONTROL_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("basketball control no longer reproduces")
    deltas = candidate_scores - control_scores
    leave_one_out = [
        float(np.mean(np.delete(deltas, fold))) for fold in range(10)
    ]
    control_views = control["holdout"]["scores"]
    candidate_views = candidate["holdout"]["scores"]
    held_delta = float(
        candidate_views["overlap_exposed_team_holdout"]["r2"]
        - control_views["overlap_exposed_team_holdout"]["r2"]
    )
    cold_delta = float(
        candidate_views["cold_player_subset"]["r2"]
        - control_views["cold_player_subset"]["r2"]
    )
    seen_delta = float(
        candidate_views["seen_player_subset"]["r2"]
        - control_views["seen_player_subset"]["r2"]
    )
    mean_delta = float(np.mean(deltas))
    gates = {
        "mean_r2_gain_at_least_0_004": mean_delta >= MIN_MEAN_R2_GAIN,
        "leave_one_fold_out_no_regression": min(leave_one_out) >= 0.0,
        "overlap_exposed_team_no_regression": held_delta >= 0.0,
        "cold_player_positive": cold_delta > 0.0,
    }
    return {
        "mean_r2_delta": mean_delta,
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "fold_losses": int(np.count_nonzero(deltas < 0.0)),
        "fold_r2_deltas": deltas.tolist(),
        "leave_one_fold_out_mean_deltas": leave_one_out,
        "overlap_exposed_team_r2_delta": held_delta,
        "cold_player_r2_delta": cold_delta,
        "seen_player_r2_delta_diagnostic": seen_delta,
        "quality_gates": gates,
        "passes_quality_gates": all(gates.values()),
    }


def _run_worker_process(args, config):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-config",
            config,
            "--data-cache",
            str(args.data_cache),
        ],
        cwd=ROOT,
        env=harness.worker_environment(args.threads),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"entity worker {config} failed with {completed.returncode}"
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_RESULT_PREFIX):])
    result["worker_stdout"] = (
        "\n".join(
            line for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_RESULT_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def run_parent(args) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = creator.git_state(ROOT)
    if not source["clean"] and not args.allow_dirty_source:
        raise RuntimeError("entity ensemble requires a clean source tree")
    dataset = harness.load_basketball_dataset(args.data_cache)
    canonical = {}
    results = []
    fingerprints = {config: set() for config in CONFIGS}

    def run_block(block, order):
        for position, config in enumerate(order):
            if creator.git_state(ROOT) != source:
                raise RuntimeError("source changed during entity ensemble screen")
            print(
                f"block {block + 1}/{len(BLOCK_ORDERS)} "
                f"position {position + 1}: {config}",
                flush=True,
            )
            result = _run_worker_process(args, config)
            result["block"] = int(block)
            result["position"] = int(position)
            canonical.setdefault(config, result)
            fingerprints[config].add(result["behavior_fingerprint_sha256"])
            results.append(result)

    run_block(0, BLOCK_ORDERS[0])
    quality = analyze_quality(canonical[CONTROL], canonical[CANDIDATE])
    if quality["passes_quality_gates"]:
        for block, order in enumerate(BLOCK_ORDERS[1:], start=1):
            run_block(block, order)
    if any(len(values) != 1 for values in fingerprints.values()):
        raise RuntimeError("entity ensemble behavior changed across repeats")
    if creator.git_state(ROOT) != source:
        raise RuntimeError("source changed during entity ensemble screen")

    paired = None
    if quality["passes_quality_gates"]:
        paired = {}
        for metric in (
            "steady_wall_seconds",
            "summed_fit_seconds",
            "summed_predict_seconds",
            "peak_rss_bytes",
        ):
            paired[metric] = campaign.paired_ratio_summary(
                [
                    row[metric] for row in results
                    if row["config"] == CANDIDATE
                ],
                [
                    row[metric] for row in results
                    if row["config"] == CONTROL
                ],
            )
    timing_pass = bool(
        paired
        and all(value["stable"] for value in paired.values())
        and paired["steady_wall_seconds"]["median_ratio"] <= MAX_WALL_RATIO
        and paired["peak_rss_bytes"]["median_ratio"] <= MAX_RSS_RATIO
    )
    passes = bool(
        source["clean"] and quality["passes_quality_gates"] and timing_pass
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_entity_ensemble_screen",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "members": N_MEMBERS,
            "minimum_mean_r2_gain": MIN_MEAN_R2_GAIN,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "executed_blocks": (
                len(BLOCK_ORDERS) if quality["passes_quality_gates"] else 1
            ),
            "paired_ratio_max_iqr_over_median": (
                campaign.MAX_PAIRED_RATIO_IQR_OVER_MEDIAN
            ),
            "max_wall_ratio": MAX_WALL_RATIO,
            "max_peak_rss_ratio": MAX_RSS_RATIO,
            "threads": int(args.threads),
            "quality_fail_fast": True,
            "lockbox_data_used": False,
            "public_api_authorized": False,
            "default_promotion_authorized": False,
            "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
        },
        "source": source,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": canonical,
        "results": results,
        "quality": quality,
        "paired_timing": paired,
        "passes_all_gates": passes,
        "recommendation": (
            "advance_to_s4_sports_confirmation"
            if passes
            else "close_entity_ensemble_as_shaped"
        ),
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {artifact['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument(
        "--threads", type=int, default=max(1, os.cpu_count() or 1)
    )
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--worker-config", choices=CONFIGS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    return args


def main(argv=None) -> int:
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
