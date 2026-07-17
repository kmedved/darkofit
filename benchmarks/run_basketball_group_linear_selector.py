#!/usr/bin/env python3
"""Run the frozen basketball group-aware linear-leaf selector gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
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
from benchmarks.run_fused_variable_hessian import (  # noqa: E402
    _canonical_model_payload_sha256,
)


CONTROL = "control"
CANDIDATE = "group_margin_selector"
CONFIGS = (CONTROL, CANDIDATE)
BLOCK_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)
VALIDATION_FRACTION = 0.2
MIN_RELATIVE_IMPROVEMENT = 0.03
MAX_WALL_RATIO = 3.5
MAX_PREDICT_RATIO = 1.25
MAX_RSS_RATIO = 2.0
EXPECTED_CONTROL_MEAN_R2 = 0.5267495183883605
PROTOCOL = ROOT / "benchmarks" / "basketball_group_linear_selector_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "basketball_group_linear_selector.json"
WORKER_RESULT_PREFIX = "BASKETBALL_GROUP_LINEAR_SELECTOR_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_sha256(values) -> str:
    value = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _strings_sha256(values) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(value) for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _players(dataset):
    values = dataset.frame.loc[
        dataset.X.index, guardrails.PLAYER_COLUMN
    ].astype(str)
    if len(values) != len(dataset.X):
        raise RuntimeError("basketball player metadata is misaligned")
    return values.reset_index(drop=True)


def selection_split(X, y, groups):
    from darkofit.sklearn_api import _make_eval_split

    train, validation, policy = _make_eval_split(
        X,
        y,
        VALIDATION_FRACTION,
        creator.RANDOM_STATE,
        validation_strategy="group",
        groups=groups,
    )
    train_groups = set(np.asarray(groups)[train].tolist())
    validation_groups = set(np.asarray(groups)[validation].tolist())
    if train_groups & validation_groups:
        raise RuntimeError("group selector split shares players")
    return train, validation, {
        "policy": policy,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_indices_sha256": _index_sha256(train),
        "validation_indices_sha256": _index_sha256(validation),
        "train_group_count": int(len(train_groups)),
        "validation_group_count": int(len(validation_groups)),
        "train_groups_sha256": _strings_sha256(sorted(train_groups)),
        "validation_groups_sha256": _strings_sha256(
            sorted(validation_groups)
        ),
        "groups_disjoint": True,
    }


def _save_state(model) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.npz"
        model.save_model(path)
        return {
            "canonical_payload_sha256": (
                _canonical_model_payload_sha256(path)
            ),
        }


def _fit_model(X, y, *, linear_leaves: bool, eval_set=None):
    from darkofit import DarkoRegressor

    params = {
        "random_state": creator.RANDOM_STATE,
        "linear_leaves": bool(linear_leaves),
        "verbose_timing": True,
    }
    if eval_set is not None:
        params.update(
            {
                "early_stopping": True,
                "early_stopping_rounds": None,
                "use_best_model": True,
                "refit": False,
            }
        )
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X, y, eval_set=eval_set)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(fit_seconds)


def _fit_control(X_train, y_train, X_test):
    model, fit_seconds = _fit_model(
        X_train, y_train, linear_leaves=False
    )
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(model.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    metadata = {
        "kind": "single",
        "selected_linear_leaves": False,
        "final_fit": harness.extract_fit_metadata(model),
        "model_state": _save_state(model),
    }
    return prediction, fit_seconds, float(predict_seconds), metadata


def _selection_record(name, model, fit_seconds):
    fitted = harness.extract_fit_metadata(model)
    validation = dict(model.model_.auto_params_.get("validation_split", {}))
    score = float(model.best_score_)
    if not math.isfinite(score) or score <= 0.0:
        raise RuntimeError("selector validation score is invalid")
    if validation.get("source") != "explicit_eval_set":
        raise RuntimeError("selector did not use explicit group validation")
    if fitted["final_fit"]["stop_reason"] != "early_stopping":
        raise RuntimeError("selector fit did not early-stop")
    return {
        "name": name,
        "linear_leaves": name == "linear",
        "validation_rmse": score,
        "fit_seconds": float(fit_seconds),
        "validation": validation,
        "fit_metadata": fitted,
    }


def _fit_selector(X_train, y_train, groups, X_test):
    train, validation, split = selection_split(
        X_train, y_train, groups
    )
    eval_set = (
        X_train.iloc[validation],
        y_train.iloc[validation],
    )
    records = []
    for name, linear in (("constant", False), ("linear", True)):
        model, seconds = _fit_model(
            X_train.iloc[train],
            y_train.iloc[train],
            linear_leaves=linear,
            eval_set=eval_set,
        )
        records.append(_selection_record(name, model, seconds))
    constant_score = records[0]["validation_rmse"]
    linear_score = records[1]["validation_rmse"]
    relative_improvement = float(
        (constant_score - linear_score) / constant_score
    )
    selected_linear = relative_improvement >= MIN_RELATIVE_IMPROVEMENT
    final, final_seconds = _fit_model(
        X_train,
        y_train,
        linear_leaves=selected_linear,
    )
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(final.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    metadata = {
        "kind": "group_margin_selector",
        "split": split,
        "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        "relative_validation_improvement": relative_improvement,
        "selected_linear_leaves": bool(selected_linear),
        "selection_fits": records,
        "final_fit_seconds": float(final_seconds),
        "final_fit": harness.extract_fit_metadata(final),
        "model_state": _save_state(final),
    }
    total_fit = sum(row["fit_seconds"] for row in records) + final_seconds
    return prediction, float(total_fit), float(predict_seconds), metadata


def fit_and_predict(config, X_train, y_train, groups, X_test):
    if config == CONTROL:
        return _fit_control(X_train, y_train, X_test)
    if config == CANDIDATE:
        return _fit_selector(
            X_train, y_train, groups, X_test
        )
    raise ValueError(f"unknown selector config: {config}")


def _warmup(config, dataset, players):
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    started = time.perf_counter_ns()
    fit_and_predict(
        config,
        dataset.X.iloc[train],
        dataset.y.iloc[train],
        players.iloc[train].reset_index(drop=True),
        dataset.X.iloc[test],
    )
    return float((time.perf_counter_ns() - started) / 1e9)


def run_worker(config: str, cache_path: Path) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    players = _players(dataset)
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
            players.iloc[train].reset_index(drop=True),
            dataset.X.iloc[test],
        )
        folds.append(
            {
                "fold": int(fold),
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_sha256": harness.prediction_sha256(prediction),
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
    fold_scores = [row["r2"] for row in folds]
    result = {
        "config": config,
        "mean_r2": float(np.mean(fold_scores)),
        "fold_scores": fold_scores,
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "summed_fit_seconds": float(
            sum(row["fit_seconds"] for row in folds)
        ),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "warmup_seconds_outside_timing": warmup_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "holdout": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "prediction_sha256": harness.prediction_sha256(prediction),
            "metadata": metadata,
            "scores": guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        result
    )
    return result


def analyze_exact(control, candidate):
    if not math.isclose(
        control["mean_r2"],
        EXPECTED_CONTROL_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("basketball control no longer reproduces")
    fold_hashes_match = all(
        a["prediction_sha256"] == b["prediction_sha256"]
        for a, b in zip(control["folds"], candidate["folds"])
    )
    model_hashes_match = all(
        a["metadata"]["model_state"]["canonical_payload_sha256"]
        == b["metadata"]["model_state"]["canonical_payload_sha256"]
        for a, b in zip(control["folds"], candidate["folds"])
    )
    candidate_records = [
        row["metadata"] for row in candidate["folds"]
    ] + [candidate["holdout"]["metadata"]]
    gates = {
        "all_candidate_splits_group_disjoint": all(
            record["split"]["groups_disjoint"]
            for record in candidate_records
        ),
        "candidate_declines_everywhere": all(
            not record["selected_linear_leaves"]
            for record in candidate_records
        ),
        "fold_prediction_hashes_match": fold_hashes_match,
        "holdout_prediction_hash_matches": (
            control["holdout"]["prediction_sha256"]
            == candidate["holdout"]["prediction_sha256"]
        ),
        "fold_model_state_hashes_match": model_hashes_match,
        "holdout_model_state_hash_matches": (
            control["holdout"]["metadata"]["model_state"][
                "canonical_payload_sha256"
            ]
            == candidate["holdout"]["metadata"]["model_state"][
                "canonical_payload_sha256"
            ]
        ),
        "fold_scores_match": control["fold_scores"] == candidate["fold_scores"],
        "guardrail_scores_match": (
            control["holdout"]["scores"] == candidate["holdout"]["scores"]
        ),
    }
    margins = [
        record["relative_validation_improvement"]
        for record in candidate_records
    ]
    return {
        "gates": gates,
        "passes": all(gates.values()),
        "selection_margins": margins,
        "maximum_selection_margin": float(max(margins)),
        "minimum_selection_margin": float(min(margins)),
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
            f"selector worker {config} failed with {completed.returncode}"
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


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = creator.git_state(ROOT)
    if not source["clean"]:
        raise RuntimeError("selector gate requires a clean source tree")
    dataset = harness.load_basketball_dataset(args.data_cache)
    canonical = {}
    results = []
    fingerprints = {config: set() for config in CONFIGS}

    def run_block(block, order):
        for position, config in enumerate(order):
            if creator.git_state(ROOT) != source:
                raise RuntimeError("source changed during selector gate")
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
    exact = analyze_exact(canonical[CONTROL], canonical[CANDIDATE])
    if exact["passes"]:
        for block, order in enumerate(BLOCK_ORDERS[1:], start=1):
            run_block(block, order)
    if any(len(values) != 1 for values in fingerprints.values()):
        raise RuntimeError("selector behavior changed across repeats")
    if creator.git_state(ROOT) != source:
        raise RuntimeError("source changed during selector gate")

    paired = None
    if exact["passes"]:
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
    timing_gates = {
        "all_paired_ratios_stable": bool(
            paired
            and all(
                paired[metric]["stable"]
                for metric in (
                    "steady_wall_seconds",
                    "summed_fit_seconds",
                    "summed_predict_seconds",
                )
            )
        ),
        "wall_ratio_at_most_3_5": bool(
            paired
            and paired["steady_wall_seconds"]["median_ratio"]
            <= MAX_WALL_RATIO
        ),
        "predict_ratio_at_most_1_25": bool(
            paired
            and paired["summed_predict_seconds"]["median_ratio"]
            <= MAX_PREDICT_RATIO
        ),
        "rss_ratio_at_most_2": bool(
            paired
            and paired["peak_rss_bytes"]["median_ratio"] <= MAX_RSS_RATIO
        ),
    }
    passes = bool(
        exact["passes"] and all(timing_gates.values()) and source["clean"]
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_group_linear_selector_gate",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "validation_fraction": VALIDATION_FRACTION,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "executed_blocks": (
                len(BLOCK_ORDERS) if exact["passes"] else 1
            ),
            "threads": int(args.threads),
            "quality_fail_fast": True,
            "lockbox_data_used": False,
            "public_selector_authorized": False,
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
        "exactness": exact,
        "paired_timing": paired,
        "timing_gates": timing_gates,
        "passes_all_gates": passes,
        "recommendation": (
            "advance_selector_to_spent_smooth_development"
            if passes
            else "close_group_margin_selector"
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
