#!/usr/bin/env python3
"""Run the preregistered basketball random-strength retirement screen."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CONTROL = "control"
CANDIDATES = ("random_strength_0_5", "random_strength_1_0")
CONFIG_ORDER = (CONTROL, *CANDIDATES)
CONFIGS = {
    CONTROL: {"random_strength": 0.0},
    "random_strength_0_5": {"random_strength": 0.5},
    "random_strength_1_0": {"random_strength": 1.0},
}
MIN_MEAN_R2_GAIN = 0.002
EXPECTED_CONTROL_MEAN_R2 = 0.5267495183883605
WORKER_RESULT_PREFIX = "BASKETBALL_RANDOM_STRENGTH_RESULT="
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_random_strength.json"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_random_strength_protocol.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _fit_predict(config: str, X_train, y_train, X_test):
    from darkofit import DarkoRegressor

    model = DarkoRegressor(
        random_state=creator.RANDOM_STATE,
        random_strength=CONFIGS[config]["random_strength"],
        verbose_timing=True,
    )
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(model.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    metadata = harness.extract_fit_metadata(model)
    if metadata["selected_tree_mode"] != "catboost":
        raise RuntimeError(f"{config} selected an unexpected tree mode")
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError(f"{config} selected an unexpected model lane")
    if metadata["refit"] or metadata["selection_fit"] is not None:
        raise RuntimeError(f"{config} unexpectedly selected or refit a model")
    if metadata["final_fit"]["iterations_requested"] != 1000:
        raise RuntimeError(f"{config} changed the default boosting horizon")
    if metadata["fitted_tree_count"] != 1000:
        raise RuntimeError(f"{config} did not retain the default horizon")
    return prediction, float(fit_seconds), float(predict_seconds), metadata


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": result["config"],
        "folds": [
            {
                "fold": row["fold"],
                "r2": row["r2"],
                "prediction_sha256": row["prediction_sha256"],
                "fit_metadata": row["fit_metadata"],
            }
            for row in result["folds"]
        ],
        "holdout": {
            "scores": result["holdout"]["scores"],
            "prediction_sha256": result["holdout"]["prediction_sha256"],
            "fit_metadata": result["holdout"]["fit_metadata"],
        },
    }


def run_worker(config: str, cache_path: Path) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    folds = []
    started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        prediction, fit_seconds, predict_seconds, metadata = _fit_predict(
            config,
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            dataset.X.iloc[test],
        )
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_indices": [int(value) for value in test],
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_sha256": harness.prediction_sha256(prediction),
                "predictions": prediction.tolist(),
                "fit_metadata": metadata,
            }
        )
    wall_seconds = (time.perf_counter_ns() - started) / 1e9

    guardrail = dataset.player_guardrail
    prediction, fit_seconds, predict_seconds, metadata = _fit_predict(
        config,
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    fold_scores = [float(row["r2"]) for row in folds]
    result = {
        "config": config,
        "config_spec": CONFIGS[config],
        "mean_r2": float(np.mean(fold_scores)),
        "fold_scores": fold_scores,
        "folds": folds,
        "wall_seconds": float(wall_seconds),
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "holdout": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "prediction_sha256": harness.prediction_sha256(prediction),
            "predictions": prediction.tolist(),
            "fit_metadata": metadata,
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
        },
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def analyze_quality(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    control_scores = np.asarray(control["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    expected_shape = (creator.N_SPLITS,)
    if (
        control_scores.shape != expected_shape
        or candidate_scores.shape != expected_shape
    ):
        raise RuntimeError("random-strength result does not contain ten folds")
    if not math.isclose(
        float(control["mean_r2"]),
        EXPECTED_CONTROL_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("basketball control no longer reproduces")

    deltas = candidate_scores - control_scores
    leave_one_out = [
        float(np.mean(np.delete(deltas, fold)))
        for fold in range(creator.N_SPLITS)
    ]
    control_holdout = control["holdout"]["scores"]
    candidate_holdout = candidate["holdout"]["scores"]

    def score_delta(view: str) -> float:
        return float(
            candidate_holdout[view]["r2"] - control_holdout[view]["r2"]
        )

    held_delta = score_delta("overlap_exposed_team_holdout")
    seen_delta = score_delta("seen_player_subset")
    cold_delta = score_delta("cold_player_subset")
    mean_delta = float(np.mean(deltas))
    gates = {
        "mean_r2_gain_at_least_0_002": mean_delta >= MIN_MEAN_R2_GAIN,
        "leave_one_fold_out_no_regression": min(leave_one_out) >= 0.0,
        "overlap_exposed_team_no_regression": held_delta >= 0.0,
        "cold_player_no_regression": cold_delta >= 0.0,
    }
    return {
        "candidate": candidate["config"],
        "mean_r2_delta": mean_delta,
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "fold_losses": int(np.count_nonzero(deltas < 0.0)),
        "fold_r2_deltas": deltas.tolist(),
        "leave_one_fold_out_mean_deltas": leave_one_out,
        "overlap_exposed_team_r2_delta": held_delta,
        "seen_player_r2_delta_diagnostic": seen_delta,
        "cold_player_r2_delta": cold_delta,
        "quality_gates": gates,
        "passes_quality_gates": all(gates.values()),
    }


def _run_worker_process(
    args: argparse.Namespace, config: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-config",
            config,
            "--data-cache",
            str(args.data_cache),
        ],
        cwd=REPO_ROOT,
        env=harness.worker_environment(args.threads),
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
            f"random-strength worker {config!r} failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
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
    if args.output.exists():
        raise RuntimeError(f"refusing to replace benchmark output: {args.output}")
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    source = _source_state(args.allow_dirty_source)
    dataset = harness.load_basketball_dataset(args.data_cache)
    results = []
    for config in CONFIG_ORDER:
        _assert_source_unchanged(
            source,
            _source_state(args.allow_dirty_source),
            f"before {config}",
        )
        print(f"running {config}...", flush=True)
        result = _run_worker_process(args, config)
        results.append(result)
        cold = result["holdout"]["scores"]["cold_player_subset"]["r2"]
        print(
            f"  mean R2={result['mean_r2']:.12f}, "
            f"cold={cold:.12f}, wall={result['wall_seconds']:.2f}s",
            flush=True,
        )
    _assert_source_unchanged(
        source,
        _source_state(args.allow_dirty_source),
        "during the experiment",
    )

    by_name = {result["config"]: result for result in results}
    decisions = {
        candidate: analyze_quality(by_name[CONTROL], by_name[candidate])
        for candidate in CANDIDATES
    }
    passing = [
        name
        for name, decision in decisions.items()
        if decision["passes_quality_gates"]
    ]
    passing.sort(
        key=lambda name: decisions[name]["mean_r2_delta"], reverse=True
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_random_strength_retirement_screen",
            "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "config_order": list(CONFIG_ORDER),
            "configs": CONFIGS,
            "minimum_mean_r2_gain": MIN_MEAN_R2_GAIN,
            "threads_per_fit": args.threads,
            "random_state": creator.RANDOM_STATE,
            "weights_used": False,
            "lockbox_data_used": False,
            "default_promotion_authorized": False,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
        },
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "source": source,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "results": results,
        "decisions": decisions,
        "advancing_candidate": passing[0] if passing else None,
        "conclusion": (
            f"keep_and_confirm_{passing[0]}"
            if passing
            else "deprecate_random_strength"
        ),
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"conclusion: {artifact['conclusion']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument(
        "--threads", type=int, default=max(1, os.cpu_count() or 1)
    )
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
