#!/usr/bin/env python3
"""Run the frozen behavior-exact fused-oblivious basketball campaign."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import tempfile
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
FUSED_CONFIG = "fused_oblivious"
CONFIG_ORDER = (DEFAULT_CONFIG, FUSED_CONFIG)
PREDICT_REPEATS = 20
MAX_WALL_RATIO = 0.85
MAX_FIT_RATIO = 0.85
MAX_PREDICT_RATIO = 1.02
MAX_PEAK_RSS_RATIO = 1.10
RUNTIME_POLICY_ORIGINAL = "original"
RUNTIME_POLICY_TRAINING_ONLY = "training-only"
RUNTIME_POLICY_AUTOMATIC = "automatic-training-only"
RUNTIME_POLICIES = (
    RUNTIME_POLICY_ORIGINAL,
    RUNTIME_POLICY_TRAINING_ONLY,
    RUNTIME_POLICY_AUTOMATIC,
)
EXPECTED_DEFAULT_MEAN_R2 = 0.5267495183883605
WORKER_RESULT_PREFIX = "BASKETBALL_FUSED_OBLIVIOUS_RESULT="
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_fused_oblivious.json"
CONFIRMATION_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_fused_oblivious_confirmation.json"
)
CONFIRMATION_PROTOCOL = (
    REPO_ROOT / "benchmarks" / "basketball_fused_oblivious_confirmation_protocol.md"
)
CONFIRMATION_PROTOCOL_SHA256 = (
    "e806fd59dd8fd1f1683fba3f0ed852104cf172cc2af0eb0a52e961c072e57ea3"
)
CONFIRMATION_DARKOFIT_SUBTREE = "033ff90c60b01a30281ffb3b88729f30571ab246"
CONFIRMATION_THREADS = 18
AUTOMATIC_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_fused_oblivious_automatic.json"
)
AUTOMATIC_PROTOCOL = (
    REPO_ROOT / "benchmarks" / "basketball_fused_oblivious_automatic_protocol.md"
)
AUTOMATIC_PROTOCOL_SHA256 = (
    "a0d36c335c06e24902efa36fad69d444c78ab07e89316f71dc317b0a5af2df87"
)
AUTOMATIC_DARKOFIT_SUBTREE = "5d8d1f0e7c9edffcb1a8e03315f231ec3e30caf4"
AUTOMATIC_THREADS = 18
_FUSED_LEVEL_COUNTER: np.ndarray | None = None


def configure_builder(config_name: str) -> None:
    """Install the private candidate builder switch inside one worker."""
    global _FUSED_LEVEL_COUNTER

    import darkofit.booster as booster_module

    if config_name not in CONFIG_ORDER:
        raise ValueError(f"unknown basketball config: {config_name}")
    original = booster_module.build_oblivious_tree
    if isinstance(original, functools.partial):
        raise RuntimeError("oblivious builder was already wrapped")
    if config_name == DEFAULT_CONFIG:
        _FUSED_LEVEL_COUNTER = None
        booster_module.build_oblivious_tree = functools.partial(
            original,
            fused_oblivious_kernel=False,
        )
        return
    _FUSED_LEVEL_COUNTER = np.zeros(1, dtype=np.int64)
    booster_module.build_oblivious_tree = functools.partial(
        original,
        fused_oblivious_kernel=True,
        fused_oblivious_counter=_FUSED_LEVEL_COUNTER,
    )


def fused_level_invocations() -> int:
    return 0 if _FUSED_LEVEL_COUNTER is None else int(_FUSED_LEVEL_COUNTER[0])


def reset_fused_level_invocations() -> None:
    if _FUSED_LEVEL_COUNTER is not None:
        _FUSED_LEVEL_COUNTER[0] = 0


def _fit_model(X_train, y_train):
    from darkofit import DarkoRegressor

    model = DarkoRegressor(random_state=creator.RANDOM_STATE)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(fit_seconds)


def _timed_prediction(model, X_test):
    predictions = None
    times = []
    for _ in range(PREDICT_REPEATS):
        started = time.perf_counter_ns()
        current = harness.validate_prediction(model.predict(X_test), len(X_test))
        times.append((time.perf_counter_ns() - started) / 1e9)
        if predictions is None:
            predictions = current
        elif not np.array_equal(current, predictions):
            raise RuntimeError("repeated basketball prediction changed")
    return predictions, float(statistics.median(times)), [float(v) for v in times]


def _archive_identity(model) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="darkofit-fused-oblivious-") as root:
        path = Path(root) / "model.npz"
        model.save_model(path)
        raw = path.read_bytes()
    if not raw:
        raise RuntimeError("serialized basketball model is empty")
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _importance_sha256(model) -> str:
    values = np.ascontiguousarray(
        np.asarray(model.feature_importances_, dtype="<f8")
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def fit_and_predict(X_train, y_train, X_test):
    model, fit_seconds = _fit_model(X_train, y_train)
    prediction, predict_seconds, predict_repeats = _timed_prediction(model, X_test)
    return {
        "prediction": prediction,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "predict_repeat_seconds": predict_repeats,
        "archive": _archive_identity(model),
        "feature_importance_sha256": _importance_sha256(model),
        "fit_metadata": harness.extract_fit_metadata(model),
    }


def validate_fitted_metadata(metadata: dict[str, Any]) -> None:
    if metadata["selected_tree_mode"] != "catboost":
        raise RuntimeError("basketball fit changed tree mode")
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError("basketball fit changed model lane")
    if int(metadata["resolved_thread_count"]) < 3:
        raise RuntimeError("fused basketball worker resolved fewer than three threads")
    if not math.isfinite(float(metadata["resolved_learning_rate"])):
        raise RuntimeError("basketball fit resolved an invalid learning rate")
    if metadata["refit"] or metadata["selection_fit"] is not None:
        raise RuntimeError("basketball fit unexpectedly used wrapper refit")
    if int(metadata["fitted_tree_count"]) != int(metadata["best_iteration"]):
        raise RuntimeError("basketball fit retained the wrong tree count")
    final = metadata["final_fit"]
    if int(final["iterations_requested"]) != 1000:
        raise RuntimeError("basketball fit changed the default horizon")
    if final["stop_reason"] != "iteration_limit":
        raise RuntimeError("basketball fit changed its stop reason")


def _warmup(dataset: harness.BasketballDataset) -> float:
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    started = time.perf_counter_ns()
    model, _ = _fit_model(dataset.X.iloc[train], dataset.y.iloc[train])
    harness.validate_prediction(model.predict(dataset.X.iloc[test]), len(test))
    return float((time.perf_counter_ns() - started) / 1e9)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _fold_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": row["fold"],
        "test_indices": row["test_indices"],
        "r2": row["r2"],
        "prediction_sha256": row["prediction_sha256"],
        "archive": row["archive"],
        "feature_importance_sha256": row["feature_importance_sha256"],
        "fit_metadata": row["fit_metadata"],
    }


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "folds": [_fold_payload(row) for row in result["folds"]],
        "holdout": {
            "scores": result["holdout"]["scores"],
            "prediction_sha256": result["holdout"]["prediction_sha256"],
            "archive": result["holdout"]["archive"],
            "feature_importance_sha256": result["holdout"][
                "feature_importance_sha256"
            ],
            "fit_metadata": result["holdout"]["fit_metadata"],
        },
    }


def run_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    configure_builder(config_name)
    dataset = harness.load_basketball_dataset(cache_path)
    warmup_seconds = _warmup(dataset)
    reset_fused_level_invocations()
    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        fitted = fit_and_predict(
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            dataset.X.iloc[test],
        )
        validate_fitted_metadata(fitted["fit_metadata"])
        prediction = fitted.pop("prediction")
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_indices": [int(value) for value in test],
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "prediction_sha256": harness.prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                **fitted,
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    guardrail = dataset.player_guardrail
    fitted = fit_and_predict(
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    validate_fitted_metadata(fitted["fit_metadata"])
    prediction = fitted.pop("prediction")
    scores = np.asarray([row["r2"] for row in folds], dtype=np.float64)
    result = {
        "config": config_name,
        "fused_oblivious_kernel_requested": config_name == FUSED_CONFIG,
        "fused_kernel_level_invocations": fused_level_invocations(),
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
        "peak_rss_bytes": _peak_rss_bytes(),
        "holdout": {
            "prediction_sha256": harness.prediction_sha256(prediction),
            "predictions": [float(value) for value in prediction],
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
            **fitted,
        },
        "guardrail": guardrail.metadata,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _run_worker_process(args: argparse.Namespace, config_name: str):
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


def analyze_exactness(canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    default = canonical[DEFAULT_CONFIG]
    fused = canonical[FUSED_CONFIG]
    gates: dict[str, bool] = {}
    gates["default_reproduces"] = math.isclose(
        float(default["mean_r2"]),
        EXPECTED_DEFAULT_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    gates["mean_r2_exact"] = fused["mean_r2"] == default["mean_r2"]
    gates["fold_scores_exact"] = fused["fold_scores"] == default["fold_scores"]
    gates["fold_count_exact"] = len(fused["folds"]) == len(default["folds"])
    fold_fields = (
        "test_indices",
        "r2",
        "prediction_sha256",
        "archive",
        "feature_importance_sha256",
        "fit_metadata",
    )
    gates["fold_payloads_exact"] = gates["fold_count_exact"] and all(
        all(fused_row[field] == default_row[field] for field in fold_fields)
        for default_row, fused_row in zip(default["folds"], fused["folds"])
    )
    holdout_fields = (
        "scores",
        "prediction_sha256",
        "archive",
        "feature_importance_sha256",
        "fit_metadata",
    )
    gates["holdout_exact"] = all(
        fused["holdout"][field] == default["holdout"][field]
        for field in holdout_fields
    )
    gates["behavior_fingerprint_exact"] = (
        fused["behavior_fingerprint_sha256"]
        == default["behavior_fingerprint_sha256"]
    )
    gates["fused_kernel_engaged"] = (
        int(default["fused_kernel_level_invocations"]) == 0
        and int(fused["fused_kernel_level_invocations"]) > 0
    )
    return {
        "exactness_gates": gates,
        "passes_exactness_gates": all(gates.values()),
    }


def analyze_runtime(
    wall_timing,
    fit_timing,
    predict_timing,
    peak_rss_values,
    canonical,
    *,
    runtime_policy=RUNTIME_POLICY_ORIGINAL,
) -> dict[str, Any]:
    if runtime_policy not in RUNTIME_POLICIES:
        raise ValueError(f"unknown runtime policy: {runtime_policy}")
    wall_ratio = float(
        wall_timing[FUSED_CONFIG]["median_seconds"]
        / wall_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    fit_ratio = float(
        fit_timing[FUSED_CONFIG]["median_seconds"]
        / fit_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    predict_ratio = float(
        predict_timing[FUSED_CONFIG]["median_seconds"]
        / predict_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    rss_ratio = float(
        statistics.median(peak_rss_values[FUSED_CONFIG])
        / statistics.median(peak_rss_values[DEFAULT_CONFIG])
    )
    default_archives = [row["archive"] for row in canonical[DEFAULT_CONFIG]["folds"]]
    default_archives.append(canonical[DEFAULT_CONFIG]["holdout"]["archive"])
    fused_archives = [row["archive"] for row in canonical[FUSED_CONFIG]["folds"]]
    fused_archives.append(canonical[FUSED_CONFIG]["holdout"]["archive"])
    gates = {
        "default_timing_stable": wall_timing[DEFAULT_CONFIG]["stable"],
        "fused_timing_stable": wall_timing[FUSED_CONFIG]["stable"],
        "wall_speedup": wall_ratio <= MAX_WALL_RATIO,
        "fit_speedup": fit_ratio <= MAX_FIT_RATIO,
        "archive_bytes_exact": fused_archives == default_archives,
        "peak_rss_within_budget": rss_ratio <= MAX_PEAK_RSS_RATIO,
    }
    prediction_timing_disposition = "gated"
    if runtime_policy == RUNTIME_POLICY_ORIGINAL:
        gates["prediction_no_regression"] = predict_ratio <= MAX_PREDICT_RATIO
    else:
        prediction_timing_disposition = (
            "diagnostic_noncausal_for_training_only_candidate"
        )
    return {
        "runtime_policy": runtime_policy,
        "prediction_timing_disposition": prediction_timing_disposition,
        "fused_over_default_median_wall_time": wall_ratio,
        "fused_over_default_median_fit_time": fit_ratio,
        "fused_over_default_median_predict_time": predict_ratio,
        "fused_over_default_median_peak_rss": rss_ratio,
        "runtime_gates": gates,
        "passes_runtime_gates": all(gates.values()),
    }


def _source_state(allow_dirty: bool) -> dict[str, Any]:
    state = creator.git_state(REPO_ROOT)
    if not allow_dirty and not state["clean"]:
        raise RuntimeError("refusing to benchmark a dirty DarkoFit source tree")
    return state


def _current_darkofit_subtree() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD:darkofit"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode or len(value) != 40:
        raise RuntimeError("could not attest the current DarkoFit package subtree")
    return value


def _runtime_policy_binding(runtime_policy: str):
    if runtime_policy == RUNTIME_POLICY_TRAINING_ONLY:
        return {
            "label": "training-only confirmation",
            "threads": CONFIRMATION_THREADS,
            "output": CONFIRMATION_OUTPUT,
            "protocol": CONFIRMATION_PROTOCOL,
            "protocol_sha256": CONFIRMATION_PROTOCOL_SHA256,
            "darkofit_subtree": CONFIRMATION_DARKOFIT_SUBTREE,
        }
    if runtime_policy == RUNTIME_POLICY_AUTOMATIC:
        return {
            "label": "automatic training confirmation",
            "threads": AUTOMATIC_THREADS,
            "output": AUTOMATIC_OUTPUT,
            "protocol": AUTOMATIC_PROTOCOL,
            "protocol_sha256": AUTOMATIC_PROTOCOL_SHA256,
            "darkofit_subtree": AUTOMATIC_DARKOFIT_SUBTREE,
        }
    raise RuntimeError(f"unknown runtime policy: {runtime_policy}")


def _validate_runtime_policy(args: argparse.Namespace) -> None:
    if args.runtime_policy == RUNTIME_POLICY_ORIGINAL:
        return
    binding = _runtime_policy_binding(args.runtime_policy)
    label = binding["label"]
    if args.threads != binding["threads"]:
        raise RuntimeError(f"{label} requires exactly 18 threads")
    if args.output != binding["output"]:
        raise RuntimeError(f"{label} output path is not exact")
    if args.allow_dirty_source:
        raise RuntimeError(f"{label} requires clean source")
    protocol_payload = binding["protocol"].read_bytes()
    if hashlib.sha256(protocol_payload).hexdigest() != binding["protocol_sha256"]:
        raise RuntimeError(f"{label} protocol changed")
    if _current_darkofit_subtree() != binding["darkofit_subtree"]:
        raise RuntimeError(f"{label} candidate subtree changed")


def _assert_source_unchanged(expected, observed, boundary: str) -> None:
    fields = ("path", "head", "branch", "clean", "status")
    changed = [field for field in fields if expected[field] != observed[field]]
    if changed:
        raise RuntimeError(f"DarkoFit source changed {boundary}: " + ", ".join(changed))


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    """Publish one new artifact atomically without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"refusing to overwrite benchmark output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _repeat_record(block: int, position: int, result: dict[str, Any]):
    return {
        "block": int(block),
        "position": int(position),
        "config": result["config"],
        "steady_wall_seconds": result["steady_wall_seconds"],
        "warmup_seconds_outside_timing": result["warmup_seconds_outside_timing"],
        "summed_fit_seconds": result["summed_fit_seconds"],
        "summed_predict_seconds": result["summed_predict_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"],
        "behavior_fingerprint_sha256": result["behavior_fingerprint_sha256"],
        "fused_kernel_level_invocations": result[
            "fused_kernel_level_invocations"
        ],
        "worker_stdout": result["worker_stdout"],
        "worker_stderr": result["worker_stderr"],
    }


def _decision_recommendation(passes_all: bool, runtime_policy: str) -> str:
    if not passes_all:
        return "advance_none"
    if runtime_policy == RUNTIME_POLICY_AUTOMATIC:
        return "promote_internal_fused_lane"
    return "advance_to_expanded_behavior_tests"


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output: {args.output}")
    _validate_runtime_policy(args)
    source = _source_state(args.allow_dirty_source)
    dataset = harness.load_basketball_dataset(args.data_cache)
    schedule = harness.reciprocal_schedule(DEFAULT_CONFIG, FUSED_CONFIG)
    canonical: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {name: set() for name in CONFIG_ORDER}
    wall_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    fit_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    predict_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    rss_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    repeats = []

    def run_block(block: int, order: tuple[str, str]) -> None:
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
            fit_values[config_name].append(result["summed_fit_seconds"])
            predict_values[config_name].append(result["summed_predict_seconds"])
            rss_values[config_name].append(result["peak_rss_bytes"])
            repeats.append(_repeat_record(block, position, result))
            canonical.setdefault(config_name, result)
            print(
                f"  mean R2={result['mean_r2']:.12f}, "
                f"steady={result['steady_wall_seconds']:.2f}s",
                flush=True,
            )

    run_block(0, schedule[0])
    exactness = analyze_exactness(canonical)
    timing_skipped = not exactness["passes_exactness_gates"]
    if not timing_skipped:
        for block, order in enumerate(schedule[1:], start=1):
            run_block(block, order)

    for config_name, values in fingerprints.items():
        if len(values) != 1:
            raise RuntimeError(f"{config_name} behavior changed across repeats")
    if not timing_skipped and (
        fingerprints[DEFAULT_CONFIG] != fingerprints[FUSED_CONFIG]
    ):
        raise RuntimeError("default and fused behavior fingerprints differ")
    _assert_source_unchanged(
        source,
        _source_state(args.allow_dirty_source),
        boundary="during the experiment",
    )

    wall_timing = fit_timing = predict_timing = runtime = None
    if not timing_skipped:
        wall_timing = {
            name: harness.timing_summary(values) for name, values in wall_values.items()
        }
        fit_timing = {
            name: harness.timing_summary(values) for name, values in fit_values.items()
        }
        predict_timing = {
            name: harness.timing_summary(values)
            for name, values in predict_values.items()
        }
        runtime = analyze_runtime(
            wall_timing,
            fit_timing,
            predict_timing,
            rss_values,
            canonical,
            runtime_policy=args.runtime_policy,
        )

    evidence_eligible = bool(source["clean"])
    passes_runtime = runtime is not None and runtime["passes_runtime_gates"]
    passes_all = (
        exactness["passes_exactness_gates"]
        and passes_runtime
        and evidence_eligible
    )
    decision = {
        "candidate": FUSED_CONFIG,
        **exactness,
        "timing_confirmation_status": (
            "skipped_exactness_failure" if timing_skipped else "completed"
        ),
        "runtime": runtime,
        "passes_runtime_gates": passes_runtime,
        "evidence_gates": {"committed_clean_source": evidence_eligible},
        "evidence_eligible": evidence_eligible,
        "passes_all_gates": passes_all,
        "recommendation": _decision_recommendation(
            passes_all, args.runtime_policy
        ),
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": (
                "basketball_behavior_exact_fused_oblivious_automatic"
                if args.runtime_policy == RUNTIME_POLICY_AUTOMATIC
                else "basketball_behavior_exact_fused_oblivious"
            ),
            "candidate_scope": (
                "automatic_internal_dispatch"
                if args.runtime_policy == RUNTIME_POLICY_AUTOMATIC
                else "private_engine_candidate"
            ),
            "public_parameter_added": False,
            "creator_benchmark_changed": False,
            "quality_tuning": False,
            "predict_repeats_per_model": PREDICT_REPEATS,
            "timing_schedule": [list(order) for order in schedule],
            "executed_timing_blocks": 1 if timing_skipped else len(schedule),
            "maximum_timing_spread_ratio": harness.MAX_TIMING_SPREAD_RATIO,
            "maximum_wall_ratio": MAX_WALL_RATIO,
            "maximum_fit_ratio": MAX_FIT_RATIO,
            "maximum_predict_ratio": MAX_PREDICT_RATIO,
            "maximum_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "runtime_policy": args.runtime_policy,
            "prediction_timing_is_decision_gate": (
                args.runtime_policy == RUNTIME_POLICY_ORIGINAL
            ),
            "threads_per_fit": args.threads,
            "random_state": creator.RANDOM_STATE,
            "weights_used": False,
            "lockbox_data_used": False,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
            "warmup": "one complete first-fold fit and prediction per worker",
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
        "fit_timing_summary": fit_timing,
        "predict_timing_summary": predict_timing,
        "peak_rss_values": rss_values,
        "decision": decision,
    }
    _atomic_write_new_bytes(
        args.output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--runtime-policy",
        choices=RUNTIME_POLICIES,
        default=RUNTIME_POLICY_ORIGINAL,
    )
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 3:
        parser.error("--threads must be at least 3 for the fused campaign")
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    if args.runtime_policy != RUNTIME_POLICY_ORIGINAL:
        binding = _runtime_policy_binding(args.runtime_policy)
        if args.threads != binding["threads"]:
            parser.error(
                f"--runtime-policy {args.runtime_policy} requires --threads 18"
            )
        if args.output != binding["output"]:
            parser.error(
                f"--runtime-policy {args.runtime_policy} requires its frozen output"
            )
        if args.allow_dirty_source:
            parser.error(
                f"--runtime-policy {args.runtime_policy} does not allow dirty source"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        result = run_worker(args.worker_config, args.data_cache)
        print(
            WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
