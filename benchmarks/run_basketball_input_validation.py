#!/usr/bin/env python3
"""Run the frozen basketball input-validation compliance campaign."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


VALIDATED = "validated"
ASSUME_FINITE = "assume_finite"
ARMS = (VALIDATED, ASSUME_FINITE)
WORKER_RESULT_PREFIX = "BASKETBALL_INPUT_VALIDATION_RESULT="
EXPECTED_THREADS = 18
TIMING_BLOCKS = 6
EXPECTED_FOLD = 0
EXPECTED_PROTOCOL_SHA256 = (
    "e1bb661237e5e9f6b12063c4ed7866e9924d445ebc810105bb3e6b16339586cb"
)
EXPECTED_PACKAGE_MANIFEST = (
    "3fed959b87aa3334a8214a073bacad9e7c0482249d1b04da5b52655e0fdb0121"
)
EXPECTED_SUPPORT_SHA256 = {
    "NOTICE": (
        "38761e49ca40218c7c6e077489c9d3c08c85a0a131fdad9805acbc66d9601ffc"
    ),
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
    "tests/test_input_validation.py": (
        "55ed72ec7070bb8bf55ece74ead73b9a1c522d03a3e9a796ab6ca2a8eee2f87f"
    ),
}
EXPECTED_PREDICTION_SHA256 = {
    "creator_fold_0": (
        "6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f"
    ),
    "held_team": (
        "1693ff2070b05bb705810aba0d9b27b5a0a01dc6f4ee51939a3ee30af3698cdf"
    ),
    "cold_player": (
        "b9dc899fcabc5a3a7892da41d839bac70f7d50da9553e2e57770501f71694c82"
    ),
}
EXPECTED_ARCHIVE = {
    "bytes": 382_557,
    "sha256": "50a7e6f0a6f8500a55a6ba088ad25137335ed4354a4b4e908ea17f023c91ec71",
}
PROTOCOL_PATH = ROOT / "benchmarks/basketball_input_validation_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks/basketball_input_validation.json"
DEFAULT_TEMP_ROOT = ROOT / ".cache/basketball-input-validation"

MAX_VALIDATED_FIT_SECONDS = 1.7674032981
MAX_VALIDATED_PREDICT_RATIO = 1.10
MAX_FIT_IQR_FRACTION = 0.25
MAX_PAIRED_PREDICT_RATIO_IQR_FRACTION = 0.25
MAX_PREDICT_IQR_FRACTION = 0.50


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def _source_state() -> dict[str, Any]:
    return {
        "repository": str(ROOT),
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "origin_main": _git("rev-parse", "origin/main"),
        "status_porcelain": _git(
            "status", "--porcelain", "--untracked-files=all"
        ),
        "package_manifest_sha256": _tracked_content_manifest(ROOT, "darkofit"),
        "support_sha256": {
            name: _sha256_file(ROOT / name)
            for name in EXPECTED_SUPPORT_SHA256
        },
    }


def require_clean_frozen_source() -> dict[str, Any]:
    state = _source_state()
    if state["branch"] != "main":
        raise RuntimeError("formal input-validation campaign requires main")
    if state["head"] != state["origin_main"]:
        raise RuntimeError("formal input-validation campaign requires pushed main")
    if state["status_porcelain"]:
        raise RuntimeError("formal input-validation campaign requires clean source")
    if state["package_manifest_sha256"] != EXPECTED_PACKAGE_MANIFEST:
        raise RuntimeError("DarkoFit package manifest changed")
    if state["support_sha256"] != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("input-validation support files changed")
    if _sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("input-validation protocol changed")
    return state


def schedule() -> tuple[tuple[str, str], ...]:
    return harness.reciprocal_schedule(
        VALIDATED, ASSUME_FINITE, repetitions=TIMING_BLOCKS
    )


def timing_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.shape != (TIMING_BLOCKS,):
        raise RuntimeError(f"timing requires exactly {TIMING_BLOCKS} values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RuntimeError("timing values must be positive and finite")
    median = float(np.median(array))
    iqr = float(np.subtract(*np.percentile(array, [75, 25])))
    return {
        "values_seconds": [float(value) for value in array],
        "minimum_seconds": float(array.min()),
        "median_seconds": median,
        "maximum_seconds": float(array.max()),
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median,
    }


def _cache_stats(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    compiled = [item for item in files if item.suffix in {".nbc", ".nbi"}]
    return {
        "file_count": len(files),
        "compiled_file_count": len(compiled),
        "bytes": int(sum(item.stat().st_size for item in files)),
        "compiled_files": [str(item.relative_to(path)) for item in compiled],
    }


def _warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]


def _archive_identity(model: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="darkofit-input-validation-archive-"
    ) as root:
        path = Path(root) / "model.npz"
        model.save_model(path)
        raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _importance_sha256(model: Any) -> str:
    values = np.ascontiguousarray(
        np.asarray(model.feature_importances_, dtype="<f8")
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def _model_behavior_payload(
    fit_metadata: dict[str, Any],
    prediction_hashes: dict[str, str],
    archive: dict[str, Any],
    feature_importance_sha256: str,
) -> dict[str, Any]:
    return {
        "fit_metadata": fit_metadata,
        "prediction_hashes": prediction_hashes,
        "archive": archive,
        "feature_importance_sha256": feature_importance_sha256,
    }


def _prediction_context(arm: str):
    if arm == ASSUME_FINITE:
        from sklearn import config_context

        return config_context(assume_finite=True)
    return contextlib.nullcontext()


def run_worker(arm: str, data_cache: Path, cache_dir: Path) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown input-validation arm {arm!r}")
    if os.environ.get("DARKOFIT_WARMUP") != "0":
        raise RuntimeError("worker import warmup must be explicitly disabled")
    if Path(os.environ.get("NUMBA_CACHE_DIR", "")) != cache_dir:
        raise RuntimeError("worker Numba cache binding changed")
    before_import = _cache_stats(cache_dir)
    if before_import["file_count"] != 0:
        raise RuntimeError("worker Numba cache was not empty")

    dataset = harness.load_basketball_dataset(data_cache)
    train_indices, test_indices = list(
        creator.creator_cv().split(dataset.X, dataset.y)
    )[EXPECTED_FOLD]
    X_train = dataset.X.iloc[train_indices]
    y_train = dataset.y.iloc[train_indices]
    X_test = dataset.X.iloc[test_indices]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import_started = time.perf_counter_ns()
        import darkofit
        from darkofit import DarkoRegressor
        from darkofit.sklearn_api import _check_predict_input
        from darkofit.tree import _build_histograms_unit_hess_and_best_split

        import_seconds = (time.perf_counter_ns() - import_started) / 1e9
        imported_path = Path(darkofit.__file__).resolve()
        if not imported_path.is_relative_to(ROOT.resolve()):
            raise RuntimeError(
                f"darkofit imported outside the repository: {imported_path}"
            )
        compiled_before_explicit_warmup = bool(
            _build_histograms_unit_hess_and_best_split.signatures
        )
        after_import = _cache_stats(cache_dir)

        warmup_started = time.perf_counter_ns()
        returned_seconds = darkofit.warmup()
        measured_seconds = (time.perf_counter_ns() - warmup_started) / 1e9
        if not math.isclose(
            float(returned_seconds),
            measured_seconds,
            rel_tol=0.05,
            abs_tol=0.05,
        ):
            raise RuntimeError("public warmup returned an inconsistent duration")
        warmup_seconds = float(measured_seconds)
        after_warmup = _cache_stats(cache_dir)

        import numba

        threads_after_warmup = int(numba.get_num_threads())
        model = DarkoRegressor(random_state=creator.RANDOM_STATE)
        fit_started = time.perf_counter_ns()
        model.fit(X_train, y_train)
        fit_seconds = (time.perf_counter_ns() - fit_started) / 1e9

        with _prediction_context(arm):
            validation_started = time.perf_counter_ns()
            checked = _check_predict_input(model, X_test)
            validation_seconds = (
                time.perf_counter_ns() - validation_started
            ) / 1e9
            if checked.shape != X_test.shape:
                raise RuntimeError("validation-only call changed basketball shape")

            predict_started = time.perf_counter_ns()
            fold_prediction = harness.validate_prediction(
                model.predict(X_test), len(X_test)
            )
            predict_seconds = (time.perf_counter_ns() - predict_started) / 1e9

            guardrail = dataset.player_guardrail
            held_prediction = harness.validate_prediction(
                model.predict(guardrail.X_holdout), len(guardrail.X_holdout)
            )
        cold_mask = guardrail.cold_player_mask
        cold_prediction = held_prediction[cold_mask]
        fit_metadata = harness.extract_fit_metadata(model)
        archive = _archive_identity(model)
        feature_importance_sha256 = _importance_sha256(model)

    prediction_hashes = {
        "creator_fold_0": harness.prediction_sha256(fold_prediction),
        "held_team": harness.prediction_sha256(held_prediction),
        "cold_player": harness.prediction_sha256(cold_prediction),
    }
    behavior_payload = _model_behavior_payload(
        fit_metadata,
        prediction_hashes,
        archive,
        feature_importance_sha256,
    )
    return {
        "arm": arm,
        "fold": EXPECTED_FOLD,
        "train_rows": int(len(train_indices)),
        "fold_rows": int(len(test_indices)),
        "held_team_rows": int(len(held_prediction)),
        "cold_player_rows": int(len(cold_prediction)),
        "import_seconds": float(import_seconds),
        "warmup_seconds": warmup_seconds,
        "first_fit_seconds": float(fit_seconds),
        "first_predict_seconds": float(predict_seconds),
        "validation_only_seconds": float(validation_seconds),
        "compiled_before_explicit_warmup": compiled_before_explicit_warmup,
        "threads_after_warmup": threads_after_warmup,
        "cache": {
            "before_import": before_import,
            "after_import": after_import,
            "after_warmup": after_warmup,
            "after_worker": _cache_stats(cache_dir),
        },
        "fit_metadata": fit_metadata,
        "prediction_hashes": prediction_hashes,
        "archive": archive,
        "feature_importance_sha256": feature_importance_sha256,
        "model_behavior_fingerprint_sha256": harness.behavior_fingerprint(
            behavior_payload
        ),
        "warnings": _warning_records(caught),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (*creator.THREAD_LIMIT_ENV_KEYS, "NUMBA_CACHE_DIR")
        },
        "_predictions": {
            "creator_fold_0": [float(value) for value in fold_prediction],
            "held_team": [float(value) for value in held_prediction],
            "cold_player": [float(value) for value in cold_prediction],
        },
    }


def _worker_environment(threads: int, cache_dir: Path) -> dict[str, str]:
    environment = harness.worker_environment(threads)
    environment["DARKOFIT_WARMUP"] = "0"
    environment["NUMBA_CACHE_DIR"] = str(cache_dir)
    return environment


def _decode_worker_result(stdout: str) -> tuple[dict[str, Any], Optional[str]]:
    lines = stdout.splitlines()
    matches = [
        line for line in lines if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "input-validation worker did not emit exactly one result"
        )
    payload = json.loads(matches[0][len(WORKER_RESULT_PREFIX) :])
    chatter = "\n".join(
        line for line in lines if not line.startswith(WORKER_RESULT_PREFIX)
    ).strip()
    return payload, chatter or None


def run_worker_process(
    arm: str,
    *,
    threads: int,
    data_cache: Path,
    temp_root: Path,
    block: int,
    position: int,
) -> dict[str, Any]:
    temp_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(
        tempfile.mkdtemp(
            prefix=f"block{block:02d}-position{position}-{arm}-",
            dir=temp_root,
        )
    )
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-arm",
            arm,
            "--data-cache",
            str(data_cache),
            "--cache-dir",
            str(cache_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_worker_environment(threads, cache_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"input-validation worker {arm!r} failed "
                f"({completed.returncode})\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result, chatter = _decode_worker_result(completed.stdout)
        result["worker_stdout"] = chatter
        result["worker_stderr"] = completed.stderr.strip() or None
        result["fresh_cache_unique"] = True
        return result
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _validate_fit_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "best_iteration": 1000,
        "fitted_tree_count": 1000,
        "resolved_learning_rate": 0.052312,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "linear_leaves_active": False,
        "resolved_thread_count": EXPECTED_THREADS,
        "refit": False,
        "refit_strategy": None,
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"basketball fit metadata {key!r} changed: "
                f"{metadata.get(key)!r}"
            )
    final_fit = metadata.get("final_fit", {})
    if (
        final_fit.get("iterations_requested") != 1000
        or final_fit.get("iterations_attempted") != 1000
        or final_fit.get("rounds_completed") != 1000
        or final_fit.get("rounds_retained") != 1000
        or final_fit.get("stop_reason") != "iteration_limit"
    ):
        raise RuntimeError("basketball final-fit stopping metadata changed")


def analyze(
    *,
    summaries: dict[str, dict[str, dict[str, Any]]],
    paired_predict_ratio: dict[str, Any],
    outputs_exact: bool,
    behavior_fingerprints: set[str],
    archives_exact: bool,
    metadata_exact: bool,
    workers_clean: bool,
    caches_isolated: bool,
) -> dict[str, Any]:
    fit = summaries["first_fit"]
    predict = summaries["first_predict"]
    validated_predict_ratio = (
        predict[VALIDATED]["median_seconds"]
        / predict[ASSUME_FINITE]["median_seconds"]
    )
    gates = {
        "outputs_array_exact": bool(outputs_exact),
        "model_behavior_repeat_exact": len(behavior_fingerprints) == 1,
        "archive_exact": bool(archives_exact),
        "fitted_metadata_exact": bool(metadata_exact),
        "workers_clean": bool(workers_clean),
        "fresh_caches_isolated": bool(caches_isolated),
        "validated_fit_within_budget": (
            fit[VALIDATED]["median_seconds"] <= MAX_VALIDATED_FIT_SECONDS
        ),
        "validated_predict_within_budget": (
            validated_predict_ratio <= MAX_VALIDATED_PREDICT_RATIO
        ),
        "fit_stable": all(
            fit[arm]["iqr_fraction"] <= MAX_FIT_IQR_FRACTION for arm in ARMS
        ),
        "predict_stable": all(
            predict[arm]["iqr_fraction"] <= MAX_PREDICT_IQR_FRACTION
            for arm in ARMS
        ),
        "paired_predict_ratio_stable": (
            paired_predict_ratio["iqr_fraction"]
            <= MAX_PAIRED_PREDICT_RATIO_IQR_FRACTION
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "validated_over_assume_finite_median_predict": validated_predict_ratio,
        "passed": passed,
        "recommendation": (
            "ship_input_validation_layer"
            if passed
            else "close_input_validation_attempt_without_threshold_changes"
        ),
        "model_default_change_authorized": False,
        "broad_quality_claim_authorized": False,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads != EXPECTED_THREADS:
        raise ValueError(
            f"input-validation campaign requires {EXPECTED_THREADS} threads"
        )
    if args.output != DEFAULT_OUTPUT:
        raise ValueError(f"formal output must be {DEFAULT_OUTPUT}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    harness.load_basketball_dataset(args.data_cache)
    source = require_clean_frozen_source()
    run_schedule = schedule()
    timing_values = {
        "first_fit": {arm: [] for arm in ARMS},
        "first_predict": {arm: [] for arm in ARMS},
        "validation_only": {arm: [] for arm in ARMS},
        "warmup": {arm: [] for arm in ARMS},
        "import": {arm: [] for arm in ARMS},
    }
    paired_predict_values = []
    reference_predictions = None
    behavior_fingerprints: set[str] = set()
    outputs_exact = True
    archives_exact = True
    metadata_exact = True
    workers_clean = True
    caches_isolated = True
    repeats = []
    canonical = {}

    for block, order in enumerate(run_schedule):
        block_predict = {}
        for position, arm in enumerate(order):
            if _source_state() != source:
                raise RuntimeError(
                    "DarkoFit source changed during input-validation campaign"
                )
            print(
                f"block {block + 1}/{TIMING_BLOCKS}, "
                f"position {position + 1}: {arm}",
                flush=True,
            )
            result = run_worker_process(
                arm,
                threads=args.threads,
                data_cache=args.data_cache,
                temp_root=args.temp_root,
                block=block,
                position=position,
            )
            _validate_fit_metadata(result["fit_metadata"])
            predictions = {
                name: np.asarray(values, dtype=np.float64)
                for name, values in result.pop("_predictions").items()
            }
            if reference_predictions is None:
                reference_predictions = {
                    name: values.copy() for name, values in predictions.items()
                }
            exact = {
                name: bool(np.array_equal(values, reference_predictions[name]))
                for name, values in predictions.items()
            }
            outputs_exact = outputs_exact and all(exact.values())
            if result["prediction_hashes"] != EXPECTED_PREDICTION_SHA256:
                raise RuntimeError("basketball prediction golden changed")
            if result["cold_player_rows"] != 585:
                raise RuntimeError("basketball cold-player row count changed")
            archives_exact = (
                archives_exact and result["archive"] == EXPECTED_ARCHIVE
            )
            behavior_fingerprints.add(
                result["model_behavior_fingerprint_sha256"]
            )
            metadata_exact = metadata_exact and bool(
                result["fit_metadata"]["best_iteration"] == 1000
                and result["fit_metadata"]["resolved_learning_rate"] == 0.052312
            )
            workers_clean = workers_clean and bool(
                result["threads_after_warmup"] == EXPECTED_THREADS
                and not result["warnings"]
                and result["worker_stdout"] is None
                and result["worker_stderr"] is None
            )
            caches_isolated = caches_isolated and bool(
                result["fresh_cache_unique"]
                and result["cache"]["before_import"]["file_count"] == 0
                and result["cache"]["after_import"]["compiled_file_count"] == 0
                and result["cache"]["after_worker"]["compiled_file_count"] > 0
            )

            for metric, result_key in (
                ("first_fit", "first_fit_seconds"),
                ("first_predict", "first_predict_seconds"),
                ("validation_only", "validation_only_seconds"),
                ("warmup", "warmup_seconds"),
                ("import", "import_seconds"),
            ):
                timing_values[metric][arm].append(float(result[result_key]))
            block_predict[arm] = float(result["first_predict_seconds"])
            canonical.setdefault(arm, result)
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "arm": arm,
                    "import_seconds": result["import_seconds"],
                    "warmup_seconds": result["warmup_seconds"],
                    "first_fit_seconds": result["first_fit_seconds"],
                    "first_predict_seconds": result["first_predict_seconds"],
                    "validation_only_seconds": result["validation_only_seconds"],
                    "prediction_hashes": result["prediction_hashes"],
                    "outputs_array_exact": exact,
                    "archive": result["archive"],
                    "feature_importance_sha256": result[
                        "feature_importance_sha256"
                    ],
                    "model_behavior_fingerprint_sha256": result[
                        "model_behavior_fingerprint_sha256"
                    ],
                    "cache": result["cache"],
                    "warnings": result["warnings"],
                    "worker_stdout": result["worker_stdout"],
                    "worker_stderr": result["worker_stderr"],
                }
            )
            print(
                f"  fit={result['first_fit_seconds']:.3f}s; "
                f"predict={result['first_predict_seconds']:.4f}s; "
                f"validate={result['validation_only_seconds']:.4f}s",
                flush=True,
            )
        paired_predict_values.append(
            block_predict[VALIDATED] / block_predict[ASSUME_FINITE]
        )

    summaries = {
        metric: {
            arm: timing_summary(values)
            for arm, values in by_arm.items()
        }
        for metric, by_arm in timing_values.items()
    }
    paired_predict_ratio = timing_summary(paired_predict_values)
    decision = analyze(
        summaries=summaries,
        paired_predict_ratio=paired_predict_ratio,
        outputs_exact=outputs_exact,
        behavior_fingerprints=behavior_fingerprints,
        archives_exact=archives_exact,
        metadata_exact=metadata_exact,
        workers_clean=workers_clean,
        caches_isolated=caches_isolated,
    )
    payload = {
        "schema_version": 1,
        "campaign": "basketball_input_validation_compliance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "primary_dataset": "basketball",
            "creator_fold": EXPECTED_FOLD,
            "cold_player_guardrail": True,
            "threads": EXPECTED_THREADS,
            "timing_blocks": TIMING_BLOCKS,
            "schedule": [list(order) for order in run_schedule],
            "fresh_numba_cache_per_arm": True,
            "explicit_warmup_outside_timing": True,
            "ctr23_used": False,
            "tabarena_used": False,
        },
        "timing": summaries,
        "paired_predict_ratio": paired_predict_ratio,
        "model_behavior_fingerprints": sorted(behavior_fingerprints),
        "canonical_results": [canonical[arm] for arm in ARMS],
        "timing_repeats": repeats,
        "decision": decision,
    }
    _write_create_only(args.output, payload)
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--data-cache", type=Path, default=harness.DEFAULT_CACHE
    )
    parser.add_argument(
        "--temp-root", type=Path, default=DEFAULT_TEMP_ROOT
    )
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.worker_arm is not None:
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required for worker mode")
        result = run_worker(args.worker_arm, args.data_cache, args.cache_dir)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    if args.cache_dir is not None:
        raise ValueError("--cache-dir is only valid in worker mode")
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
