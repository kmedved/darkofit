#!/usr/bin/env python3
"""Run the frozen fresh-cache basketball warmup campaign."""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CONTROL = "control"
CANDIDATE = "warmup"
ARMS = (CONTROL, CANDIDATE)
WORKER_RESULT_PREFIX = "BASKETBALL_WARMUP_RESULT="
EXPECTED_THREADS = 18
TIMING_BLOCKS = 6
EXPECTED_FOLD = 0
EXPECTED_FOLD_PREDICTION_SHA256 = (
    "6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f"
)
EXPECTED_PACKAGE_MANIFEST = (
    "157a660db4e3f14d3f0cac0bad8d527bdb9268d0b2339285c71e04925c8da1bd"
)
EXPECTED_PROTOCOL_SHA256 = (
    "06a965820b237ce56b31465c002deb3b3eb230a4268e434c7957adeb7705764d"
)
EXPECTED_SUPPORT_SHA256 = {
    "NOTICE": "c245b425c96e7b43a9c8762d6effdb7857659d053978b9c0106084c60ee94ca3",
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
    "tests/test_warmup.py": (
        "eefc7afa503d447d1fb4ba2d6acf9878afbdc69afa7e79e471105335149f993b"
    ),
}
PROTOCOL_PATH = ROOT / "benchmarks/basketball_warmup_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks/basketball_warmup.json"
DEFAULT_TEMP_ROOT = ROOT / ".cache/basketball-warmup"

MAX_WARMUP_SECONDS = 15.0
MAX_FIT_RATIO = 0.70
MAX_PREDICT_RATIO = 0.25
MAX_FIT_IQR_FRACTION = 0.25
MAX_PAIRED_FIT_RATIO_IQR_FRACTION = 0.20
MAX_CANDIDATE_PREDICT_IQR_FRACTION = 0.50


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
        raise RuntimeError("formal warmup campaign requires main")
    if state["head"] != state["origin_main"]:
        raise RuntimeError("formal warmup campaign requires pushed main")
    if state["status_porcelain"]:
        raise RuntimeError("formal warmup campaign requires clean source")
    if state["package_manifest_sha256"] != EXPECTED_PACKAGE_MANIFEST:
        raise RuntimeError("DarkoFit package manifest changed")
    if state["support_sha256"] != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("warmup support files changed")
    if _sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("warmup protocol changed")
    return state


def schedule() -> tuple[tuple[str, str], ...]:
    return harness.reciprocal_schedule(
        CONTROL, CANDIDATE, repetitions=TIMING_BLOCKS
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
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]


def _model_behavior_payload(
    fit_metadata: dict[str, Any],
    prediction_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "fit_metadata": fit_metadata,
        "prediction_hashes": prediction_hashes,
    }


def run_worker(arm: str, data_cache: Path, cache_dir: Path) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown warmup arm {arm!r}")
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
        from darkofit.tree import _build_histograms_unit_hess_and_best_split

        import_seconds = (time.perf_counter_ns() - import_started) / 1e9
        imported_path = Path(darkofit.__file__).resolve()
        if not imported_path.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"darkofit imported outside the repository: {imported_path}")
        compiled_before_explicit_warmup = bool(
            _build_histograms_unit_hess_and_best_split.signatures
        )
        after_import = _cache_stats(cache_dir)

        warmup_seconds = 0.0
        if arm == CANDIDATE:
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
        model = DarkoRegressor(
            random_state=creator.RANDOM_STATE,
            verbose_timing=True,
        )
        fit_started = time.perf_counter_ns()
        model.fit(X_train, y_train)
        fit_seconds = (time.perf_counter_ns() - fit_started) / 1e9

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

    prediction_hashes = {
        "creator_fold_0": harness.prediction_sha256(fold_prediction),
        "held_team": harness.prediction_sha256(held_prediction),
        "cold_player": harness.prediction_sha256(cold_prediction),
    }
    behavior_payload = _model_behavior_payload(
        fit_metadata, prediction_hashes
    )
    result = {
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
    return result


def _worker_environment(threads: int, cache_dir: Path) -> dict[str, str]:
    environment = harness.worker_environment(threads)
    environment["DARKOFIT_WARMUP"] = "0"
    environment["NUMBA_CACHE_DIR"] = str(cache_dir)
    return environment


def _decode_worker_result(stdout: str) -> tuple[dict[str, Any], str | None]:
    lines = stdout.splitlines()
    matches = [
        line for line in lines if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("warmup worker did not emit exactly one result")
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
                f"warmup worker {arm!r} failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
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
                f"basketball fit metadata {key!r} changed: {metadata.get(key)!r}"
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
    paired_fit_ratio: dict[str, Any],
    all_outputs_exact: bool,
    behavior_fingerprints: set[str],
    import_cold: bool,
    state_clean: bool,
    caches_isolated: bool,
) -> dict[str, Any]:
    fit = summaries["first_fit"]
    predict = summaries["first_predict"]
    warmup = summaries["warmup"]
    fit_ratio = (
        fit[CANDIDATE]["median_seconds"] / fit[CONTROL]["median_seconds"]
    )
    predict_ratio = (
        predict[CANDIDATE]["median_seconds"]
        / predict[CONTROL]["median_seconds"]
    )
    gates = {
        "outputs_array_exact": bool(all_outputs_exact),
        "model_behavior_repeat_exact": len(behavior_fingerprints) == 1,
        "ordinary_import_stays_cold": bool(import_cold),
        "caller_state_preserved": bool(state_clean),
        "fresh_caches_isolated": bool(caches_isolated),
        "warmup_within_budget": (
            0.0 < warmup[CANDIDATE]["median_seconds"] <= MAX_WARMUP_SECONDS
            and warmup[CANDIDATE]["maximum_seconds"] <= MAX_WARMUP_SECONDS
        ),
        "first_fit_speedup": fit_ratio <= MAX_FIT_RATIO,
        "first_predict_speedup": predict_ratio <= MAX_PREDICT_RATIO,
        "first_fit_stable": all(
            fit[arm]["iqr_fraction"] <= MAX_FIT_IQR_FRACTION for arm in ARMS
        ),
        "paired_fit_ratio_stable": (
            paired_fit_ratio["iqr_fraction"]
            <= MAX_PAIRED_FIT_RATIO_IQR_FRACTION
        ),
        "candidate_predict_stable": (
            predict[CANDIDATE]["iqr_fraction"]
            <= MAX_CANDIDATE_PREDICT_IQR_FRACTION
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "candidate_over_control_median_first_fit": fit_ratio,
        "candidate_over_control_median_first_predict": predict_ratio,
        "passed": passed,
        "recommendation": (
            "ship_explicit_warmup"
            if passed
            else "close_warmup_attempt_without_threshold_changes"
        ),
        "model_default_change_authorized": False,
        "hidden_import_warmup_authorized": False,
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
        raise ValueError(f"warmup campaign requires {EXPECTED_THREADS} threads")
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
        "warmup": {CANDIDATE: []},
        "import": {arm: [] for arm in ARMS},
    }
    paired_fit_values = []
    reference_predictions = None
    behavior_fingerprints: set[str] = set()
    all_outputs_exact = True
    import_cold = True
    state_clean = True
    caches_isolated = True
    repeats = []
    canonical = {}

    for block, order in enumerate(run_schedule):
        block_fit = {}
        for position, arm in enumerate(order):
            if _source_state() != source:
                raise RuntimeError("DarkoFit source changed during warmup campaign")
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
            all_outputs_exact = all_outputs_exact and all(exact.values())
            if (
                result["prediction_hashes"]["creator_fold_0"]
                != EXPECTED_FOLD_PREDICTION_SHA256
            ):
                raise RuntimeError("basketball fold-0 prediction golden changed")
            if result["cold_player_rows"] != 585:
                raise RuntimeError("basketball cold-player row count changed")
            behavior_fingerprints.add(
                result["model_behavior_fingerprint_sha256"]
            )
            import_cold = import_cold and bool(
                not result["compiled_before_explicit_warmup"]
                and result["cache"]["after_import"]["compiled_file_count"] == 0
            )
            state_clean = state_clean and bool(
                result["threads_after_warmup"] == EXPECTED_THREADS
                and not result["warnings"]
                and result["worker_stdout"] is None
                and result["worker_stderr"] is None
            )
            caches_isolated = caches_isolated and bool(
                result["fresh_cache_unique"]
                and result["cache"]["before_import"]["file_count"] == 0
                and result["cache"]["after_worker"]["compiled_file_count"] > 0
            )

            fit_seconds = float(result["first_fit_seconds"])
            predict_seconds = float(result["first_predict_seconds"])
            timing_values["first_fit"][arm].append(fit_seconds)
            timing_values["first_predict"][arm].append(predict_seconds)
            timing_values["import"][arm].append(float(result["import_seconds"]))
            if arm == CANDIDATE:
                timing_values["warmup"][CANDIDATE].append(
                    float(result["warmup_seconds"])
                )
            block_fit[arm] = fit_seconds
            canonical.setdefault(arm, result)
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "arm": arm,
                    "import_seconds": result["import_seconds"],
                    "warmup_seconds": result["warmup_seconds"],
                    "first_fit_seconds": fit_seconds,
                    "first_predict_seconds": predict_seconds,
                    "prediction_hashes": result["prediction_hashes"],
                    "outputs_array_exact": exact,
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
                f"  fit={fit_seconds:.3f}s; predict={predict_seconds:.4f}s; "
                f"warmup={result['warmup_seconds']:.3f}s",
                flush=True,
            )
        paired_fit_values.append(
            block_fit[CANDIDATE] / block_fit[CONTROL]
        )

    if _source_state() != source:
        raise RuntimeError("DarkoFit source changed during warmup campaign")
    summaries = {
        "first_fit": {
            arm: timing_summary(values)
            for arm, values in timing_values["first_fit"].items()
        },
        "first_predict": {
            arm: timing_summary(values)
            for arm, values in timing_values["first_predict"].items()
        },
        "warmup": {
            CANDIDATE: timing_summary(timing_values["warmup"][CANDIDATE])
        },
        "import": {
            arm: timing_summary(values)
            for arm, values in timing_values["import"].items()
        },
    }
    paired_summary = timing_summary(paired_fit_values)
    decision = analyze(
        summaries=summaries,
        paired_fit_ratio=paired_summary,
        all_outputs_exact=all_outputs_exact,
        behavior_fingerprints=behavior_fingerprints,
        import_cold=import_cold,
        state_clean=state_clean,
        caches_isolated=caches_isolated,
    )
    payload = {
        "schema_version": 1,
        "campaign": "basketball_fresh_worker_warmup",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "primary_dataset": "basketball",
            "creator_fold": EXPECTED_FOLD,
            "cold_player_guardrail": True,
            "ctr23_used": False,
            "threads": EXPECTED_THREADS,
            "timing_blocks": TIMING_BLOCKS,
            "schedule": [list(order) for order in run_schedule],
            "fresh_numba_cache_per_arm": True,
            "import_warmup_environment": "0",
        },
        "runner_sha256": _sha256_file(Path(__file__)),
        "canonical_results": [canonical[arm] for arm in ARMS],
        "timing_repeats": repeats,
        "timing": summaries,
        "paired_first_fit_ratio": paired_summary,
        "model_behavior_fingerprints": sorted(behavior_fingerprints),
        "decision": decision,
    }
    _write_create_only(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("data_cache", "output", "temp_root"):
        path = getattr(args, name)
        setattr(args, name, Path(os.path.abspath(path.expanduser())))
    if args.cache_dir is not None:
        args.cache_dir = Path(os.path.abspath(args.cache_dir.expanduser()))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_arm is not None:
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required for worker mode")
        result = run_worker(args.worker_arm, args.data_cache, args.cache_dir)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.cache_dir is not None:
        raise ValueError("--cache-dir is only valid in worker mode")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
