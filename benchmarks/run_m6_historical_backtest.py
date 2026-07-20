#!/usr/bin/env python3
"""Replay the predeclared historical verdict subset before M6 may rank work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import mean_squared_error

try:
    from benchmark_adapters import build_dataset, split_case
    from campaign_lib.provenance import canonical_json_sha256, file_sha256
    from standing_evidence import (
        M6_BACKTEST_COMPLETE,
        M6_BACKTEST_VERDICTS,
        M6_CONTRACT_FROZEN,
        M6_THREADS,
        contract_payload,
    )
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import build_dataset, split_case
    from benchmarks.campaign_lib.provenance import (
        canonical_json_sha256,
        file_sha256,
    )
    from benchmarks.standing_evidence import (
        M6_BACKTEST_COMPLETE,
        M6_BACKTEST_VERDICTS,
        M6_CONTRACT_FROZEN,
        M6_THREADS,
        contract_payload,
    )


RUNNER_VERSION = "m6-historical-backtest-v1"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
SELECTOR_PREFIX = "M6_SELECTOR_REPLAY_RESULT="
SELECTOR_RANDOM_STATE = 4
SELECTOR_VALIDATION_FRACTION = 0.20
SELECTOR_MIN_RELATIVE_IMPROVEMENT = 0.03
SELECTOR_CASES = (
    ("friedman_numeric", "small"),
    ("friedman_numeric", "medium"),
    ("wide_numeric_reg", "small"),
    ("wide_numeric_reg", "medium"),
    ("categorical_reg", "small"),
    ("categorical_reg", "medium"),
)
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "branch": _git(repository, "branch", "--show-current"),
        "clean": not status,
        "status": status,
    }


def _verdict(mechanism_id: str):
    matches = [
        verdict
        for verdict in M6_BACKTEST_VERDICTS
        if verdict.mechanism_id == mechanism_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique backtest verdict: {mechanism_id}")
    return matches[0]


def validate_sources(
    harness: dict[str, Any],
    fused: dict[str, Any],
    packed: dict[str, Any],
    selector: dict[str, Any],
    chimeraboost_015: dict[str, Any],
) -> None:
    states = {
        "harness": harness,
        "fused": fused,
        "packed": packed,
        "selector": selector,
        "chimeraboost_015": chimeraboost_015,
    }
    dirty = [name for name, state in states.items() if not state["clean"]]
    if dirty:
        raise RuntimeError(f"M6 backtest sources are dirty: {dirty}")
    expected = {
        "fused": _verdict("fused_variable_hessian").candidate_source,
        "packed": _verdict("forest_work_packed_router").candidate_source,
        "selector": _verdict("linear_leaf_selector_3pct").candidate_source,
        "chimeraboost_015": (
            "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
        ),
    }
    for name, expected_head in expected.items():
        if states[name]["head"] != expected_head:
            raise RuntimeError(
                f"{name} source is {states[name]['head']}, expected "
                f"{expected_head}"
            )


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.size == 0
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
    ):
        raise RuntimeError("backtest geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(array))))


def analyze_fused(artifact: dict[str, Any]) -> dict[str, Any]:
    verdict = _verdict("fused_variable_hessian")
    analysis = artifact["analysis"]
    cases = analysis["cases"]
    expected_cases = {"binary_logloss", "weighted_rmse"}
    if set(cases) != expected_cases:
        raise RuntimeError("fused replay cases drifted")
    exact = all(cases[name]["exact"] for name in expected_cases)
    fit_ratios = [
        float(cases[name]["paired_ratios"]["fit_seconds"]["median_ratio"])
        for name in sorted(expected_cases)
    ]
    stability = {
        name: float(
            cases[name]["paired_ratios"]["fit_seconds"]["iqr_over_median"]
        )
        for name in sorted(expected_cases)
    }
    fit_geomean = _geomean(fit_ratios)
    gates = {
        "behavior_and_engagement_exact": exact,
        "fit_geomean_at_most_0_90": fit_geomean <= 0.90,
        "fit_series_stable": all(
            value <= verdict.max_stability_iqr_fraction
            for value in stability.values()
        ),
    }
    disposition = "advance" if all(gates.values()) else "kill"
    return {
        "mechanism_id": verdict.mechanism_id,
        "expected_disposition": verdict.expected_disposition,
        "observed_disposition": disposition,
        "agreement": disposition == verdict.expected_disposition,
        "fit_geomean_ratio": fit_geomean,
        "fit_iqr_over_median": stability,
        "gates": gates,
    }


def analyze_packed(artifact: dict[str, Any]) -> dict[str, Any]:
    verdict = _verdict("forest_work_packed_router")
    cases = artifact["cases"]
    expected_cases = {
        "tiny_127",
        "confirmation_fold",
        "cold_player",
        "held_team",
        "repeated_8192",
        "repeated_100000",
    }
    if set(cases) != expected_cases:
        raise RuntimeError("packed replay cases drifted")
    small = ("confirmation_fold", "cold_player")
    large = ("repeated_8192", "repeated_100000")
    exact = all(
        all(bool(value) for value in cases[name]["exactness"].values())
        for name in expected_cases
    )
    routes = (
        cases["tiny_127"]["candidate_route"] == "serial"
        and all(cases[name]["candidate_route"] == "parallel" for name in small)
        and all(cases[name]["candidate_route"] == "parallel" for name in large)
    )
    small_speedups = {
        name: float(cases[name]["ratios"]["legacy_over_candidate_speedup"])
        for name in small
    }
    large_ratios = {
        name: float(cases[name]["ratios"]["candidate_over_legacy"])
        for name in large
    }
    timing_stability = {}
    for name in sorted(expected_cases):
        timing_stability[name] = max(
            float(summary["iqr_fraction"])
            for scope in ("core_timing", "public_timing")
            for summary in cases[name][scope].values()
        )
    gates = {
        "predictions_exact": exact,
        "observed_routes_exact": routes,
        "small_core_speedup_at_least_2x": all(
            value >= 2.0 for value in small_speedups.values()
        ),
        "large_candidate_over_legacy_at_most_1_10": all(
            value <= 1.10 for value in large_ratios.values()
        ),
        "all_timing_series_stable": all(
            value <= verdict.max_stability_iqr_fraction
            for value in timing_stability.values()
        ),
    }
    disposition = "advance" if all(gates.values()) else "kill"
    return {
        "mechanism_id": verdict.mechanism_id,
        "expected_disposition": verdict.expected_disposition,
        "observed_disposition": disposition,
        "agreement": disposition == verdict.expected_disposition,
        "small_legacy_over_candidate_speedup": small_speedups,
        "large_candidate_over_legacy_ratio": large_ratios,
        "maximum_iqr_fraction_by_case": timing_stability,
        "gates": gates,
    }


def analyze_selector(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict = _verdict("linear_leaf_selector_3pct")
    expected = {
        (dataset, size) for dataset, size in SELECTOR_CASES
    }
    observed = {(row["dataset"], row["size"]) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError("selector replay grid drifted")
    ratios = []
    wins = 0
    for row in rows:
        default = float(row["default_rmse"])
        selector = float(row["selector_rmse"])
        if (
            not math.isfinite(default)
            or not math.isfinite(selector)
            or default <= 0.0
            or selector <= 0.0
        ):
            raise RuntimeError("selector replay RMSE is invalid")
        ratio = selector / default
        row["selector_over_default_rmse_ratio"] = ratio
        ratios.append(ratio)
        wins += int(ratio < 1.0)
    geomean = _geomean(ratios)
    gates = {
        "geomean_rmse_ratio_at_most_0_98": geomean <= 0.98,
        "wins_at_least_4_of_6": wins >= 4,
        "no_cell_ratio_above_1_02": max(ratios) <= 1.02,
        "selection_policy_exact": all(
            bool(row["selected_linear"])
            == (
                float(row["relative_validation_improvement"])
                >= SELECTOR_MIN_RELATIVE_IMPROVEMENT
            )
            for row in rows
        ),
    }
    disposition = "advance" if all(gates.values()) else "kill"
    return {
        "mechanism_id": verdict.mechanism_id,
        "expected_disposition": verdict.expected_disposition,
        "observed_disposition": disposition,
        "agreement": disposition == verdict.expected_disposition,
        "geomean_rmse_ratio": geomean,
        "wins": wins,
        "worst_cell_ratio": max(ratios),
        "selection_count": sum(bool(row["selected_linear"]) for row in rows),
        "gates": gates,
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("selector worker peak RSS is unavailable")
    return value


def _fit_regressor(
    X_train: Any,
    y_train: Any,
    cat_features: list[int],
    *,
    linear_leaves: bool,
    eval_set: Optional[tuple[Any, Any]] = None,
):
    from darkofit import DarkoRegressor

    params = {
        "random_state": SELECTOR_RANDOM_STATE,
        "thread_count": M6_THREADS,
        "linear_leaves": linear_leaves,
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
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=eval_set,
    )
    return model, (time.perf_counter_ns() - started) / 1e9


def selector_worker(payload: dict[str, Any]) -> dict[str, Any]:
    source = Path(payload["selector_source"]).resolve()
    sys.path = [
        entry
        for entry in sys.path
        if entry
        and Path(entry).resolve() not in {REPO_ROOT, source}
    ]
    sys.path.insert(0, str(source))
    spec, X, y, cat_features = build_dataset(
        payload["dataset"], payload["size"], 0
    )
    if spec.task != "regression":
        raise RuntimeError("selector replay received a non-regression case")
    cat_features = list(cat_features or [])
    split = split_case(X, y, spec.task, 0)
    X_train = np.concatenate((split["X_fit"], split["X_val"]), axis=0)
    y_train = np.concatenate((split["y_fit"], split["y_val"]), axis=0)

    default, default_fit_seconds = _fit_regressor(
        X_train,
        y_train,
        cat_features,
        linear_leaves=False,
    )
    default_prediction = np.asarray(
        default.predict(split["X_test"]), dtype=np.float64
    )

    from darkofit.sklearn_api import _make_eval_split

    selection_train, validation, _policy = _make_eval_split(
        X_train,
        y_train,
        SELECTOR_VALIDATION_FRACTION,
        SELECTOR_RANDOM_STATE,
        validation_strategy="weighted_stratified",
    )
    selection_rows = []
    selection_models = {}
    for name, linear in (("constant", False), ("linear", True)):
        model, seconds = _fit_regressor(
            X_train[selection_train],
            y_train[selection_train],
            cat_features,
            linear_leaves=linear,
            eval_set=(X_train[validation], y_train[validation]),
        )
        score = float(model.best_score_)
        if not math.isfinite(score) or score <= 0.0:
            raise RuntimeError("selector validation score is invalid")
        selection_models[name] = model
        selection_rows.append(
            {
                "name": name,
                "validation_rmse": score,
                "fit_seconds": float(seconds),
            }
        )
    constant_score = selection_rows[0]["validation_rmse"]
    linear_score = selection_rows[1]["validation_rmse"]
    improvement = (constant_score - linear_score) / constant_score
    selected_linear = improvement >= SELECTOR_MIN_RELATIVE_IMPROVEMENT
    final, final_seconds = _fit_regressor(
        X_train,
        y_train,
        cat_features,
        linear_leaves=selected_linear,
    )
    selector_prediction = np.asarray(
        final.predict(split["X_test"]), dtype=np.float64
    )
    if (
        default_prediction.shape != selector_prediction.shape
        or default_prediction.shape != (len(split["y_test"]),)
        or not np.isfinite(default_prediction).all()
        or not np.isfinite(selector_prediction).all()
    ):
        raise RuntimeError("selector replay prediction is invalid")
    return {
        "status": "ok",
        "dataset": payload["dataset"],
        "size": payload["size"],
        "train_rows": int(len(y_train)),
        "test_rows": int(len(split["y_test"])),
        "default_rmse": float(
            mean_squared_error(split["y_test"], default_prediction) ** 0.5
        ),
        "selector_rmse": float(
            mean_squared_error(split["y_test"], selector_prediction) ** 0.5
        ),
        "selected_linear": bool(selected_linear),
        "relative_validation_improvement": float(improvement),
        "minimum_relative_improvement": SELECTOR_MIN_RELATIVE_IMPROVEMENT,
        "selection_fits": selection_rows,
        "default_fit_seconds": float(default_fit_seconds),
        "selector_final_fit_seconds": float(final_seconds),
        "worker_peak_rss_bytes": _peak_rss_bytes(),
    }


def _selector_worker_main(payload_path: Path) -> None:
    try:
        result = selector_worker(json.loads(payload_path.read_text()))
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}
    print(
        SELECTOR_PREFIX
        + json.dumps(result, sort_keys=True, allow_nan=False)
    )


def _environment(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in THREAD_ENV_KEYS:
        environment[key] = str(M6_THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_CACHE_DIR": str(cache_dir),
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    label: str,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{label} replay failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _run_selector_cases(
    selector_source: Path,
    temporary: Path,
    environment: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    executions = []
    for dataset, size in SELECTOR_CASES:
        payload_path = temporary / f"selector-{dataset}-{size}.json"
        payload_path.write_text(
            json.dumps(
                {
                    "selector_source": str(selector_source),
                    "dataset": dataset,
                    "size": size,
                },
                sort_keys=True,
            )
        )
        execution = _run_command(
            [
                sys.executable,
                str(RUNNER_PATH),
                "--selector-worker",
                str(payload_path),
            ],
            cwd=REPO_ROOT,
            environment=environment,
            label=f"selector {dataset}/{size}",
        )
        matches = [
            line[len(SELECTOR_PREFIX) :]
            for line in execution["stdout"].splitlines()
            if line.startswith(SELECTOR_PREFIX)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"selector {dataset}/{size} returned invalid worker output"
            )
        row = json.loads(matches[0])
        if row.get("status") != "ok":
            raise RuntimeError(
                f"selector {dataset}/{size} failed:\n{row.get('error')}"
            )
        rows.append(row)
        execution["stdout"] = "\n".join(
            line
            for line in execution["stdout"].splitlines()
            if not line.startswith(SELECTOR_PREFIX)
        ).strip()
        executions.append(execution)
        print(f"ok selector {dataset:22s} {size}", flush=True)
    return rows, executions


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-source", type=Path, required=True)
    parser.add_argument("--packed-source", type=Path, required=True)
    parser.add_argument("--selector-source", type=Path, required=True)
    parser.add_argument("--chimeraboost-015-source", type=Path, required=True)
    parser.add_argument("--basketball-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if not M6_CONTRACT_FROZEN or M6_BACKTEST_COMPLETE:
        raise RuntimeError(
            "M6 replay requires a frozen contract with backtest incomplete"
        )
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    sources = {
        "harness": source_state(REPO_ROOT),
        "fused": source_state(args.fused_source),
        "packed": source_state(args.packed_source),
        "selector": source_state(args.selector_source),
        "chimeraboost_015": source_state(args.chimeraboost_015_source),
    }
    validate_sources(**sources)
    cache = args.basketball_cache.expanduser().resolve()
    if not cache.is_file() or cache.is_symlink():
        raise RuntimeError("pinned basketball cache is unavailable")

    with tempfile.TemporaryDirectory(prefix="darkofit-m6-backtest-") as directory:
        temporary = Path(directory)
        environment = _environment(temporary / "numba-cache")
        fused_output = temporary / "fused.json"
        fused_execution = _run_command(
            [
                sys.executable,
                str(
                    args.fused_source
                    / "benchmarks"
                    / "run_fused_variable_hessian.py"
                ),
                "--output",
                str(fused_output),
            ],
            cwd=args.fused_source,
            environment=environment,
            label="fused variable Hessian",
        )
        fused_artifact = json.loads(fused_output.read_text())
        fused_analysis = analyze_fused(fused_artifact)
        print(
            "fused replay: "
            f"{fused_analysis['observed_disposition']} "
            f"(agreement={fused_analysis['agreement']})",
            flush=True,
        )

        packed_output = temporary / "packed.json"
        packed_execution = _run_command(
            [
                sys.executable,
                str(
                    args.packed_source
                    / "benchmarks"
                    / "run_basketball_packed_prediction.py"
                ),
                "--threads",
                "18",
                "--chimeraboost-repo",
                str(args.chimeraboost_015_source),
                "--cache-path",
                str(cache),
                "--output",
                str(packed_output),
            ],
            cwd=args.packed_source,
            environment=environment,
            label="forest-work packed router",
        )
        packed_artifact = json.loads(packed_output.read_text())
        packed_analysis = analyze_packed(packed_artifact)
        print(
            "packed replay: "
            f"{packed_analysis['observed_disposition']} "
            f"(agreement={packed_analysis['agreement']})",
            flush=True,
        )

        selector_rows, selector_executions = _run_selector_cases(
            args.selector_source,
            temporary,
            environment,
        )
        selector_analysis = analyze_selector(selector_rows)
        print(
            "selector replay: "
            f"{selector_analysis['observed_disposition']} "
            f"(agreement={selector_analysis['agreement']})",
            flush=True,
        )

        sources_after = {
            "harness": source_state(REPO_ROOT),
            "fused": source_state(args.fused_source),
            "packed": source_state(args.packed_source),
            "selector": source_state(args.selector_source),
            "chimeraboost_015": source_state(args.chimeraboost_015_source),
        }
        if sources_after != sources:
            raise RuntimeError("M6 backtest source changed during execution")
        analyses = [fused_analysis, packed_analysis, selector_analysis]
        all_agree = all(item["agreement"] for item in analyses)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_status": "historical_backtest",
            "backtest_complete": all_agree,
            "candidate_ranking_authorized": all_agree,
            "shipping_or_default_claim_authorized": False,
            "contract": contract_payload(),
            "contract_sha256": canonical_json_sha256(contract_payload()),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "sources": sources,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "logical_cpu_count": os.cpu_count(),
            },
            "replays": {
                "fused_variable_hessian": {
                    "execution": fused_execution,
                    "raw_artifact_sha256": file_sha256(fused_output),
                    "raw_artifact": fused_artifact,
                    "analysis": fused_analysis,
                },
                "forest_work_packed_router": {
                    "execution": packed_execution,
                    "raw_artifact_sha256": file_sha256(packed_output),
                    "raw_artifact": packed_artifact,
                    "analysis": packed_analysis,
                },
                "linear_leaf_selector_3pct": {
                    "executions": selector_executions,
                    "rows": selector_rows,
                    "analysis": selector_analysis,
                },
            },
            "summary": {
                "declared_replays": len(analyses),
                "agreements": sum(item["agreement"] for item in analyses),
                "all_agree": all_agree,
                "dispositions": {
                    item["mechanism_id"]: item["observed_disposition"]
                    for item in analyses
                },
            },
        }
        _write_create_only(
            output,
            (
                json.dumps(
                    artifact, indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            ).encode("utf-8"),
        )
    print(f"wrote M6 historical backtest to {output}")
    print(f"artifact sha256: {file_sha256(output)}")
    return output


def main(argv: Optional[list[str]] = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--selector-worker"]:
        if len(arguments) != 2:
            raise SystemExit("--selector-worker requires one payload path")
        _selector_worker_main(Path(arguments[1]))
        return
    run(parse_args(arguments))


if __name__ == "__main__":
    main()
