#!/usr/bin/env python3
"""Compare T7b control/candidate state on paths that must remain exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import run_t7b_automatic_depth_v1 as campaign
except ImportError:  # pragma: no cover
    from benchmarks import run_t7b_automatic_depth_v1 as campaign


IDENTITY = "t7b-automatic-scalar-rmse-depth-v1-invariants-20260722"
WORKER_PREFIX = "T7B_DEPTH_INVARIANTS="
RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
THREADS = 2
NOOP_CASES = (
    "explicit_catboost_rmse",
    "literal_auto_catboost_rmse",
    "catboost_classifier_default",
    "catboost_mae_default",
    "lightgbm_rmse_default",
    "hybrid_rmse_default",
    "depthwise_rmse_default",
)
ENGAGED_CASES = (
    "default_low_density",
    "default_middle_density",
    "default_high_density",
)
ALL_CASES = NOOP_CASES + ENGAGED_CASES
DEPTH_RULE = "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _logical_booster_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="darkofit-t7b-depth-state-") as temp:
        path = Path(temp) / "booster.npz"
        model.model_.save_model(path)
        with np.load(path, allow_pickle=False) as archive:
            for name in sorted(archive.files):
                value = np.ascontiguousarray(archive[name])
                digest.update(name.encode())
                digest.update(value.dtype.str.encode())
                digest.update(_canonical_json(list(value.shape)))
                digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _dataset(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case == "default_middle_density":
        n_samples, n_features = 200, 2
    elif case == "default_high_density":
        n_samples, n_features = 2_500, 1
    else:
        n_samples, n_features = 200, 8
    rng = np.random.default_rng(20_260_722 + n_features)
    X = rng.normal(size=(n_samples, n_features))
    y = 1.7 * np.sin(X[:, 0])
    if n_features > 1:
        y = y - 0.8 * X[:, 1]
    if n_features > 3:
        y = y + 0.4 * X[:, 2] * X[:, 3]
    y = y + rng.normal(0.0, 0.2, size=len(X))
    labels = (y > np.median(y)).astype(np.int64)
    return X, y, labels


def _case_model(case: str):
    from darkofit import DarkoClassifier, DarkoRegressor

    common = {
        "iterations": 6,
        "learning_rate": 0.1,
        "early_stopping": False,
        "ordered_boosting": False,
        "thread_count": THREADS,
        "random_state": 17,
        "diagnostic_warnings": "never",
    }
    if case == "explicit_catboost_rmse":
        return DarkoRegressor(
            **common, tree_mode="catboost", depth=6, l2_leaf_reg="auto"
        )
    if case == "literal_auto_catboost_rmse":
        return DarkoRegressor(
            **common, tree_mode="catboost", depth="auto", l2_leaf_reg="auto"
        )
    if case == "catboost_classifier_default":
        return DarkoClassifier(
            **common, tree_mode="catboost", l2_leaf_reg="auto"
        )
    if case == "catboost_mae_default":
        return DarkoRegressor(
            **common,
            tree_mode="catboost",
            loss="MAE",
            l2_leaf_reg="auto",
        )
    if case == "lightgbm_rmse_default":
        return DarkoRegressor(
            **common,
            tree_mode="lightgbm",
            num_leaves=15,
            l2_leaf_reg="auto",
        )
    if case == "hybrid_rmse_default":
        return DarkoRegressor(
            **common,
            tree_mode="hybrid",
            num_leaves=15,
            l2_leaf_reg="auto",
        )
    if case == "depthwise_rmse_default":
        return DarkoRegressor(
            **common, tree_mode="depthwise", l2_leaf_reg="auto"
        )
    if case in ENGAGED_CASES:
        return DarkoRegressor(
            **common, tree_mode="catboost", l2_leaf_reg="auto"
        )
    raise RuntimeError(f"unknown invariant case: {case}")


def _activate_source(source: Path) -> None:
    source = source.resolve()
    for name in tuple(sys.modules):
        if name == "darkofit" or name.startswith("darkofit."):
            del sys.modules[name]
    sys.path = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() not in {ROOT, source}
    ]
    sys.path.insert(0, str(source))


def _worker(source: Path) -> dict[str, Any]:
    import numba

    _activate_source(source)
    import darkofit

    module_path = Path(darkofit.__file__).resolve()
    try:
        module_path.relative_to(source.resolve())
    except ValueError as exc:
        raise RuntimeError("worker imported DarkoFit outside its source") from exc
    ambient_before = int(numba.get_num_threads())
    cases = {}
    for case in ALL_CASES:
        X, y, labels = _dataset(case)
        X_test = X[-32:].copy()
        model = _case_model(case)
        target = labels if case == "catboost_classifier_default" else y
        model.fit(X, target)
        prediction = (
            model.predict_proba(X_test)
            if case == "catboost_classifier_default"
            else model.predict(X_test)
        )
        core = model.model_
        cases[case] = {
            "prediction_sha256": _array_sha256(prediction),
            "logical_booster_sha256": _logical_booster_sha256(model),
            "requested_depth": core._depth_input,
            "resolved_depth": int(core.depth),
            "resolved_l2_leaf_reg": float(
                getattr(core, "l2_leaf_reg_", core.l2_leaf_reg)
            ),
            "auto_structure": core.auto_params_.get("auto_structure"),
            "tree_count": len(core.trees_),
        }
    ambient_after = int(numba.get_num_threads())
    if ambient_after != ambient_before:
        raise RuntimeError("invariant worker leaked its Numba thread mask")
    return {
        "source": campaign.source_state(source),
        "implementation_path": str(module_path),
        "ambient_numba_threads_before": ambient_before,
        "ambient_numba_threads_after": ambient_after,
        "cases": cases,
    }


def _worker_environment(cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    prefixes = ("NUMBA_", "OMP_", "KMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_")
    for key in tuple(environment):
        if key.startswith("PYTHON") or key.startswith(prefixes):
            environment.pop(key)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "MKL_DYNAMIC": "FALSE",
            "NUMBA_CACHE_DIR": str(cache),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(THREADS),
            "OMP_THREAD_LIMIT": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "VECLIB_MAXIMUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _run_worker(source: Path, *, cache: Path) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--worker", str(source.resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(cache),
    )
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in process.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if process.returncode or len(matches) != 1:
        raise RuntimeError(
            "T7b depth invariant worker failed:\n"
            + (process.stderr.strip() or process.stdout.strip())
        )
    return json.loads(matches[0])


def analyze(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if set(control["cases"]) != set(ALL_CASES) or set(candidate["cases"]) != set(
        ALL_CASES
    ):
        raise RuntimeError("T7b depth invariant cases drifted")
    comparisons = {}
    for case in NOOP_CASES:
        left = control["cases"][case]
        right = candidate["cases"][case]
        exact_prediction = left["prediction_sha256"] == right["prediction_sha256"]
        exact_state = left["logical_booster_sha256"] == right["logical_booster_sha256"]
        if not exact_prediction or not exact_state:
            raise RuntimeError(f"T7b depth no-op invariant changed: {case}")
        comparisons[case] = {
            "prediction_exact": exact_prediction,
            "fitted_state_exact": exact_state,
            "resolved_depth": right["resolved_depth"],
            "resolved_l2_leaf_reg": right["resolved_l2_leaf_reg"],
        }
    expected = {
        "default_low_density": (4, 200.0, 8, 25.0, "low_density"),
        "default_middle_density": (6, 200.0, 2, 100.0, "middle_density"),
        "default_high_density": (8, 2_500.0, 1, 2_500.0, "high_density"),
    }
    engagement = {}
    for case, (depth, n_eff, n_features, density, branch) in expected.items():
        left = control["cases"][case]
        right = candidate["cases"][case]
        structure = right["auto_structure"]
        resolved = structure["resolved"]["depth"]
        policy = structure["candidates"]["depth"]
        if (
            left["resolved_depth"] != 6
            or right["requested_depth"] is not None
            or right["resolved_depth"] != depth
            or right["resolved_l2_leaf_reg"] != 3.0
            or resolved
            != {"input": None, "resolved": depth, "source": "auto"}
            or policy.get("rule") != DEPTH_RULE
            or policy.get("branch") != branch
            or float(policy.get("n_eff")) != n_eff
            or int(policy.get("input_feature_count")) != n_features
            or float(policy.get("effective_rows_per_feature")) != density
            or float(policy.get("low_threshold")) != 100.0
            or float(policy.get("high_threshold")) != 2_500.0
        ):
            raise RuntimeError(f"T7b depth engagement invariant failed: {case}")
        engagement[case] = {
            "control_depth": left["resolved_depth"],
            "candidate_depth": right["resolved_depth"],
            "n_eff": n_eff,
            "input_feature_count": n_features,
            "effective_rows_per_feature": density,
            "branch": branch,
            "l2_leaf_reg": right["resolved_l2_leaf_reg"],
        }
    return {
        "all_noop_cases_exact": True,
        "all_depth_branches_engaged": True,
        "comparisons": comparisons,
        "engagement": engagement,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("T7b depth invariant output must be outside the harness")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"T7b depth invariant output is create-only: {output}")
    sources = campaign.validate_sources(args.control, args.candidate)
    with tempfile.TemporaryDirectory(prefix="darkofit-t7b-depth-invariants-") as temp:
        temp_path = Path(temp)
        control = _run_worker(args.control, cache=temp_path / "control-cache")
        candidate = _run_worker(
            args.candidate, cache=temp_path / "candidate-cache"
        )
    after = campaign.validate_sources(args.control, args.candidate)
    if after != sources:
        raise RuntimeError("T7b depth source state changed during invariant probe")
    result = {
        "schema_version": 1,
        "identity": IDENTITY,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_outcomes_inspected": False,
        "sources": sources,
        "bindings": {
            "contract_sha256": campaign.file_sha256(campaign.CONTRACT_PATH),
            "campaign_runner_sha256": campaign.file_sha256(campaign.RUNNER_PATH),
            "invariant_runner_sha256": campaign.file_sha256(RUNNER_PATH),
        },
        "control": control,
        "candidate": candidate,
        "analysis": analyze(control, candidate),
    }
    _write_create_only(output, result)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args(argv)
    if args.worker is None and any(
        value is None for value in (args.control, args.candidate, args.output)
    ):
        parser.error("main mode requires --control, --candidate, and --output")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker is not None:
        print(
            WORKER_PREFIX
            + json.dumps(_worker(parsed.worker), allow_nan=False, sort_keys=True)
        )
        raise SystemExit(0)
    raise SystemExit(0 if run(parsed) else 1)
