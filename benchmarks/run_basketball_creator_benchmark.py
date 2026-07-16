#!/usr/bin/env python3
"""Reproduce bbstats' basketball default-regressor comparison.

The primary lane intentionally preserves the creator's narrow protocol:
unweighted, unshuffled 10-fold cross-validated R2 on the training-team rows,
with the folds evaluated through ``n_jobs=-1``.  A separate steady lane uses
the same folds and estimators, but warms one fold and evaluates folds
sequentially so each estimator owns the machine during its fit.

Run from the DarkoFit repository root with the benchmark environment::

    python benchmarks/run_basketball_creator_benchmark.py --lane author
    python benchmarks/run_basketball_creator_benchmark.py --lane steady
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHIMERABOOST_REPO = REPO_ROOT.parent / "chimeraboost"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "basketball-creator-benchmark"

CREATOR_GIST_URL = (
    "https://gist.github.com/bbstats/b9f5c0c60a186f21d0574ad0220789c6"
)
CREATOR_GIST_REVISION = "cbaa9666f632a9891afb8e91959088d944d8c8b2"
CREATOR_SCRIPT_SHA256 = (
    "40011048376dbc1af27c200568c0ba9c7608524c87503b8a5210cf689b98329b"
)
DATA_URL = (
    "https://gist.githubusercontent.com/bbstats/"
    "16d332857c1d9ebbb439b90a55439270/raw/"
    "ab65fac8769cffdbe15221a64ec86d1097a7fe53/"
    "basketball_reference_toy_data.csv"
)
DATA_SHA256 = "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
DATA_BYTES = 2_549_434
X_TRAIN_SHA256 = (
    "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
)
Y_TRAIN_SHA256 = (
    "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf"
)

CHIMERABOOST_BASELINE_REVISION = (
    "29602d3452b1754042006ad2b14bca320c94b4b7"
)
RANDOM_STATE = 4
N_SPLITS = 10
SCORING = "r2"

FEATURES = (
    "3P",
    "3PA",
    "2P",
    "2PA",
    "FT",
    "FTA",
    "ORB",
    "DRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "Age",
)

ARM_ORDER = (
    "darkofit_default",
    "chimeraboost_default",
    "chimeraboost_ensemble5",
    "catboost_default",
)
WORKER_RESULT_PREFIX = "BASKETBALL_BENCHMARK_RESULT="
THREAD_LIMIT_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
EXECUTION_ENV_PREFIXES = (
    "NUMBA_",
    "JOBLIB_",
    "LOKY_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "BLIS_",
    "VECLIB_",
    "NUMEXPR_",
    "TBB_",
    "KMP_",
)
THREAD_ENV_KEYS = (
    "CHIMERABOOST_WARMUP",
    *THREAD_LIMIT_ENV_KEYS,
    "NUMBA_DISABLE_JIT",
    "NUMBA_CPU_NAME",
    "NUMBA_CPU_FEATURES",
    "NUMBA_THREADING_LAYER",
    "NUMBA_THREADING_LAYER_PRIORITY",
    "NUMBA_CACHE_DIR",
    "ENABLE_IPC",
    "JOBLIB_MULTIPROCESSING",
    "JOBLIB_START_METHOD",
    "LOKY_MAX_CPU_COUNT",
    "LOKY_MAX_DEPTH",
    "LOKY_PICKLER",
    "JOBLIB_TEMP_FOLDER",
    "PYTHONHASHSEED",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path = _absolute_lexical_path(path)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlink destination: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_raw_data(cache_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the pinned CSV, refusing content that differs from the creator data."""
    cache_path = _absolute_lexical_path(cache_path)
    if cache_path.is_symlink():
        raise RuntimeError(f"refusing symlink data cache: {cache_path}")
    if cache_path.exists():
        raw = cache_path.read_bytes()
        source = "cache"
    else:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
        source = "network"

    digest = sha256_bytes(raw)
    if len(raw) != DATA_BYTES or digest != DATA_SHA256:
        raise RuntimeError(
            "basketball CSV does not match the frozen creator data: "
            f"bytes={len(raw)}, sha256={digest}"
        )
    if source == "network":
        _atomic_write_bytes(cache_path, raw)

    frame = pd.read_csv(cache_path)
    return frame, {
        "url": DATA_URL,
        "cache_path": str(cache_path),
        "load_source": source,
        "bytes": len(raw),
        "sha256": digest,
        "raw_rows": int(frame.shape[0]),
        "raw_columns": int(frame.shape[1]),
    }


def prepare_creator_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply the gist's transform and alphabetical team holdout exactly."""
    filtered = frame.loc[frame["MP"] > 500].copy()
    filtered["MPG"] = filtered["MP"] / filtered["G"]
    filtered["starter"] = np.where(
        filtered["GS"] / filtered["G"] >= 0.5, 1, 0
    )

    teams = filtered["Tm"].sort_values().drop_duplicates().tolist()
    test_size = len(teams) // 3
    test_teams = teams[:test_size]
    train_teams = [team for team in teams if team not in test_teams]
    train = filtered.loc[filtered["Tm"].isin(train_teams)]
    test = filtered.loc[filtered["Tm"].isin(test_teams)]

    X_train = train.loc[:, FEATURES]
    y_train = train.loc[:, "MPG"]
    x_digest = sha256_bytes(
        np.ascontiguousarray(X_train.to_numpy(dtype="<f8")).tobytes()
    )
    y_digest = sha256_bytes(
        np.ascontiguousarray(y_train.to_numpy(dtype="<f8")).tobytes()
    )
    if x_digest != X_TRAIN_SHA256 or y_digest != Y_TRAIN_SHA256:
        raise RuntimeError(
            "processed creator data fingerprint mismatch: "
            f"X={x_digest}, y={y_digest}"
        )

    metadata = {
        "filter": "MP > 500",
        "filtered_rows": int(filtered.shape[0]),
        "team_count": len(teams),
        "train_team_count": len(train_teams),
        "test_team_count": len(test_teams),
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "features": list(FEATURES),
        "target": "MPG",
        "defined_but_unused_weight": "G",
        "test_teams": test_teams,
        "x_train_sha256": x_digest,
        "y_train_sha256": y_digest,
        "missing_train_feature_cells": int(X_train.isna().sum().sum()),
    }
    return X_train, y_train, metadata


def creator_cv() -> KFold:
    """Return the explicit equivalent of ``cv=10`` for a regressor."""
    return KFold(n_splits=N_SPLITS, shuffle=False)


def fold_fingerprint(X: pd.DataFrame, y: pd.Series) -> tuple[str, list[int]]:
    digest = hashlib.sha256()
    test_sizes = []
    for fold, (train_indices, test_indices) in enumerate(creator_cv().split(X, y)):
        digest.update(np.asarray([fold], dtype="<i8").tobytes())
        digest.update(np.asarray(train_indices, dtype="<i8").tobytes())
        digest.update(np.asarray([-1], dtype="<i8").tobytes())
        digest.update(np.asarray(test_indices, dtype="<i8").tobytes())
        test_sizes.append(int(len(test_indices)))
    return digest.hexdigest(), test_sizes


def _prepend_import_path(path: Path) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    current = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    if resolved not in current:
        os.environ["PYTHONPATH"] = os.pathsep.join([resolved, *current])


def build_estimator(arm: str, chimeraboost_repo: Path, lane: str = "author"):
    _prepend_import_path(REPO_ROOT)
    if arm.startswith("chimeraboost"):
        _prepend_import_path(chimeraboost_repo)

    if arm == "darkofit_default":
        from darkofit import DarkoRegressor

        return DarkoRegressor(random_state=RANDOM_STATE)
    if arm == "chimeraboost_default":
        from chimeraboost import ChimeraBoostRegressor

        return ChimeraBoostRegressor(random_state=RANDOM_STATE)
    if arm == "chimeraboost_ensemble5":
        from chimeraboost import ChimeraBoostRegressor

        return ChimeraBoostRegressor(
            random_state=RANDOM_STATE,
            n_ensembles=5,
        )
    if arm == "catboost_default":
        from catboost import CatBoostRegressor

        # These two switches suppress notebook/file noise only.  All training
        # defaults, including the creator's random_state=4, remain unchanged.
        return CatBoostRegressor(
            random_state=RANDOM_STATE,
            thread_count=(1 if lane == "author" else max(1, os.cpu_count() or 1)),
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown benchmark arm: {arm}")


def _assert_estimator_source(arm: str, chimeraboost_repo: Path) -> None:
    if arm == "darkofit_default":
        package_name = "darkofit"
        repository = REPO_ROOT
    elif arm.startswith("chimeraboost"):
        package_name = "chimeraboost"
        repository = chimeraboost_repo
    else:
        return

    module = importlib.import_module(package_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{package_name} module has no source file")
    repository_top = Path(
        _git_output(repository, "rev-parse", "--show-toplevel")
    ).resolve()
    resolved_module = Path(module_file).resolve()
    if not resolved_module.is_relative_to(repository_top):
        raise RuntimeError(
            f"{package_name} imported from {resolved_module}, outside the "
            f"attested checkout {repository_top}"
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _module_details(estimator) -> dict[str, Any]:
    package_name = estimator.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package_name)
    try:
        distribution_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "package": package_name,
        "module_version": getattr(module, "__version__", None),
        "distribution_version": distribution_version,
        "module_file": str(Path(module.__file__).resolve()),
        "estimator_class": (
            f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"
        ),
    }


def run_worker(
    arm: str,
    lane: str,
    cache_path: Path,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    frame, raw_metadata = load_raw_data(cache_path)
    X, y, processed_metadata = prepare_creator_data(frame)
    estimator = build_estimator(arm, chimeraboost_repo, lane)
    _assert_estimator_source(arm, chimeraboost_repo)
    cv_jobs = -1 if lane == "author" else 1

    warmup_seconds = None
    if lane == "steady":
        train_indices, test_indices = next(creator_cv().split(X, y))
        warmup_model = clone(estimator)
        warmup_started = time.perf_counter_ns()
        warmup_model.fit(X.iloc[train_indices], y.iloc[train_indices])
        warmup_model.predict(X.iloc[test_indices])
        warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9

    started = time.perf_counter_ns()
    scores = cross_val_score(
        estimator,
        X,
        y,
        scoring=SCORING,
        cv=creator_cv(),
        n_jobs=cv_jobs,
        error_score="raise",
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9

    return {
        "arm": arm,
        "lane": lane,
        "cv_jobs": cv_jobs,
        "scoring": SCORING,
        "fold_scores": [float(score) for score in scores],
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
        "wall_seconds": float(elapsed_seconds),
        "warmup_seconds_outside_timing": warmup_seconds,
        "estimator_params": _jsonable(estimator.get_params(deep=False)),
        "implementation": _module_details(estimator),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
    }


def _git_output(repo: Path, *args: str, check: bool = True) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        if check:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}"
            )
        return None
    return completed.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    status_text = _git_output(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ) or ""
    remotes = {}
    remote_text = _git_output(repo, "remote", "-v", check=False) or ""
    for line in remote_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "(fetch)":
            remotes[parts[0]] = sanitize_git_remote(parts[1])
    refs = {}
    for ref in ("origin/main", "upstream/main"):
        value = _git_output(repo, "rev-parse", "--verify", ref, check=False)
        if value:
            refs[ref] = value
    return {
        "path": str(repo),
        "head": _git_output(repo, "rev-parse", "HEAD"),
        "branch": _git_output(repo, "branch", "--show-current"),
        "clean": not bool(status_text),
        "status": status_text.splitlines(),
        "describe": _git_output(repo, "describe", "--tags", "--always", check=False),
        "remotes": remotes,
        "tracked_main_refs": refs,
    }


def sanitize_git_remote(value: str) -> str:
    """Remove credentials and query secrets from a git remote URL."""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    if "::" in value:
        helper = value.split("::", 1)[0]
        return f"{helper}::<redacted>"
    if "@" in value and ":" in value.rsplit("@", 1)[-1]:
        return value.rsplit("@", 1)[-1]
    return value


def _machine_details() -> dict[str, Any]:
    cpu_brand = platform.processor() or None
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            cpu_brand = completed.stdout.strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu_brand,
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "numba",
        "catboost",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _worker_command(args: argparse.Namespace, arm: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--lane",
        args.lane,
        "--data-cache",
        str(args.data_cache),
        "--chimeraboost-repo",
        str(args.chimeraboost_repo),
    ]


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key == "ENABLE_IPC" or key.startswith(EXECUTION_ENV_PREFIXES):
            environment.pop(key)
    # The upstream package can warm Numba kernels at import time.  Disable the
    # opt-in hook explicitly so an inherited shell setting cannot contaminate
    # or overlap the author-lane timer.
    environment["CHIMERABOOST_WARMUP"] = "0"
    inner_threads = 1 if args.lane == "author" else max(1, os.cpu_count() or 1)
    for key in THREAD_LIMIT_ENV_KEYS:
        environment[key] = str(inner_threads)
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["ENABLE_IPC"] = "1"
    environment["JOBLIB_MULTIPROCESSING"] = "1"
    environment["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count() or 1))
    environment["LOKY_MAX_DEPTH"] = "10"
    environment["LOKY_PICKLER"] = "cloudpickle"
    environment["PYTHONHASHSEED"] = "0"
    for key in (
        "NUMBA_CPU_NAME",
        "NUMBA_CPU_FEATURES",
        "NUMBA_THREADING_LAYER",
        "NUMBA_CACHE_DIR",
        "JOBLIB_START_METHOD",
        "JOBLIB_TEMP_FOLDER",
    ):
        environment.pop(key, None)
    source_paths = [str(REPO_ROOT.resolve())]
    if arm.startswith("chimeraboost"):
        source_paths.insert(0, str(args.chimeraboost_repo.resolve()))
    existing = environment.get("PYTHONPATH")
    if existing:
        source_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)

    completed = subprocess.run(
        _worker_command(args, arm),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    )
    result_lines = [
        line for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(result_lines) != 1:
        raise RuntimeError(
            f"benchmark worker {arm!r} failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(result_lines[0][len(WORKER_RESULT_PREFIX):])
    extra_stdout = [
        line for line in completed.stdout.splitlines()
        if not line.startswith(WORKER_RESULT_PREFIX)
    ]
    result["worker_stdout"] = "\n".join(extra_stdout).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    darkofit = git_state(REPO_ROOT)
    chimeraboost = git_state(args.chimeraboost_repo)
    if not args.allow_dirty_source:
        dirty = [
            name for name, state in (
                ("DarkoFit", darkofit),
                ("ChimeraBoost", chimeraboost),
            ) if not state["clean"]
        ]
        if dirty:
            raise RuntimeError(
                "refusing to benchmark dirty source trees: " + ", ".join(dirty)
            )
    if (
        not args.allow_chimeraboost_drift
        and chimeraboost["head"] != CHIMERABOOST_BASELINE_REVISION
    ):
        raise RuntimeError(
            "ChimeraBoost source drifted from the frozen comparator: "
            f"expected {CHIMERABOOST_BASELINE_REVISION}, "
            f"found {chimeraboost['head']}"
        )
    return {"darkofit": darkofit, "chimeraboost": chimeraboost}


def _assert_sources_unchanged(
    expected: dict[str, Any],
    observed: dict[str, Any],
    *,
    boundary: str,
) -> None:
    identity_fields = ("path", "head", "branch", "clean", "status")
    for name in ("darkofit", "chimeraboost"):
        changed = [
            field for field in identity_fields
            if expected[name][field] != observed[name][field]
        ]
        if changed:
            raise RuntimeError(
                f"{name} source changed {boundary}; mismatched fields: "
                + ", ".join(changed)
            )


def _baseline_eligibility(
    args: argparse.Namespace,
    sources: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if args.lane != "author":
        reasons.append("lane_not_author")
    if tuple(args.arms) != ARM_ORDER:
        reasons.append("incomplete_or_reordered_arms")
    if args.allow_dirty_source:
        reasons.append("dirty_source_override_enabled")
    if args.allow_chimeraboost_drift:
        reasons.append("chimeraboost_drift_override_enabled")
    if not sources["darkofit"]["clean"] or not sources["chimeraboost"]["clean"]:
        reasons.append("source_not_clean")
    if sources["chimeraboost"]["head"] != CHIMERABOOST_BASELINE_REVISION:
        reasons.append("chimeraboost_revision_mismatch")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "expected_chimeraboost_revision": CHIMERABOOST_BASELINE_REVISION,
        "overrides": {
            "allow_dirty_source": bool(args.allow_dirty_source),
            "allow_chimeraboost_drift": bool(args.allow_chimeraboost_drift),
        },
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    sources = _validate_sources(args)
    frame, raw_metadata = load_raw_data(args.data_cache)
    X, y, processed_metadata = prepare_creator_data(frame)
    cv_digest, fold_test_sizes = fold_fingerprint(X, y)

    results = []
    for arm in args.arms:
        _assert_sources_unchanged(
            sources,
            _validate_sources(args),
            boundary=f"before {arm}",
        )
        print(f"running {arm} ({args.lane} lane)...", flush=True)
        result = _run_worker_process(args, arm)
        _assert_sources_unchanged(
            sources,
            _validate_sources(args),
            boundary=f"while running {arm}",
        )
        results.append(result)
        print(
            f"  mean R2={result['mean_r2']:.12f}, "
            f"wall={result['wall_seconds']:.2f}s",
            flush=True,
        )

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_eligibility": _baseline_eligibility(args, sources),
        "protocol": {
            "name": "bbstats_basketball_creator_default_regressor_benchmark",
            "creator_gist_url": CREATOR_GIST_URL,
            "creator_gist_revision": CREATOR_GIST_REVISION,
            "creator_script_sha256": CREATOR_SCRIPT_SHA256,
            "lane": args.lane,
            "scoring": SCORING,
            "cv": {
                "kind": "KFold",
                "n_splits": N_SPLITS,
                "shuffle": False,
                "n_jobs": -1 if args.lane == "author" else 1,
                "fold_fingerprint_sha256": cv_digest,
                "fold_test_sizes": fold_test_sizes,
            },
            "inner_thread_limit": (
                1 if args.lane == "author" else max(1, os.cpu_count() or 1)
            ),
            "warmup": (
                "none; preserves the creator call shape"
                if args.lane == "author"
                else "one full first-fold fit and predict per arm outside timing"
            ),
            "random_state": RANDOM_STATE,
            "weights_used": False,
            "held_out_teams_scored": False,
        },
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
        "sources": sources,
        "environment": {
            "machine": _machine_details(),
            "dependencies": _dependency_versions(),
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
        },
        "arms": list(args.arms),
        "results": results,
    }
    _atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("author", "steady"), default="author")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_ORDER,
        default=list(ARM_ORDER),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON artifact path (default: .cache/basketball-creator-benchmark)",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=DEFAULT_CACHE / "basketball_reference_toy_data.csv",
    )
    parser.add_argument(
        "--chimeraboost-repo",
        type=Path,
        default=DEFAULT_CHIMERABOOST_REPO,
    )
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--allow-chimeraboost-drift", action="store_true")
    parser.add_argument("--worker-arm", choices=ARM_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = DEFAULT_CACHE / f"results-{args.lane}.json"
    args.output = _absolute_lexical_path(args.output)
    args.data_cache = _absolute_lexical_path(args.data_cache)
    args.chimeraboost_repo = args.chimeraboost_repo.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_arm:
        result = run_worker(
            args.worker_arm,
            args.lane,
            args.data_cache,
            args.chimeraboost_repo,
        )
        print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
