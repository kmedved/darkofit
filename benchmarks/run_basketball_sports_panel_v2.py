#!/usr/bin/env python3
"""Run the frozen T10 basketball sports automatic-policy campaign."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import build_basketball_sports_panel_v2 as panel_builder  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_basketball_sports_panel as base  # noqa: E402


CONTROL = "darkofit_control"
CANDIDATE = "darkofit_sports_oob_ensemble5"
CHIMERABOOST = "chimeraboost_0_15_0"
CATBOOST = "catboost_1_2_10"
ARM_ORDER = (CONTROL, CANDIDATE, CHIMERABOOST, CATBOOST)
BLOCK_ORDERS = (
    (CONTROL, CANDIDATE, CHIMERABOOST, CATBOOST),
    (CATBOOST, CHIMERABOOST, CANDIDATE, CONTROL),
    (CANDIDATE, CONTROL, CATBOOST, CHIMERABOOST),
)

EXPECTED_THREADS = 18
EXPECTED_BRANCH = "codex/product-offense"
EXPECTED_CHIMERABOOST_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_CATBOOST_VERSION = "1.2.10"
PROTOCOL_PATH = panel_builder.PROTOCOL_PATH
MANIFEST_PATH = panel_builder.DEFAULT_MANIFEST
DEFAULT_CACHE = panel_builder.DEFAULT_CACHE
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_sports_panel_v2_raw.json"
DEFAULT_CHIMERABOOST_REPO = REPO_ROOT.parent / "chimeraboost"
WORKER_RESULT_PREFIX = "BASKETBALL_SPORTS_PANEL_V2_RESULT="


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _json_sha256(value: Any) -> str:
    return base._json_sha256(value)


def _atomic_create(path: Path, value: bytes) -> None:
    base._atomic_create(path, value)


def load_panel(
    cache_path: Path,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["processed_panel"]
    if not cache_path.is_file() or cache_path.is_symlink():
        raise RuntimeError(f"sports panel 2 cache is unavailable: {cache_path}")
    if (
        cache_path.stat().st_size != expected["bytes"]
        or _sha256(cache_path) != expected["sha256"]
    ):
        raise RuntimeError("sports panel 2 cache differs from its manifest")
    frame = pd.read_csv(cache_path)
    columns = [
        *panel_builder.IDENTITY_COLUMNS,
        *panel_builder.FEATURE_COLUMNS,
        *panel_builder.TARGET_COLUMNS,
    ]
    if frame.columns.tolist() != columns or len(frame) != expected["rows"]:
        raise RuntimeError("sports panel 2 shape differs from its manifest")
    identities = frame.loc[:, list(panel_builder.IDENTITY_COLUMNS)].values.tolist()
    if _json_sha256(identities) != expected["identities_sha256"]:
        raise RuntimeError("sports panel 2 identities differ from its manifest")
    values = frame.loc[
        :, [*panel_builder.FEATURE_COLUMNS, *panel_builder.TARGET_COLUMNS]
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("sports panel 2 contains non-finite values")
    return frame


def build_estimator(arm: str, chimeraboost_repo: Path):
    if arm in {CONTROL, CANDIDATE}:
        from darkofit import DarkoRegressor

        return DarkoRegressor(
            random_state=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
            n_ensembles=5 if arm == CANDIDATE else 1,
            ensemble_bootstrap="rows",
            ensemble_shared_preprocessing=True,
            diagnostic_warnings="never",
        )
    if arm == CHIMERABOOST:
        base._prepend_import_path(chimeraboost_repo)
        from chimeraboost import ChimeraBoostRegressor

        return ChimeraBoostRegressor(
            random_state=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
        )
    if arm == CATBOOST:
        from catboost import CatBoostRegressor

        if importlib.metadata.version("catboost") != EXPECTED_CATBOOST_VERSION:
            raise RuntimeError("CatBoost version differs from the frozen runner")
        return CatBoostRegressor(
            random_seed=panel_builder.RANDOM_STATE,
            thread_count=EXPECTED_THREADS,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown basketball sports panel 2 arm: {arm}")


def _implementation(model: Any) -> dict[str, Any]:
    return base._implementation(model)


def _assert_implementation_source(
    model: Any,
    arm: str,
    chimeraboost_repo: Path,
) -> None:
    if arm == CATBOOST:
        return
    expected = REPO_ROOT if arm in {CONTROL, CANDIDATE} else chimeraboost_repo
    package = model.__class__.__module__.split(".", 1)[0]
    module_path = Path(importlib.import_module(package).__file__).resolve()
    if not module_path.is_relative_to(expected.resolve()):
        raise RuntimeError(f"{arm} imported outside the frozen source checkout")


def extract_metadata(model: Any, arm: str) -> dict[str, Any]:
    if arm == CANDIDATE:
        members = [harness.extract_fit_metadata(member) for member in model.estimators_]
        if len(members) != 5:
            raise RuntimeError("sports ensemble did not fit five members")
        return {
            "kind": "oob_ensemble",
            "member_count": len(members),
            "members": members,
            "ensemble": model.ensemble_metadata_,
            "resolved_thread_count": int(members[0]["resolved_thread_count"]),
            "fitted_tree_count": int(sum(row["fitted_tree_count"] for row in members)),
        }
    if arm == CONTROL:
        return {"kind": "single", **harness.extract_fit_metadata(model)}
    if arm == CHIMERABOOST:
        return {"kind": "single", **base._chimera_metadata(model)}
    return {"kind": "single", **base._catboost_metadata(model)}


def _prediction_sha256(values: np.ndarray) -> str:
    return base._prediction_sha256(values)


def _fit_predict(
    arm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    model = build_estimator(arm, chimeraboost_repo)
    _assert_implementation_source(model, arm, chimeraboost_repo)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(X_test),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"{arm} returned invalid panel 2 predictions")
    metadata = extract_metadata(model, arm)
    if int(metadata["resolved_thread_count"]) != EXPECTED_THREADS:
        raise RuntimeError(f"{arm} did not resolve {EXPECTED_THREADS} threads")
    if arm == CONTROL and metadata["fitted_tree_count"] != 1000:
        raise RuntimeError("DarkoFit control changed its fixed default horizon")
    if arm == CANDIDATE:
        ensemble = metadata["ensemble"]
        if (
            ensemble["bootstrap"] != "rows"
            or ensemble["member_count"] != 5
            or not ensemble["oob_early_stopping"]
            or ensemble["shared_preprocessing"] != "numeric_target_free"
        ):
            raise RuntimeError("sports ensemble fitted outside its frozen route")
    return {
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": _prediction_sha256(prediction),
        "fit_metadata": metadata,
    }


def _score_view(y_true: pd.Series, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(y_true, dtype=np.float64)
    if values.shape != prediction.shape or len(values) < 2:
        raise RuntimeError("sports panel 2 view has an invalid shape")
    return {
        "rows": int(len(values)),
        "rmse": float(mean_squared_error(values, prediction) ** 0.5),
        "r2": float(r2_score(values, prediction)),
        "target_sha256": _prediction_sha256(values),
        "prediction_sha256": _prediction_sha256(prediction),
    }


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": result["arm"],
        "implementation": result["implementation"],
        "cells": [
            {
                "season": cell["season"],
                "target": cell["target"],
                "primary": cell["primary"],
                "folds": [
                    {
                        "fold": row["fold"],
                        "train_indices": row["train_indices"],
                        "test_indices": row["test_indices"],
                        "prediction_sha256": row["prediction_sha256"],
                        "fit_metadata": row["fit_metadata"],
                    }
                    for row in cell["folds"]
                ],
                "guardrail": {
                    "scores": cell["guardrail"]["scores"],
                    "prediction_sha256": cell["guardrail"]["prediction_sha256"],
                    "fit_metadata": cell["guardrail"]["fit_metadata"],
                },
            }
            for cell in result["cells"]
        ],
    }


def _season_views(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
    split = manifest["split"]["seasons"][str(season)]
    held = frozenset(split["held_teams"])
    primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
    holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
    train_players = frozenset(primary["bref_id"].astype(str))
    seen = holdout["bref_id"].astype(str).isin(train_players).to_numpy()
    if (
        len(primary) != split["primary_rows"]
        or len(holdout) != split["held_team_rows"]
        or int(np.sum(seen)) != split["seen_player_rows"]
    ):
        raise RuntimeError("sports panel 2 split differs from its manifest")
    return primary, holdout, seen, split["folds"]


def run_worker(
    arm: str,
    cache_path: Path,
    manifest_path: Path,
    chimeraboost_repo: Path,
) -> dict[str, Any]:
    panel = load_panel(cache_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first, _, _, first_folds = _season_views(panel, manifest, panel_builder.SEASONS[0])
    warmup_fold = first_folds[0]
    X_first = first.loc[:, list(panel_builder.FEATURE_COLUMNS)]
    warmup_started = time.perf_counter_ns()
    warmup = _fit_predict(
        arm,
        X_first.iloc[warmup_fold["train_indices"]],
        first[panel_builder.TARGET_COLUMNS[0]].iloc[warmup_fold["train_indices"]],
        X_first.iloc[warmup_fold["test_indices"]],
        chimeraboost_repo,
    )
    warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9
    del warmup
    gc.collect()

    cells = []
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    started = time.perf_counter_ns()
    for season in panel_builder.SEASONS:
        primary, holdout, seen, fold_plan = _season_views(panel, manifest, season)
        X_primary = primary.loc[:, list(panel_builder.FEATURE_COLUMNS)]
        X_holdout = holdout.loc[:, list(panel_builder.FEATURE_COLUMNS)]
        for target in panel_builder.TARGET_COLUMNS:
            folds = []
            oof_prediction = np.empty(len(primary), dtype=np.float64)
            for frozen in fold_plan:
                train = np.asarray(frozen["train_indices"], dtype=np.int64)
                test = np.asarray(frozen["test_indices"], dtype=np.int64)
                fitted = _fit_predict(
                    arm,
                    X_primary.iloc[train],
                    primary[target].iloc[train],
                    X_primary.iloc[test],
                    chimeraboost_repo,
                )
                prediction = fitted.pop("prediction")
                oof_prediction[test] = prediction
                total_fit_seconds += fitted["fit_seconds"]
                total_predict_seconds += fitted["predict_seconds"]
                folds.append(
                    {
                        "fold": int(frozen["fold"]),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_indices": frozen["train_indices"],
                        "test_indices": frozen["test_indices"],
                        "rmse": float(
                            mean_squared_error(primary[target].iloc[test], prediction)
                            ** 0.5
                        ),
                        "r2": float(r2_score(primary[target].iloc[test], prediction)),
                        "predictions": prediction.tolist(),
                        **fitted,
                    }
                )
            primary_score = _score_view(primary[target], oof_prediction)
            guardrail_fit = _fit_predict(
                arm,
                X_primary,
                primary[target],
                X_holdout,
                chimeraboost_repo,
            )
            holdout_prediction = guardrail_fit.pop("prediction")
            total_fit_seconds += guardrail_fit["fit_seconds"]
            total_predict_seconds += guardrail_fit["predict_seconds"]
            scores = {
                "held_team": _score_view(holdout[target], holdout_prediction),
                "seen_player": _score_view(
                    holdout.loc[seen, target], holdout_prediction[seen]
                ),
                "cold_player": _score_view(
                    holdout.loc[~seen, target], holdout_prediction[~seen]
                ),
            }
            cells.append(
                {
                    "season": int(season),
                    "target": target,
                    "primary_rows": int(len(primary)),
                    "held_team_rows": int(len(holdout)),
                    "seen_player_rows": int(np.sum(seen)),
                    "cold_player_rows": int(np.sum(~seen)),
                    "primary": primary_score,
                    "folds": folds,
                    "guardrail": {
                        "scores": scores,
                        "predictions": holdout_prediction.tolist(),
                        **guardrail_fit,
                    },
                }
            )
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    result = {
        "arm": arm,
        "implementation": _implementation(build_estimator(arm, chimeraboost_repo)),
        "cells": cells,
        "total_fit_seconds": float(total_fit_seconds),
        "total_predict_seconds": float(total_predict_seconds),
        "steady_wall_seconds": float(wall_seconds),
        "warmup_seconds_outside_timing": float(warmup_seconds),
        "peak_rss_bytes": base._peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _source_states(chimeraboost_repo: Path) -> dict[str, Any]:
    darkofit = creator.git_state(REPO_ROOT)
    chimeraboost = creator.git_state(chimeraboost_repo)
    if not darkofit["clean"] or not chimeraboost["clean"]:
        raise RuntimeError("sports panel 2 requires both repositories clean")
    if darkofit["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"sports panel 2 requires {EXPECTED_BRANCH}")
    branch_ref = f"origin/{EXPECTED_BRANCH}"
    published_head = creator._git_output(
        REPO_ROOT, "rev-parse", "--verify", branch_ref, check=False
    )
    if published_head != darkofit["head"]:
        raise RuntimeError("sports panel 2 source branch is not published")
    darkofit["published_branch_ref"] = branch_ref
    darkofit["published_branch_head"] = published_head
    if chimeraboost["head"] != EXPECTED_CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost differs from frozen 0.15.0")
    for ref in ("origin/main", "upstream/main"):
        value = chimeraboost["tracked_main_refs"].get(ref)
        if value is not None and value != EXPECTED_CHIMERABOOST_HEAD:
            raise RuntimeError(f"ChimeraBoost {ref} differs from frozen 0.15.0")
    return {"darkofit": darkofit, "chimeraboost": chimeraboost}


def _assert_sources_unchanged(
    expected: dict[str, Any],
    observed: dict[str, Any],
    boundary: str,
) -> None:
    base._assert_sources_unchanged(expected, observed, boundary)


def _validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError("sports panel 2 requires exactly 18 threads")
    exact = (
        (args.output, DEFAULT_OUTPUT),
        (args.data_cache, DEFAULT_CACHE),
        (args.manifest, MANIFEST_PATH),
    )
    if any(left.resolve() != right.resolve() for left, right in exact):
        raise RuntimeError("sports panel 2 paths are not exact")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest["candidate_data_scored"]
        or manifest["comparators_scored"]
        or manifest["panel_spent"]
    ):
        raise RuntimeError("sports panel 2 manifest is not pre-score")
    if _sha256(PROTOCOL_PATH) != manifest["protocol"]["sha256"]:
        raise RuntimeError("sports panel 2 protocol changed after manifest")
    builder_path = Path(panel_builder.__file__).resolve()
    if _sha256(builder_path) != manifest["builder"]["sha256"]:
        raise RuntimeError("sports panel 2 builder changed after manifest")
    shared_path = Path(panel_builder.base.__file__).resolve()
    if _sha256(shared_path) != manifest["builder"]["shared_builder_sha256"]:
        raise RuntimeError("shared sports builder changed after manifest")
    if not manifest["power_analysis"]["passes"]:
        raise RuntimeError("sports panel 2 lacks preregistered power")
    load_panel(args.data_cache, args.manifest)
    return manifest


def _worker_environment(
    threads: int,
    chimeraboost_repo: Path,
) -> dict[str, str]:
    environment = harness.worker_environment(threads)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(chimeraboost_repo.resolve()), str(REPO_ROOT))
    )
    environment["CHIMERABOOST_WARMUP"] = "0"
    environment["DARKOFIT_WARMUP"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--threads",
        str(args.threads),
        "--data-cache",
        str(args.data_cache),
        "--manifest",
        str(args.manifest),
        "--chimeraboost-repo",
        str(args.chimeraboost_repo),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(args.threads, args.chimeraboost_repo),
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
            f"worker {arm} failed with exit code {completed.returncode}"
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
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


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to replace sports result: {args.output}")
    manifest = _validate_frozen_inputs(args)
    source_states = _source_states(args.chimeraboost_repo)
    repeats = []
    fingerprints = {arm: set() for arm in ARM_ORDER}
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            _assert_sources_unchanged(
                source_states,
                _source_states(args.chimeraboost_repo),
                f"before block {block} position {position}",
            )
            print(
                f"[sports-panel-v2] block={block} position={position} arm={arm}",
                flush=True,
            )
            result = _run_worker_process(args, arm)
            fingerprints[arm].add(result["behavior_fingerprint_sha256"])
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "order": list(order),
                    "arm": arm,
                    "result": result,
                }
            )
    _assert_sources_unchanged(
        source_states,
        _source_states(args.chimeraboost_repo),
        "after final worker",
    )
    changed = {
        arm: sorted(values) for arm, values in fingerprints.items() if len(values) != 1
    }
    if changed:
        raise RuntimeError(f"sports panel 2 behavior changed: {changed}")
    artifact = {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_raw_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "panel_manifest": {
            "path": str(args.manifest.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(args.manifest),
            "processed_panel_sha256": manifest["processed_panel"]["sha256"],
            "split_manifest_sha256": manifest["split"]["split_manifest_sha256"],
            "power_pass_probability": manifest["power_analysis"]["pass_probability"],
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source": source_states,
        "execution": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threads": args.threads,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "worker_count": len(repeats),
            "candidate_or_comparator_outcomes_previously_scored": False,
        },
        "behavior_fingerprints": {
            arm: next(iter(values)) for arm, values in fingerprints.items()
        },
        "repeats": repeats,
        "panel_spent_by_this_run": True,
    }
    _atomic_create(
        args.output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-arm", choices=ARM_ORDER)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--chimeraboost-repo",
        type=Path,
        default=DEFAULT_CHIMERABOOST_REPO,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_arm is not None:
        result = run_worker(
            args.worker_arm,
            args.data_cache,
            args.manifest,
            args.chimeraboost_repo,
        )
        print(
            WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    artifact = run_parent(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "workers": len(artifact["repeats"]),
                "panel_spent": artifact["panel_spent_by_this_run"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
