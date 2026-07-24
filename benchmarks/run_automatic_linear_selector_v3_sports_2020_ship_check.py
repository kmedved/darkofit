#!/usr/bin/env python3
"""Run selector-v3 on the newest complete 2020 basketball season."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import build_basketball_sports_panel as sports
from benchmarks import build_basketball_sports_panel_v2 as sports_v2
from benchmarks import run_t7b_automatic_depth_ctr23_ship_check_v1 as audit_base
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


SHIP_CHECK_ID = "automatic-linear-selector-v3-sports-2020-ship-check"
PROTOCOL = (
    ROOT
    / "benchmarks"
    / "automatic_linear_selector_v3_sports_2020_ship_check.md"
)
SEASON = 2020
TARGETS = sports.TARGET_COLUMNS
FEATURES = sports.FEATURE_COLUMNS
THREADS = 4
ARMS = {"control": False, "automatic": "auto"}


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(sports.panel_csv_bytes(frame)).hexdigest()


def _prediction_sha256(values) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_case(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, source_metadata = sports.load_raw_source(source)
    panel = sports.prepare_panel(raw, seasons=(SEASON,))
    held = frozenset(sports_v2.held_teams(panel, SEASON))
    primary = panel.loc[~panel["Tm"].isin(held)].reset_index(drop=True)
    holdout = panel.loc[panel["Tm"].isin(held)].reset_index(drop=True)
    primary_players = frozenset(primary["bref_id"].astype(str))
    cold = ~holdout["bref_id"].astype(str).isin(primary_players).to_numpy()
    if (
        len(panel) == 0
        or len(primary) == 0
        or len(holdout) == 0
        or panel["Tm"].nunique() != 30
        or len(held) != 10
        or not set(FEATURES).issubset(panel.columns)
        or not set(TARGETS).issubset(panel.columns)
    ):
        raise RuntimeError("2020 sports panel is incomplete")
    metadata = {
        "source": source_metadata,
        "season": SEASON,
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "panel_rows": len(panel),
        "primary_rows": len(primary),
        "holdout_rows": len(holdout),
        "cold_player_rows": int(np.sum(cold)),
        "seen_player_rows": int(np.sum(~cold)),
        "held_teams": sorted(held),
        "panel_sha256": _frame_sha256(panel),
        "primary_identities_sha256": sports._json_sha256(
            primary.loc[:, list(sports.IDENTITY_COLUMNS)].values.tolist()
        ),
        "holdout_identities_sha256": sports._json_sha256(
            holdout.loc[:, list(sports.IDENTITY_COLUMNS)].values.tolist()
        ),
        "target_sha256": {
            target: sports._json_sha256(
                panel[target].to_numpy(dtype=np.float64).tolist()
            )
            for target in TARGETS
        },
    }
    return panel, metadata


def _selector_integrity(arm: str, model, restored) -> bool:
    selector = getattr(model, "automatic_linear_selector_", None)
    restored_selector = getattr(restored, "automatic_linear_selector_", None)
    if arm == "control":
        return bool(
            selector is None
            and restored_selector is None
            and not bool(getattr(model.model_, "linear_leaves", False))
        )
    if (
        not isinstance(selector, Mapping)
        or selector != restored_selector
        or selector.get("version") != 2
        or selector.get("requested") != "auto"
        or selector.get("minimum_gain_z") != 2.0
    ):
        return False
    selected = selector.get("resolved_linear_leaves")
    if not isinstance(selected, bool):
        return False
    return bool(
        selector.get("final_booster_linear_leaves") is selected
        and bool(getattr(model.model_, "linear_leaves", False)) is selected
    )


def run_worker(
    spec: Mapping[str, Any],
    *,
    arm: str,
    target: str,
    panel_path: Path,
    source: Path,
    expected_head: str,
) -> dict[str, Any]:
    import numba

    source = source.expanduser().resolve()
    state = helpers.source_state(source)
    if not state["clean"] or state["head"] != expected_head:
        raise RuntimeError("sports selector source state changed")
    sys.path.insert(0, str(source))
    from darkofit import DarkoRegressor
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("worker imported DarkoFit from the wrong source")
    if helpers.file_sha256(panel_path) != spec["panel_sha256"]:
        raise RuntimeError("sports selector panel bytes changed")
    panel = pd.read_csv(panel_path)
    if (
        target not in TARGETS
        or arm not in ARMS
    ):
        raise RuntimeError("sports selector worker input changed")
    held = frozenset(spec["held_teams"])
    primary = panel.loc[~panel["Tm"].isin(held)].reset_index(drop=True)
    holdout = panel.loc[panel["Tm"].isin(held)].reset_index(drop=True)
    if (
        sports._json_sha256(
            primary.loc[:, list(sports.IDENTITY_COLUMNS)].values.tolist()
        )
        != spec["primary_identities_sha256"]
        or sports._json_sha256(
            holdout.loc[:, list(sports.IDENTITY_COLUMNS)].values.tolist()
        )
        != spec["holdout_identities_sha256"]
    ):
        raise RuntimeError("sports selector split fingerprint changed")
    primary_players = frozenset(primary["bref_id"].astype(str))
    cold = ~holdout["bref_id"].astype(str).isin(primary_players).to_numpy()
    seen = ~cold
    X_train = primary.loc[:, list(FEATURES)]
    X_test = holdout.loc[:, list(FEATURES)]
    y_train = primary[target].to_numpy(dtype=np.float64)
    y_test = holdout[target].to_numpy(dtype=np.float64)
    groups = primary["bref_id"].astype(str).to_numpy()

    helpers._warmup(DarkoRegressor)
    model = DarkoRegressor(
        iterations=600,
        early_stopping=True,
        early_stopping_rounds=30,
        validation_fraction=0.15,
        validation_strategy="group",
        use_best_model=True,
        refit=False,
        random_state=4,
        thread_count=THREADS,
        diagnostic_warnings="never",
        linear_leaves=ARMS[arm],
    )
    ambient = int(numba.get_num_threads())
    with helpers._PeakRSS() as rss:
        started = time.perf_counter()
        model.fit(X_train, y_train, groups=groups)
        fit_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        prediction = np.asarray(model.predict(X_test), dtype=np.float64)
        predict_seconds = time.perf_counter() - prediction_started
        with tempfile.TemporaryDirectory(prefix="selector-v3-sports-worker-") as temp:
            archive = Path(temp) / "model.npz"
            model.save_model(archive)
            restored = DarkoRegressor.load_model(archive)
            archive_exact = np.array_equal(
                restored.predict(X_test), prediction
            )
    selector = getattr(model, "automatic_linear_selector_", None)
    integrity = bool(
        archive_exact
        and int(numba.get_num_threads()) == ambient
        and _selector_integrity(arm, model, restored)
        and prediction.shape == y_test.shape
        and np.isfinite(prediction).all()
    )

    def score(mask) -> dict[str, Any]:
        truth = y_test[mask]
        predicted = prediction[mask]
        return {
            "rows": int(len(truth)),
            "rmse": (
                None
                if len(truth) == 0
                else helpers._rmse(truth, predicted, None)
            ),
            "prediction_sha256": _prediction_sha256(predicted),
        }

    return {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "status": "ok" if integrity else "integrity_failed",
        "arm": arm,
        "target": target,
        "source": state,
        "panel_sha256": spec["panel_sha256"],
        "train_rows": len(primary),
        "test_rows": len(holdout),
        "input_features": len(FEATURES),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_process_tree_rss_bytes": rss.peak,
        "automatic_linear_selector": (
            None if selector is None else dict(selector)
        ),
        "safe_npz_exact": bool(archive_exact),
        "ambient_thread_restored": int(numba.get_num_threads()) == ambient,
        "views": {
            "all_held": score(np.ones(len(y_test), dtype=bool)),
            "seen_player": score(seen),
            "cold_player": score(cold),
        },
        "integrity_passes": integrity,
    }


def _geomean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or np.any(array <= 0.0)
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("invalid sports selector geometric-mean input")
    return float(np.exp(np.mean(np.log(array))))


def analyze_rows(rows) -> dict[str, Any]:
    indexed = defaultdict(dict)
    for row in rows:
        key = str(row["target"])
        arm = str(row["arm"])
        if arm in indexed[key]:
            raise RuntimeError("duplicate sports selector arm")
        indexed[key][arm] = row
    if set(indexed) != set(TARGETS):
        raise RuntimeError("sports selector target census changed")
    ratios = defaultdict(list)
    target_rows = []
    all_exact = True
    all_pairs_safe = True
    fit_ratios = []
    predict_ratios = []
    rss_ratios = []
    for target in TARGETS:
        arms = indexed[target]
        if set(arms) != {"control", "automatic"}:
            raise RuntimeError("incomplete sports selector pair")
        control, automatic = arms["control"], arms["automatic"]
        if (
            control.get("integrity_passes") is not True
            or automatic.get("integrity_passes") is not True
            or control["panel_sha256"] != automatic["panel_sha256"]
            or control["train_rows"] != automatic["train_rows"]
            or control["test_rows"] != automatic["test_rows"]
        ):
            raise RuntimeError("sports selector pair integrity failed")
        pair_exact = all(
            control["views"][view]["prediction_sha256"]
            == automatic["views"][view]["prediction_sha256"]
            for view in ("all_held", "seen_player", "cold_player")
        )
        all_exact &= pair_exact
        view_ratios = {}
        for view in ("all_held", "seen_player", "cold_player"):
            control_rmse = control["views"][view]["rmse"]
            automatic_rmse = automatic["views"][view]["rmse"]
            if control_rmse is None or automatic_rmse is None:
                ratio = None
            elif control_rmse == 0.0:
                ratio = 1.0 if automatic_rmse == 0.0 else float("inf")
            else:
                ratio = automatic_rmse / control_rmse
                ratios[view].append(ratio)
            view_ratios[view] = ratio
        pair_safe = bool(
            pair_exact
            or all(
                ratio is None or ratio <= 1.0
                for ratio in view_ratios.values()
            )
        )
        all_pairs_safe &= pair_safe
        target_rows.append({
            "target": target,
            "prediction_exact": pair_exact,
            "pair_safe": pair_safe,
            "view_ratios": view_ratios,
            "selector_reason": automatic["automatic_linear_selector"]["reason"],
        })
        fit_ratios.append(automatic["fit_seconds"] / control["fit_seconds"])
        predict_ratios.append(
            automatic["predict_seconds"] / control["predict_seconds"]
        )
        rss_ratios.append(
            automatic["peak_process_tree_rss_bytes"]
            / control["peak_process_tree_rss_bytes"]
        )
    aggregate = {
        view: (_geomean(values) if values else None)
        for view, values in ratios.items()
    }
    no_harm = bool(all_pairs_safe)
    return {
        "disposition": (
            "eligible_for_automatic_default" if no_harm else "keep_explicit_opt_in"
        ),
        "default_eligible": no_harm,
        "all_prediction_vectors_exact": all_exact,
        "aggregate_view_ratios": aggregate,
        "targets": target_rows,
        "costs": {
            "fit_pair_geomean_ratio": _geomean(fit_ratios),
            "predict_pair_geomean_ratio": _geomean(predict_ratios),
            "peak_rss_pair_geomean_ratio": _geomean(rss_ratios),
        },
        "integrity": {
            "passes": True,
            "rows": len(rows),
            "pairs": len(indexed),
            "targets": len(indexed),
        },
    }


def _worker_command(
    spec_path: Path,
    *,
    arm: str,
    target: str,
    panel_path: Path,
    source: Path,
    expected_head: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--spec",
        str(spec_path),
        "--arm",
        arm,
        "--target",
        target,
        "--panel",
        str(panel_path),
        "--source",
        str(source),
        "--expected-head",
        expected_head,
    ]


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    helpers._write_create_only(path, payload)


def exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    audit = audit_base.exclusive_machine_audit()
    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    marker = Path(__file__).name
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and marker in command:
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another sports selector ship-check is active: {conflicts}")
    return audit


def execute(
    *,
    raw_source: Path,
    source: Path,
    expected_head: str,
    output_prefix: Path,
) -> dict[str, Any]:
    output_prefix = output_prefix.expanduser().resolve()
    try:
        output_prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("sports ship-check outputs must be outside source")
    paths = {
        name: Path(f"{output_prefix}_{name}.json")
        for name in ("launch", "raw", "result")
    }
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"sports selector output collision: {collisions}")
    state = helpers.source_state(source)
    if (
        not state["clean"]
        or state["head"] != expected_head
        or helpers.source_state(ROOT) != state
    ):
        raise RuntimeError("sports selector source pin changed")
    audit = exclusive_machine_audit()
    panel, spec = prepare_case(raw_source)
    launch = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": state,
        "source_hashes": {
            "runner": helpers.file_sha256(Path(__file__)),
            "protocol": helpers.file_sha256(PROTOCOL),
        },
        "environment": helpers._environment(),
        "exclusive_machine_audit": audit,
        "case": spec,
        "planned_rows": len(TARGETS) * len(ARMS),
    }
    _write_create_only(paths["launch"], launch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="selector-v3-sports-2020-") as temp:
        temp_path = Path(temp)
        panel_path = temp_path / "panel.csv"
        panel_path.write_bytes(sports.panel_csv_bytes(panel))
        spec_path = temp_path / "spec.json"
        spec_path.write_bytes(helpers.canonical_json_bytes(spec))
        caches = {arm: temp_path / f"numba-{arm}" for arm in ARMS}
        for cache in caches.values():
            cache.mkdir()
        for target_index, target in enumerate(TARGETS):
            order = (
                ("control", "automatic")
                if target_index % 2 == 0
                else ("automatic", "control")
            )
            for arm in order:
                completed = subprocess.run(
                    _worker_command(
                        spec_path,
                        arm=arm,
                        target=target,
                        panel_path=panel_path,
                        source=source,
                        expected_head=expected_head,
                    ),
                    cwd=ROOT,
                    env=helpers._worker_env(source, caches[arm]),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"sports selector worker failed for {target}/{arm}: "
                        f"{completed.stderr[-4000:]}"
                    )
                lines = [
                    line for line in completed.stdout.splitlines() if line.strip()
                ]
                if not lines:
                    raise RuntimeError("sports selector worker returned no row")
                row = json.loads(lines[-1])
                if row.get("status") != "ok":
                    raise RuntimeError(
                        f"sports selector worker integrity failed: {row}"
                    )
                rows.append(row)
                print(
                    f"ok {len(rows)}/{len(TARGETS) * len(ARMS)} "
                    f"target={target} arm={arm}",
                    flush=True,
                )
    raw = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "complete": True,
        "launch_sha256": helpers.file_sha256(paths["launch"]),
        "rows": rows,
    }
    _write_create_only(paths["raw"], raw)
    result = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "holdout_ship_check",
        "holdout": "newest complete sports season 2020",
        "analysis": analyze_rows(rows),
        "source_hashes": {
            "launch": helpers.file_sha256(paths["launch"]),
            "raw": helpers.file_sha256(paths["raw"]),
            "runner": helpers.file_sha256(Path(__file__)),
            "protocol": helpers.file_sha256(PROTOCOL),
        },
    }
    _write_create_only(paths["result"], result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    worker.add_argument("--arm", choices=tuple(ARMS), required=True)
    worker.add_argument("--target", choices=TARGETS, required=True)
    worker.add_argument("--panel", type=Path, required=True)
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--expected-head", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--raw-source", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--expected-head", required=True)
    run.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "worker":
        result = run_worker(
            helpers._load_json(args.spec),
            arm=args.arm,
            target=args.target,
            panel_path=args.panel,
            source=args.source,
            expected_head=args.expected_head,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        raw_source=args.raw_source,
        source=args.source,
        expected_head=args.expected_head,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result["analysis"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
