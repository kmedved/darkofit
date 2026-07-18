#!/usr/bin/env python3
"""Analyze the T7 CatBoost attribution development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_native_ordinal_c2 as c2  # noqa: E402
from benchmarks import run_t7_catboost_attribution as runner  # noqa: E402


DEFAULT_INPUT = runner.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t7_catboost_attribution_summary.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "t7_catboost_attribution_result.md"
ELIGIBLE = (
    "ordered",
    "border_128",
    "leaf10_no_backtracking",
    "leaf10_any_improvement",
    "ctr_complexity_2",
    "depth_by_n_p",
)


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or not values.size
        or np.any(values <= 0)
        or not np.all(np.isfinite(values))
    ):
        raise RuntimeError("T7 geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(values))))


def depth_policy_arm(fit_rows, n_features):
    density = float(fit_rows) / max(1, int(n_features))
    if density < 100:
        return "depth_4"
    if density >= 2_500:
        return "depth_8"
    return "default"


def _behavior(result):
    return {
        "task_id": result["task_id"],
        "fold": result["fold"],
        "arms": [
            {
                "arm": arm["arm"],
                "validation": arm["validation"],
                "test": arm["test"],
                "resolved_params": arm["resolved_params"],
            }
            for arm in result["arms"]
        ],
    }


def _validate(raw):
    if raw["name"] != "darkofit_t7_catboost_attribution_raw_v1":
        raise RuntimeError("T7 raw artifact name changed")
    expected_hash = raw["raw_sha256"]
    unhashed = dict(raw)
    unhashed.pop("raw_sha256")
    if expected_hash != runner._json_sha256(unhashed):
        raise RuntimeError("T7 raw artifact hash changed")
    if (
        not raw["development_data_only"]
        or raw["confirmation_outcomes_inspected"]
        or raw["lockbox_data_used"]
        or raw["default_change_authorized"]
        or raw["protocol"]["runner_sha256"]
        != runner._sha256(Path(runner.__file__).resolve())
        or raw["protocol"]["protocol_sha256"]
        != runner._sha256(runner.PROTOCOL)
        or tuple(raw["protocol"]["arms"]) != runner.ARM_NAMES
        or raw["task_count"] != 8
        or raw["coordinate_count"] != 24
        or raw["fit_count"] != 216
    ):
        raise RuntimeError("T7 raw protocol changed")
    registry, rows = runner._rows()
    expected = {
        (int(row["task_id"]), fold)
        for row in registry["development_tasks"]
        for fold in runner.FOLDS
    }
    results = raw["results"]
    observed = {
        (int(row["task_id"]), int(row["fold"])) for row in results
    }
    if len(results) != 24 or observed != expected:
        raise RuntimeError("T7 coordinate matrix is incomplete")
    by_coordinate = {}
    for result in results:
        key = (int(result["task_id"]), int(result["fold"]))
        if key in by_coordinate:
            raise RuntimeError(f"T7 duplicate coordinate: {key}")
        row = rows[key[0]]
        expected_split = c2._expected_outer_split(row, key[1])
        outer = result["outer_split"]
        if (
            int(result["dataset_id"]) != int(row["dataset_id"])
            or result["dataset_name"] != row["dataset_name"]
            or result["lineage_cluster"] != row["lineage_cluster"]
            or result["n_features"]
            != row["task_record"]["fingerprint"]["n_features"]
            or outer["train_size"] != expected_split["train_size"]
            or outer["test_size"] != expected_split["test_size"]
            or outer["train_index_sha256"]
            != expected_split["train_index_sha256"]
            or outer["test_index_sha256"]
            != expected_split["test_index_sha256"]
            or tuple(result["arm_order"])
            != runner._arm_order(result["coordinate_index"])
            or result["behavior_sha256"]
            != runner._json_sha256(_behavior(result))
        ):
            raise RuntimeError(f"T7 coordinate binding changed: {key}")
        arms = {arm["arm"]: arm for arm in result["arms"]}
        if len(arms) != len(runner.ARM_NAMES) or set(arms) != set(
            runner.ARM_NAMES
        ):
            raise RuntimeError(f"T7 arm matrix changed: {key}")
        for arm in arms.values():
            for split in ("validation", "test"):
                metric = arm[split]
                if (
                    not math.isfinite(float(metric["rmse"]))
                    or float(metric["rmse"]) <= 0
                    or len(metric["prediction_sha256"]) != 64
                ):
                    raise RuntimeError(f"T7 metric changed: {key}")
            if (
                not math.isfinite(float(arm["fit_seconds"]))
                or float(arm["fit_seconds"]) <= 0
                or arm["prediction_timing"]["calls"]
                != runner.PREDICTION_CALLS
                or float(arm["prediction_timing"]["median_seconds"]) <= 0
            ):
                raise RuntimeError(f"T7 timing changed: {key}")
        by_coordinate[key] = {
            "result": result,
            "arms": arms,
        }
    if len(raw["spool_records"]) != 24:
        raise RuntimeError("T7 spool matrix changed")
    return by_coordinate, rows


def _selected_arm(coordinate, name):
    if name == "depth_by_n_p":
        result = coordinate["result"]
        return depth_policy_arm(
            result["inner_split"]["fit_rows"],
            result["n_features"],
        )
    return name


def _contrast(by_coordinate, rows, numerator, denominator="default"):
    per_task = {}
    for task_id, row in rows.items():
        test_ratios = []
        validation_ratios = []
        selected_arms = []
        for fold in runner.FOLDS:
            coordinate = by_coordinate[(task_id, fold)]
            top_name = _selected_arm(coordinate, numerator)
            bottom_name = _selected_arm(coordinate, denominator)
            top = coordinate["arms"][top_name]
            bottom = coordinate["arms"][bottom_name]
            test_ratios.append(
                float(top["test"]["rmse"]) / float(bottom["test"]["rmse"])
            )
            validation_ratios.append(
                float(top["validation"]["rmse"])
                / float(bottom["validation"]["rmse"])
            )
            selected_arms.append(top_name)
        per_task[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "test_ratio": _geomean(test_ratios),
            "validation_ratio": _geomean(validation_ratios),
            "split_test_ratios": test_ratios,
            "selected_arms": selected_arms,
        }
    task_ratios = [row["test_ratio"] for row in per_task.values()]
    validation_ratios = [
        row["validation_ratio"] for row in per_task.values()
    ]
    logs = np.log(task_ratios)
    loo = [
        {
            "omitted_lineage": lineage,
            "ratio": float(
                np.exp((logs.sum() - logs[index]) / (len(logs) - 1))
            ),
        }
        for index, lineage in enumerate(per_task)
    ]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "equal_dataset_test_ratio": _geomean(task_ratios),
        "equal_dataset_validation_ratio": _geomean(validation_ratios),
        "worst_task_test_ratio": float(max(task_ratios)),
        "worst_split_test_ratio": float(
            max(
                max(row["split_test_ratios"])
                for row in per_task.values()
            )
        ),
        "least_favorable_loo_test_ratio": float(
            max(row["ratio"] for row in loo)
        ),
        "wins": int(np.count_nonzero(np.asarray(task_ratios) < 1)),
        "losses": int(np.count_nonzero(np.asarray(task_ratios) > 1)),
        "ties": int(np.count_nonzero(np.asarray(task_ratios) == 1)),
        "leave_one_out": loo,
        "per_task": per_task,
    }


def _darkofit_anchor(by_coordinate, rows, arm):
    c2_raw = json.loads(runner.C2_RAW.read_text())
    controls = {
        (int(row["task_id"]), int(row["fold"])): row
        for row in c2_raw["results"]
        if row["arm"] == "control"
    }
    if len(controls) != 24:
        raise RuntimeError("T7 DarkoFit anchor matrix changed")
    per_task = {}
    for task_id, row in rows.items():
        ratios = []
        for fold in runner.FOLDS:
            coordinate = by_coordinate[(task_id, fold)]
            arm_name = _selected_arm(coordinate, arm)
            cat_rmse = coordinate["arms"][arm_name]["test"]["rmse"]
            darko_rmse = controls[(task_id, fold)]["test"]["rmse"]
            ratios.append(float(darko_rmse) / float(cat_rmse))
        per_task[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "darkofit_over_catboost_ratio": _geomean(ratios),
            "split_ratios": ratios,
        }
    ratios = [
        row["darkofit_over_catboost_ratio"] for row in per_task.values()
    ]
    return {
        "catboost_arm": arm,
        "equal_dataset_darkofit_over_catboost_ratio": _geomean(ratios),
        "darkofit_wins": int(np.count_nonzero(np.asarray(ratios) < 1)),
        "darkofit_losses": int(np.count_nonzero(np.asarray(ratios) > 1)),
        "per_task": per_task,
        "evidence_scope": (
            "immutable C2 DarkoFit control; descriptive, not current-release "
            "confirmation"
        ),
    }


def analyze(raw):
    by_coordinate, rows = _validate(raw)
    contrasts = {
        "ordered_over_plain": _contrast(
            by_coordinate, rows, "ordered", "plain"
        ),
        "plain_over_default": _contrast(
            by_coordinate, rows, "plain", "default"
        ),
        "border_128_over_default": _contrast(
            by_coordinate, rows, "border_128"
        ),
        "leaf10_no_backtracking_over_default": _contrast(
            by_coordinate, rows, "leaf10_no_backtracking"
        ),
        "backtracking_over_no_backtracking": _contrast(
            by_coordinate,
            rows,
            "leaf10_any_improvement",
            "leaf10_no_backtracking",
        ),
        "ctr_complexity_2_over_default": _contrast(
            by_coordinate, rows, "ctr_complexity_2"
        ),
        "depth_4_over_default": _contrast(
            by_coordinate, rows, "depth_4"
        ),
        "depth_8_over_default": _contrast(
            by_coordinate, rows, "depth_8"
        ),
        "depth_by_n_p_over_default": _contrast(
            by_coordinate, rows, "depth_by_n_p"
        ),
    }
    candidate_contrasts = {
        name: _contrast(by_coordinate, rows, name)
        for name in ELIGIBLE
    }
    nominations = []
    for name, contrast in candidate_contrasts.items():
        gates = {
            "test_ratio_at_most_0_995": (
                contrast["equal_dataset_test_ratio"] <= 0.995
            ),
            "validation_ratio_at_most_1_005": (
                contrast["equal_dataset_validation_ratio"] <= 1.005
            ),
            "worst_task_at_most_1_02": (
                contrast["worst_task_test_ratio"] <= 1.02
            ),
            "least_favorable_loo_at_most_1": (
                contrast["least_favorable_loo_test_ratio"] <= 1.0
            ),
        }
        nominations.append(
            {
                "candidate": name,
                "passes": all(gates.values()),
                "gates": gates,
                "test_ratio": contrast["equal_dataset_test_ratio"],
            }
        )
    survivors = sorted(
        (row for row in nominations if row["passes"]),
        key=lambda row: (row["test_ratio"], row["candidate"]),
    )[:3]
    anchors = {
        arm: _darkofit_anchor(by_coordinate, rows, arm)
        for arm in ("default", *[row["candidate"] for row in survivors])
    }
    default_resolution = {}
    for task_id, row in rows.items():
        params = [
            by_coordinate[(task_id, fold)]["arms"]["default"][
                "resolved_params"
            ]
            for fold in runner.FOLDS
        ]
        default_resolution[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "resolved_params_by_fold": params,
        }
    summary = {
        "schema_version": 1,
        "name": "darkofit_t7_catboost_attribution_summary_v1",
        "raw_sha256": raw["raw_sha256"],
        "analyzer_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "contrasts": contrasts,
        "candidate_contrasts": candidate_contrasts,
        "candidate_gate_results": nominations,
        "frozen_research_candidates": [
            row["candidate"] for row in survivors
        ],
        "candidate_limit": 3,
        "darkofit_anchors": anchors,
        "catboost_default_resolution": default_resolution,
        "development_data_only": True,
        "confirmation_outcomes_inspected": False,
        "lockbox_data_used": False,
        "default_change_authorized": False,
        "decision": (
            "freeze_t7_research_candidates"
            if survivors
            else "close_t7_without_candidates"
        ),
    }
    summary["summary_sha256"] = _json_sha256(summary)
    return summary


def _markdown(summary):
    rows = []
    for name, contrast in summary["contrasts"].items():
        rows.append(
            f"| `{name}` | {contrast['equal_dataset_test_ratio']:.6f} | "
            f"{contrast['equal_dataset_validation_ratio']:.6f} | "
            f"{contrast['worst_task_test_ratio']:.6f} | "
            f"{contrast['wins']}/{contrast['losses']}/"
            f"{contrast['ties']} |"
        )
    table = "\n".join(rows)
    candidates = summary["frozen_research_candidates"]
    candidate_text = (
        ", ".join(f"`{name}`" for name in candidates)
        if candidates
        else "none"
    )
    anchor = summary["darkofit_anchors"]["default"]
    candidate_rows = []
    for name in candidates:
        contrast = summary["candidate_contrasts"][name]
        for lineage, row in contrast["per_task"].items():
            candidate_rows.append(
                f"| `{name}` | {row['dataset_name']} | "
                f"{row['test_ratio']:.6f} | "
                f"{row['validation_ratio']:.6f} | "
                f"`{row['selected_arms'][0]}` |"
            )
    candidate_table = (
        "\n".join(candidate_rows)
        if candidate_rows
        else "| — | — | — | — | — |"
    )
    anchor_ratio_key = "equal_dataset_darkofit_over_catboost_ratio"
    candidate_anchor_rows = "\n".join(
        f"| `{name}` | "
        f"{summary['darkofit_anchors'][name][anchor_ratio_key]:.6f} |"
        for name in candidates
    )
    candidate_anchor_table = (
        candidate_anchor_rows
        if candidate_anchor_rows
        else "| — | — |"
    )
    return f"""# T7 CatBoost mechanism attribution

**Decision: `{summary['decision']}`.**

| Contrast | Test ratio | Validation ratio | Worst task | W/L/T |
|---|---:|---:|---:|---:|
{table}

Frozen research candidates (maximum three): {candidate_text}.

## Surviving candidate by dataset

| Candidate | Dataset | Test ratio | Validation ratio | Selected arm |
|---|---|---:|---:|---|
{candidate_table}

The fixed depth policy uses depth 4 below 100 inner-fit rows per feature,
depth 8 at or above 2,500, and CatBoost's default depth 6 otherwise. It
declines exactly to the default on the five middle-density datasets.

## DarkoFit anchor

Against CatBoost's product default, the immutable C2 DarkoFit control has an
equal-dataset RMSE ratio of
`{anchor['equal_dataset_darkofit_over_catboost_ratio']:.6f}`.

| CatBoost arm | DarkoFit / CatBoost RMSE |
|---|---:|
| `default` | {anchor['equal_dataset_darkofit_over_catboost_ratio']:.6f} |
{candidate_anchor_table}

This is a descriptive historical anchor, not a current-release confirmation
claim. The surviving CatBoost depth policy widens rather than closes that
historical competitive gap; porting the rule to DarkoFit would require a
separate implementation and outcome-unseen evaluation.

All results use spent development tasks. No confirmation panel or lockbox was
opened, and no default change is authorized.
"""


def _atomic_create(path, value):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    creator._atomic_write_bytes(path, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw = json.loads(args.input.read_text())
    summary = analyze(raw)
    _atomic_create(
        args.output,
        (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    )
    _atomic_create(args.markdown, _markdown(summary).encode())
    print(summary["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
