#!/usr/bin/env python3
"""Validate and analyze the immutable raw T5 composite campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_t5_composite_confirmation as runner  # noqa: E402


DEFAULT_INPUT = ROOT / "benchmarks" / "t5_composite_confirmation_raw.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t5_composite_confirmation_summary.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "t5_composite_confirmation_result.md"
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_REPLICATES = 100_000
QUALITY_BAR = 0.995
UNCERTAINTY_BAR = 1.002
LOO_BAR = 0.998
HARM_BAR = 1.005
FIT_AGGREGATE_BAR = 6.0
FIT_WORST_DATASET_BAR = 12.0
PREDICT_AGGREGATE_BAR = 1.5
RSS_AGGREGATE_BAR = 2.5


def _json_sha256(value: Any) -> str:
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
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise RuntimeError("T5 geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(values))))


def _validate(raw):
    if raw.get("name") != "darkofit_t5_composite_confirmation_raw_v1":
        raise RuntimeError("T5 raw artifact name changed")
    expected_hash = raw.get("raw_artifact_sha256")
    unhashed = dict(raw)
    unhashed.pop("raw_artifact_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise RuntimeError("T5 raw artifact hash is invalid")
    if (
        not raw.get("outcomes_scored")
        or raw.get("analysis_performed")
        or raw.get("default_promotion_authorized")
        or raw.get("lockbox_data_used")
    ):
        raise RuntimeError("T5 raw artifact state is invalid")
    protocol = raw["protocol"]
    if (
        protocol["sha256"] != runner._sha256(runner.PROTOCOL)
        or protocol["runner_sha256"]
        != runner._sha256(Path(runner.__file__).resolve())
        or protocol["registry_file_sha256"]
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or protocol["registry_canonical_sha256"]
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or tuple(protocol["configs"]) != runner.CONFIGS
        or tuple(protocol["folds"]) != runner.FOLDS
        or protocol["task_count"] != 25
        or protocol["coordinate_count"] != 75
        or protocol["worker_count"] != 100
        or protocol["lockbox_data_used"]
        or protocol["task_drop_allowed"]
        or protocol["task_imputation_allowed"]
    ):
        raise RuntimeError("T5 raw protocol changed")
    registry, _rows = runner._registry()
    expected_task_ids = {
        int(row["task_id"]) for row in registry["tasks"]
    }
    results = raw["results"]
    expected = {
        (task_id, config)
        for task_id in expected_task_ids
        for config in runner.CONFIGS
    }
    observed = {(int(row["task_id"]), row["config"]) for row in results}
    if (
        len(results) != 100
        or len({task_id for task_id, _config in observed}) != 25
        or observed != expected
    ):
        raise RuntimeError("T5 raw worker matrix is incomplete")
    spool = raw["spool"]
    binding = spool["binding"]
    if (
        binding["runner_sha256"] != protocol["runner_sha256"]
        or binding["protocol_sha256"] != protocol["sha256"]
        or binding["registry_file_sha256"]
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or binding["registry_canonical_sha256"]
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or binding["darkofit_head"] != raw["sources"]["darkofit"]["head"]
        or binding["chimeraboost_head"]
        != raw["sources"]["chimeraboost"]["head"]
        or tuple(binding["configs"]) != runner.CONFIGS
        or tuple(binding["folds"]) != runner.FOLDS
        or int(spool["record_count"]) != len(expected)
        or int(spool["resumed_record_count"])
        != sum(bool(row["resumed"]) for row in spool["records"])
    ):
        raise RuntimeError("T5 raw spool binding changed")
    spool_coordinates = {
        (int(row["task_id"]), str(row["config"]))
        for row in spool["records"]
    }
    if (
        int(binding["schema_version"]) != 1
        or len(spool["records"]) != len(expected)
        or spool_coordinates != expected
        or len({row["sha256"] for row in spool["records"]})
        != len(expected)
        or any(
            row["filename"]
            != runner._spool_path(
                Path("."), row["task_id"], row["config"]
            ).name
            or len(str(row["sha256"])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(row["sha256"])
            )
            for row in spool["records"]
        )
    ):
        raise RuntimeError("T5 raw spool matrix is incomplete")
    sources = raw["sources"]
    if (
        not sources["darkofit"]["clean"]
        or not sources["chimeraboost"]["clean"]
        or sources["chimeraboost"]["head"] != runner.EXPECTED_CHIMERA_HEAD
    ):
        raise RuntimeError("T5 raw source state is invalid")
    by_key = {}
    identity = {}
    for row in results:
        key = (int(row["task_id"]), str(row["config"]))
        if key in by_key:
            raise RuntimeError(f"T5 raw duplicate worker: {key}")
        by_key[key] = row
        current_identity = (
            str(row["dataset_name"]),
            str(row["lineage_cluster"]),
            str(row["stratum"]),
        )
        registry_row = _rows[key[0]]
        expected_identity = (
            str(registry_row["dataset_name"]),
            str(registry_row["lineage_cluster"]),
            str(registry_row["stratum"]),
        )
        if current_identity != expected_identity:
            raise RuntimeError(f"T5 registry identity changed: {key[0]}")
        if (
            int(row["dataset_id"]) != int(registry_row["dataset_id"])
            or row["ordinal_features"] != registry_row["ordinal_features"]
            or int(row["fold_count"]) != len(runner.FOLDS)
        ):
            raise RuntimeError(f"T5 worker declaration changed: {key}")
        previous = identity.setdefault(key[0], current_identity)
        if previous != current_identity:
            raise RuntimeError(f"T5 task identity changed: {key[0]}")
        folds = row["folds"]
        if tuple(int(fold["fold"]) for fold in folds) != runner.FOLDS:
            raise RuntimeError(f"T5 fold order changed: {key}")
        if float(row["warmup_seconds"]) <= 0 or int(row["peak_rss_bytes"]) <= 0:
            raise RuntimeError(f"T5 resource record is invalid: {key}")
        for fold in folds:
            expected_split = runner._expected_split(
                registry_row, int(fold["fold"])
            )
            if (
                int(fold["train_rows"]) != expected_split["train_size"]
                or int(fold["test_rows"]) != expected_split["test_size"]
                or fold["train_index_sha256"]
                != expected_split["train_index_sha256"]
                or fold["test_index_sha256"]
                != expected_split["test_index_sha256"]
                or not math.isfinite(float(fold["rmse"]))
                or float(fold["rmse"]) <= 0
                or not math.isfinite(float(fold["fit_seconds"]))
                or float(fold["fit_seconds"]) <= 0
            ):
                raise RuntimeError(f"T5 metric is invalid: {key}")
            timing = fold["prediction_timing"]
            if (
                int(timing["call_count"]) < runner.PREDICTION_MIN_CALLS
                or float(timing["total_seconds"])
                < runner.PREDICTION_BLOCK_SECONDS
                or float(timing["per_call_median_seconds"]) <= 0
            ):
                raise RuntimeError(f"T5 prediction block is invalid: {key}")
        expected_behavior = {
            "task_id": key[0],
            "config": key[1],
            "folds": [
                {
                    "fold": fold["fold"],
                    "rmse": fold["rmse"],
                    "prediction_sha256": fold["prediction_sha256"],
                    "metadata": fold["metadata"],
                }
                for fold in folds
            ],
        }
        if (
            row["behavior_fingerprint_sha256"]
            != _json_sha256(expected_behavior)
            or not math.isclose(
                float(row["summed_fit_seconds"]),
                sum(float(fold["fit_seconds"]) for fold in folds),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(f"T5 worker behavior changed: {key}")
    if len(set(identity.values())) != 25:
        raise RuntimeError("T5 lineage identity is not unique")

    for task_id in identity:
        control = by_key[(task_id, runner.CONTROL)]
        composite = by_key[(task_id, runner.COMPOSITE)]
        for control_fold, composite_fold in zip(
            control["folds"], composite["folds"]
        ):
            engaged = bool(composite_fold["metadata"]["engaged"])
            if not engaged and (
                composite_fold["prediction_sha256"]
                != control_fold["prediction_sha256"]
                or float(composite_fold["rmse"])
                != float(control_fold["rmse"])
            ):
                raise RuntimeError(
                    f"T5 exact decline differs from control: "
                    f"{task_id}/{control_fold['fold']}"
                )
    return by_key, identity


def _quality_contrast(by_key, identity, numerator, denominator):
    per_dataset = {}
    logs = []
    for task_id, (dataset, lineage, stratum) in identity.items():
        top = by_key[(task_id, numerator)]
        bottom = by_key[(task_id, denominator)]
        ratios = np.asarray(
            [
                float(a["rmse"]) / float(b["rmse"])
                for a, b in zip(top["folds"], bottom["folds"])
            ],
            dtype=np.float64,
        )
        ratio = _geomean(ratios)
        logs.append(np.log(ratios))
        per_dataset[lineage] = {
            "task_id": task_id,
            "dataset_name": dataset,
            "stratum": stratum,
            "ratio": ratio,
            "split_ratios": ratios.tolist(),
        }
    dataset_ratios = [row["ratio"] for row in per_dataset.values()]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios),
        "equal_dataset_pct": (_geomean(dataset_ratios) - 1.0) * 100.0,
        "worst_dataset_ratio": float(max(dataset_ratios)),
        "worst_split_ratio": float(
            max(max(row["split_ratios"]) for row in per_dataset.values())
        ),
        "dataset_wins": int(np.count_nonzero(np.asarray(dataset_ratios) < 1)),
        "dataset_losses": int(np.count_nonzero(np.asarray(dataset_ratios) > 1)),
        "dataset_ties": int(np.count_nonzero(np.asarray(dataset_ratios) == 1)),
        "per_dataset": per_dataset,
        "_log_split_ratios": np.asarray(logs, dtype=np.float64),
    }


def _hierarchical_bootstrap_upper(log_split_ratios):
    logs = np.asarray(log_split_ratios, dtype=np.float64)
    if logs.shape != (25, 3):
        raise RuntimeError("T5 bootstrap requires 25 datasets x 3 folds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    block = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, block):
        count = min(block, BOOTSTRAP_REPLICATES - start)
        datasets = rng.integers(0, 25, size=(count, 25))
        folds = rng.integers(0, 3, size=(count, 25, 3))
        selected = logs[datasets[..., None], folds]
        estimates[start : start + count] = np.exp(
            selected.mean(axis=(1, 2))
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "one_sided_95_upper": float(np.quantile(estimates, 0.95)),
        "median": float(np.median(estimates)),
        "lower_5": float(np.quantile(estimates, 0.05)),
    }


def _cost_contrast(by_key, identity):
    per_dataset = {}
    for task_id, (dataset, lineage, stratum) in identity.items():
        candidate = by_key[(task_id, runner.COMPOSITE)]
        control = by_key[(task_id, runner.CONTROL)]
        fit_ratio = float(
            candidate["summed_fit_seconds"] / control["summed_fit_seconds"]
        )
        predict_ratios = [
            float(a["prediction_timing"]["per_call_median_seconds"])
            / float(b["prediction_timing"]["per_call_median_seconds"])
            for a, b in zip(candidate["folds"], control["folds"])
        ]
        rss_ratio = float(
            candidate["peak_rss_bytes"] / control["peak_rss_bytes"]
        )
        per_dataset[lineage] = {
            "task_id": task_id,
            "dataset_name": dataset,
            "stratum": stratum,
            "fit_seconds_ratio": fit_ratio,
            "prediction_seconds_ratio": _geomean(predict_ratios),
            "peak_rss_ratio": rss_ratio,
        }
    return {
        "equal_dataset_fit_seconds_ratio": _geomean(
            [row["fit_seconds_ratio"] for row in per_dataset.values()]
        ),
        "worst_dataset_fit_seconds_ratio": float(
            max(row["fit_seconds_ratio"] for row in per_dataset.values())
        ),
        "equal_dataset_prediction_seconds_ratio": _geomean(
            [row["prediction_seconds_ratio"] for row in per_dataset.values()]
        ),
        "equal_dataset_peak_rss_ratio": _geomean(
            [row["peak_rss_ratio"] for row in per_dataset.values()]
        ),
        "per_dataset": per_dataset,
    }


def analyze(raw):
    by_key, identity = _validate(raw)
    primary = _quality_contrast(
        by_key, identity, runner.COMPOSITE, runner.CONTROL
    )
    bootstrap = _hierarchical_bootstrap_upper(
        primary.pop("_log_split_ratios")
    )
    dataset_logs = np.log(
        [row["ratio"] for row in primary["per_dataset"].values()]
    )
    loo = []
    lineages = list(primary["per_dataset"])
    for index, lineage in enumerate(lineages):
        ratio = float(
            np.exp(
                (dataset_logs.sum() - dataset_logs[index])
                / (len(dataset_logs) - 1)
            )
        )
        loo.append({"omitted_lineage": lineage, "ratio": ratio})
    least_favorable_loo = max(loo, key=lambda row: row["ratio"])
    cost = _cost_contrast(by_key, identity)
    comparisons = {}
    for name, denominator in (
        ("composite_over_chimeraboost", runner.CHIMERA),
        ("composite_over_catboost", runner.CATBOOST),
        ("control_over_chimeraboost", runner.CHIMERA),
        ("control_over_catboost", runner.CATBOOST),
    ):
        numerator = (
            runner.COMPOSITE if name.startswith("composite") else runner.CONTROL
        )
        contrast = _quality_contrast(by_key, identity, numerator, denominator)
        contrast.pop("_log_split_ratios")
        comparisons[name] = contrast
    gates = {
        "quality_ratio_at_most_0_995": (
            primary["equal_dataset_geomean_ratio"] <= QUALITY_BAR
        ),
        "bootstrap_upper_at_most_1_002": (
            bootstrap["one_sided_95_upper"] <= UNCERTAINTY_BAR
        ),
        "least_favorable_loo_at_most_0_998": (
            least_favorable_loo["ratio"] <= LOO_BAR
        ),
        "worst_dataset_at_most_1_005": (
            primary["worst_dataset_ratio"] <= HARM_BAR
        ),
        "fit_aggregate_at_most_6": (
            cost["equal_dataset_fit_seconds_ratio"] <= FIT_AGGREGATE_BAR
        ),
        "fit_worst_dataset_at_most_12": (
            cost["worst_dataset_fit_seconds_ratio"]
            <= FIT_WORST_DATASET_BAR
        ),
        "prediction_aggregate_at_most_1_5": (
            cost["equal_dataset_prediction_seconds_ratio"]
            <= PREDICT_AGGREGATE_BAR
        ),
        "rss_aggregate_at_most_2_5": (
            cost["equal_dataset_peak_rss_ratio"] <= RSS_AGGREGATE_BAR
        ),
        "complete_without_imputation_or_lockbox": True,
    }
    composite_folds = [
        fold
        for task_id in identity
        for fold in by_key[(task_id, runner.COMPOSITE)]["folds"]
    ]
    engaged = sum(bool(fold["metadata"]["engaged"]) for fold in composite_folds)
    passes = all(gates.values())
    summary = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_summary_v1",
        "raw_artifact_sha256": raw["raw_artifact_sha256"],
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary": primary,
        "hierarchical_bootstrap": bootstrap,
        "leave_one_out": loo,
        "least_favorable_leave_one_out": least_favorable_loo,
        "cost": cost,
        "competitive_comparisons": comparisons,
        "selection": {
            "coordinate_count": len(composite_folds),
            "engaged_count": engaged,
            "declined_count": len(composite_folds) - engaged,
            "exact_declines_verified": True,
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": (
            "promote_t5_composite_automatic_policy"
            if passes
            else "close_t5_composite_candidate"
        ),
        "default_change_implemented": False,
        "lockbox_data_used": False,
    }
    summary["summary_sha256"] = _json_sha256(summary)
    return summary


def _markdown(summary):
    primary = summary["primary"]
    cost = summary["cost"]
    comp = summary["competitive_comparisons"]
    rows = [
        ("T5 / current default", primary["equal_dataset_geomean_ratio"]),
        (
            "T5 / ChimeraBoost 0.15.0",
            comp["composite_over_chimeraboost"][
                "equal_dataset_geomean_ratio"
            ],
        ),
        (
            "T5 / CatBoost 1.2.10",
            comp["composite_over_catboost"]["equal_dataset_geomean_ratio"],
        ),
    ]
    table = "\n".join(
        f"| {name} | {ratio:.6f} | {(ratio - 1) * 100:+.3f}% |"
        for name, ratio in rows
    )
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in summary["gates"].items()
    )
    return f"""# T5 composite confirmation result

**Decision: `{summary['decision']}`.**

All 25 outcome-unseen lineages and 75 frozen coordinates completed. The
selection-guarded candidate engaged on
{summary['selection']['engaged_count']} coordinates and declined exactly on
{summary['selection']['declined_count']}.

| Contrast | Equal-dataset RMSE ratio | Difference |
|---|---:|---:|
{table}

The one-sided hierarchical 95% upper bound is
`{summary['hierarchical_bootstrap']['one_sided_95_upper']:.6f}`; the
least-favorable leave-one-out ratio is
`{summary['least_favorable_leave_one_out']['ratio']:.6f}`; and the worst
dataset ratio is `{primary['worst_dataset_ratio']:.6f}`.

Cost versus current default: fit `{cost['equal_dataset_fit_seconds_ratio']:.3f}x`
(worst dataset `{cost['worst_dataset_fit_seconds_ratio']:.3f}x`), prediction
`{cost['equal_dataset_prediction_seconds_ratio']:.3f}x`, and peak RSS
`{cost['equal_dataset_peak_rss_ratio']:.3f}x`.

## Frozen gates

{gates}

Competitive comparisons are descriptive. No earlier lockbox was opened, no
task was dropped or imputed, and no default changes until the recorded
decision is implemented and verified.
"""


def _atomic_create(path, text):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    creator._atomic_write_bytes(path, text.encode())


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
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_create(args.markdown, _markdown(summary))
    print(summary["decision"])
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
