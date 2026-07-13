"""Analyze the frozen remaining-nine TabArena regression confirmation.

The runner writes gzip-compressed ``results.pkl`` files.  This module treats
those files and TabArena's registered ``CHIMERA (default)`` rows as a closed,
predeclared panel: missing, duplicate, failed, imputed, or unexpected rows are
fatal.  It must therefore only be run after the 330 local result files exist.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

try:
    from benchmarks.run_tabarena_regression_remaining9 import (
        FROZEN_CANDIDATE,
        TASK_SPLIT_COUNTS,
    )
except ModuleNotFoundError:  # Direct execution: python benchmarks/analyze_*.py
    from run_tabarena_regression_remaining9 import (
        FROZEN_CANDIDATE,
        TASK_SPLIT_COUNTS,
    )
LOCAL_METRICS = (
    "rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "peak_memory_bytes",
)
RATIO_PREFIXES = {
    "rmse": "candidate_default_rmse",
    "val_rmse": "candidate_default_val_rmse",
    "train_time_s": "candidate_default_train_time",
    "infer_time_s": "candidate_default_infer_time",
    "peak_memory_bytes": "candidate_default_peak_memory",
}
GATE_THRESHOLDS = {
    "equal_dataset_rmse_ratio_max": 0.995,
    "dataset_rmse_ratio_max": 1.005,
    "split_rmse_ratio_max": 1.02,
    "validation_equal_dataset_ratio_max": 1.0,
    "train_time_ratio_max": 1.20,
    "infer_time_ratio_max": 1.10,
    "peak_memory_ratio_max": 1.10,
}


def _positive_finite(value, field: str) -> float:
    """Return a positive finite float or raise a schema error."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise RuntimeError(f"{field} must be positive and finite, got {value!r}")
    return out


def geometric_mean(values: Iterable[float]) -> float:
    """Return a geometric mean after validating every input."""
    checked = [_positive_finite(value, "geometric-mean input") for value in values]
    if not checked:
        raise RuntimeError("cannot take the geometric mean of an empty sequence")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def geometric_mean_ratio(
    numerators: Iterable[float], denominators: Iterable[float]
) -> float:
    """Return the paired geometric-mean numerator/denominator ratio."""
    numerator_values = list(numerators)
    denominator_values = list(denominators)
    if len(numerator_values) != len(denominator_values):
        raise RuntimeError("paired-ratio inputs have different lengths")
    pairs = list(zip(numerator_values, denominator_values))
    if not pairs:
        raise RuntimeError("cannot aggregate an empty set of paired ratios")
    logs = [
        math.log(_positive_finite(numerator, "ratio numerator"))
        - math.log(_positive_finite(denominator, "ratio denominator"))
        for numerator, denominator in pairs
    ]
    return math.exp(math.fsum(logs) / len(logs))


def _user_hyperparameters(record: Mapping) -> dict:
    try:
        hyperparameters = dict(record["method_metadata"]["model_hyperparameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("missing method_metadata.model_hyperparameters") from exc
    hyperparameters.pop("ag_args", None)
    hyperparameters.pop("ag_args_ensemble", None)
    return hyperparameters


def local_result_row(
    record: Mapping,
    *,
    source: Union[Path, str],
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
    frozen_candidate: Mapping[str, object] = FROZEN_CANDIDATE,
) -> dict:
    """Validate and normalize one local TabArena result payload."""
    source = str(source)
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: expected regression/rmse result")

    task = record.get("task_metadata")
    if not isinstance(task, Mapping):
        raise RuntimeError(f"{source}: missing task_metadata")
    dataset = task.get("name")
    if dataset not in task_split_counts:
        raise RuntimeError(f"{source}: unexpected dataset {dataset!r}")
    expected_task_id, expected_splits = task_split_counts[dataset]
    task_id = int(task.get("tid", -1))
    if task_id != expected_task_id:
        raise RuntimeError(
            f"{source}: task id {task_id} does not match {dataset} ({expected_task_id})"
        )
    repeat = int(task.get("repeat", -1))
    fold = int(task.get("fold", -1))
    expected_repeats, remainder = divmod(expected_splits, 3)
    if remainder or repeat not in range(expected_repeats) or fold not in range(3):
        raise RuntimeError(f"{source}: unexpected split r{repeat}f{fold} for {dataset}")
    registered_fold = 3 * repeat + fold
    split_idx = task.get("split_idx")
    if split_idx is not None and int(split_idx) != registered_fold:
        raise RuntimeError(
            f"{source}: split_idx={split_idx} does not equal {registered_fold}"
        )

    info = record.get("method_metadata", {}).get("info", {})
    if info.get("is_valid") is not True or info.get("can_infer") is not True:
        raise RuntimeError(f"{source}: result is not a successful inferable model")

    hyperparameters = _user_hyperparameters(record)
    if hyperparameters == {}:
        config = "default"
    elif hyperparameters == dict(frozen_candidate):
        config = "candidate"
    else:
        raise RuntimeError(
            f"{source}: unexpected non-AutoGluon hyperparameters {hyperparameters!r}"
        )

    memory = record.get("memory_usage")
    if not isinstance(memory, Mapping):
        raise RuntimeError(f"{source}: missing memory_usage")
    return {
        "dataset": dataset,
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "registered_fold": registered_fold,
        "config": config,
        "rmse": _positive_finite(
            record.get("metric_error"), f"{source}: metric_error"
        ),
        "val_rmse": _positive_finite(
            record.get("metric_error_val"), f"{source}: metric_error_val"
        ),
        "train_time_s": _positive_finite(
            record.get("time_train_s"), f"{source}: time_train_s"
        ),
        "infer_time_s": _positive_finite(
            record.get("time_infer_s"), f"{source}: time_infer_s"
        ),
        "peak_memory_bytes": _positive_finite(
            memory.get("peak_mem_cpu"), f"{source}: peak_mem_cpu"
        ),
        "framework": str(record.get("framework", "")),
        "source": source,
    }


def load_local_rows(
    input_dir: Path,
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
) -> list[dict]:
    """Read every gzip result in ``input_dir`` and require the complete panel."""
    paths = sorted(input_dir.rglob("results.pkl"))
    paths.extend(sorted(input_dir.rglob("results.pkl.gz")))
    if not paths:
        raise RuntimeError(f"no gzip result pickles found under {input_dir}")

    rows = []
    for path in paths:
        try:
            with gzip.open(path, "rb") as stream:
                record = pickle.load(stream)
        except Exception as exc:
            raise RuntimeError(f"failed to read gzip result {path}: {exc}") from exc
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{path}: result payload must be a mapping")
        rows.append(
            local_result_row(
                record,
                source=path,
                task_split_counts=task_split_counts,
            )
        )
    validate_local_rows(rows, task_split_counts=task_split_counts)
    return rows


def validate_local_rows(
    rows: Sequence[Mapping],
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
) -> None:
    """Require one successful default and candidate row per expected split."""
    expected_per_config = sum(count for _, count in task_split_counts.values())
    seen: set[tuple[str, str, int, int]] = set()
    counts = defaultdict(int)
    for row in rows:
        config = row.get("config")
        dataset = row.get("dataset")
        if config not in {"default", "candidate"} or dataset not in task_split_counts:
            raise RuntimeError(f"unexpected local row: {row!r}")
        key = (str(config), str(dataset), int(row["repeat"]), int(row["fold"]))
        if key in seen:
            raise RuntimeError(f"duplicate local result for {key}")
        seen.add(key)
        counts[config] += 1

    for config in ("default", "candidate"):
        if counts[config] != expected_per_config:
            raise RuntimeError(
                f"expected {expected_per_config} unique successful {config} rows, "
                f"got {counts[config]}"
            )
        for dataset, (_, split_count) in task_split_counts.items():
            for registered_fold in range(split_count):
                key = (
                    config,
                    dataset,
                    registered_fold // 3,
                    registered_fold % 3,
                )
                if key not in seen:
                    raise RuntimeError(f"missing local result for {key}")
    if len(rows) != 2 * expected_per_config:
        raise RuntimeError(
            f"expected exactly {2 * expected_per_config} local rows, got {len(rows)}"
        )


def registered_chimera_rows(
    results,
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
) -> list[dict]:
    """Validate and normalize registered, non-imputed CHIMERA default rows."""
    records = results.to_dict("records") if hasattr(results, "to_dict") else list(results)
    selected = [
        row
        for row in records
        if row.get("dataset") in task_split_counts
        and row.get("method") == "CHIMERA (default)"
    ]
    expected = sum(count for _, count in task_split_counts.values())
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} registered CHIMERA rows, got {len(selected)}"
        )

    out = []
    seen = set()
    for row in selected:
        if bool(row.get("imputed")):
            raise RuntimeError("registered CHIMERA coverage contains imputed rows")
        dataset = row["dataset"]
        registered_fold = int(row["fold"])
        split_count = task_split_counts[dataset][1]
        if registered_fold not in range(split_count):
            raise RuntimeError(
                f"unexpected registered CHIMERA fold {registered_fold} for {dataset}"
            )
        key = (dataset, registered_fold)
        if key in seen:
            raise RuntimeError(f"duplicate registered CHIMERA row for {key}")
        seen.add(key)
        if row.get("problem_type") != "regression" or row.get("metric") != "rmse":
            raise RuntimeError(f"registered CHIMERA row {key} is not regression/rmse")
        out.append(
            {
                "dataset": dataset,
                "repeat": registered_fold // 3,
                "fold": registered_fold % 3,
                "registered_fold": registered_fold,
                "rmse": _positive_finite(row.get("metric_error"), f"CHIMERA {key} RMSE"),
                "val_rmse": _positive_finite(
                    row.get("metric_error_val"), f"CHIMERA {key} validation RMSE"
                ),
            }
        )
    for dataset, (_, split_count) in task_split_counts.items():
        for registered_fold in range(split_count):
            if (dataset, registered_fold) not in seen:
                raise RuntimeError(
                    f"missing registered CHIMERA row for {(dataset, registered_fold)}"
                )
    return out


def _ratio_fields(prefix: str, numerator: float, denominator: float) -> dict:
    ratio = numerator / denominator
    return {
        f"{prefix}_log_ratio": math.log(ratio),
        f"{prefix}_ratio": ratio,
        f"{prefix}_pct": 100.0 * (ratio - 1.0),
    }


def _aggregate_ratio_fields(rows: Sequence[Mapping], ratio_prefix: str) -> dict:
    log_key = f"{ratio_prefix}_log_ratio"
    mean_log = math.fsum(float(row[log_key]) for row in rows) / len(rows)
    ratio = math.exp(mean_log)
    return {"log_ratio": mean_log, "ratio": ratio, "pct": 100.0 * (ratio - 1.0)}


def _dataset_equal_aggregate(
    rows: Sequence[Mapping], datasets: Sequence[str], ratio_prefix: str
) -> dict:
    dataset_logs = []
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            raise RuntimeError(f"no rows for aggregate dataset {dataset}")
        dataset_logs.append(
            math.fsum(float(row[f"{ratio_prefix}_log_ratio"]) for row in selected)
            / len(selected)
        )
    mean_log = math.fsum(dataset_logs) / len(dataset_logs)
    ratio = math.exp(mean_log)
    return {"log_ratio": mean_log, "ratio": ratio, "pct": 100.0 * (ratio - 1.0)}


def analyze_rows(
    local_rows: Sequence[Mapping],
    chimera_rows: Sequence[Mapping],
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
) -> tuple[list[dict], dict]:
    """Build the tidy paired rows and exact predeclared decision summary."""
    validate_local_rows(local_rows, task_split_counts=task_split_counts)
    expected = sum(count for _, count in task_split_counts.values())
    if len(chimera_rows) != expected:
        raise RuntimeError(f"expected {expected} normalized CHIMERA rows, got {len(chimera_rows)}")

    local_index = {
        (row["config"], row["dataset"], int(row["repeat"]), int(row["fold"])): row
        for row in local_rows
    }
    chimera_index = {}
    for row in chimera_rows:
        key = (row["dataset"], int(row["repeat"]), int(row["fold"]))
        if key in chimera_index:
            raise RuntimeError(f"duplicate normalized CHIMERA row for {key}")
        chimera_index[key] = row

    datasets = list(task_split_counts)
    tidy = []
    for dataset in datasets:
        task_id, split_count = task_split_counts[dataset]
        for registered_fold in range(split_count):
            repeat, fold = divmod(registered_fold, 3)
            default = local_index[("default", dataset, repeat, fold)]
            candidate = local_index[("candidate", dataset, repeat, fold)]
            chimera_key = (dataset, repeat, fold)
            if chimera_key not in chimera_index:
                raise RuntimeError(f"missing normalized CHIMERA row for {chimera_key}")
            chimera = chimera_index[chimera_key]
            row = {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered_fold,
            }
            for metric in LOCAL_METRICS:
                row[f"default_{metric}"] = float(default[metric])
                row[f"candidate_{metric}"] = float(candidate[metric])
                row.update(
                    _ratio_fields(
                        RATIO_PREFIXES[metric],
                        float(candidate[metric]),
                        float(default[metric]),
                    )
                )
            row["chimera_rmse"] = float(chimera["rmse"])
            row["chimera_val_rmse"] = float(chimera["val_rmse"])
            row.update(
                _ratio_fields(
                    "default_chimera_rmse",
                    row["default_rmse"],
                    row["chimera_rmse"],
                )
            )
            row.update(
                _ratio_fields(
                    "candidate_chimera_rmse",
                    row["candidate_rmse"],
                    row["chimera_rmse"],
                )
            )
            row.update(
                _ratio_fields(
                    "default_chimera_val_rmse",
                    row["default_val_rmse"],
                    row["chimera_val_rmse"],
                )
            )
            row.update(
                _ratio_fields(
                    "candidate_chimera_val_rmse",
                    row["candidate_val_rmse"],
                    row["chimera_val_rmse"],
                )
            )
            row["candidate_wins"] = row["candidate_default_rmse_ratio"] < 1.0
            tidy.append(row)

    dataset_summaries = []
    repeat_details = []
    for dataset in datasets:
        dataset_rows = [row for row in tidy if row["dataset"] == dataset]
        summary = {"dataset": dataset, "split_count": len(dataset_rows)}
        for metric in LOCAL_METRICS:
            prefix = RATIO_PREFIXES[metric]
            aggregate = _aggregate_ratio_fields(dataset_rows, prefix)
            summary[f"{prefix}_log_ratio"] = aggregate["log_ratio"]
            summary[f"{prefix}_ratio"] = aggregate["ratio"]
            summary[f"{prefix}_pct"] = aggregate["pct"]
        for prefix in (
            "default_chimera_rmse",
            "candidate_chimera_rmse",
            "default_chimera_val_rmse",
            "candidate_chimera_val_rmse",
        ):
            aggregate = _aggregate_ratio_fields(dataset_rows, prefix)
            summary[f"{prefix}_log_ratio"] = aggregate["log_ratio"]
            summary[f"{prefix}_ratio"] = aggregate["ratio"]
            summary[f"{prefix}_pct"] = aggregate["pct"]

        split_ratios = [row["candidate_default_rmse_ratio"] for row in dataset_rows]
        summary["split_wins"] = sum(ratio < 1.0 for ratio in split_ratios)
        summary["split_losses"] = sum(ratio > 1.0 for ratio in split_ratios)
        summary["split_ties"] = sum(ratio == 1.0 for ratio in split_ratios)
        worst = max(dataset_rows, key=lambda row: row["candidate_default_rmse_ratio"])
        summary["worst_split_ratio"] = worst["candidate_default_rmse_ratio"]
        summary["worst_split_pct"] = worst["candidate_default_rmse_pct"]
        summary["worst_split"] = f"r{worst['repeat']}f{worst['fold']}"

        repeat_ratios = []
        repeat_count = task_split_counts[dataset][1] // 3
        for repeat in range(repeat_count):
            selected = [row for row in dataset_rows if row["repeat"] == repeat]
            aggregate = _aggregate_ratio_fields(selected, "candidate_default_rmse")
            repeat_ratios.append(aggregate["ratio"])
            repeat_details.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "split_count": len(selected),
                    "candidate_default_rmse_log_ratio": aggregate["log_ratio"],
                    "candidate_default_rmse_ratio": aggregate["ratio"],
                    "candidate_default_rmse_pct": aggregate["pct"],
                }
            )
        summary["repeat_count"] = repeat_count
        summary["repeat_wins"] = sum(ratio < 1.0 for ratio in repeat_ratios)
        summary["repeat_losses"] = sum(ratio > 1.0 for ratio in repeat_ratios)
        summary["repeat_ties"] = sum(ratio == 1.0 for ratio in repeat_ratios)
        summary["required_repeat_wins"] = 2 if repeat_count == 3 else 7
        summary["repeat_sign_gate"] = (
            summary["repeat_wins"] >= summary["required_repeat_wins"]
        )
        dataset_summaries.append(summary)

    equal_dataset = {}
    for metric in LOCAL_METRICS:
        equal_dataset[RATIO_PREFIXES[metric]] = _dataset_equal_aggregate(
            tidy, datasets, RATIO_PREFIXES[metric]
        )
    chimera_equal_dataset = {
        prefix: _dataset_equal_aggregate(tidy, datasets, prefix)
        for prefix in (
            "default_chimera_rmse",
            "candidate_chimera_rmse",
            "default_chimera_val_rmse",
            "candidate_chimera_val_rmse",
        )
    }

    common_repeats = []
    for repeat in range(3):
        selected = [row for row in tidy if row["repeat"] == repeat]
        if len(selected) != 3 * len(datasets):
            raise RuntimeError(
                f"common repeat {repeat} has {len(selected)} rows; "
                f"expected {3 * len(datasets)}"
            )
        item = {"repeat": repeat, "split_count": len(selected)}
        for metric in LOCAL_METRICS:
            prefix = RATIO_PREFIXES[metric]
            item[prefix] = _dataset_equal_aggregate(selected, datasets, prefix)
        common_repeats.append(item)

    rmse_ratios = [row["candidate_default_rmse_ratio"] for row in tidy]
    worst = max(tidy, key=lambda row: row["candidate_default_rmse_ratio"])
    split_summary = {
        "count": len(tidy),
        "wins": sum(ratio < 1.0 for ratio in rmse_ratios),
        "losses": sum(ratio > 1.0 for ratio in rmse_ratios),
        "ties": sum(ratio == 1.0 for ratio in rmse_ratios),
        "worst_dataset": worst["dataset"],
        "worst_split": f"r{worst['repeat']}f{worst['fold']}",
        "worst_ratio": worst["candidate_default_rmse_ratio"],
        "worst_pct": worst["candidate_default_rmse_pct"],
    }

    rmse_ratio = equal_dataset["candidate_default_rmse"]["ratio"]
    val_ratio = equal_dataset["candidate_default_val_rmse"]["ratio"]
    train_ratio = equal_dataset["candidate_default_train_time"]["ratio"]
    infer_ratio = equal_dataset["candidate_default_infer_time"]["ratio"]
    memory_ratio = equal_dataset["candidate_default_peak_memory"]["ratio"]
    gates = {
        "complete_unique_successful_local_and_non_imputed_chimera": True,
        "equal_dataset_rmse_improves_at_least_0_5pct": (
            rmse_ratio <= GATE_THRESHOLDS["equal_dataset_rmse_ratio_max"]
        ),
        "no_dataset_regresses_more_than_0_5pct": max(
            item["candidate_default_rmse_ratio"] for item in dataset_summaries
        )
        <= GATE_THRESHOLDS["dataset_rmse_ratio_max"],
        "no_split_regresses_more_than_2pct": max(rmse_ratios)
        <= GATE_THRESHOLDS["split_rmse_ratio_max"],
        "common_repeats_0_to_2_all_improve": all(
            item["candidate_default_rmse"]["ratio"] < 1.0
            for item in common_repeats
        ),
        "per_dataset_repeat_sign_counts_pass": all(
            item["repeat_sign_gate"] for item in dataset_summaries
        ),
        "validation_has_no_broad_direction_reversal": val_ratio
        <= GATE_THRESHOLDS["validation_equal_dataset_ratio_max"],
        "train_time_regression_within_20pct": train_ratio
        <= GATE_THRESHOLDS["train_time_ratio_max"],
        "inference_time_regression_within_10pct": infer_ratio
        <= GATE_THRESHOLDS["infer_time_ratio_max"],
        "peak_memory_regression_within_10pct": memory_ratio
        <= GATE_THRESHOLDS["peak_memory_ratio_max"],
    }
    gates["runtime_and_memory_acceptable"] = all(
        gates[key]
        for key in (
            "train_time_regression_within_20pct",
            "inference_time_regression_within_10pct",
            "peak_memory_regression_within_10pct",
        )
    )
    gates["advance"] = all(
        value for key, value in gates.items() if key not in {"advance"}
    )

    summary = {
        "protocol": "remaining-nine frozen candidate confirmation",
        "candidate": dict(FROZEN_CANDIDATE),
        "counts": {
            "datasets": len(datasets),
            "dataset_splits": expected,
            "local_default_rows": expected,
            "local_candidate_rows": expected,
            "registered_chimera_rows": expected,
        },
        "thresholds": dict(GATE_THRESHOLDS),
        "equal_dataset": equal_dataset,
        "split_summary": split_summary,
        "common_repeat_aggregates": common_repeats,
        "repeat_aggregates": repeat_details,
        "datasets": dataset_summaries,
        "matched_chimera": {"equal_dataset": chimera_equal_dataset},
        "gates": gates,
    }
    return tidy, summary


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty analysis CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    local_rows = load_local_rows(args.input_dir)

    from tabarena.contexts import TabArenaContext

    registered = TabArenaContext().load_results(methods=["ChimeraBoost"])
    chimera_rows = registered_chimera_rows(registered)
    tidy, summary = analyze_rows(local_rows, chimera_rows)
    _write_csv(args.csv, tidy)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"analyzed {len(tidy)} paired splits; advance={summary['gates']['advance']}; "
        f"wrote {args.csv} and {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
