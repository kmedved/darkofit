"""Analyze the frozen 81-job same-machine product performance campaign.

The input is the output directory produced by
``run_tabarena_same_machine_performance.py``. Results are accepted only when
all 27 coordinates contain exactly one canonical row for each of the three
frozen product configurations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_tabarena_regression_remaining9 import (  # noqa: E402
    FROZEN_CANDIDATE,
    TASK_SPLIT_COUNTS,
)
from benchmarks.run_tabarena_same_machine_performance import (  # noqa: E402
    FROZEN_CHIMERA_COMMIT,
    FROZEN_CHIMERA_VERSION,
    SPLIT_INDICES,
    TIME_LIMIT_SECONDS,
)


ANALYSIS_SCHEMA_VERSION = 1
SPLIT_COORDINATES = (
    (0, 0, 0),
    (1, 1, 4),
    (2, 2, 8),
)
CONFIG_SPECS = {
    "darkofit_default": {
        "framework": (
            "DarkoFitSameMachine_c1_same_machine_darkofit_default_BAG_L1"
        ),
        "model_cls": "SameMachineDarkoFitModel",
        "model_type": "DARKOPERF",
        "name_prefix": "DarkoFitSameMachine",
        "name_suffix": "_c1_same_machine_darkofit_default",
        "parameters": {},
    },
    "darkofit_candidate": {
        "framework": (
            "DarkoFitSameMachine_c1_same_machine_darkofit_candidate_BAG_L1"
        ),
        "model_cls": "SameMachineDarkoFitModel",
        "model_type": "DARKOPERF",
        "name_prefix": "DarkoFitSameMachine",
        "name_suffix": "_c1_same_machine_darkofit_candidate",
        "parameters": dict(FROZEN_CANDIDATE),
    },
    "chimeraboost_default": {
        "framework": (
            "ChimeraBoostSameMachine_c1_same_machine_chimera_default_BAG_L1"
        ),
        "model_cls": "SameMachineChimeraBoostModel",
        "model_type": "CHIMERAPERF",
        "name_prefix": "ChimeraBoostSameMachine",
        "name_suffix": "_c1_same_machine_chimera_default",
        "parameters": {},
    },
}
FRAMEWORK_TO_CONFIG = {
    spec["framework"]: config for config, spec in CONFIG_SPECS.items()
}
METRICS = (
    "rmse",
    "val_rmse",
    "train_time_s",
    "preprocessing_time_s",
    "infer_time_s",
    "peak_rss_bytes",
    "model_size_all_children_bytes",
    "model_size_low_memory_bytes",
)
PAIRWISE_COMPARISONS = (
    ("candidate_vs_darkofit_default", "darkofit_candidate", "darkofit_default"),
    ("darkofit_default_vs_chimeraboost", "darkofit_default", "chimeraboost_default"),
    ("candidate_vs_chimeraboost", "darkofit_candidate", "chimeraboost_default"),
)
EXPECTED_COORDINATES = {
    (dataset, repeat, fold)
    for dataset in TASK_SPLIT_COUNTS
    for repeat, fold, _ in SPLIT_COORDINATES
}
EXPECTED_ROWS = len(EXPECTED_COORDINATES) * len(CONFIG_SPECS)


def _finite(value, field: str, *, positive: bool = True) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        condition = "positive and finite" if positive else "finite"
        raise RuntimeError(f"{field} must be {condition}, got {value!r}")
    return out


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be a positive integer, got {value!r}")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not an integer: {value!r}") from exc
    if out <= 0 or float(value) != out:
        raise RuntimeError(f"{field} must be a positive integer, got {value!r}")
    return out


def geometric_mean(values: Iterable[float]) -> float:
    checked = [_finite(value, "geometric-mean input") for value in values]
    if not checked:
        raise RuntimeError("cannot aggregate an empty sequence")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("cannot aggregate an empty sequence")
    return math.fsum(values) / len(values)


def validate_provenance(provenance: Mapping) -> dict:
    """Require the runner's canonical source and machine provenance."""
    required_equal = {
        "package": "chimeraboost",
        "commit": FROZEN_CHIMERA_COMMIT,
        "version_expected": FROZEN_CHIMERA_VERSION,
        "version_imported": FROZEN_CHIMERA_VERSION,
        "dirty": False,
        "darkofit_package": "darkofit",
        "darkofit_dirty": False,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "split_indices": list(SPLIT_INDICES),
        "candidate": dict(FROZEN_CANDIDATE),
    }
    for field, expected in required_equal.items():
        if provenance.get(field) != expected:
            raise RuntimeError(
                f"noncanonical provenance {field}: "
                f"{provenance.get(field)!r} != {expected!r}"
            )
    for field in (
        "repository",
        "module_file",
        "darkofit_repository",
        "darkofit_module_file",
        "darkofit_version_imported",
    ):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise RuntimeError(f"provenance {field} must be a nonempty string")
    darkofit_commit = provenance.get("darkofit_commit")
    if not isinstance(darkofit_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", darkofit_commit
    ):
        raise RuntimeError("provenance darkofit_commit must be a full Git hash")
    warmup_seconds = _finite(
        provenance.get("chimeraboost_warmup_seconds"),
        "chimeraboost_warmup_seconds",
        positive=False,
    )
    if warmup_seconds < 0.0:
        raise RuntimeError("chimeraboost_warmup_seconds must be nonnegative")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("provenance runtime must be a mapping")
    for field in (
        "python_version",
        "python_implementation",
        "python_executable",
        "platform",
        "machine",
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RuntimeError(f"provenance runtime.{field} must be nonempty")
    _positive_int(runtime.get("logical_cpu_count"), "runtime.logical_cpu_count")
    return json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))


def load_provenance(input_dir: Path) -> dict:
    path = input_dir / "provenance.json"
    if not path.is_file():
        raise RuntimeError(f"missing performance provenance: {path}")
    try:
        provenance = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read provenance {path}: {exc}") from exc
    if not isinstance(provenance, Mapping):
        raise RuntimeError("performance provenance must be a JSON object")
    return validate_provenance(provenance)


def _classify_config(record: Mapping, source: str) -> str:
    framework = record.get("framework")
    config = FRAMEWORK_TO_CONFIG.get(framework)
    if config is None:
        raise RuntimeError(f"{source}: unexpected framework {framework!r}")
    spec = CONFIG_SPECS[config]
    metadata = record.get("method_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"{source}: missing method_metadata")
    for field in ("model_cls", "model_type", "name_prefix"):
        if metadata.get(field) != spec[field]:
            raise RuntimeError(
                f"{source}: {field}={metadata.get(field)!r}; "
                f"expected {spec[field]!r}"
            )
    hyperparameters = metadata.get("model_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise RuntimeError(f"{source}: missing model_hyperparameters")
    hyperparameters = dict(hyperparameters)
    ag_args = hyperparameters.pop("ag_args", None)
    if not isinstance(ag_args, Mapping) or dict(ag_args) != {
        "name_suffix": spec["name_suffix"]
    }:
        raise RuntimeError(f"{source}: noncanonical ag_args name suffix")
    ensemble = hyperparameters.pop("ag_args_ensemble", None)
    if not isinstance(ensemble, Mapping):
        raise RuntimeError(f"{source}: missing ag_args_ensemble")
    ensemble_expected = {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
    }
    for field, expected in ensemble_expected.items():
        if ensemble.get(field) != expected:
            raise RuntimeError(
                f"{source}: ag_args_ensemble.{field} is noncanonical"
            )
    max_time = ensemble.get("ag.max_time_limit")
    if max_time is None or int(max_time) != TIME_LIMIT_SECONDS:
        raise RuntimeError(f"{source}: noncanonical ag.max_time_limit")
    if set(ensemble) != set(ensemble_expected) | {"ag.max_time_limit"}:
        raise RuntimeError(f"{source}: unexpected ag_args_ensemble fields")
    if hyperparameters != spec["parameters"]:
        raise RuntimeError(
            f"{source}: product parameters {hyperparameters!r}; "
            f"expected {spec['parameters']!r}"
        )
    return config


def performance_result_row(record: Mapping, *, source: str) -> dict:
    """Validate and normalize one canonical same-machine result payload."""
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: expected regression/rmse result")
    config = _classify_config(record, source)
    task = record.get("task_metadata")
    if not isinstance(task, Mapping):
        raise RuntimeError(f"{source}: missing task_metadata")
    dataset = task.get("name")
    if dataset not in TASK_SPLIT_COUNTS:
        raise RuntimeError(f"{source}: unexpected dataset {dataset!r}")
    task_id = int(task.get("tid", -1))
    if task_id != TASK_SPLIT_COUNTS[dataset][0]:
        raise RuntimeError(f"{source}: task id does not match {dataset}")
    repeat = int(task.get("repeat", -1))
    fold = int(task.get("fold", -1))
    coordinate = next(
        (
            registered
            for expected_repeat, expected_fold, registered in SPLIT_COORDINATES
            if (repeat, fold) == (expected_repeat, expected_fold)
        ),
        None,
    )
    if coordinate is None:
        raise RuntimeError(f"{source}: noncanonical split r{repeat}f{fold}")
    split_idx = task.get("split_idx")
    if split_idx is not None and int(split_idx) != coordinate:
        raise RuntimeError(f"{source}: split_idx does not match repeat/fold")
    if task.get("sample", 0) != 0:
        raise RuntimeError(f"{source}: noncanonical task sample")

    metadata = record["method_metadata"]
    info = metadata.get("info")
    if not isinstance(info, Mapping):
        raise RuntimeError(f"{source}: missing model info")
    if info.get("is_valid") is not True or info.get("can_infer") is not True:
        raise RuntimeError(f"{source}: model is not successful and inferable")
    bagged = info.get("bagged_info")
    children = info.get("children_info")
    if not isinstance(bagged, Mapping) or not isinstance(children, Mapping):
        raise RuntimeError(f"{source}: missing AutoGluon bag/child info")
    child_count = _positive_int(
        bagged.get("num_child_models"), f"{source}: num_child_models"
    )
    if child_count != 8 or len(children) != 8:
        raise RuntimeError(f"{source}: expected exactly 8 bagged children")
    preprocessing_seconds = 0.0
    preprocessing_calls = 0
    for child_name, child in sorted(children.items()):
        if not isinstance(child, Mapping):
            raise RuntimeError(f"{source}: invalid child info for {child_name}")
        preprocessing_seconds += _finite(
            child.get("preprocessing_fit_transform_seconds"),
            f"{source}: {child_name} preprocessing time",
        )
        preprocessing_calls += _positive_int(
            child.get("preprocessing_fit_transform_calls"),
            f"{source}: {child_name} preprocessing calls",
        )
        if config == "chimeraboost_default":
            if (
                child.get("benchmark_package") != "chimeraboost"
                or child.get("benchmark_package_version")
                != FROZEN_CHIMERA_VERSION
                or child.get("benchmark_source_commit") != FROZEN_CHIMERA_COMMIT
            ):
                raise RuntimeError(
                    f"{source}: noncanonical ChimeraBoost child provenance"
                )

    maximum_size = _positive_int(
        bagged.get("max_memory_size"), f"{source}: max_memory_size"
    )
    minimum_size = _positive_int(
        bagged.get("min_memory_size"), f"{source}: min_memory_size"
    )
    if maximum_size < minimum_size:
        raise RuntimeError(f"{source}: max model size is below min model size")
    memory = record.get("memory_usage")
    if not isinstance(memory, Mapping):
        raise RuntimeError(f"{source}: missing memory_usage")
    return {
        "dataset": dataset,
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "registered_fold": coordinate,
        "config": config,
        "framework": record["framework"],
        "status": "ok",
        "imputed": False,
        "rmse": _finite(record.get("metric_error"), f"{source}: RMSE"),
        "val_rmse": _finite(
            record.get("metric_error_val"), f"{source}: validation RMSE"
        ),
        "train_time_s": _finite(
            record.get("time_train_s"), f"{source}: training time"
        ),
        "preprocessing_time_s": preprocessing_seconds,
        "preprocessing_fit_transform_calls": preprocessing_calls,
        "infer_time_s": _finite(
            record.get("time_infer_s"), f"{source}: inference time"
        ),
        "peak_rss_bytes": _finite(
            memory.get("peak_mem_cpu"), f"{source}: peak RSS"
        ),
        "model_size_all_children_bytes": maximum_size,
        "model_size_low_memory_bytes": minimum_size,
        "child_model_count": child_count,
        "source": source,
    }


def validate_performance_rows(rows: Sequence[Mapping]) -> None:
    """Require the exact canonical 27-coordinate by three-config panel."""
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected exactly {EXPECTED_ROWS} rows, got {len(rows)}")
    seen = set()
    for row in rows:
        key = (row["config"], row["dataset"], row["repeat"], row["fold"])
        if key in seen:
            raise RuntimeError(f"duplicate same-machine result for {key}")
        seen.add(key)
    expected = {
        (config, dataset, repeat, fold)
        for config in CONFIG_SPECS
        for dataset, repeat, fold in EXPECTED_COORDINATES
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"noncanonical same-machine panel; missing={missing}, "
            f"unexpected={unexpected}"
        )


def load_performance_rows(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.rglob("results.pkl"))
    paths.extend(sorted(input_dir.rglob("results.pkl.gz")))
    if len(paths) != EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_ROWS} gzip result pickles, found {len(paths)}"
        )
    rows = []
    for path in paths:
        try:
            with gzip.open(path, "rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise RuntimeError(f"failed to read gzip result {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{path}: result payload must be a mapping")
        source = str(path.relative_to(input_dir))
        rows.append(performance_result_row(payload, source=source))
    validate_performance_rows(rows)
    return sorted(
        rows,
        key=lambda row: (
            list(TASK_SPLIT_COUNTS).index(row["dataset"]),
            row["repeat"],
            row["fold"],
            list(CONFIG_SPECS).index(row["config"]),
        ),
    )


def _descriptive(values: Sequence[float]) -> dict:
    return {
        "geometric_mean": geometric_mean(values),
        "arithmetic_mean": _mean(values),
        "total": math.fsum(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _ratio_summary(log_ratios: Sequence[float]) -> dict:
    mean_log = _mean(log_ratios)
    ratio = math.exp(mean_log)
    return {
        "log_ratio": mean_log,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def analyze_performance_rows(rows: Sequence[Mapping], provenance: Mapping) -> dict:
    """Return deterministic raw, equal-dataset, and paired aggregates."""
    validate_performance_rows(rows)
    provenance = validate_provenance(provenance)
    datasets = list(TASK_SPLIT_COUNTS)
    row_index = {
        (row["config"], row["dataset"], row["repeat"], row["fold"]): row
        for row in rows
    }

    config_aggregates = {}
    for config in CONFIG_SPECS:
        dataset_aggregates = {}
        for dataset in datasets:
            selected = [
                row
                for row in rows
                if row["config"] == config and row["dataset"] == dataset
            ]
            dataset_aggregates[dataset] = {
                metric: _descriptive([float(row[metric]) for row in selected])
                for metric in METRICS
            }
        equal_dataset = {}
        for metric in METRICS:
            dataset_geometric_means = [
                dataset_aggregates[dataset][metric]["geometric_mean"]
                for dataset in datasets
            ]
            dataset_arithmetic_means = [
                dataset_aggregates[dataset][metric]["arithmetic_mean"]
                for dataset in datasets
            ]
            equal_dataset[metric] = {
                "geometric_mean_of_dataset_geometric_means": geometric_mean(
                    dataset_geometric_means
                ),
                "arithmetic_mean_of_dataset_means": _mean(
                    dataset_arithmetic_means
                ),
            }
        config_rows = [row for row in rows if row["config"] == config]
        config_aggregates[config] = {
            "equal_dataset": equal_dataset,
            "overall_totals": {
                metric: math.fsum(float(row[metric]) for row in config_rows)
                for metric in METRICS
            },
            "datasets": dataset_aggregates,
        }

    pairwise = {}
    for name, numerator, denominator in PAIRWISE_COMPARISONS:
        dataset_summaries = {}
        equal_dataset_logs = {metric: [] for metric in METRICS}
        coordinate_logs = {metric: [] for metric in METRICS}
        for dataset in datasets:
            metric_summaries = {}
            for metric in METRICS:
                logs = []
                for repeat, fold, _ in SPLIT_COORDINATES:
                    numerator_value = float(
                        row_index[(numerator, dataset, repeat, fold)][metric]
                    )
                    denominator_value = float(
                        row_index[(denominator, dataset, repeat, fold)][metric]
                    )
                    log_ratio = math.log(numerator_value) - math.log(
                        denominator_value
                    )
                    logs.append(log_ratio)
                    coordinate_logs[metric].append(log_ratio)
                metric_summaries[metric] = _ratio_summary(logs)
                equal_dataset_logs[metric].append(_mean(logs))
            dataset_summaries[dataset] = metric_summaries
        equal_dataset = {
            metric: _ratio_summary(equal_dataset_logs[metric])
            for metric in METRICS
        }
        coordinate_summary = {}
        for metric in METRICS:
            ratios = [math.exp(value) for value in coordinate_logs[metric]]
            coordinate_summary[metric] = {
                "count": len(ratios),
                "numerator_better": sum(ratio < 1.0 for ratio in ratios),
                "ties": sum(ratio == 1.0 for ratio in ratios),
                "numerator_worse": sum(ratio > 1.0 for ratio in ratios),
                "best_ratio": min(ratios),
                "worst_ratio": max(ratios),
            }
        pairwise[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "equal_dataset": equal_dataset,
            "coordinate_summary": coordinate_summary,
            "datasets": dataset_summaries,
        }

    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "protocol": "conditional same-machine product performance",
        "counts": {
            "datasets": len(datasets),
            "coordinates": len(EXPECTED_COORDINATES),
            "configs": len(CONFIG_SPECS),
            "rows": len(rows),
            "bagged_children": sum(int(row["child_model_count"]) for row in rows),
        },
        "metric_units": {
            "rmse": "target units; lower is better",
            "val_rmse": "target units; lower is better",
            "train_time_s": "seconds; lower is better",
            "preprocessing_time_s": "seconds; lower is better",
            "infer_time_s": "seconds; lower is better",
            "peak_rss_bytes": "process-wide bytes; lower is better",
            "model_size_all_children_bytes": "AutoGluon bytes; lower is better",
            "model_size_low_memory_bytes": "AutoGluon bytes; lower is better",
        },
        "provenance": provenance,
        "config_aggregates": config_aggregates,
        "pairwise": pairwise,
        "limitations": [
            "Peak RSS is process-wide and can retain allocations from earlier jobs.",
            "The comparison is between product defaults, not matched hyperparameters.",
            "ChimeraBoost warmup is recorded separately and excluded from fit times.",
            "Preprocessing time sums every instrumented child fit_transform lane.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty performance CSV")
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
    input_dir = args.input_dir.resolve()
    provenance = load_provenance(input_dir)
    rows = load_performance_rows(input_dir)
    summary = analyze_performance_rows(rows, provenance)
    _write_csv(args.csv, rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"analyzed {len(rows)} canonical same-machine jobs; "
        f"wrote {args.csv} and {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
