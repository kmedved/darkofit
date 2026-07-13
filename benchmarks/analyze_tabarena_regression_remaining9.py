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
import hashlib
import io
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

try:
    from benchmarks.remaining9_run_manifest import (
        ATTESTATION_SCHEMA_VERSION,
        SCHEMA_VERSION,
        load_and_verify_completion_attestation,
        load_and_verify_manifest,
        sha256_file,
    )
    from benchmarks.run_tabarena_regression_remaining9 import (
        FROZEN_CANDIDATE,
        TASK_SPLIT_COUNTS,
    )
except ModuleNotFoundError:  # Direct execution: python benchmarks/analyze_*.py
    from remaining9_run_manifest import (
        ATTESTATION_SCHEMA_VERSION,
        SCHEMA_VERSION,
        load_and_verify_completion_attestation,
        load_and_verify_manifest,
        sha256_file,
    )
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
EXPECTED_AG_ENSEMBLE = {
    "model_random_seed": 0,
    "vary_seed_across_folds": True,
    "fold_fitting_strategy": "sequential_local",
    "ag.max_time_limit": 3_600,
}
EXPECTED_CHILD_DEFAULTS = {
    "iterations": 1_000,
    "early_stopping": True,
    "tree_mode": "catboost",
    "diagnostic_warnings": "never",
}
EXPECTED_CONFIG_NAMES = {
    "default": {
        "suffix": "_c1_remaining9_confirm",
        "framework": "DarkoFit_c1_remaining9_confirm_BAG_L1",
    },
    "candidate": {
        "suffix": "_c2_remaining9_confirm",
        "framework": "DarkoFit_c2_remaining9_confirm_BAG_L1",
    },
}
FROZEN_CHIMERA_ARTIFACT_SHA256 = (
    "02a093f42931b1b53dd4fae7b88d5dd545ee51083b49142136410f28a4232275"
)
FROZEN_CHIMERA_ARTIFACT_SIZE_BYTES = 83_420


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


def _as_mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def _validate_fitted_model_metadata(
    record: Mapping,
    *,
    source: str,
    frozen_candidate: Mapping[str, object],
) -> tuple[str, Mapping]:
    """Verify the exact bagging contract and all eight fitted child models."""
    method_metadata = _as_mapping(
        record.get("method_metadata"), f"{source}: method_metadata"
    )
    raw_hyperparameters = dict(
        _as_mapping(
            method_metadata.get("model_hyperparameters"),
            f"{source}: method_metadata.model_hyperparameters",
        )
    )
    ag_args = raw_hyperparameters.pop("ag_args", None)
    ag_args_ensemble = raw_hyperparameters.pop("ag_args_ensemble", None)
    if raw_hyperparameters == {}:
        config = "default"
    elif raw_hyperparameters == dict(frozen_candidate):
        config = "candidate"
    else:
        raise RuntimeError(
            f"{source}: unexpected non-AutoGluon hyperparameters "
            f"{raw_hyperparameters!r}"
        )

    expected_name = EXPECTED_CONFIG_NAMES[config]
    if ag_args != {"name_suffix": expected_name["suffix"]}:
        raise RuntimeError(f"{source}: unexpected ag_args for {config}: {ag_args!r}")
    if ag_args_ensemble != EXPECTED_AG_ENSEMBLE:
        raise RuntimeError(
            f"{source}: unexpected ag_args_ensemble: {ag_args_ensemble!r}"
        )
    if record.get("framework") != expected_name["framework"]:
        raise RuntimeError(
            f"{source}: unexpected framework {record.get('framework')!r} for {config}"
        )

    experiment = _as_mapping(
        record.get("experiment_metadata"), f"{source}: experiment_metadata"
    )
    if (
        experiment.get("experiment_cls") != "OOFExperimentRunner"
        or experiment.get("method_cls") != "AGSingleBagWrapper"
    ):
        raise RuntimeError(f"{source}: unexpected experiment implementation")

    info = _as_mapping(
        method_metadata.get("info"), f"{source}: method_metadata.info"
    )
    if info.get("is_valid") is not True or info.get("can_infer") is not True:
        raise RuntimeError(f"{source}: result is not a successful inferable model")
    if info.get("model_type") != "StackerEnsembleModel":
        raise RuntimeError(
            f"{source}: unexpected top-level model type {info.get('model_type')!r}"
        )

    bagged = _as_mapping(info.get("bagged_info"), f"{source}: bagged_info")
    expected_bagged = {
        "child_model_type": "DarkoFitModel",
        "num_child_models": 8,
        "child_model_names": [f"S1F{fold}" for fold in range(1, 9)],
        "_n_repeats": 1,
        "_k_per_n_repeat": [8],
        "child_hyperparameters_user": raw_hyperparameters,
        "child_hyperparameters_fit": {},
    }
    for field, expected in expected_bagged.items():
        if bagged.get(field) != expected:
            raise RuntimeError(
                f"{source}: bagged_info.{field}={bagged.get(field)!r}; "
                f"expected {expected!r}"
            )
    expected_base_child = {
        **EXPECTED_CHILD_DEFAULTS,
        **raw_hyperparameters,
        "random_state": 0,
    }
    if bagged.get("child_hyperparameters") != expected_base_child:
        raise RuntimeError(f"{source}: unexpected bagged child hyperparameters")

    children = _as_mapping(info.get("children_info"), f"{source}: children_info")
    if len(children) != 8:
        raise RuntimeError(f"{source}: expected 8 fitted child models, got {len(children)}")
    expected_child_names = {f"S1F{fold}" for fold in range(1, 9)}
    if set(children) != expected_child_names:
        raise RuntimeError(
            f"{source}: unexpected child model names {sorted(children)!r}"
        )
    child_seeds = []
    for child_name, child_value in children.items():
        child = _as_mapping(child_value, f"{source}: child {child_name}")
        if child.get("model_type") != "DarkoFitModel":
            raise RuntimeError(
                f"{source}: child {child_name} is not a DarkoFitModel"
            )
        if child.get("is_valid") is not True or child.get("can_infer") is not True:
            raise RuntimeError(
                f"{source}: child {child_name} is not valid and inferable"
            )
        if child.get("hyperparameters_user") != raw_hyperparameters:
            raise RuntimeError(
                f"{source}: child {child_name} user hyperparameters do not match"
            )
        if child.get("name") != child_name:
            raise RuntimeError(
                f"{source}: child {child_name} reports name {child.get('name')!r}"
            )
        if child.get("hyperparameters_fit") != {}:
            raise RuntimeError(
                f"{source}: child {child_name} has fitted hyperparameter overrides"
            )
        child_hyperparameters = _as_mapping(
            child.get("hyperparameters"),
            f"{source}: child {child_name} hyperparameters",
        )
        seed = child_hyperparameters.get("random_state")
        expected_child = {
            **EXPECTED_CHILD_DEFAULTS,
            **raw_hyperparameters,
            "random_state": seed,
        }
        if dict(child_hyperparameters) != expected_child:
            raise RuntimeError(
                f"{source}: child {child_name} effective hyperparameters do not match"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError(f"{source}: child {child_name} has invalid seed {seed!r}")
        expected_seed = int(str(child_name).removeprefix("S1F")) - 1
        if seed != expected_seed:
            raise RuntimeError(
                f"{source}: child {child_name} has seed {seed}; expected {expected_seed}"
            )
        child_seeds.append(seed)
    if sorted(child_seeds) != list(range(8)):
        raise RuntimeError(
            f"{source}: expected fold-wise child seeds 0..7, got {sorted(child_seeds)!r}"
        )
    return config, info


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

    config, _ = _validate_fitted_model_metadata(
        record,
        source=source,
        frozen_candidate=frozen_candidate,
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
        "framework": str(record["framework"]),
        "source": source,
    }


def load_local_rows(
    input_dir: Path,
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
    verified_result_payloads: Mapping[str, bytes] | None = None,
) -> list[dict]:
    """Read every gzip result in ``input_dir`` and require the complete panel."""
    if verified_result_payloads is None:
        paths = sorted(input_dir.rglob("results.pkl"))
        paths.extend(sorted(input_dir.rglob("results.pkl.gz")))
        inputs = [(path, None) for path in paths]
    else:
        inputs = [
            (input_dir / relative, payload)
            for relative, payload in sorted(verified_result_payloads.items())
        ]
    if not inputs:
        raise RuntimeError(f"no gzip result pickles found under {input_dir}")

    rows = []
    for path, verified_payload in inputs:
        try:
            if verified_payload is None:
                with gzip.open(path, "rb") as stream:
                    record = pickle.load(stream)
            else:
                record = pickle.loads(gzip.decompress(verified_payload))
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


def load_frozen_chimera_results(context) -> tuple[object, dict]:
    """Load the exact registered ChimeraBoost parquet bytes frozen for this run."""
    method = context.method_metadata(method="ChimeraBoost")
    if method.method_type != "config":
        raise RuntimeError(
            f"registered ChimeraBoost method type is {method.method_type!r}, not 'config'"
        )
    path = Path(method.path_results_hpo()).resolve()
    try:
        stat_before = path.stat()
        payload = path.read_bytes()
        stat_after = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"failed to read frozen ChimeraBoost results artifact {path}: {exc}"
        ) from exc
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError("ChimeraBoost results artifact changed while being read")
    digest = hashlib.sha256(payload).hexdigest()
    if (
        stat_after.st_size != FROZEN_CHIMERA_ARTIFACT_SIZE_BYTES
        or digest != FROZEN_CHIMERA_ARTIFACT_SHA256
    ):
        raise RuntimeError(
            "registered ChimeraBoost results artifact does not match the frozen bytes"
        )

    import pandas as pd

    try:
        results = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(
            f"failed to decode frozen ChimeraBoost results artifact {path}: {exc}"
        ) from exc
    return results, {
        "path": str(path),
        "sha256": digest,
        "size_bytes": stat_after.st_size,
        "mtime_ns": stat_after.st_mtime_ns,
        "method": "ChimeraBoost",
        "method_type": method.method_type,
    }


def normalized_chimera_rows_sha256(rows: Sequence[Mapping]) -> str:
    """Return a stable digest of the exact normalized comparison coordinates."""
    canonical = [
        {
            "dataset": str(row["dataset"]),
            "repeat": int(row["repeat"]),
            "fold": int(row["fold"]),
            "registered_fold": int(row["registered_fold"]),
            "rmse_hex": float(row["rmse"]).hex(),
            "val_rmse_hex": float(row["val_rmse"]).hex(),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["dataset"]),
                int(item["registered_fold"]),
            ),
        )
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
            "expected_child_fits": 16 * expected,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest, manifest_sha256 = load_and_verify_manifest(
        args.manifest, input_dir=args.input_dir
    )
    (
        attestation,
        attestation_sha256,
        verified_result_payloads,
    ) = load_and_verify_completion_attestation(
        args.attestation,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        input_dir=args.input_dir,
    )
    local_rows = load_local_rows(
        args.input_dir,
        verified_result_payloads=verified_result_payloads,
    )

    from tabarena.contexts import TabArenaContext

    registered, chimera_artifact = load_frozen_chimera_results(TabArenaContext())
    chimera_rows = registered_chimera_rows(registered)
    chimera_artifact["selected_rows"] = len(chimera_rows)
    chimera_artifact["selected_rows_sha256"] = normalized_chimera_rows_sha256(
        chimera_rows
    )
    tidy, summary = analyze_rows(local_rows, chimera_rows)
    summary["counts"]["validated_child_fits"] = summary["counts"][
        "expected_child_fits"
    ]
    summary["provenance"] = {
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "provenance_validator": {
            "path": str(Path(load_and_verify_manifest.__code__.co_filename).resolve()),
            "sha256": sha256_file(
                Path(load_and_verify_manifest.__code__.co_filename).resolve()
            ),
            "manifest_schema_version": SCHEMA_VERSION,
            "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        },
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "captured_at_utc": manifest["captured_at_utc"],
        "completion_attestation": {
            "path": str(args.attestation.resolve()),
            "sha256": attestation_sha256,
            "watch_started_utc": attestation["watch_started_utc"],
            "completed_utc": attestation["completed_utc"],
            "runner_pid": attestation["runner_pid"],
            "expected_results": attestation["expected_results"],
            "observed_results_count": len(attestation["observed_results"]),
            "run_manifest_sha256": attestation["run_manifest_sha256"],
        },
        "registered_chimera_artifact": chimera_artifact,
        "runner": manifest["runner"],
        "adapter": manifest["adapter"],
        "darkofit": manifest["darkofit"],
        "tabarena": manifest["tabarena"],
        "python": manifest["python"],
        "packages": manifest["packages"],
        "runtime_configuration": manifest["runtime_configuration"],
        "environment": manifest["environment"],
        "process": manifest["process"],
        "result_snapshot": manifest["result_snapshot"],
    }
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
