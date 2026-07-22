"""Immutable decision rule and exact grid for M6 quality successor v2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_ID = "m6-quality-successor-v2"
DATASETS = (
    "diabetes_resampled",
    "friedman_numeric",
    "wide_numeric_reg",
    "categorical_reg",
    "breast_cancer_resampled",
    "numeric_binary",
    "wine_resampled",
    "numeric_multiclass",
    "categorical_binary",
    "categorical_multiclass",
)
SIZES = ("medium",)
SEEDS = (0, 1, 2)
WEIGHT_MODES = ("none", "stress")
REPEAT = 3
THREADS = 4
ARMS = ("control_default", "candidate_default")

MAX_GEOMEAN_RATIO = 0.98
MIN_WIN_FRACTION = 0.60
MAX_CELL_RATIO = 1.02


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise RuntimeError("M6 quality ratios must be nonnegative and finite")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def quality_decision(ratios: Mapping[str, float]) -> dict[str, Any]:
    if not isinstance(ratios, Mapping) or not ratios:
        raise RuntimeError("M6 quality decision requires named ratios")
    numeric = {}
    for name, value in ratios.items():
        if not isinstance(name, str) or not name or isinstance(value, bool):
            raise RuntimeError("M6 quality ratios have invalid names or values")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("M6 quality ratios must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("M6 quality ratios must be nonnegative and finite")
        numeric[name] = value
    count = len(numeric)
    wins = sum(value < 1.0 for value in numeric.values())
    minimum_wins = int(math.ceil(MIN_WIN_FRACTION * count))
    geomean = _geomean(tuple(numeric.values()))
    worst_name, worst_ratio = max(numeric.items(), key=lambda item: item[1])
    gates = {
        "geomean_at_most_0_98": geomean <= MAX_GEOMEAN_RATIO,
        "wins_at_least_60_percent": wins >= minimum_wins,
        "no_cell_above_1_02": worst_ratio <= MAX_CELL_RATIO,
    }
    return {
        "disposition": "advance" if all(gates.values()) else "kill",
        "gates": gates,
        "case_count": count,
        "geometric_mean_ratio": geomean,
        "wins": wins,
        "minimum_wins": minimum_wins,
        "worst_case": worst_name,
        "worst_ratio": worst_ratio,
        "ratios": numeric,
    }


def expected_pair_keys() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (dataset, size, str(seed), weight)
        for size in SIZES
        for dataset in DATASETS
        for seed in SEEDS
        for weight in WEIGHT_MODES
    )


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["size"]),
        str(row["seed"]),
        str(row["weight_mode"]),
    )


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = set(expected_pair_keys())
    pairs: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = _pair_key(row)
        arm = str(row["variant"])
        if key not in expected or arm not in ARMS:
            raise RuntimeError("M6 quality rows contain an unexpected identity")
        pair = pairs.setdefault(key, {})
        if arm in pair:
            raise RuntimeError("M6 quality rows contain a duplicate arm")
        pair[arm] = row
    if set(pairs) != expected or any(
        set(pair) != set(ARMS) for pair in pairs.values()
    ):
        raise RuntimeError("M6 quality rows do not cover the exact paired grid")

    ratios = {}
    dataset_ratios = {dataset: [] for dataset in DATASETS}
    for key in expected_pair_keys():
        control = pairs[key]["control_default"]
        candidate = pairs[key]["candidate_default"]
        if control["primary_metric"] != candidate["primary_metric"]:
            raise RuntimeError("M6 paired primary metrics differ")
        try:
            denominator = float(control["primary_value"])
            numerator = float(candidate["primary_value"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("M6 paired primary values are invalid") from exc
        if (
            not math.isfinite(denominator)
            or not math.isfinite(numerator)
            or denominator <= 0.0
            or numerator < 0.0
        ):
            raise RuntimeError("M6 paired primary values are invalid")
        ratio = numerator / denominator
        ratios["/".join(key)] = ratio
        dataset_ratios[key[0]].append(ratio)

    decision = quality_decision(ratios)
    per_dataset = {
        dataset: _geomean(values) for dataset, values in dataset_ratios.items()
    }
    leave_one_dataset_out = {
        omitted: _geomean([
            ratio
            for dataset, values in dataset_ratios.items()
            if dataset != omitted
            for ratio in values
        ])
        for omitted in DATASETS
    }
    return {
        **decision,
        "per_dataset_geometric_mean_ratio": per_dataset,
        "leave_one_dataset_out_geometric_mean_ratio": leave_one_dataset_out,
    }
