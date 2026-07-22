"""Immutable decision rule and exact grid for M6 quality successor v3."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_ID = "m6-quality-successor-v3"
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

# These are development-triage harm and concentration bounds, not shipping
# bars or a minimum worthwhile movement.  See the v3 contract.
MAX_AGGREGATE_RATIO = 1.0
MAX_GROUP_RATIO = 1.02
MAX_LOO_RATIO = 1.003


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise RuntimeError("M6 quality ratios must be nonnegative and finite")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _validated_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(ratios, Mapping) or not ratios:
        raise RuntimeError("M6 quality decision requires named ratios")
    numeric: dict[str, float] = {}
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
    return numeric


def _validated_groups(
    ratios: Mapping[str, float], groups: Mapping[str, str] | None
) -> dict[str, str]:
    if groups is None:
        normalized = {name: name for name in ratios}
    else:
        if not isinstance(groups, Mapping) or set(groups) != set(ratios):
            raise RuntimeError("M6 quality groups must cover every ratio exactly")
        normalized = {}
        for name, group in groups.items():
            if not isinstance(group, str) or not group:
                raise RuntimeError("M6 quality groups must have nonempty names")
            normalized[name] = group
    if len(set(normalized.values())) < 2:
        raise RuntimeError("M6 quality concentration requires at least two groups")
    return normalized


def quality_decision(
    ratios: Mapping[str, float], *, groups: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Rank spent development evidence without a breadth or effect-size gate.

    Each historical replay case is its own group by default.  The standing
    runner passes dataset identities so seeds and weight modes do not pretend
    to be independent datasets.
    """

    numeric = _validated_ratios(ratios)
    normalized_groups = _validated_groups(numeric, groups)
    grouped_values: dict[str, list[float]] = {}
    for name, value in numeric.items():
        grouped_values.setdefault(normalized_groups[name], []).append(value)

    aggregate = _geomean(tuple(numeric.values()))
    group_ratios = {
        group: _geomean(values) for group, values in grouped_values.items()
    }
    leave_one_group_out = {
        omitted: _geomean([
            value
            for name, value in numeric.items()
            if normalized_groups[name] != omitted
        ])
        for omitted in grouped_values
    }
    worst_case, worst_case_ratio = max(numeric.items(), key=lambda item: item[1])
    worst_group, worst_group_ratio = max(
        group_ratios.items(), key=lambda item: item[1]
    )
    worst_loo_omission, worst_loo_ratio = max(
        leave_one_group_out.items(), key=lambda item: item[1]
    )
    gates = {
        "aggregate_not_worse": aggregate <= MAX_AGGREGATE_RATIO,
        "worst_group_at_most_1_02": worst_group_ratio <= MAX_GROUP_RATIO,
        "loo_concentration_at_most_1_003": worst_loo_ratio <= MAX_LOO_RATIO,
    }
    return {
        "disposition": "advance" if all(gates.values()) else "kill",
        "gates": gates,
        "case_count": len(numeric),
        "group_count": len(grouped_values),
        "geometric_mean_ratio": aggregate,
        "worst_case": worst_case,
        "worst_case_ratio": worst_case_ratio,
        "worst_group": worst_group,
        "worst_group_ratio": worst_group_ratio,
        "worst_loo_omission": worst_loo_omission,
        "worst_loo_ratio": worst_loo_ratio,
        "group_geometric_mean_ratio": group_ratios,
        "leave_one_group_out_geometric_mean_ratio": leave_one_group_out,
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

    ratios: dict[str, float] = {}
    groups: dict[str, str] = {}
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
        name = "/".join(key)
        ratios[name] = numerator / denominator
        groups[name] = key[0]

    return quality_decision(ratios, groups=groups)
