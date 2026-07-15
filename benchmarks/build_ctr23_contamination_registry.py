"""Freeze the OpenML-CTR23 inventory, contamination registry, and panel split.

The builder may download task data and official split indices, but it never
fits a model or reads benchmark results.  Target values are used only inside
``dataset_fingerprint`` to create opaque digests.  No target-derived statistic
is returned, printed, or used by the panel allocator.

Three independent artifacts are emitted so a later confirmation/lockbox runner
can bind provenance, exclusion judgments, and allocation separately.  Existing
artifacts are never overwritten; ``--verify-existing`` rebuilds all three and
requires byte equality.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


SCHEMA_VERSION = 3
ALGORITHM_VERSION = "ctr23-contamination-registry-v3"
EXPECTED_CTR23_TASK_COUNT = 35
VALUE_SKETCH_SIZE = 128
BLOOM_BITS_PER_ITEM = 10
BLOOM_HASH_COUNT = 7
SCHEMA_ROW_LANE_SALTS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
)
ARTIFACT_FILENAMES = {
    "suite": "ctr23_suite_snapshot.json",
    "registry": "ctr23_contamination_registry.json",
    "partition": "ctr23_partition.json",
}
DEFAULT_DECLARATIONS = Path(__file__).with_name(
    "ctr23_contamination_sources.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent


def canonical_json_bytes(value: Any) -> bytes:
    """Render the repository's canonical, newline-terminated JSON form."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"^https?://", "", text)
    return text.rstrip("/")


def _dtype_family(series: pd.Series) -> str:
    dtype = series.dtype
    if is_bool_dtype(dtype):
        return "bool"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    if is_numeric_dtype(dtype):
        return "numeric"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if is_string_dtype(dtype) or is_object_dtype(dtype):
        return "string"
    return normalize_name(dtype)


def _validate_numeric_dtype(dtype: Any) -> None:
    candidate = getattr(dtype, "numpy_dtype", dtype)
    kind = getattr(candidate, "kind", None)
    itemsize = getattr(candidate, "itemsize", None)
    if kind is None or itemsize is None:
        try:
            candidate = np.dtype(candidate)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"unsupported numeric dtype {dtype}") from exc
        kind = candidate.kind
        itemsize = candidate.itemsize
    if kind == "c" or is_complex_dtype(dtype):
        raise TypeError("complex numeric fingerprints are unsupported")
    if kind == "f" and int(itemsize) > 8:
        raise TypeError("floating dtypes wider than float64 are unsupported")
    if kind not in {"i", "u", "f"}:
        raise TypeError(f"unsupported numeric dtype {dtype}")


def _canonical_series(series: pd.Series, family: str) -> pd.Series:
    """Normalize values before the version-pinned pandas hash backend."""
    series = series.reset_index(drop=True)
    if family == "numeric":
        _validate_numeric_dtype(series.dtype)
        missing = series.isna().to_numpy()
        values = series.astype(object).to_numpy(copy=True)
        for index in np.flatnonzero(~missing):
            value = values[index]
            if isinstance(value, (int, np.integer)):
                values[index] = f"n:{int(value)}"
                continue
            if isinstance(value, np.floating) and value.dtype.itemsize > 8:
                raise TypeError(
                    "floating dtypes wider than float64 are unsupported"
                )
            if not isinstance(value, (float, np.floating)):
                raise TypeError(
                    f"unsupported numeric scalar type {type(value).__name__}"
                )
            numeric = float(value)
            if math.isnan(numeric):
                values[index] = "<missing>"
            elif math.isinf(numeric):
                values[index] = "n:+inf" if numeric > 0 else "n:-inf"
            else:
                numerator, denominator = numeric.as_integer_ratio()
                values[index] = (
                    f"n:{numerator}"
                    if denominator == 1
                    else f"q:{numerator}/{denominator}"
                )
        values[missing] = "<missing>"
        return pd.Series(values, dtype=object)
    if family == "datetime":
        return pd.Series(pd.to_datetime(series, errors="raise").astype("int64"))

    missing = series.isna().to_numpy()
    values = series.astype(object).to_numpy(copy=True)
    for index in np.flatnonzero(~missing):
        value = values[index]
        if family == "bool":
            values[index] = "b:1" if bool(value) else "b:0"
        else:
            values[index] = "s:" + unicodedata.normalize("NFKC", str(value))
    values[missing] = "<missing>"
    return pd.Series(values, dtype=object)


def _series_hashes(series: pd.Series, family: str) -> np.ndarray:
    canonical = _canonical_series(series, family)
    hashes = pd.util.hash_pandas_object(
        canonical, index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=True)
    salt = int.from_bytes(
        hashlib.sha256(("dtype:" + family).encode("utf-8")).digest()[:8],
        "little",
    )
    hashes ^= np.uint64(salt)
    return hashes


def _comparison_series_hashes(series: pd.Series, family: str) -> np.ndarray:
    """Hash representation-compatible values for near-lineage matching."""
    if family not in {"bool", "categorical", "string"}:
        return _series_hashes(series, family)
    series = series.reset_index(drop=True)
    missing = series.isna().to_numpy()
    values = series.astype(object).to_numpy(copy=True)
    for index in np.flatnonzero(~missing):
        value = values[index]
        if isinstance(value, (bool, np.bool_)):
            text = "true" if bool(value) else "false"
        else:
            text = str(value)
        values[index] = unicodedata.normalize("NFKC", text).casefold()
    values[missing] = "<missing>"
    hashes = pd.util.hash_pandas_object(
        pd.Series(values, dtype=object), index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=True)
    salt = int.from_bytes(
        hashlib.sha256(b"dtype:textual-comparison").digest()[:8], "little"
    )
    hashes ^= np.uint64(salt)
    return hashes


def _hash_multiset(values: np.ndarray, *, role: str) -> str:
    ordered = np.sort(np.asarray(values, dtype=np.uint64))
    digest = hashlib.sha256()
    digest.update(role.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(len(ordered)).encode("ascii"))
    digest.update(b"\0")
    digest.update(ordered.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _bottom_k(values: np.ndarray, size: int = VALUE_SKETCH_SIZE) -> list[str]:
    unique = np.unique(np.asarray(values, dtype=np.uint64))
    return [f"{int(value):016x}" for value in unique[:size]]


def _mix_uint64(values: np.ndarray, salt: int) -> np.ndarray:
    """Return an independently salted SplitMix64 lane."""
    mixed = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        mixed += np.uint64(salt)
        mixed = (mixed ^ (mixed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> 27)) * np.uint64(0x94D049BB133111EB)
    return mixed ^ (mixed >> 31)


def _rotate_left_uint64(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    return (values << bits) | (values >> (64 - bits))


def _bottom_k_uint128(
    high: np.ndarray,
    low: np.ndarray,
    *,
    size: int = VALUE_SKETCH_SIZE,
) -> tuple[list[str], int]:
    pairs = np.empty(len(high), dtype=[("high", ">u8"), ("low", ">u8")])
    pairs["high"] = np.asarray(high, dtype=np.uint64)
    pairs["low"] = np.asarray(low, dtype=np.uint64)
    unique = np.unique(pairs)
    return (
        [
            f"{int(value['high']):016x}{int(value['low']):016x}"
            for value in unique[:size]
        ],
        int(len(unique)),
    )


def _bloom_filter_uint128(
    high: np.ndarray, low: np.ndarray, *, unique_count: int
) -> dict[str, Any]:
    bit_count = max(
        64,
        8 * math.ceil(max(1, unique_count) * BLOOM_BITS_PER_ITEM / 8),
    )
    bits = np.zeros(bit_count // 8, dtype=np.uint8)
    high = np.asarray(high, dtype=np.uint64)
    low = np.asarray(low, dtype=np.uint64)
    for index in range(BLOOM_HASH_COUNT):
        offset = np.uint64(
            ((index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        )
        with np.errstate(over="ignore"):
            positions = (high + np.uint64(2 * index + 1) * low + offset) % (
                np.uint64(bit_count)
            )
        positions = positions.astype(np.int64, copy=False)
        byte_indices = positions >> 3
        masks = np.left_shift(
            np.uint8(1), (positions & 7).astype(np.uint8, copy=False)
        )
        np.bitwise_or.at(bits, byte_indices, masks)
    return {
        "algorithm": "double_hash_uint128_bloom_v1",
        "bit_count": bit_count,
        "hash_count": BLOOM_HASH_COUNT,
        "bits_base64": base64.b64encode(bits.tobytes()).decode("ascii"),
    }


def _bloom_membership_fraction(
    sample_hashes: Sequence[str], bloom: Mapping[str, Any]
) -> float:
    if not sample_hashes:
        return 1.0
    if str(bloom.get("algorithm")) != "double_hash_uint128_bloom_v1":
        raise ValueError("unsupported row-membership Bloom algorithm")
    bit_count = int(bloom["bit_count"])
    hash_count = int(bloom["hash_count"])
    if bit_count < 64 or bit_count % 8 or hash_count != BLOOM_HASH_COUNT:
        raise ValueError("invalid row-membership Bloom parameters")
    bits = np.frombuffer(
        base64.b64decode(bloom["bits_base64"], validate=True), dtype=np.uint8
    )
    if len(bits) * 8 != bit_count:
        raise ValueError("row-membership Bloom payload length mismatch")
    high = np.asarray(
        [int(value[:16], 16) for value in sample_hashes], dtype=np.uint64
    )
    low = np.asarray(
        [int(value[16:], 16) for value in sample_hashes], dtype=np.uint64
    )
    present = np.ones(len(sample_hashes), dtype=bool)
    for index in range(hash_count):
        offset = np.uint64(
            ((index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        )
        with np.errstate(over="ignore"):
            positions = (high + np.uint64(2 * index + 1) * low + offset) % (
                np.uint64(bit_count)
            )
        positions = positions.astype(np.int64, copy=False)
        present &= (
            bits[positions >> 3]
            & np.left_shift(
                np.uint8(1), (positions & 7).astype(np.uint8, copy=False)
            )
        ) != 0
    return float(np.mean(present))


def _schema_row_commitments(
    lane_sums: Sequence[np.ndarray], *, included_feature_count: int
) -> tuple[np.ndarray, np.ndarray]:
    count_salt = (included_feature_count * 0x9E3779B97F4A7C15) & (
        (1 << 64) - 1
    )
    first = _mix_uint64(
        lane_sums[0]
        ^ _rotate_left_uint64(lane_sums[1], 19)
        ^ np.uint64(count_salt),
        0xD6E8FEB86659FD93,
    )
    second = _mix_uint64(
        lane_sums[2]
        ^ _rotate_left_uint64(lane_sums[0], 41)
        ^ np.uint64(count_salt ^ 0xA5A3564E27F8862B),
        0xA0761D6478BD642F,
    )
    return first, second


def _schema_row_lane_sums(
    columns: Sequence[np.ndarray], *, n_rows: int
) -> list[np.ndarray]:
    totals = [np.zeros(n_rows, dtype=np.uint64) for _ in SCHEMA_ROW_LANE_SALTS]
    for column in columns:
        for lane, salt in enumerate(SCHEMA_ROW_LANE_SALTS):
            totals[lane] += _mix_uint64(column, salt)
    return totals


def _schema_row_view(
    lane_sums: Sequence[np.ndarray], *, included_feature_count: int
) -> dict[str, Any]:
    first, second = _schema_row_commitments(
        lane_sums, included_feature_count=included_feature_count
    )
    bottom_k, unique_count = _bottom_k_uint128(first, second)
    view = {
        "included_feature_count": int(included_feature_count),
        "row_unique_count": unique_count,
        "bottom_k_row_hashes": bottom_k,
        "row_membership_bloom": _bloom_filter_uint128(
            first, second, unique_count=unique_count
        ),
    }
    view["view_sha256"] = sha256_json(view)
    return view


def _schema_deletion_row_sketch_deck(
    columns: Sequence[np.ndarray], *, n_rows: int
) -> list[dict[str, Any]]:
    """Build a full view plus every one-feature-deletion view in O(n*p)."""
    totals = _schema_row_lane_sums(columns, n_rows=n_rows)

    feature_count = len(columns)
    views = [
        _schema_row_view(totals, included_feature_count=feature_count)
    ]
    for column in columns:
        omitted = [
            totals[lane] - _mix_uint64(column, salt)
            for lane, salt in enumerate(SCHEMA_ROW_LANE_SALTS)
        ]
        views.append(
            _schema_row_view(
                omitted, included_feature_count=feature_count - 1
            )
        )
    return sorted(
        views,
        key=lambda view: (
            int(view["included_feature_count"]),
            str(view["view_sha256"]),
        ),
    )


def _row_hashes(columns: Sequence[np.ndarray], *, n_rows: int) -> np.ndarray:
    if not columns:
        return np.zeros(n_rows, dtype=np.uint64)
    frame = pd.DataFrame(
        {str(index): values for index, values in enumerate(columns)}
    )
    return pd.util.hash_pandas_object(
        frame, index=False, categorize=False
    ).to_numpy(dtype=np.uint64, copy=False)


def _row_table_sha256(
    columns: Sequence[np.ndarray],
    *,
    n_rows: int,
    role: str,
    logical_column_count: int | None = None,
) -> str:
    row_hashes = _row_hashes(columns, n_rows=n_rows)
    return sha256_json(
        {
            "role": role,
            "n_rows": int(n_rows),
            "n_columns": (
                len(columns)
                if logical_column_count is None
                else int(logical_column_count)
            ),
            "row_hash_multiset_sha256": _hash_multiset(
                row_hashes, role="row_multiset"
            ),
        }
    )


def _canonical_feature_columns(
    ordered_positions: Sequence[int],
    column_records: Sequence[Mapping[str, Any]],
    column_hashes: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], int, bool]:
    """Order columns under global permutations, failing closed on unresolved ties."""
    groups: list[tuple[tuple[str, str], list[int], np.ndarray]] = []
    tie_group_count = 0
    for key, group_iterator in itertools.groupby(
        ordered_positions,
        key=lambda index: (
            column_records[index]["dtype_family"],
            column_records[index]["value_multiset_sha256"],
        ),
    ):
        group = list(group_iterator)
        if len(group) == 1:
            group_rows = column_hashes[group[0]]
        else:
            tie_group_count += 1
            values = np.column_stack([column_hashes[index] for index in group])
            values.sort(axis=1)
            group_rows = _row_hashes(
                [values[:, index] for index in range(values.shape[1])],
                n_rows=values.shape[0],
            ).copy()
            salt = int.from_bytes(
                hashlib.sha256(
                    f"feature-group:{key[0]}:{key[1]}:{len(group)}".encode(
                        "ascii"
                    )
                ).digest()[:8],
                "little",
            )
            group_rows ^= np.uint64(salt)
        groups.append((key, group, group_rows))

    canonical_positions: list[int] = []
    unresolved = False
    for group_index, (_, group, _) in enumerate(groups):
        if len(group) == 1:
            canonical_positions.extend(group)
            continue
        anchor_columns = [
            row_hashes
            for other_index, (_, _, row_hashes) in enumerate(groups)
            if other_index != group_index
        ]
        n_rows = len(column_hashes[group[0]])
        anchor = _row_hashes(anchor_columns, n_rows=n_rows)
        conditional = []
        for position in group:
            pair_hashes = _row_hashes(
                [anchor, column_hashes[position]], n_rows=n_rows
            )
            conditional.append(
                (
                    _hash_multiset(
                        pair_hashes, role="conditional_feature_column"
                    ),
                    position,
                )
            )
        conditional.sort(key=lambda item: item[0])
        for _, tied_iterator in itertools.groupby(
            conditional, key=lambda item: item[0]
        ):
            tied = [position for _, position in tied_iterator]
            if len(tied) > 1 and any(
                not np.array_equal(column_hashes[tied[0]], column_hashes[position])
                for position in tied[1:]
            ):
                unresolved = True
            canonical_positions.extend(tied)
    return (
        [column_hashes[position] for position in canonical_positions],
        tie_group_count,
        unresolved,
    )


def dataset_fingerprint(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Return full exact digests and target-blind near-match sketches.

    Exact feature/table hashes are invariant to row order, feature-column
    order, and feature renaming.  Duplicate rows remain significant.  The
    target-marked digest preserves the feature/target row relationship, while
    the feature-only digest deliberately survives target transforms.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y, name="__target__").reset_index(drop=True)
    if len(X) != len(y):
        raise ValueError("feature and target row counts differ")
    for position in range(X.shape[1]):
        series = X.iloc[:, position]
        if _dtype_family(series) == "numeric":
            _validate_numeric_dtype(series.dtype)
    if _dtype_family(y) == "numeric":
        _validate_numeric_dtype(y.dtype)
    if len(X) == 0:
        raise ValueError("dataset fingerprint requires at least one row")
    if X.shape[1] == 0:
        raise ValueError("dataset fingerprint requires at least one feature")
    if X.columns.duplicated().any():
        raise ValueError("duplicate feature names are ambiguous")

    column_records: list[dict[str, Any]] = []
    column_hashes: list[np.ndarray] = []
    column_comparison_hashes: list[np.ndarray] = []
    for position, column_name in enumerate(X.columns):
        series = X.iloc[:, position]
        family = _dtype_family(series)
        hashes = _series_hashes(series, family)
        comparison_hashes = _comparison_series_hashes(series, family)
        content_sha256 = _hash_multiset(hashes, role="feature_column")
        column_hashes.append(hashes)
        column_comparison_hashes.append(comparison_hashes)
        column_records.append(
            {
                "normalized_name": normalize_name(column_name),
                "dtype_family": family,
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "value_multiset_sha256": content_sha256,
                "bottom_k_value_hashes": _bottom_k(hashes),
                "comparison_unique_count": int(
                    np.unique(comparison_hashes).size
                ),
                "bottom_k_comparison_hashes": _bottom_k(comparison_hashes),
            }
        )

    ordered_positions = sorted(
        range(len(column_records)),
        key=lambda index: (
            column_records[index]["dtype_family"],
            column_records[index]["value_multiset_sha256"],
        ),
    )
    ordered_hashes, tie_group_count, canonicalization_ambiguous = (
        _canonical_feature_columns(
        ordered_positions, column_records, column_hashes
        )
    )
    target_family = _dtype_family(y)
    target_hashes = _series_hashes(y, target_family)
    feature_table_sha256 = None
    target_marked_table_sha256 = None
    feature_row_hashes = None
    if not canonicalization_ambiguous:
        feature_row_hashes = _row_hashes(ordered_hashes, n_rows=len(X))
        feature_table_sha256 = _row_table_sha256(
            ordered_hashes,
            n_rows=len(X),
            role="features",
            logical_column_count=X.shape[1],
        )
        target_marked_table_sha256 = _row_table_sha256(
            [*ordered_hashes, target_hashes],
            n_rows=len(X),
            role="features_plus_marked_target",
            logical_column_count=X.shape[1] + 1,
        )

    categorical_count = sum(
        record["dtype_family"] in {"bool", "categorical", "string"}
        for record in column_records
    )
    feature_content = sorted(
        (
            record["dtype_family"],
            record["value_multiset_sha256"],
        )
        for record in column_records
    )
    if column_comparison_hashes:
        unlabeled_values = np.column_stack(column_comparison_hashes)
        unlabeled_values.sort(axis=1)
        unlabeled_row_hashes = _row_hashes(
            [
                unlabeled_values[:, index]
                for index in range(unlabeled_values.shape[1])
            ],
            n_rows=len(X),
        )
    else:
        unlabeled_row_hashes = np.zeros(len(X), dtype=np.uint64)
    return {
        "semantic_hash_backend": "pandas_hash_pandas_object_uint64_v1",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "categorical_feature_count": int(categorical_count),
        "has_categorical": bool(categorical_count),
        "has_missing_features": bool(X.isna().to_numpy().any()),
        "feature_name_multiset_sha256": sha256_json(
            sorted(record["normalized_name"] for record in column_records)
        ),
        "feature_content_multiset_sha256": sha256_json(feature_content),
        "feature_table_sha256": feature_table_sha256,
        "opaque_target_value_sha256": _hash_multiset(
            target_hashes, role="target_column"
        ),
        "target_marked_table_sha256": target_marked_table_sha256,
        "feature_row_unique_count": (
            None
            if feature_row_hashes is None
            else int(np.unique(feature_row_hashes).size)
        ),
        "bottom_k_feature_row_hashes": (
            [] if feature_row_hashes is None else _bottom_k(feature_row_hashes)
        ),
        "unlabeled_feature_row_unique_count": int(
            np.unique(unlabeled_row_hashes).size
        ),
        "bottom_k_unlabeled_feature_row_hashes": _bottom_k(
            unlabeled_row_hashes
        ),
        "schema_deletion_row_sketch_deck": _schema_deletion_row_sketch_deck(
            column_comparison_hashes, n_rows=len(X)
        ),
        "canonicalization_ambiguous": canonicalization_ambiguous,
        "marginal_tie_group_count": tie_group_count,
        "columns": column_records,
    }


def _compatible_families(left: str, right: str) -> bool:
    textual = {"bool", "categorical", "string"}
    return left == right or (left in textual and right in textual)


def _kmv_containment(
    left: Sequence[str],
    right: Sequence[str],
    *,
    left_cardinality: int,
    right_cardinality: int,
) -> float:
    """Estimate smaller-set containment from two bottom-k KMV sketches."""
    left_set = set(left)
    right_set = set(right)
    minimum_cardinality = min(left_cardinality, right_cardinality)
    if minimum_cardinality == 0:
        return 1.0 if not left_set and not right_set else 0.0
    sample_size = min(VALUE_SKETCH_SIZE, len(left_set | right_set))
    union_sample = set(sorted(left_set | right_set)[:sample_size])
    if not union_sample:
        return 0.0
    jaccard = sum(
        value in left_set and value in right_set for value in union_sample
    ) / len(union_sample)
    if jaccard == 0.0:
        return 0.0
    estimated_intersection = (
        jaccard * (left_cardinality + right_cardinality) / (1.0 + jaccard)
    )
    return float(min(1.0, estimated_intersection / minimum_cardinality))


def _column_match_key(column: Mapping[str, Any]) -> tuple[Any, ...]:
    family = str(column["dtype_family"])
    comparison_family = (
        "textual" if family in {"bool", "categorical", "string"} else family
    )
    return (
        comparison_family,
        int(column["comparison_unique_count"]),
        tuple(column["bottom_k_comparison_hashes"]),
        str(column["value_multiset_sha256"]),
    )


def _maximum_column_matches(
    left_columns: Sequence[Mapping[str, Any]],
    right_columns: Sequence[Mapping[str, Any]],
    *,
    column_containment_threshold: float,
    minimum_informative_unique_values: int,
) -> tuple[list[float], int]:
    """Return stable maximum-cardinality, then maximum-weight matches."""
    from scipy.optimize import linear_sum_assignment

    left = sorted(left_columns, key=_column_match_key)
    right = sorted(right_columns, key=_column_match_key)
    minimum_features = min(len(left), len(right))
    if not left or not right:
        return [], minimum_features
    weights = np.zeros((len(left), len(right)), dtype=np.float64)
    cardinality_weight = minimum_features + 1.0
    for left_index, left_column in enumerate(left):
        for right_index, right_column in enumerate(right):
            if not _compatible_families(
                str(left_column["dtype_family"]),
                str(right_column["dtype_family"]),
            ):
                continue
            left_unique = int(left_column["comparison_unique_count"])
            right_unique = int(right_column["comparison_unique_count"])
            textual = str(left_column["dtype_family"]) in {
                "bool",
                "categorical",
                "string",
            }
            if (
                not textual
                and (
                    left_unique < minimum_informative_unique_values
                    or right_unique < minimum_informative_unique_values
                )
            ):
                score = float(
                    left_column["value_multiset_sha256"]
                    == right_column["value_multiset_sha256"]
                )
            else:
                score = _kmv_containment(
                    left_column["bottom_k_comparison_hashes"],
                    right_column["bottom_k_comparison_hashes"],
                    left_cardinality=left_unique,
                    right_cardinality=right_unique,
                )
            if score >= column_containment_threshold:
                weights[left_index, right_index] = cardinality_weight + score
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    scores = [
        float(weights[left_index, right_index] - cardinality_weight)
        for left_index, right_index in zip(row_indices, column_indices)
        if weights[left_index, right_index] >= cardinality_weight
    ]
    return scores, minimum_features


def _row_view_bloom_containment(
    left_view: Mapping[str, Any],
    right_view: Mapping[str, Any],
) -> float:
    """Query the lower-cardinality unique-row sample against the larger set."""
    left_unique = int(left_view["row_unique_count"])
    right_unique = int(right_view["row_unique_count"])
    if left_unique < right_unique:
        return _bloom_membership_fraction(
            left_view["bottom_k_row_hashes"],
            right_view["row_membership_bloom"],
        )
    if right_unique < left_unique:
        return _bloom_membership_fraction(
            right_view["bottom_k_row_hashes"],
            left_view["row_membership_bloom"],
        )
    return min(
        _bloom_membership_fraction(
            left_view["bottom_k_row_hashes"],
            right_view["row_membership_bloom"],
        ),
        _bloom_membership_fraction(
            right_view["bottom_k_row_hashes"],
            left_view["row_membership_bloom"],
        ),
    )


def _schema_deletion_row_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    row_ratio: float,
    row_ratio_threshold: float,
    feature_coverage_threshold: float,
    schema_deletion_membership_threshold: float,
) -> dict[str, Any]:
    """Compare full-schema subsets and p-of-(p+1) deletion views."""
    left_width = int(left["n_features"])
    right_width = int(right["n_features"])
    default = {
        "schema_row_comparison_supported": False,
        "schema_deletion_supported": False,
        "schema_deletion_row_alarm": False,
        "schema_deletion_row_containment": 0.0,
        "schema_deletion_included_feature_count": None,
        "schema_deletion_tied_max_count": 0,
        "schema_deletion_winner_digest": None,
    }
    width_difference = abs(left_width - right_width)
    if width_difference > 1:
        return default

    if width_difference == 0:
        included_width = left_width
        left_full = [
            view
            for view in left["schema_deletion_row_sketch_deck"]
            if int(view["included_feature_count"]) == included_width
        ]
        right_full = [
            view
            for view in right["schema_deletion_row_sketch_deck"]
            if int(view["included_feature_count"]) == included_width
        ]
        if len(left_full) != 1 or len(right_full) != 1:
            raise ValueError("invalid full-schema sketch deck")

        left_view = left_full[0]
        right_view = right_full[0]
        maximum = _row_view_bloom_containment(left_view, right_view)
        winner = sha256_json(
            sorted([left_view["view_sha256"], right_view["view_sha256"]])
        )
        return {
            "schema_row_comparison_supported": True,
            "schema_deletion_supported": False,
            "schema_deletion_row_alarm": bool(
                maximum >= schema_deletion_membership_threshold
                and row_ratio >= row_ratio_threshold
            ),
            "schema_deletion_row_containment": float(f"{maximum:.15g}"),
            "schema_deletion_included_feature_count": included_width,
            "schema_deletion_tied_max_count": 1,
            "schema_deletion_winner_digest": sha256_json([winner]),
        }

    smaller, larger = (
        (left, right) if left_width < right_width else (right, left)
    )
    included_width = min(left_width, right_width)
    minimum_width = min(left_width, right_width)
    coverage = included_width / minimum_width if minimum_width else 1.0
    if coverage < feature_coverage_threshold:
        return default

    smaller_full = [
        view
        for view in smaller["schema_deletion_row_sketch_deck"]
        if int(view["included_feature_count"]) == included_width
    ]
    larger_deletions = [
        view
        for view in larger["schema_deletion_row_sketch_deck"]
        if int(view["included_feature_count"]) == included_width
    ]
    if len(smaller_full) != 1 or not larger_deletions:
        raise ValueError("invalid one-feature-deletion sketch deck")

    source_view = smaller_full[0]
    scored = []
    for candidate in larger_deletions:
        score = _row_view_bloom_containment(source_view, candidate)
        scored.append((score, str(candidate["view_sha256"])))
    maximum = max(score for score, _ in scored)
    winners = sorted(
        digest for score, digest in scored if score == maximum
    )
    return {
        "schema_row_comparison_supported": True,
        "schema_deletion_supported": True,
        "schema_deletion_row_alarm": bool(
            maximum >= schema_deletion_membership_threshold
            and row_ratio >= row_ratio_threshold
            and min(left_width, right_width) / max(left_width, right_width)
            >= feature_coverage_threshold
        ),
        "schema_deletion_row_containment": float(f"{maximum:.15g}"),
        "schema_deletion_included_feature_count": included_width,
        "schema_deletion_tied_max_count": len(winners),
        "schema_deletion_winner_digest": sha256_json(winners),
    }


def near_match_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    column_containment_threshold: float,
    feature_coverage_threshold: float,
    mean_containment_threshold: float,
    row_ratio_threshold: float,
    minimum_informative_unique_values: int,
    minimum_informative_matches: int,
    row_sketch_containment_threshold: float,
    schema_deletion_membership_threshold: float,
) -> dict[str, Any]:
    """Compare target-blind sketches; a positive result is only ambiguous."""
    scores, minimum_informative_features = _maximum_column_matches(
        left["columns"],
        right["columns"],
        column_containment_threshold=column_containment_threshold,
        minimum_informative_unique_values=minimum_informative_unique_values,
    )
    minimum_features = min(int(left["n_features"]), int(right["n_features"]))
    coverage = len(scores) / minimum_features if minimum_features else 1.0
    mean_containment = float(np.mean(scores)) if scores else 0.0
    left_rows = int(left["n_rows"])
    right_rows = int(right["n_rows"])
    row_ratio = min(left_rows, right_rows) / max(left_rows, right_rows)
    column_ambiguous = bool(
        len(scores) >= minimum_informative_matches
        and coverage >= feature_coverage_threshold
        and mean_containment >= mean_containment_threshold
        and row_ratio >= row_ratio_threshold
    )
    feature_count_ratio = min(
        int(left["n_features"]), int(right["n_features"])
    ) / max(int(left["n_features"]), int(right["n_features"]))
    row_containment = 0.0
    if (
        left["bottom_k_unlabeled_feature_row_hashes"]
        and right["bottom_k_unlabeled_feature_row_hashes"]
        and feature_count_ratio >= feature_coverage_threshold
    ):
        row_containment = _kmv_containment(
            left["bottom_k_unlabeled_feature_row_hashes"],
            right["bottom_k_unlabeled_feature_row_hashes"],
            left_cardinality=int(left["unlabeled_feature_row_unique_count"]),
            right_cardinality=int(right["unlabeled_feature_row_unique_count"]),
        )
    row_ambiguous = bool(
        row_containment >= row_sketch_containment_threshold
        and row_ratio >= row_ratio_threshold
        and feature_count_ratio >= feature_coverage_threshold
    )
    deletion_evidence = _schema_deletion_row_evidence(
        left,
        right,
        row_ratio=row_ratio,
        row_ratio_threshold=row_ratio_threshold,
        feature_coverage_threshold=feature_coverage_threshold,
        schema_deletion_membership_threshold=(
            schema_deletion_membership_threshold
        ),
    )
    return {
        "ambiguous": bool(
            column_ambiguous
            or row_ambiguous
            or deletion_evidence["schema_deletion_row_alarm"]
        ),
        "column_sketch_alarm": column_ambiguous,
        "row_sketch_alarm": row_ambiguous,
        "matched_feature_count": len(scores),
        "minimum_informative_feature_count": minimum_informative_features,
        "minimum_feature_count": minimum_features,
        "minimum_feature_coverage": float(f"{coverage:.15g}"),
        "mean_column_containment": float(f"{mean_containment:.15g}"),
        "row_count_ratio": float(f"{row_ratio:.15g}"),
        "feature_count_ratio": float(f"{feature_count_ratio:.15g}"),
        "feature_row_containment": float(f"{row_containment:.15g}"),
        **deletion_evidence,
    }


def _cache_file_sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required OpenML cache artifact is missing: {path}")
    return sha256_file(path)


def _split_snapshot(
    task: Any, cache_root: Path, *, n_rows: int
) -> dict[str, Any]:
    repeats, folds, samples = (int(value) for value in task.get_split_dimensions())
    coordinates = []
    test_indices_by_repeat_sample: dict[tuple[int, int], list[np.ndarray]] = {}
    for repeat in range(repeats):
        for fold in range(folds):
            for sample in range(samples):
                train, test = task.get_train_test_split_indices(
                    repeat=repeat, fold=fold, sample=sample
                )
                train = np.asarray(train, dtype="<i8")
                test = np.asarray(test, dtype="<i8")
                if np.intersect1d(train, test).size:
                    raise RuntimeError(
                        f"task {task.task_id} coordinate "
                        f"r{repeat}f{fold}s{sample} overlaps train and test"
                    )
                covered = np.sort(np.concatenate([train, test]))
                if not np.array_equal(covered, np.arange(n_rows, dtype="<i8")):
                    raise RuntimeError(
                        f"task {task.task_id} coordinate "
                        f"r{repeat}f{fold}s{sample} does not cover every row once"
                    )
                test_indices_by_repeat_sample.setdefault(
                    (repeat, sample), []
                ).append(test)
                coordinates.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "sample": sample,
                        "train_size": int(train.size),
                        "test_size": int(test.size),
                        "train_index_sha256": hashlib.sha256(
                            train.tobytes()
                        ).hexdigest(),
                        "test_index_sha256": hashlib.sha256(
                            test.tobytes()
                        ).hexdigest(),
                    }
                )
    if str(task.estimation_procedure.get("type", "")).casefold() == "crossvalidation":
        expected_rows = np.arange(n_rows, dtype="<i8")
        for (repeat, sample), test_parts in test_indices_by_repeat_sample.items():
            partition = np.sort(np.concatenate(test_parts))
            if not np.array_equal(partition, expected_rows):
                raise RuntimeError(
                    f"task {task.task_id} repeat {repeat}, sample {sample} "
                    "test folds do not partition the dataset"
                )
    task_dir = cache_root / "tasks" / str(int(task.task_id))
    return {
        "dimensions": {
            "repeats": repeats,
            "folds": folds,
            "samples": samples,
        },
        "coordinate_count": len(coordinates),
        "estimation_procedure": task.estimation_procedure,
        "raw_split_file_sha256": _cache_file_sha256(
            task_dir / "datasplits.arff"
        ),
        "semantic_split_sha256": sha256_json(coordinates),
        "integrity": {
            "coordinate_keys_complete": True,
            "train_test_disjoint": True,
            "train_test_cover_all_rows_once": True,
            "crossvalidation_test_folds_partition_rows": True,
        },
        "coordinates": coordinates,
    }


def _optional_file_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _attribute_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    return sorted(str(item) for item in value)


def _task_record(task_id: int, *, include_splits: bool) -> dict[str, Any]:
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - CLI dependency error.
        raise RuntimeError("openml is required to build the registry") from exc

    task = openml.tasks.get_task(
        int(task_id),
        download_splits=include_splits,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    dataset = openml.datasets.get_dataset(
        int(task.dataset_id), download_data=False, download_qualities=False
    )
    X, y, _, _ = dataset.get_data(
        target=task.target_name,
        include_row_id=False,
        include_ignore_attribute=False,
        dataset_format="dataframe",
    )
    if y is None:
        raise RuntimeError(f"task {task_id} did not return a target")
    default_target = str(dataset.default_target_attribute)
    if str(task.target_name) != default_target:
        raise RuntimeError(
            f"task {task_id} target {task.target_name!r} differs from the "
            f"dataset default target {default_target!r}"
        )
    fingerprint = dataset_fingerprint(X, y)

    cache_root = Path(openml.config.get_cache_directory())
    task_dir = cache_root / "tasks" / str(int(task_id))
    dataset_dir = cache_root / "datasets" / str(int(task.dataset_id))
    raw_path = dataset.parquet_file or dataset.data_file
    raw_artifact = None
    if raw_path is not None:
        raw_path = Path(raw_path)
        raw_artifact = {
            "format": raw_path.suffix.lstrip(".") or "unknown",
            "sha256": sha256_file(raw_path),
        }

    record = {
        "openml_task_id": int(task_id),
        "openml_dataset_id": int(task.dataset_id),
        "openml_dataset_version": int(dataset.version),
        "dataset_name": str(dataset.name),
        "normalized_name": normalize_name(dataset.name),
        "target_name": str(task.target_name),
        "dataset_default_target_attribute": default_target,
        "openml_task_type_id": int(task.task_type_id.value),
        "openml_estimation_procedure_id": int(task.estimation_procedure_id),
        "ignore_attributes": _attribute_list(dataset.ignore_attribute),
        "row_id_attributes": _attribute_list(dataset.row_id_attribute),
        "original_data_url": getattr(dataset, "original_data_url", None),
        "normalized_original_data_url": normalize_url(
            getattr(dataset, "original_data_url", None)
        ),
        "openml_download_url": str(dataset.url),
        "openml_declared_md5": str(dataset.md5_checksum),
        "raw_download_artifact": raw_artifact,
        "task_xml_sha256": _cache_file_sha256(task_dir / "task.xml"),
        "dataset_description_xml_sha256": _cache_file_sha256(
            dataset_dir / "description.xml"
        ),
        "dataset_features_xml_sha256": _optional_file_sha256(
            dataset_dir / "features.xml"
        ),
        "fingerprint": fingerprint,
    }
    if include_splits:
        record["official_splits"] = _split_snapshot(
            task, cache_root, n_rows=len(X)
        )
    return record


def _allocation_values(task: Mapping[str, Any]) -> dict[str, float]:
    fingerprint = task["fingerprint"]
    coordinate_count = int(task["official_splits"]["coordinate_count"])
    rows = int(fingerprint["n_rows"])
    features = int(fingerprint["n_features"])
    return {
        "log_rows": math.log1p(rows),
        "log_raw_predictors": math.log1p(features),
        "has_categorical": float(bool(fingerprint["has_categorical"])),
        "has_missing_features": float(bool(fingerprint["has_missing_features"])),
        "log_compute_proxy": math.log1p(rows * max(1, features) * coordinate_count),
    }


def _resampling_regime(task: Mapping[str, Any]) -> str:
    dimensions = task["official_splits"]["dimensions"]
    return (
        f"r{int(dimensions['repeats'])}f{int(dimensions['folds'])}"
        f"s{int(dimensions['samples'])}"
    )


def balanced_panel_split(
    tasks: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Exhaustively choose the target-blind, lineage-atomic panel allocation."""
    if len(tasks) < 4:
        raise ValueError("at least four eligible tasks are required")
    ordered = sorted(tasks, key=lambda task: int(task["openml_task_id"]))
    ids = [int(task["openml_task_id"]) for task in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("eligible tasks contain duplicate ids")

    clusters: dict[str, list[int]] = {}
    task_by_id = {int(task["openml_task_id"]): task for task in ordered}
    for task in ordered:
        task_id = int(task["openml_task_id"])
        cluster = str(task.get("lineage_cluster", f"openml_task:{task_id}"))
        clusters.setdefault(cluster, []).append(task_id)
    cluster_items = sorted(
        (cluster, tuple(sorted(task_ids)))
        for cluster, task_ids in clusters.items()
    )

    confirmation_size = len(ids) // 2
    fields = tuple(_allocation_values(ordered[0]))
    matrix = np.asarray(
        [[_allocation_values(task)[field] for field in fields] for task in ordered],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    row_by_id = {task_id: index for index, task_id in enumerate(ids)}
    all_ids = set(ids)
    all_regimes = {
        task_id: _resampling_regime(task_by_id[task_id]) for task_id in ids
    }
    regimes = sorted(set(all_regimes.values()))

    best: tuple[tuple[Any, ...], tuple[int, ...]] | None = None
    evaluated = 0
    feasible = 0
    for cluster_count in range(1, len(cluster_items)):
        for chosen_indices in itertools.combinations(
            range(len(cluster_items)), cluster_count
        ):
            confirmation_tuple = tuple(
                sorted(
                    task_id
                    for index in chosen_indices
                    for task_id in cluster_items[index][1]
                )
            )
            if len(confirmation_tuple) != confirmation_size:
                continue
            evaluated += 1
            confirmation = set(confirmation_tuple)
            lockbox = all_ids - confirmation
            regime_difference = {
                regime: abs(
                    sum(all_regimes[task_id] == regime for task_id in confirmation)
                    - sum(all_regimes[task_id] == regime for task_id in lockbox)
                )
                for regime in regimes
            }
            if any(difference > 1 for difference in regime_difference.values()):
                continue
            feasible += 1
            confirmation_rows = [row_by_id[task_id] for task_id in confirmation]
            lockbox_rows = [row_by_id[task_id] for task_id in lockbox]
            imbalance = np.abs(
                standardized[confirmation_rows].mean(axis=0)
                - standardized[lockbox_rows].mean(axis=0)
            )
            confirmation_coordinates = sum(
                int(task_by_id[task_id]["official_splits"]["coordinate_count"])
                for task_id in confirmation
            )
            lockbox_coordinates = sum(
                int(task_by_id[task_id]["official_splits"]["coordinate_count"])
                for task_id in lockbox
            )
            tie = hashlib.sha256(
                f"{seed}:".encode("ascii")
                + b",".join(
                    str(task_id).encode("ascii")
                    for task_id in confirmation_tuple
                )
            ).hexdigest()
            objective = (
                abs(confirmation_coordinates - lockbox_coordinates),
                float(f"{float(np.max(imbalance)):.15g}"),
                float(f"{float(np.dot(imbalance, imbalance)):.15g}"),
                tie,
            )
            item = (objective, confirmation_tuple)
            if best is None or item < best:
                best = item

    if best is None:
        raise RuntimeError("no lineage-atomic allocation satisfies hard balance")
    confirmation = list(best[1])
    lockbox = sorted(all_ids - set(confirmation))
    allocation = {
        "confirmation_task_ids": confirmation,
        "lockbox_task_ids": lockbox,
    }
    diagnostics = {
        "algorithm": "exhaustive_lineage_atomic_target_blind_lexicographic_v1",
        "seed_used_only_for_exact_ties": int(seed),
        "eligible_task_count": len(ids),
        "lineage_cluster_count": len(cluster_items),
        "evaluated_size_balanced_assignments": evaluated,
        "feasible_regime_balanced_assignments": feasible,
        "objective": {
            "coordinate_count_imbalance": int(best[0][0]),
            "max_standardized_metadata_imbalance": best[0][1],
            "sum_squared_standardized_metadata_imbalance": best[0][2],
            "tie_sha256": best[0][3],
        },
        "metadata_fields": list(fields),
        "hard_constraints": {
            "panel_task_count_difference_at_most": 1,
            "per_resampling_regime_task_count_difference_at_most": 1,
            "lineage_clusters_are_atomic": True,
            "target_information_used": False,
        },
        "allocation_sha256": sha256_json(allocation),
    }
    return confirmation, lockbox, diagnostics


def _reason_key(reason: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(reason["reason_code"]),
        str(reason["reason"]),
        str(reason.get("source", "")),
    )


def _manual_exclusions(
    declarations: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    entries = declarations.get("manual_task_exclusions", [])
    result = {int(entry["openml_task_id"]): entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("manual exclusions contain duplicate task ids")
    return result


def _eligible_lineage_clusters(
    declarations: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    entries = declarations.get("eligible_task_lineage_clusters", [])
    result = {int(entry["openml_task_id"]): entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("eligible lineage declarations contain duplicate task ids")
    return result


def _validate_expected_suite(
    ctr_tasks: Sequence[Mapping[str, Any]], declarations: Mapping[str, Any]
) -> None:
    expected = {
        int(entry["openml_task_id"]): str(entry["expected_normalized_name"])
        for entry in declarations["expected_ctr23_tasks"]
    }
    actual = {
        int(task["openml_task_id"]): str(task["normalized_name"])
        for task in ctr_tasks
    }
    if len(actual) != EXPECTED_CTR23_TASK_COUNT or actual != expected:
        raise ValueError(
            "CTR23 task membership/name drifted from the frozen declaration"
        )


def _safe_repository_path(repository_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"evidence path must be repository-relative: {relative}")
    root = repository_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes repository: {relative}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"evidence artifact is missing: {relative}")
    return candidate


def _git_object_sha1(object_type: str, payload: bytes) -> str:
    header = f"{object_type} {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _parse_git_tree(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    offset = 0
    while offset < len(payload):
        space = payload.find(b" ", offset)
        nul = payload.find(b"\0", space + 1)
        if space < 0 or nul < 0 or nul + 21 > len(payload):
            raise ValueError("malformed raw Git tree evidence")
        mode = payload[offset:space].decode("ascii")
        name = payload[space + 1 : nul].decode("utf-8")
        object_sha1 = payload[nul + 1 : nul + 21].hex()
        if name in entries:
            raise ValueError(f"duplicate Git tree entry {name!r}")
        entries[name] = (mode, object_sha1)
        offset = nul + 21
    return entries


def _verify_git_membership_proof(
    proof_path: Path,
    *,
    proof_sha256: str,
    source_commit_sha1: str,
    source_path: str,
    source_git_blob_sha1: str,
) -> None:
    if sha256_file(proof_path) != proof_sha256:
        raise ValueError(f"Git membership proof hash mismatch: {proof_path}")
    proof = _load_json(proof_path)
    if int(proof.get("schema_version", -1)) != 1:
        raise ValueError("unsupported Git membership proof schema")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit_sha1):
        raise ValueError("historical evidence requires a full commit SHA-1")
    if proof.get("source_commit_sha1") != source_commit_sha1:
        raise ValueError("Git membership proof commit mismatch")

    objects: dict[str, tuple[str, bytes]] = {}
    for record in proof.get("objects", []):
        object_type = str(record["type"])
        object_sha1 = str(record["sha1"])
        payload = base64.b64decode(record["payload_base64"], validate=True)
        if _git_object_sha1(object_type, payload) != object_sha1:
            raise ValueError(f"invalid {object_type} object {object_sha1}")
        if object_sha1 in objects:
            raise ValueError(f"duplicate Git proof object {object_sha1}")
        objects[object_sha1] = (object_type, payload)

    commit_type, commit_payload = objects.get(source_commit_sha1, (None, None))
    if commit_type != "commit" or commit_payload is None:
        raise ValueError("Git membership proof omits the source commit")
    first_line = commit_payload.splitlines()[0].decode("ascii")
    match = re.fullmatch(r"tree ([0-9a-f]{40})", first_line)
    if match is None:
        raise ValueError("source commit has no valid root tree")
    current_sha1 = match.group(1)
    visited = {source_commit_sha1}
    parts = Path(source_path).parts
    if not parts or Path(source_path).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe historical source path: {source_path}")
    for index, part in enumerate(parts):
        tree_type, tree_payload = objects.get(current_sha1, (None, None))
        if tree_type != "tree" or tree_payload is None:
            raise ValueError(f"Git proof omits tree {current_sha1}")
        visited.add(current_sha1)
        entries = _parse_git_tree(tree_payload)
        if part not in entries:
            raise ValueError(f"Git proof path is absent: {source_path}")
        mode, next_sha1 = entries[part]
        is_last = index == len(parts) - 1
        if is_last:
            if mode not in {"100644", "100755"}:
                raise ValueError(f"Git proof path is not a file: {source_path}")
            if next_sha1 != source_git_blob_sha1:
                raise ValueError("Git proof blob identity mismatch")
        else:
            if mode not in {"40000", "040000"}:
                raise ValueError(f"Git proof path component is not a tree: {part}")
            current_sha1 = next_sha1
    if set(objects) != visited:
        raise ValueError("Git membership proof contains unused objects")


def _verify_required_claims(text: str, claims: Sequence[Mapping[str, Any]]) -> None:
    for claim in claims:
        literal = str(claim["literal"])
        expected_count = int(claim["count"])
        if not literal or expected_count < 1:
            raise ValueError("evidence claims require a literal and positive count")
        actual_count = text.count(literal)
        if actual_count != expected_count:
            raise ValueError(
                f"evidence claim count mismatch for {literal!r}: "
                f"actual={actual_count}, expected={expected_count}"
            )


def _resolve_manual_evidence(
    declarations: Mapping[str, Any],
    source_tasks: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    binding = declarations["manual_evidence_catalog"]
    catalog_path = _safe_repository_path(repository_root, str(binding["path"]))
    if sha256_file(catalog_path) != str(binding["sha256"]):
        raise ValueError("manual evidence catalog hash mismatch")
    catalog = _load_json(catalog_path)
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError("unsupported manual evidence catalog schema")
    entries = catalog.get("entries", [])
    by_id = {str(entry["evidence_id"]): entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("manual evidence catalog contains duplicate ids")
    manual = _manual_exclusions(declarations)
    used_ids = {str(entry["evidence_id"]) for entry in manual.values()}
    if set(by_id) != used_ids:
        raise ValueError(
            "manual evidence ids must be defined and used exactly once as a set"
        )
    source_by_id = {
        int(task["openml_task_id"]): task for task in source_tasks
    }

    resolved: list[dict[str, Any]] = []
    for evidence_id in sorted(by_id):
        entry = by_id[evidence_id]
        kind = str(entry["kind"])
        record: dict[str, Any] = {
            "evidence_id": evidence_id,
            "kind": kind,
        }
        if kind == "openml_task_snapshot":
            task_id = int(entry["openml_task_id"])
            if task_id not in source_by_id:
                raise ValueError(
                    f"manual evidence references unknown source task {task_id}"
                )
            record.update(
                {
                    "openml_task_id": task_id,
                    "source_task_record_sha256": sha256_json(source_by_id[task_id]),
                }
            )
        elif kind == "repository_file":
            path = _safe_repository_path(repository_root, str(entry["path"]))
            digest = sha256_file(path)
            if digest != str(entry["sha256"]):
                raise ValueError(f"repository evidence hash mismatch: {path}")
            _verify_required_claims(
                path.read_text(encoding="utf-8"), entry["required_claims"]
            )
            record.update(
                {
                    "path": str(entry["path"]),
                    "sha256": digest,
                    "required_claims_sha256": sha256_json(
                        entry["required_claims"]
                    ),
                }
            )
        elif kind == "historical_git_snapshot":
            resolved_sources = []
            for source in entry.get("sources", []):
                artifact_path = _safe_repository_path(
                    repository_root, str(source["artifact_path"])
                )
                artifact_bytes = artifact_path.read_bytes()
                artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
                if artifact_sha256 != str(source["artifact_sha256"]):
                    raise ValueError(
                        f"historical evidence hash mismatch: {artifact_path}"
                    )
                artifact_kind = str(source["artifact_kind"])
                _verify_required_claims(
                    artifact_bytes.decode("utf-8"),
                    source["required_claims"],
                )
                source_record = {
                    "artifact_kind": artifact_kind,
                    "artifact_path": str(source["artifact_path"]),
                    "artifact_sha256": artifact_sha256,
                    "required_claims_sha256": sha256_json(
                        source["required_claims"]
                    ),
                    "source_commit_sha1": str(source["source_commit_sha1"]),
                }
                if artifact_kind == "git_commit":
                    if (
                        _git_object_sha1("commit", artifact_bytes)
                        != source_record["source_commit_sha1"]
                    ):
                        raise ValueError("historical commit snapshot mismatch")
                elif artifact_kind == "git_blob":
                    source_bytes = artifact_bytes
                    if bool(source.get("strip_artifact_final_newline", False)):
                        if not source_bytes.endswith(b"\n"):
                            raise ValueError("normalized Git blob lacks final newline")
                        source_bytes = source_bytes[:-1]
                    if hashlib.sha256(source_bytes).hexdigest() != str(
                        source["source_blob_sha256"]
                    ):
                        raise ValueError("historical source blob SHA-256 mismatch")
                    if _git_object_sha1("blob", source_bytes) != str(
                        source["source_git_blob_sha1"]
                    ):
                        raise ValueError("historical Git blob identity mismatch")
                    proof_path = _safe_repository_path(
                        repository_root, str(source["proof_path"])
                    )
                    _verify_git_membership_proof(
                        proof_path,
                        proof_sha256=str(source["proof_sha256"]),
                        source_commit_sha1=source_record["source_commit_sha1"],
                        source_path=str(source["source_path"]),
                        source_git_blob_sha1=str(source["source_git_blob_sha1"]),
                    )
                    source_record.update(
                        {
                            "source_path": str(source["source_path"]),
                            "source_git_blob_sha1": str(
                                source["source_git_blob_sha1"]
                            ),
                            "source_blob_sha256": str(
                                source["source_blob_sha256"]
                            ),
                            "proof_path": str(source["proof_path"]),
                            "proof_sha256": str(source["proof_sha256"]),
                        }
                    )
                else:
                    raise ValueError(
                        f"unsupported historical artifact kind {artifact_kind!r}"
                    )
                resolved_sources.append(source_record)
            if not resolved_sources:
                raise ValueError(f"historical evidence {evidence_id} has no sources")
            record["sources"] = resolved_sources
        else:
            raise ValueError(f"unsupported manual evidence kind {kind!r}")
        resolved.append(record)
    digest = sha256_json(resolved)
    return resolved, digest


def _validate_builder_runtime(
    declarations: Mapping[str, Any],
) -> dict[str, str]:
    def distribution_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "<missing>"

    runtime = {
        "python": sys.version.split()[0],
        "openml": distribution_version("openml"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": distribution_version("scipy"),
        "pyarrow": distribution_version("pyarrow"),
        "liac-arff": distribution_version("liac-arff"),
    }
    expected = {
        str(key): str(value)
        for key, value in declarations["builder_runtime"].items()
    }
    if runtime != expected:
        raise ValueError(
            "registry builder runtime differs from the frozen declaration: "
            f"actual={runtime}, expected={expected}"
        )
    return runtime


def build_artifacts(
    ctr_tasks: Sequence[Mapping[str, Any]],
    source_tasks: Sequence[Mapping[str, Any]],
    declarations: Mapping[str, Any],
    *,
    builder_source_sha256: str,
    repository_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the three frozen payloads from already-fetched task records."""
    runtime = _validate_builder_runtime(declarations)
    _validate_expected_suite(ctr_tasks, declarations)
    manual = _manual_exclusions(declarations)
    manual_evidence, manual_evidence_sha256 = _resolve_manual_evidence(
        declarations,
        source_tasks,
        repository_root=(
            Path(__file__).parents[1]
            if repository_root is None
            else repository_root
        ),
    )
    eligible_lineages = _eligible_lineage_clusters(declarations)
    overlap = set(manual) & set(eligible_lineages)
    if overlap:
        raise ValueError(
            "tasks cannot be both manually excluded and declared eligible: "
            f"{sorted(overlap)}"
        )
    thresholds = declarations["near_match_thresholds"]
    source_by_name: dict[str, list[Mapping[str, Any]]] = {}
    source_by_features: dict[str, list[Mapping[str, Any]]] = {}
    for source in source_tasks:
        source_by_name.setdefault(str(source["normalized_name"]), []).append(source)
        feature_hash = source["fingerprint"]["feature_table_sha256"]
        if feature_hash is not None:
            source_by_features.setdefault(str(feature_hash), []).append(source)

    resolved = []
    eligible_records = []
    for task in sorted(ctr_tasks, key=lambda row: int(row["openml_task_id"])):
        task_id = int(task["openml_task_id"])
        name = str(task["normalized_name"])
        reasons: list[dict[str, Any]] = []
        ambiguous_matches: list[dict[str, Any]] = []
        for source in source_by_name.get(name, []):
            reasons.append(
                {
                    "reason_code": "normalized_name_match",
                    "reason": "Normalized name matches a spent source task.",
                    "source": f"openml_task:{source['openml_task_id']}",
                }
            )
        feature_hash = task["fingerprint"]["feature_table_sha256"]
        if feature_hash is not None:
            for source in source_by_features.get(str(feature_hash), []):
                reasons.append(
                    {
                        "reason_code": "exact_feature_table_match",
                        "reason": "Full feature-table digest matches a spent source task.",
                        "source": f"openml_task:{source['openml_task_id']}",
                    }
                )
        for source in source_tasks:
            if source in source_by_name.get(name, []):
                continue
            evidence = near_match_evidence(
                task["fingerprint"], source["fingerprint"], **thresholds
            )
            if evidence["ambiguous"]:
                ambiguous_matches.append(
                    {
                        "source": f"openml_task:{source['openml_task_id']}",
                        **evidence,
                    }
                )
        if task_id in manual:
            entry = manual[task_id]
            expected_name = str(entry["expected_normalized_name"])
            if name != expected_name:
                raise ValueError(
                    f"manual exclusion {task_id} expected {expected_name!r}, "
                    f"got {name!r}"
                )
            reasons.append(
                {
                    "reason_code": str(entry["reason_code"]),
                    "reason": str(entry["reason"]),
                    "source": f"manual_evidence:{entry['evidence_id']}",
                }
            )
        if task["fingerprint"]["canonicalization_ambiguous"]:
            ambiguous_matches.append(
                {
                    "source": f"openml_task:{task_id}",
                    "ambiguous": True,
                    "reason": "feature canonicalization has a tied content signature",
                }
            )

        deduped = {_reason_key(reason): reason for reason in reasons}
        reasons = [deduped[key] for key in sorted(deduped)]
        declared_status = (
            str(manual[task_id].get("status", "excluded"))
            if task_id in manual
            else None
        )
        if reasons:
            status = "ambiguous" if declared_status == "ambiguous" else "excluded"
        elif ambiguous_matches:
            status = "ambiguous"
        else:
            status = "eligible"
        if task_id in manual:
            lineage_cluster = str(manual[task_id]["lineage_cluster"])
        elif task_id in eligible_lineages:
            lineage_entry = eligible_lineages[task_id]
            expected_name = str(lineage_entry["expected_normalized_name"])
            if name != expected_name:
                raise ValueError(
                    f"eligible lineage {task_id} expected {expected_name!r}, "
                    f"got {name!r}"
                )
            lineage_cluster = str(lineage_entry["lineage_cluster"])
        else:
            raise ValueError(
                f"task {task_id} has no source-reviewed lineage declaration"
            )
        exposure_scope = None
        if task_id in manual:
            exposure_scope = str(
                manual[task_id].get(
                    "model_scope",
                    declarations["registry_scope"][
                        "default_manual_exclusion_model_scope"
                    ],
                )
            )
        row = {
            "openml_task_id": task_id,
            "openml_dataset_id": int(task["openml_dataset_id"]),
            "normalized_name": name,
            "lineage_cluster": lineage_cluster,
            "exposure_scope": exposure_scope,
            "status": status,
            "exclusion_reasons": reasons,
            "ambiguous_matches": sorted(
                ambiguous_matches, key=lambda match: str(match["source"])
            ),
            "fingerprint_evidence": {
                "feature_table_sha256": feature_hash,
                "target_marked_table_sha256": task["fingerprint"][
                    "target_marked_table_sha256"
                ],
                "openml_declared_md5": task["openml_declared_md5"],
            },
        }
        resolved.append(row)
        if status == "eligible":
            eligible = dict(task)
            eligible["lineage_cluster"] = lineage_cluster
            eligible_records.append(eligible)

    status_ids = {
        status: sorted(
            row["openml_task_id"] for row in resolved if row["status"] == status
        )
        for status in ("excluded", "ambiguous", "eligible")
    }
    expected_excluded = sorted(
        int(task_id) for task_id in declarations["expected_excluded_ctr23_task_ids"]
    )
    expected_ambiguous = sorted(
        int(task_id) for task_id in declarations["expected_ambiguous_ctr23_task_ids"]
    )
    if status_ids["excluded"] != expected_excluded:
        raise ValueError(
            "resolved exclusions differ from the declaration: "
            f"actual={status_ids['excluded']}, expected={expected_excluded}"
        )
    if status_ids["ambiguous"] != expected_ambiguous:
        raise ValueError(
            "resolved ambiguities differ from the declaration: "
            f"actual={status_ids['ambiguous']}, expected={expected_ambiguous}"
        )

    suite = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "ctr23_suite_id": int(declarations["ctr23_suite_id"]),
        "builder_source_sha256": builder_source_sha256,
        "declarations_sha256": sha256_json(declarations),
        "manual_evidence_sha256": manual_evidence_sha256,
        "runtime": runtime,
        "task_count": len(ctr_tasks),
        "official_coordinate_count": sum(
            int(task["official_splits"]["coordinate_count"])
            for task in ctr_tasks
        ),
        "ctr23_tasks": sorted(
            ctr_tasks, key=lambda task: int(task["openml_task_id"])
        ),
        "spent_source_tasks": sorted(
            source_tasks, key=lambda task: int(task["openml_task_id"])
        ),
    }
    suite["suite_snapshot_sha256"] = sha256_json(suite)

    registry = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "ctr23_suite_id": int(declarations["ctr23_suite_id"]),
        "registry_scope": declarations["registry_scope"],
        "suite_snapshot_sha256": suite["suite_snapshot_sha256"],
        "declarations_sha256": sha256_json(declarations),
        "manual_evidence_sha256": manual_evidence_sha256,
        "manual_evidence": manual_evidence,
        "counts": {status: len(ids) for status, ids in status_ids.items()},
        "excluded_task_ids": status_ids["excluded"],
        "ambiguous_task_ids": status_ids["ambiguous"],
        "eligible_task_ids": status_ids["eligible"],
        "tasks": resolved,
    }
    registry["contamination_registry_sha256"] = sha256_json(registry)

    confirmation, lockbox, diagnostics = balanced_panel_split(
        eligible_records, seed=int(declarations["split_seed"])
    )
    eligible_by_id = {
        int(task["openml_task_id"]): task for task in eligible_records
    }
    partition = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "ctr23_suite_id": int(declarations["ctr23_suite_id"]),
        "suite_snapshot_sha256": suite["suite_snapshot_sha256"],
        "contamination_registry_sha256": registry[
            "contamination_registry_sha256"
        ],
        "manual_evidence_sha256": manual_evidence_sha256,
        "split_seed": int(declarations["split_seed"]),
        "confirmation_task_ids": confirmation,
        "lockbox_task_ids": lockbox,
        "confirmation_coordinate_count": sum(
            int(eligible_by_id[task_id]["official_splits"]["coordinate_count"])
            for task_id in confirmation
        ),
        "lockbox_coordinate_count": sum(
            int(eligible_by_id[task_id]["official_splits"]["coordinate_count"])
            for task_id in lockbox
        ),
        "task_allocation_metadata": {
            str(task_id): {
                **_allocation_values(eligible_by_id[task_id]),
                "coordinate_count": int(
                    eligible_by_id[task_id]["official_splits"]["coordinate_count"]
                ),
                "resampling_regime": _resampling_regime(eligible_by_id[task_id]),
                "lineage_cluster": str(eligible_by_id[task_id]["lineage_cluster"]),
            }
            for task_id in sorted(eligible_by_id)
        },
        "split_diagnostics": diagnostics,
    }
    partition["partition_sha256"] = sha256_json(partition)
    partition["registry_bundle_sha256"] = sha256_json(
        {
            "suite_snapshot_sha256": suite["suite_snapshot_sha256"],
            "contamination_registry_sha256": registry[
                "contamination_registry_sha256"
            ],
            "partition_sha256": partition["partition_sha256"],
            "builder_source_sha256": builder_source_sha256,
            "manual_evidence_sha256": manual_evidence_sha256,
        }
    )
    return {"suite": suite, "registry": registry, "partition": partition}


def fetch_registry_inputs(
    declarations: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - CLI dependency error.
        raise RuntimeError("openml is required to build the registry") from exc

    suite_id = int(declarations["ctr23_suite_id"])
    suite = openml.study.get_suite(suite_id)
    ctr_ids = [int(task_id) for task_id in suite.tasks]
    if len(ctr_ids) != EXPECTED_CTR23_TASK_COUNT:
        raise ValueError(
            f"OpenML suite {suite_id} contains {len(ctr_ids)} tasks; "
            f"expected {EXPECTED_CTR23_TASK_COUNT}"
        )
    source_ids = sorted(
        {
            int(task_id)
            for group in declarations.get("source_task_groups", [])
            for task_id in group.get("openml_task_ids", [])
        }
    )
    ctr_tasks = [_task_record(task_id, include_splits=True) for task_id in ctr_ids]
    source_tasks = [
        _task_record(task_id, include_splits=False) for task_id in source_ids
    ]
    return ctr_tasks, source_tasks


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_or_verify_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    verify: bool,
) -> None:
    paths = {
        key: output_dir / filename for key, filename in ARTIFACT_FILENAMES.items()
    }
    rendered = {key: canonical_json_bytes(artifacts[key]) for key in paths}
    if verify:
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"cannot verify missing artifact {path}")
            if path.read_bytes() != rendered[key]:
                raise RuntimeError(
                    f"{path} does not match the deterministic rebuild"
                )
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing artifacts: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        path.write_bytes(rendered[key])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, default=DEFAULT_DECLARATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="rebuild and require byte equality with all existing artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    declarations = _load_json(args.declarations)
    if int(declarations.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported declaration schema")
    _validate_builder_runtime(declarations)
    ctr_tasks, source_tasks = fetch_registry_inputs(declarations)
    artifacts = build_artifacts(
        ctr_tasks,
        source_tasks,
        declarations,
        builder_source_sha256=sha256_file(Path(__file__)),
    )
    _write_or_verify_artifacts(
        args.output_dir, artifacts, verify=bool(args.verify_existing)
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "suite_snapshot_sha256": artifacts["suite"][
                    "suite_snapshot_sha256"
                ],
                "contamination_registry_sha256": artifacts["registry"][
                    "contamination_registry_sha256"
                ],
                "partition_sha256": artifacts["partition"]["partition_sha256"],
                "counts": artifacts["registry"]["counts"],
                "confirmation_coordinates": artifacts["partition"][
                    "confirmation_coordinate_count"
                ],
                "lockbox_coordinates": artifacts["partition"][
                    "lockbox_coordinate_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
