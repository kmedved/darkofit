"""Target-free feature transforms shared by panel-3 freeze and execution."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


DATETIME_COMPONENTS = frozenset(
    {
        "ordinal_day",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "quarter",
        "hour",
    }
)
LEXICAL_COUNTS = ("char_count", "token_count")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def feature_schema(X: pd.DataFrame) -> list[dict[str, str]]:
    if not isinstance(X, pd.DataFrame):
        raise RuntimeError("panel-3 feature input must be a DataFrame")
    return [
        {"name": str(name), "dtype": str(dtype)}
        for name, dtype in zip(X.columns, X.dtypes, strict=True)
    ]


def feature_schema_sha256(X: pd.DataFrame) -> str:
    return canonical_json_sha256(feature_schema(X))


def generated_values_sha256(
    generated: dict[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    for name, supplied in generated.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("panel-3 generated feature name is invalid")
        array = np.asarray(supplied)
        if array.ndim != 1 or array.dtype.kind not in {"i", "u", "f"}:
            raise RuntimeError("panel-3 generated feature values are invalid")
        if array.dtype.kind == "f":
            normalized = np.ascontiguousarray(array, dtype="<f8")
            dtype = b"<f8"
        elif array.dtype.kind == "u":
            normalized = np.ascontiguousarray(array, dtype="<u8")
            dtype = b"<u8"
        else:
            normalized = np.ascontiguousarray(array, dtype="<i8")
            dtype = b"<i8"
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(dtype)
        digest.update(len(normalized).to_bytes(8, "big"))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def canonical_group_hashes(
    X: pd.DataFrame,
    group_key_spec: Any,
) -> list[str]:
    """Hash target-free grouping fields under one frozen declaration."""
    if not isinstance(X, pd.DataFrame):
        raise RuntimeError("panel-3 grouping input must be a DataFrame")
    validate_group_key_spec(group_key_spec)
    columns = group_key_spec["source_columns"]
    fields = _string_list(
        columns,
        "grouping-column",
        allow_empty=False,
    )
    missing = [column for column in fields if column not in X.columns]
    if missing:
        raise RuntimeError(
            f"panel-3 grouping columns are missing: {missing}"
        )
    hashes = []
    for values in X.loc[:, fields].itertuples(index=False, name=None):
        digest = hashlib.sha256()
        for value in values:
            canonical = _canonical_group_value(value, group_key_spec)
            digest.update(len(canonical).to_bytes(8, "big", signed=False))
            digest.update(canonical)
        hashes.append(digest.hexdigest())
    return hashes


def validate_group_key_spec(group_key_spec: Any) -> dict[str, Any]:
    required = {"kind", "source_columns", "missing", "whitespace"}
    if (
        not isinstance(group_key_spec, dict)
        or set(group_key_spec) != required
        or group_key_spec["kind"]
        not in {
            "length_prefixed_nfkc_casefold_sha256_v1",
            "typed_value_tuple_sha256_v1",
        }
        or group_key_spec["missing"] not in {"reject", "empty_string"}
        or group_key_spec["whitespace"] not in {"collapse", "preserve"}
    ):
        raise RuntimeError("panel-3 group-key declaration is invalid")
    _string_list(
        group_key_spec["source_columns"],
        "grouping-column",
        allow_empty=False,
    )
    return group_key_spec


def _is_missing_scalar(value: Any) -> bool:
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise RuntimeError("panel-3 grouping value is nonscalar")
    return bool(missing)


def _canonical_text(value: Any, whitespace: str) -> str:
    try:
        text = unicodedata.normalize("NFKC", str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("panel-3 grouping value is invalid") from exc
    if whitespace == "collapse":
        return " ".join(text.split())
    return text


def _canonical_group_value(
    value: Any,
    group_key_spec: dict[str, Any],
) -> bytes:
    missing = _is_missing_scalar(value)
    if missing:
        if group_key_spec["missing"] == "reject":
            raise RuntimeError("panel-3 grouping value is missing")
        value = ""
    whitespace = group_key_spec["whitespace"]
    if (
        group_key_spec["kind"]
        == "length_prefixed_nfkc_casefold_sha256_v1"
    ):
        text = _canonical_text(value, "preserve").casefold()
        if whitespace == "collapse":
            text = " ".join(text.split())
        return text.encode("utf-8")
    if isinstance(value, (str, np.str_)):
        token = "s:" + _canonical_text(value, whitespace)
    elif isinstance(value, (bool, np.bool_)):
        token = "b:1" if bool(value) else "b:0"
    elif isinstance(value, (int, np.integer)):
        token = f"i:{int(value)}"
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("panel-3 grouping float is nonfinite")
        token = f"f:{number.hex()}"
    else:
        raise RuntimeError(
            "panel-3 typed grouping value has unsupported type"
        )
    return token.encode("utf-8")


def greedy_group_fold_ids(
    group_hashes: list[str],
    *,
    n_splits: int = 3,
) -> list[int]:
    """Assign whole groups by frozen size/hash order to smallest folds."""
    if (
        type(n_splits) is not int
        or n_splits < 2
        or not isinstance(group_hashes, list)
        or not group_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in group_hashes
        )
    ):
        raise RuntimeError("panel-3 group-fold inputs are invalid")
    counts = Counter(group_hashes)
    if len(counts) < n_splits:
        raise RuntimeError("panel-3 has fewer groups than folds")
    fold_sizes = [0] * n_splits
    assignment = {}
    for group, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    ):
        fold = min(range(n_splits), key=lambda value: (fold_sizes[value], value))
        assignment[group] = fold
        fold_sizes[fold] += count
    if any(value == 0 for value in fold_sizes):
        raise RuntimeError("panel-3 group split produced an empty fold")
    return [assignment[value] for value in group_hashes]


def _string_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeError(f"panel-3 {label} declaration is invalid")
    return list(value)


def _validate_datetime_declaration(
    declaration: Any,
) -> tuple[str, str, list[str], bool]:
    required = {
        "source_column",
        "output_prefix",
        "format",
        "utc",
        "components",
        "drop_source",
        "missing",
    }
    if (
        not isinstance(declaration, dict)
        or set(declaration) != required
        or not isinstance(declaration["source_column"], str)
        or not declaration["source_column"]
        or not isinstance(declaration["output_prefix"], str)
        or not declaration["output_prefix"]
        or not isinstance(declaration["format"], str)
        or not declaration["format"]
        or type(declaration["utc"]) is not bool
        or type(declaration["drop_source"]) is not bool
        or declaration["missing"] not in {"reject", "NaT"}
    ):
        raise RuntimeError(
            "panel-3 datetime-calendar declaration is invalid"
        )
    components = _string_list(
        declaration["components"],
        "datetime component",
        allow_empty=False,
    )
    if not set(components) <= DATETIME_COMPONENTS:
        raise RuntimeError(
            "panel-3 datetime-calendar component is unsupported"
        )
    return (
        declaration["source_column"],
        declaration["output_prefix"],
        components,
        declaration["drop_source"],
    )


def _validate_lexical_declaration(
    declaration: Any,
) -> tuple[str, str, bool]:
    required = {
        "source_column",
        "output_prefix",
        "counts",
        "unicode_normalization",
        "missing",
        "drop_source",
    }
    if (
        not isinstance(declaration, dict)
        or set(declaration) != required
        or not isinstance(declaration["source_column"], str)
        or not declaration["source_column"]
        or not isinstance(declaration["output_prefix"], str)
        or not declaration["output_prefix"]
        or declaration["counts"] != list(LEXICAL_COUNTS)
        or declaration["unicode_normalization"] != "NFKC"
        or declaration["missing"] != "empty_string"
        or type(declaration["drop_source"]) is not bool
    ):
        raise RuntimeError("panel-3 lexical-count declaration is invalid")
    return (
        declaration["source_column"],
        declaration["output_prefix"],
        declaration["drop_source"],
    )


def validate_feature_policy(policy: Any) -> dict[str, Any]:
    """Validate one declaration without loading features or targets."""
    if not isinstance(policy, dict):
        raise RuntimeError("panel-3 feature policy is invalid")
    kind = policy.get("kind")
    if kind == "none":
        if set(policy) != {"kind"}:
            raise RuntimeError("panel-3 no-op feature policy changed")
        return policy
    if kind == "drop_columns":
        if set(policy) != {"kind", "columns"}:
            raise RuntimeError("panel-3 drop-column policy changed")
        _string_list(
            policy["columns"],
            "dropped-column",
            allow_empty=False,
        )
        return policy
    if kind != "target_free_transform_v1" or set(policy) != {
        "kind",
        "drop_columns",
        "datetime_calendar",
        "lexical_counts",
    }:
        raise RuntimeError("panel-3 feature policy kind is unsupported")
    drops = _string_list(
        policy["drop_columns"],
        "dropped-column",
        allow_empty=True,
    )
    datetime_declarations = policy["datetime_calendar"]
    lexical_declarations = policy["lexical_counts"]
    if (
        not isinstance(datetime_declarations, list)
        or not isinstance(lexical_declarations, list)
        or (
            not drops
            and not datetime_declarations
            and not lexical_declarations
        )
    ):
        raise RuntimeError(
            "panel-3 target-free feature declarations are invalid"
        )
    sources = []
    generated = []
    source_drops = []
    for declaration in datetime_declarations:
        source, prefix, components, drop_source = (
            _validate_datetime_declaration(declaration)
        )
        sources.append(source)
        generated.extend(f"{prefix}_{value}" for value in components)
        if drop_source:
            source_drops.append(source)
    for declaration in lexical_declarations:
        source, prefix, drop_source = _validate_lexical_declaration(
            declaration
        )
        sources.append(source)
        generated.extend(f"{prefix}_{value}" for value in LEXICAL_COUNTS)
        if drop_source:
            source_drops.append(source)
    if (
        len(set(sources)) != len(sources)
        or len(set(generated)) != len(generated)
        or len(set(source_drops)) != len(source_drops)
        or set(source_drops) & set(drops)
    ):
        raise RuntimeError(
            "panel-3 target-free transform declarations collide"
        )
    return policy


def _datetime_values(
    X: pd.DataFrame,
    declaration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], bool]:
    source, _prefix, components, drop_source = (
        _validate_datetime_declaration(declaration)
    )
    if source not in X.columns:
        raise RuntimeError(
            f"panel-3 datetime-calendar source is missing: {source}"
        )
    try:
        parsed = pd.to_datetime(
            X[source],
            format=declaration["format"],
            utc=declaration["utc"],
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"panel-3 datetime-calendar parse failed: {source}"
        ) from exc
    missing = bool(parsed.isna().any())
    if missing and declaration["missing"] == "reject":
        raise RuntimeError(
            f"panel-3 datetime-calendar source has missing values: {source}"
        )
    accessor = parsed.dt
    calendar = {
        "year": accessor.year,
        "month": accessor.month,
        "day": accessor.day,
        "dayofweek": accessor.dayofweek,
        "dayofyear": accessor.dayofyear,
        "quarter": accessor.quarter,
        "hour": accessor.hour,
    }
    wall_time = parsed
    if getattr(parsed.dt, "tz", None) is not None:
        wall_time = parsed.dt.tz_localize(None)
    ordinal = (
        wall_time.dt.normalize() - pd.Timestamp("1970-01-01")
    ) / pd.Timedelta(days=1)
    calendar["ordinal_day"] = ordinal
    values = {}
    output_dtype = np.float64 if missing else np.int64
    for component in components:
        name = f"{declaration['output_prefix']}_{component}"
        values[name] = np.asarray(
            calendar[component],
            dtype=output_dtype,
        )
    return values, bool(drop_source)


def _lexical_values(
    X: pd.DataFrame,
    declaration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], bool]:
    source, prefix, drop_source = _validate_lexical_declaration(
        declaration
    )
    if source not in X.columns:
        raise RuntimeError(
            f"panel-3 lexical-count source is missing: {source}"
        )
    normalized = []
    for value in X[source].tolist():
        missing = pd.isna(value)
        if not isinstance(missing, (bool, np.bool_)):
            raise RuntimeError(
                f"panel-3 lexical-count value is nonscalar: {source}"
            )
        if bool(missing):
            text = ""
        else:
            try:
                text = unicodedata.normalize("NFKC", str(value))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"panel-3 lexical-count value is invalid: {source}"
                ) from exc
        normalized.append(text)
    values = {
        f"{prefix}_char_count": np.asarray(
            [len(text) for text in normalized], dtype=np.int64
        ),
        f"{prefix}_token_count": np.asarray(
            [len(text.split()) for text in normalized], dtype=np.int64
        ),
    }
    return values, bool(drop_source)


def apply_feature_policy(
    X: pd.DataFrame,
    policy: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one frozen policy without consulting a target or split."""
    if not isinstance(X, pd.DataFrame):
        raise RuntimeError("panel-3 feature input must be a DataFrame")
    source_columns = list(X.columns)
    if (
        any(not isinstance(name, str) or not name for name in source_columns)
        or len(set(source_columns)) != len(source_columns)
    ):
        raise RuntimeError(
            "panel-3 feature names must be unique nonempty strings"
        )
    validate_feature_policy(policy)
    kind = policy.get("kind")
    datetime_declarations: list[dict[str, Any]] = []
    lexical_declarations: list[dict[str, Any]] = []
    if kind == "none":
        if set(policy) != {"kind"}:
            raise RuntimeError("panel-3 no-op feature policy changed")
        drop_columns = []
    elif kind == "drop_columns":
        if set(policy) != {"kind", "columns"}:
            raise RuntimeError("panel-3 drop-column policy changed")
        drop_columns = _string_list(
            policy["columns"],
            "dropped-column",
            allow_empty=False,
        )
    elif kind == "target_free_transform_v1":
        if set(policy) != {
            "kind",
            "drop_columns",
            "datetime_calendar",
            "lexical_counts",
        }:
            raise RuntimeError(
                "panel-3 target-free feature policy changed"
            )
        drop_columns = _string_list(
            policy["drop_columns"],
            "dropped-column",
            allow_empty=True,
        )
        datetime_declarations = policy["datetime_calendar"]
        lexical_declarations = policy["lexical_counts"]
        if (
            not isinstance(datetime_declarations, list)
            or not isinstance(lexical_declarations, list)
            or (
                not drop_columns
                and not datetime_declarations
                and not lexical_declarations
            )
        ):
            raise RuntimeError(
                "panel-3 target-free feature declarations are invalid"
            )
    else:
        raise RuntimeError("panel-3 feature policy kind is unsupported")
    missing_drops = [
        name for name in drop_columns if name not in source_columns
    ]
    if missing_drops:
        raise RuntimeError(
            f"panel-3 dropped columns are missing: {missing_drops}"
        )

    generated: dict[str, np.ndarray] = {}
    transform_sources = []
    source_drops = []
    for declaration in datetime_declarations:
        values, drop_source = _datetime_values(X, declaration)
        source = declaration["source_column"]
        if source in transform_sources:
            raise RuntimeError(
                "panel-3 transform source is repeated"
            )
        transform_sources.append(source)
        if drop_source:
            source_drops.append(source)
        for name, value in values.items():
            if name in source_columns or name in generated:
                raise RuntimeError(
                    f"panel-3 generated-column collision: {name}"
                )
            generated[name] = value
    for declaration in lexical_declarations:
        values, drop_source = _lexical_values(X, declaration)
        source = declaration["source_column"]
        if source in transform_sources:
            raise RuntimeError(
                "panel-3 transform source is repeated"
            )
        transform_sources.append(source)
        if drop_source:
            source_drops.append(source)
        for name, value in values.items():
            if name in source_columns or name in generated:
                raise RuntimeError(
                    f"panel-3 generated-column collision: {name}"
                )
            generated[name] = value
    if len(set(source_drops)) != len(source_drops) or set(
        source_drops
    ) & set(drop_columns):
        raise RuntimeError("panel-3 transform source is dropped twice")
    all_drops = [*drop_columns, *source_drops]
    retained = [
        name for name in source_columns if name not in set(all_drops)
    ]
    if not retained and not generated:
        raise RuntimeError("panel-3 feature policy drops every feature")
    result = X.loc[:, retained].copy()
    for name, values in generated.items():
        result[name] = values
    metadata = {
        "kind": kind,
        "policy_sha256": canonical_json_sha256(policy),
        "source_columns_sha256": canonical_json_sha256(source_columns),
        "dropped_columns": all_drops,
        "generated_columns": list(generated),
        "generated_values_sha256": generated_values_sha256(generated),
        "retained_source_columns": retained,
        "retained_columns": list(result.columns),
        "retained_feature_count": int(result.shape[1]),
        "retained_columns_sha256": canonical_json_sha256(
            list(result.columns)
        ),
        "output_schema": feature_schema(result),
        "output_schema_sha256": feature_schema_sha256(result),
    }
    return result, metadata


def categorical_flags_after_policy(
    source_columns: list[str],
    declared_categorical: list[bool],
    metadata: dict[str, Any],
) -> list[bool]:
    """Map original flags onto retained columns; transforms are numeric."""
    if (
        len(source_columns) != len(declared_categorical)
        or len(set(source_columns)) != len(source_columns)
        or any(type(value) is not bool for value in declared_categorical)
    ):
        raise RuntimeError(
            "panel-3 categorical declaration width changed"
        )
    by_name = dict(zip(source_columns, declared_categorical, strict=True))
    retained = metadata.get("retained_source_columns")
    generated = metadata.get("generated_columns")
    if (
        not isinstance(retained, list)
        or not isinstance(generated, list)
        or any(name not in by_name for name in retained)
    ):
        raise RuntimeError("panel-3 feature metadata is invalid")
    return [by_name[name] for name in retained] + [False] * len(generated)
