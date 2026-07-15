import base64
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.build_ctr23_contamination_registry import (
    _bloom_filter_uint128,
    _bloom_membership_fraction,
    _comparison_series_hashes,
    _dtype_family,
    _eligible_lineage_clusters,
    _git_object_sha1,
    _kmv_containment,
    _resolve_manual_evidence,
    _row_hashes,
    _schema_row_commitments,
    _schema_row_lane_sums,
    _validate_numeric_dtype,
    _validate_expected_suite,
    balanced_panel_split,
    build_artifacts,
    dataset_fingerprint,
    main as registry_main,
    near_match_evidence,
    sha256_file,
    sha256_json,
)


DECLARATIONS = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "ctr23_contamination_sources.json"
)
BENCHMARKS = DECLARATIONS.parent
NEAR_MATCH_THRESHOLDS = json.loads(
    DECLARATIONS.read_text(encoding="utf-8")
)["near_match_thresholds"]


def _exact_keys(fingerprint):
    return {
        key: fingerprint[key]
        for key in (
            "feature_content_multiset_sha256",
            "feature_table_sha256",
            "opaque_target_value_sha256",
            "target_marked_table_sha256",
            "canonicalization_ambiguous",
        )
    }


def test_fingerprint_rejects_empty_rows_and_featureless_tables():
    with pytest.raises(ValueError, match="at least one row"):
        dataset_fingerprint(
            pd.DataFrame({"x": pd.Series(dtype=np.float64)}),
            pd.Series(dtype=np.float64),
        )
    with pytest.raises(ValueError, match="at least one feature"):
        dataset_fingerprint(
            pd.DataFrame(index=range(3)),
            pd.Series([1.0, 2.0, 3.0]),
        )


def test_dtype_validation_precedes_empty_row_guard_for_features_and_target(
    monkeypatch,
):
    def reject_float32(dtype):
        if np.dtype(dtype) == np.dtype(np.float32):
            raise TypeError("synthetic wide dtype rejection")

    monkeypatch.setattr(
        "benchmarks.build_ctr23_contamination_registry._validate_numeric_dtype",
        reject_float32,
    )
    with pytest.raises(TypeError, match="synthetic wide dtype rejection"):
        dataset_fingerprint(
            pd.DataFrame({"feature": pd.Series(dtype=np.float32)}),
            pd.Series(dtype=np.int64),
        )
    with pytest.raises(TypeError, match="synthetic wide dtype rejection"):
        dataset_fingerprint(
            pd.DataFrame({"feature": pd.Series(dtype="string")}),
            pd.Series(dtype=np.float32),
        )


def test_exact_fingerprint_is_invariant_to_row_column_order_and_feature_names():
    X = pd.DataFrame(
        {
            "numeric": [1.0, -0.0, 3.5, np.nan, 1.0],
            "label": pd.Series(["a", "b", "a", None, "c"], dtype="category"),
            "flag": [True, False, True, False, True],
        }
    )
    y = pd.Series([4.0, 2.0, 9.0, 1.0, 4.0])
    expected = dataset_fingerprint(X, y)

    order = [3, 0, 4, 1, 2]
    changed = X.iloc[order, [2, 0, 1]].copy()
    changed.columns = ["renamed c", "renamed a", "renamed b"]
    actual = dataset_fingerprint(changed, y.iloc[order])

    assert _exact_keys(actual) == _exact_keys(expected)


def test_exact_fingerprint_detects_cell_duplicate_and_target_role_changes():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    y = pd.Series([100, 200, 300])
    original = dataset_fingerprint(X, y)

    edited = X.copy()
    edited.loc[1, "a"] = 99
    assert (
        dataset_fingerprint(edited, y)["feature_table_sha256"]
        != original["feature_table_sha256"]
    )

    duplicated = dataset_fingerprint(
        pd.concat([X, X.iloc[[0]]], ignore_index=True),
        pd.concat([y, y.iloc[[0]]], ignore_index=True),
    )
    assert duplicated["feature_table_sha256"] != original["feature_table_sha256"]

    role_changed = dataset_fingerprint(
        pd.DataFrame({"a": X["a"], "target_as_feature": y}), X["b"]
    )
    assert (
        role_changed["target_marked_table_sha256"]
        != original["target_marked_table_sha256"]
    )


def test_numeric_fingerprint_preserves_int64_precision_and_exact_equivalence():
    boundary = 2**53
    first = dataset_fingerprint(
        pd.DataFrame({"value": pd.Series([boundary], dtype="int64")}),
        pd.Series([boundary], dtype="int64"),
    )
    adjacent = dataset_fingerprint(
        pd.DataFrame({"value": pd.Series([boundary + 1], dtype="int64")}),
        pd.Series([boundary + 1], dtype="int64"),
    )
    equivalent_float = dataset_fingerprint(
        pd.DataFrame({"value": pd.Series([float(boundary)], dtype="float64")}),
        pd.Series([float(boundary)], dtype="float64"),
    )

    assert (
        first["feature_content_multiset_sha256"]
        != adjacent["feature_content_multiset_sha256"]
    )
    assert first["feature_table_sha256"] != adjacent["feature_table_sha256"]
    assert (
        first["opaque_target_value_sha256"]
        != adjacent["opaque_target_value_sha256"]
    )
    assert _exact_keys(first) == _exact_keys(equivalent_float)


def test_numeric_fingerprint_rejects_unsupported_wide_and_complex_dtypes():
    class SyntheticWideFloatDtype:
        kind = "f"
        itemsize = 16

        def __str__(self):
            return "synthetic-float128"

    with pytest.raises(TypeError, match="wider than float64"):
        _validate_numeric_dtype(SyntheticWideFloatDtype())

    if np.dtype(np.longdouble).itemsize > 8:
        one = np.longdouble(1)
        adjacent = np.nextafter(one, np.longdouble(2))
        assert float(one) == float(adjacent)
        wide_cases = (
            np.asarray([one, adjacent], dtype=np.longdouble),
            np.asarray([np.nan], dtype=np.longdouble),
            np.asarray([], dtype=np.longdouble),
        )
        for values in wide_cases:
            with pytest.raises(TypeError, match="wider than float64"):
                dataset_fingerprint(
                    pd.DataFrame({"wide": values}),
                    pd.Series(np.arange(len(values))),
                )

    with pytest.raises(TypeError, match="complex numeric"):
        dataset_fingerprint(
            pd.DataFrame({"complex": np.asarray([1 + 2j, 3 + 4j])}),
            pd.Series([0, 1]),
        )


def test_feature_identity_survives_target_rename_and_transform():
    X = pd.DataFrame({"a": np.arange(20), "b": np.arange(20) ** 2})
    first = dataset_fingerprint(X, pd.Series(np.arange(20), name="old"))
    second = dataset_fingerprint(
        X, pd.Series(np.log1p(np.arange(20)), name="new")
    )

    assert first["feature_table_sha256"] == second["feature_table_sha256"]
    assert (
        first["target_marked_table_sha256"]
        != second["target_marked_table_sha256"]
    )
    assert (
        first["opaque_target_value_sha256"]
        != second["opaque_target_value_sha256"]
    )


def test_unresolved_tied_column_multisets_fail_closed():
    X = pd.DataFrame({"forward": [1, 2, 3], "reverse": [3, 2, 1]})
    y = pd.Series([0, 1, 2])
    fingerprint = dataset_fingerprint(X, y)

    assert fingerprint["canonicalization_ambiguous"] is True
    assert fingerprint["marginal_tie_group_count"] == 1
    assert fingerprint["feature_table_sha256"] is None


def test_anchored_tied_columns_have_global_order_invariant_canonicalization():
    X = pd.DataFrame(
        {
            "forward": [1, 2, 3, 4],
            "reverse": [4, 3, 2, 1],
            "anchor": [10, 30, 20, 40],
        }
    )
    y = pd.Series([0, 1, 2, 3])
    fingerprint = dataset_fingerprint(X, y)
    order = [2, 0, 3, 1]
    changed = X.iloc[order, [1, 2, 0]].copy()
    changed.columns = ["renamed one", "renamed anchor", "renamed two"]
    reordered = dataset_fingerprint(changed, y.iloc[order])

    assert fingerprint["canonicalization_ambiguous"] is False
    assert _exact_keys(fingerprint) == _exact_keys(reordered)


def test_near_match_flags_subsets_and_feature_add_drop_but_not_unrelated_data():
    rng = np.random.default_rng(7)
    n_rows = 1_000
    X = pd.DataFrame(
        {
            "a": np.arange(n_rows, dtype=float),
            "b": rng.normal(size=n_rows),
            "c": np.tile(["x", "y", "z", "w"], n_rows // 4),
        }
    )
    y = pd.Series(rng.normal(size=n_rows))
    base = dataset_fingerprint(X, y)
    subset = dataset_fingerprint(X.iloc[:700], y.iloc[:700])
    dropped = dataset_fingerprint(X.drop(columns="c"), y)
    added = dataset_fingerprint(X.assign(extra=np.arange(n_rows) * 3), y)
    unrelated_X = pd.DataFrame(
        {
            "u": rng.normal(size=n_rows),
            "v": rng.normal(size=n_rows),
            "w": rng.integers(1000, 2000, size=n_rows),
        }
    )
    unrelated = dataset_fingerprint(unrelated_X, y)

    for candidate in (subset, dropped, added):
        assert near_match_evidence(
            base, candidate, **NEAR_MATCH_THRESHOLDS
        )["ambiguous"]
    assert not near_match_evidence(
        base, unrelated, **NEAR_MATCH_THRESHOLDS
    )["ambiguous"]

    reordered = dataset_fingerprint(
        X.iloc[:, [2, 0, 1]].rename(columns={"c": "z", "a": "x", "b": "y"}),
        y,
    )
    assert near_match_evidence(
        base, reordered, **NEAR_MATCH_THRESHOLDS
    ) == near_match_evidence(
        base, dataset_fingerprint(X, y), **NEAR_MATCH_THRESHOLDS
    )


def test_near_match_uses_common_textual_representation():
    values = [f"level-{index}" for index in range(40)] * 8
    source = pd.DataFrame(
        {
            "first": pd.Series(values, dtype="category"),
            "second": pd.Series(list(reversed(values)), dtype="category"),
        }
    )
    reingested = source.astype("string").rename(
        columns={"first": "renamed-a", "second": "renamed-b"}
    )
    y = pd.Series(np.arange(len(source)))

    evidence = near_match_evidence(
        dataset_fingerprint(source, y),
        dataset_fingerprint(reingested, y),
        **NEAR_MATCH_THRESHOLDS,
    )
    assert evidence["ambiguous"]
    assert evidence["column_sketch_alarm"]


def test_production_kmv_thresholds_recognize_high_cardinality_row_subset():
    n_rows = 1_000
    X = pd.DataFrame(
        {
            "first": np.arange(n_rows, dtype=float),
            "second": np.arange(n_rows, dtype=float) ** 2 + 0.5,
        }
    )
    y = pd.Series(np.arange(n_rows))
    evidence = near_match_evidence(
        dataset_fingerprint(X, y),
        dataset_fingerprint(X.iloc[:700], y.iloc[:700]),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert evidence["ambiguous"]
    assert evidence["column_sketch_alarm"]
    assert evidence["mean_column_containment"] >= 0.95


def test_binary_matrix_row_subset_uses_joint_low_cardinality_evidence():
    n_rows = 300
    X = pd.DataFrame(
        {
            f"bit-{column}": (
                (np.arange(n_rows) * (column + 3) + column) % 97
                < (column + 5)
            ).astype(np.int8)
            for column in range(32)
        }
    )
    y = pd.Series(np.arange(n_rows))
    evidence = near_match_evidence(
        dataset_fingerprint(X, y),
        dataset_fingerprint(X.iloc[:210], y.iloc[:210]),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert evidence["ambiguous"]
    assert evidence["row_sketch_alarm"]
    assert evidence["matched_feature_count"] == 0


def test_added_low_cardinality_numeric_feature_raises_column_alarm():
    n_rows = 300
    X = pd.DataFrame(
        {
            f"bit-{column}": (
                (np.arange(n_rows) * (column + 3) + column) % 97
                < (column + 5)
            ).astype(np.int8)
            for column in range(32)
        }
    )
    y = pd.Series(np.arange(n_rows))
    added = X.assign(
        extra=((np.arange(n_rows) * 37) % 101 < 43).astype(np.int8)
    )
    evidence = near_match_evidence(
        dataset_fingerprint(X, y),
        dataset_fingerprint(added, y),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert evidence["ambiguous"]
    assert evidence["column_sketch_alarm"]
    assert evidence["matched_feature_count"] == X.shape[1]


def test_binary_schema_deletion_alarm_is_permutation_and_operand_invariant():
    n_rows = 300
    X = pd.DataFrame(
        {
            f"bit-{column}": (
                (np.arange(n_rows) * (column + 3) + column) % 97
                < (column + 5)
            ).astype(np.int8)
            for column in range(32)
        }
    )
    y = pd.Series(np.arange(n_rows))
    subset = X.iloc[:210].copy()
    subset["extra"] = (
        (np.arange(len(subset)) * 37) % 101 < 43
    ).astype(np.int8)
    base_fingerprint = dataset_fingerprint(X, y)
    changed_fingerprint = dataset_fingerprint(subset, y.iloc[:210])
    forward = near_match_evidence(
        base_fingerprint,
        changed_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    reverse = near_match_evidence(
        changed_fingerprint,
        base_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    row_order = np.random.default_rng(12).permutation(len(subset))
    permuted = subset.iloc[row_order, ::-1].copy()
    permuted.columns = [f"renamed-{index}" for index in range(permuted.shape[1])]
    permuted_evidence = near_match_evidence(
        base_fingerprint,
        dataset_fingerprint(permuted, y.iloc[:210].iloc[row_order]),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert len(base_fingerprint["schema_deletion_row_sketch_deck"]) == 33
    assert len(changed_fingerprint["schema_deletion_row_sketch_deck"]) == 34
    assert forward == reverse
    assert forward == permuted_evidence
    assert forward["ambiguous"]
    assert not forward["column_sketch_alarm"]
    assert not forward["row_sketch_alarm"]
    assert forward["schema_row_comparison_supported"]
    assert forward["schema_deletion_supported"]
    assert forward["schema_deletion_row_alarm"]
    assert forward["schema_deletion_row_containment"] >= 0.95
    assert forward["schema_deletion_included_feature_count"] == 32
    assert forward["schema_deletion_tied_max_count"] >= 1


def test_distinct_low_card_domains_identify_only_the_added_feature_deletion():
    rng = np.random.default_rng(19)
    n_rows = 1_000
    n_features = 12
    bits = rng.integers(0, 2, size=(n_rows, n_features), dtype=np.int8)
    X = pd.DataFrame(
        {
            f"domain-{column}": 100 * column + bits[:, column].astype(np.int64)
            for column in range(n_features)
        }
    )
    y = pd.Series(np.arange(n_rows))
    selected = rng.choice(n_rows, size=700, replace=False)
    extra = 9_999 + rng.integers(0, 2, size=n_rows, dtype=np.int64)
    base_fingerprint = dataset_fingerprint(X, y)
    expected = None
    for insertion in (0, n_features // 2, n_features):
        candidate = X.iloc[selected].copy()
        candidate.insert(insertion, "extra", extra[selected])
        row_order = rng.permutation(len(candidate))
        column_order = rng.permutation(candidate.shape[1])
        candidate = candidate.iloc[row_order, column_order].copy()
        candidate.columns = [
            f"renamed-{index}" for index in range(candidate.shape[1])
        ]
        candidate_fingerprint = dataset_fingerprint(
            candidate, y.iloc[selected].iloc[row_order]
        )
        forward = near_match_evidence(
            base_fingerprint,
            candidate_fingerprint,
            **NEAR_MATCH_THRESHOLDS,
        )
        reverse = near_match_evidence(
            candidate_fingerprint,
            base_fingerprint,
            **NEAR_MATCH_THRESHOLDS,
        )
        assert forward == reverse
        if expected is None:
            expected = forward
        else:
            assert forward == expected

    assert expected is not None
    assert expected["ambiguous"]
    assert not expected["column_sketch_alarm"]
    assert not expected["row_sketch_alarm"]
    assert expected["schema_deletion_row_alarm"]
    assert expected["schema_deletion_row_containment"] == 1.0
    assert expected["schema_deletion_tied_max_count"] == 1
    assert expected["schema_deletion_included_feature_count"] == n_features


def test_bloom_witness_catches_subset_that_omits_parent_bottom_k():
    parent_high = np.arange(1_000, dtype=np.uint64)
    parent_low = np.arange(10_000, 11_000, dtype=np.uint64)
    subset_high = parent_high[300:]
    subset_low = parent_low[300:]
    parent_bottom = [
        f"{int(high):016x}{int(low):016x}"
        for high, low in zip(parent_high[:128], parent_low[:128])
    ]
    subset_bottom = [
        f"{int(high):016x}{int(low):016x}"
        for high, low in zip(subset_high[:128], subset_low[:128])
    ]

    assert _kmv_containment(
        parent_bottom,
        subset_bottom,
        left_cardinality=1_000,
        right_cardinality=700,
    ) == 0.0
    bloom = _bloom_filter_uint128(
        parent_high, parent_low, unique_count=1_000
    )
    assert _bloom_membership_fraction(subset_bottom, bloom) == 1.0


def test_full_matcher_catches_subset_that_omits_parent_bottom_k():
    rng = np.random.default_rng(91)
    n_rows = 1_000
    n_features = 12
    bits = rng.integers(0, 2, size=(n_rows, n_features), dtype=np.int8)
    X = pd.DataFrame(
        {
            f"domain-{column}": 100 * column + bits[:, column].astype(np.int64)
            for column in range(n_features)
        }
    )
    y = pd.Series(np.arange(n_rows))
    comparison_columns = [
        _comparison_series_hashes(X.iloc[:, index], _dtype_family(X.iloc[:, index]))
        for index in range(X.shape[1])
    ]
    lane_sums = _schema_row_lane_sums(comparison_columns, n_rows=n_rows)
    high, low = _schema_row_commitments(
        lane_sums, included_feature_count=n_features
    )
    row_commitments = np.asarray(
        [
            f"{int(row_high):016x}{int(row_low):016x}"
            for row_high, row_low in zip(high, low)
        ],
        dtype=object,
    )
    unlabeled_values = np.column_stack(comparison_columns)
    unlabeled_values.sort(axis=1)
    legacy_row_hashes = _row_hashes(
        [
            unlabeled_values[:, index]
            for index in range(unlabeled_values.shape[1])
        ],
        n_rows=n_rows,
    )
    legacy_row_commitments = np.asarray(
        [f"{int(value):016x}" for value in legacy_row_hashes],
        dtype=object,
    )
    base_fingerprint = dataset_fingerprint(X, y)
    base_full = next(
        view
        for view in base_fingerprint["schema_deletion_row_sketch_deck"]
        if view["included_feature_count"] == n_features
    )
    parent_bottom = set(base_full["bottom_k_row_hashes"])
    parent_legacy_bottom = set(
        base_fingerprint["bottom_k_unlabeled_feature_row_hashes"]
    )
    eligible_rows = np.flatnonzero(
        np.asarray(
            [
                deck_value not in parent_bottom
                and legacy_value not in parent_legacy_bottom
                for deck_value, legacy_value in zip(
                    row_commitments, legacy_row_commitments
                )
            ]
        )
    )
    assert len(eligible_rows) >= 700
    selected = eligible_rows[:700]
    shared_subset = X.iloc[selected].copy()
    shared_fingerprint = dataset_fingerprint(shared_subset, y.iloc[selected])
    shared_full = next(
        view
        for view in shared_fingerprint["schema_deletion_row_sketch_deck"]
        if view["included_feature_count"] == n_features
    )
    assert parent_bottom.isdisjoint(shared_full["bottom_k_row_hashes"])
    assert parent_legacy_bottom.isdisjoint(
        shared_fingerprint["bottom_k_unlabeled_feature_row_hashes"]
    )
    assert _kmv_containment(
        base_full["bottom_k_row_hashes"],
        shared_full["bottom_k_row_hashes"],
        left_cardinality=base_full["row_unique_count"],
        right_cardinality=shared_full["row_unique_count"],
    ) == 0.0

    same_schema_forward = near_match_evidence(
        base_fingerprint,
        shared_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    same_schema_reverse = near_match_evidence(
        shared_fingerprint,
        base_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    same_schema_order = np.random.default_rng(92).permutation(
        len(shared_subset)
    )
    same_schema_permuted = shared_subset.iloc[
        same_schema_order, ::-1
    ].copy()
    same_schema_permuted.columns = [
        f"same-schema-renamed-{index}"
        for index in range(same_schema_permuted.shape[1])
    ]
    same_schema_permuted_evidence = near_match_evidence(
        base_fingerprint,
        dataset_fingerprint(
            same_schema_permuted,
            y.iloc[selected].iloc[same_schema_order],
        ),
        **NEAR_MATCH_THRESHOLDS,
    )
    assert (
        same_schema_forward
        == same_schema_reverse
        == same_schema_permuted_evidence
    )
    assert same_schema_forward["ambiguous"]
    assert not same_schema_forward["column_sketch_alarm"]
    assert not same_schema_forward["row_sketch_alarm"]
    assert same_schema_forward["schema_row_comparison_supported"]
    assert not same_schema_forward["schema_deletion_supported"]
    assert same_schema_forward["schema_deletion_row_alarm"]
    assert same_schema_forward["schema_deletion_row_containment"] == 1.0
    assert same_schema_forward["schema_deletion_tied_max_count"] == 1
    assert (
        same_schema_forward["schema_deletion_included_feature_count"]
        == n_features
    )

    duplicate_expanded = pd.concat(
        [shared_subset, shared_subset.iloc[:300]],
        ignore_index=True,
    )
    duplicate_target = pd.concat(
        [y.iloc[selected], y.iloc[selected[:300]]],
        ignore_index=True,
    )
    duplicate_fingerprint = dataset_fingerprint(
        duplicate_expanded,
        duplicate_target,
    )
    duplicate_evidence = near_match_evidence(
        base_fingerprint,
        duplicate_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    duplicate_reverse = near_match_evidence(
        duplicate_fingerprint,
        base_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    assert duplicate_fingerprint["n_rows"] == base_fingerprint["n_rows"]
    assert duplicate_evidence == duplicate_reverse
    assert duplicate_evidence["ambiguous"]
    assert not duplicate_evidence["column_sketch_alarm"]
    assert not duplicate_evidence["row_sketch_alarm"]
    assert duplicate_evidence["schema_row_comparison_supported"]
    assert not duplicate_evidence["schema_deletion_supported"]
    assert duplicate_evidence["schema_deletion_row_alarm"]
    assert duplicate_evidence["schema_deletion_row_containment"] == 1.0

    duplicate_extra_values = 20_000 + (
        np.arange(len(shared_subset), dtype=np.int64) % 2
    )
    duplicate_with_extra = duplicate_expanded.copy()
    duplicate_with_extra.insert(
        3,
        "duplicate-extra",
        np.concatenate(
            [duplicate_extra_values, duplicate_extra_values[:300]]
        ),
    )
    duplicate_extra_fingerprint = dataset_fingerprint(
        duplicate_with_extra,
        duplicate_target,
    )
    duplicate_extra_evidence = near_match_evidence(
        base_fingerprint,
        duplicate_extra_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    duplicate_extra_reverse = near_match_evidence(
        duplicate_extra_fingerprint,
        base_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    assert duplicate_extra_evidence == duplicate_extra_reverse
    assert duplicate_extra_evidence["ambiguous"]
    assert duplicate_extra_evidence["schema_row_comparison_supported"]
    assert duplicate_extra_evidence["schema_deletion_supported"]
    assert duplicate_extra_evidence["schema_deletion_row_alarm"]
    assert duplicate_extra_evidence["schema_deletion_row_containment"] == 1.0

    candidate = shared_subset.copy()
    candidate.insert(
        4,
        "extra",
        9_999 + rng.integers(0, 2, size=len(candidate), dtype=np.int64),
    )
    candidate_fingerprint = dataset_fingerprint(candidate, y.iloc[selected])
    correct_deletion = next(
        view
        for view in candidate_fingerprint["schema_deletion_row_sketch_deck"]
        if view["view_sha256"] == shared_full["view_sha256"]
    )
    assert correct_deletion["bottom_k_row_hashes"] == shared_full[
        "bottom_k_row_hashes"
    ]
    forward = near_match_evidence(
        base_fingerprint,
        candidate_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    reverse = near_match_evidence(
        candidate_fingerprint,
        base_fingerprint,
        **NEAR_MATCH_THRESHOLDS,
    )
    row_order = rng.permutation(len(candidate))
    permuted = candidate.iloc[row_order, ::-1].copy()
    permuted.columns = [f"renamed-{index}" for index in range(permuted.shape[1])]
    permuted_evidence = near_match_evidence(
        base_fingerprint,
        dataset_fingerprint(permuted, y.iloc[selected].iloc[row_order]),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert forward == reverse == permuted_evidence
    assert forward["ambiguous"]
    assert not forward["column_sketch_alarm"]
    assert not forward["row_sketch_alarm"]
    assert forward["schema_deletion_row_alarm"]
    assert forward["schema_deletion_row_containment"] == 1.0
    assert forward["schema_deletion_tied_max_count"] == 1


def test_schema_deletion_alarm_does_not_claim_two_feature_drift_support():
    n_rows = 300
    X = pd.DataFrame(
        {
            f"bit-{column}": (
                (np.arange(n_rows) * (column + 3) + column) % 97
                < (column + 5)
            ).astype(np.int8)
            for column in range(32)
        }
    )
    y = pd.Series(np.arange(n_rows))
    changed = X.iloc[:210].assign(
        extra_one=((np.arange(210) * 37) % 101 < 43).astype(np.int8),
        extra_two=((np.arange(210) * 41) % 103 < 47).astype(np.int8),
    )
    evidence = near_match_evidence(
        dataset_fingerprint(X, y),
        dataset_fingerprint(changed, y.iloc[:210]),
        **NEAR_MATCH_THRESHOLDS,
    )

    assert not evidence["schema_deletion_supported"]
    assert not evidence["schema_row_comparison_supported"]
    assert not evidence["schema_deletion_row_alarm"]


def _allocation_task(task_id, *, regime="r1f10s1", cluster=None):
    match = re.fullmatch(r"r(\d+)f(\d+)s(\d+)", regime)
    assert match is not None
    repeats, folds, samples = (int(value) for value in match.groups())
    rows = 100 + 17 * task_id
    features = 3 + task_id % 5
    return {
        "openml_task_id": task_id,
        "lineage_cluster": cluster or f"lineage:{task_id}",
        "fingerprint": {
            "n_rows": rows,
            "n_features": features,
            "has_categorical": bool(task_id % 2),
            "has_missing_features": bool(task_id % 3 == 0),
            "opaque_target_value_sha256": f"target-{task_id}",
        },
        "official_splits": {
            "coordinate_count": repeats * folds * samples,
            "dimensions": {
                "repeats": repeats,
                "folds": folds,
                "samples": samples,
            },
        },
    }


def test_partition_is_deterministic_target_blind_exhaustive_and_lineage_atomic():
    tasks = [_allocation_task(task_id) for task_id in range(1, 9)]
    tasks[0]["lineage_cluster"] = "shared"
    tasks[1]["lineage_cluster"] = "shared"

    confirmation, lockbox, diagnostics = balanced_panel_split(tasks, seed=11)
    reversed_result = balanced_panel_split(list(reversed(tasks)), seed=11)
    target_changed = copy.deepcopy(tasks)
    for task in target_changed:
        task["fingerprint"]["opaque_target_value_sha256"] += "-changed"

    assert (confirmation, lockbox) == reversed_result[:2]
    assert (confirmation, lockbox) == balanced_panel_split(
        target_changed, seed=11
    )[:2]
    assert set(confirmation).isdisjoint(lockbox)
    assert set(confirmation) | set(lockbox) == set(range(1, 9))
    assert ({1, 2} <= set(confirmation)) or ({1, 2} <= set(lockbox))
    assert diagnostics["hard_constraints"]["target_information_used"] is False
    assert diagnostics["allocation_sha256"] == (
        "bbb535034d33b71ecdcb5623ee67194d99aadf55f0875837ddfb4b341968bc41"
    )


def test_declaration_covers_all_suite_tasks_and_fails_closed_on_drift():
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    expected = declarations["expected_ctr23_tasks"]
    records = [
        {
            "openml_task_id": entry["openml_task_id"],
            "normalized_name": entry["expected_normalized_name"],
        }
        for entry in expected
    ]
    _validate_expected_suite(records, declarations)

    drifted = copy.deepcopy(records)
    drifted[0]["normalized_name"] = "changed"
    with pytest.raises(ValueError, match="membership/name drifted"):
        _validate_expected_suite(drifted, declarations)

    manual = {
        entry["openml_task_id"]: entry
        for entry in declarations["manual_task_exclusions"]
    }
    assert set(declarations["expected_excluded_ctr23_task_ids"]) == {
        task_id for task_id, entry in manual.items() if entry["status"] == "excluded"
    }
    assert set(declarations["expected_ambiguous_ctr23_task_ids"]) == {
        task_id for task_id, entry in manual.items() if entry["status"] == "ambiguous"
    }
    eligible_lineages = _eligible_lineage_clusters(declarations)
    expected_ids = {entry["openml_task_id"] for entry in expected}
    assert set(manual).isdisjoint(eligible_lineages)
    assert set(manual) | set(eligible_lineages) == expected_ids


def test_build_rejects_runtime_drift_before_validating_task_inputs(monkeypatch):
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    frozen_runtime = dict(declarations["builder_runtime"])
    monkeypatch.setattr(
        "benchmarks.build_ctr23_contamination_registry.metadata.version",
        lambda package: frozen_runtime[package],
    )
    declarations["builder_runtime"] = {
        "python": sys.version.split()[0],
        "openml": frozen_runtime["openml"],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": "0.0.0-adversarial",
        "pyarrow": frozen_runtime["pyarrow"],
        "liac-arff": frozen_runtime["liac-arff"],
    }

    with pytest.raises(ValueError, match="runtime differs"):
        build_artifacts(
            [],
            [],
            declarations,
            builder_source_sha256="not-reached",
        )


def test_cli_rejects_runtime_drift_before_fetching_inputs(
    tmp_path,
    monkeypatch,
):
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    frozen_runtime = dict(declarations["builder_runtime"])
    monkeypatch.setattr(
        "benchmarks.build_ctr23_contamination_registry.metadata.version",
        lambda package: frozen_runtime[package],
    )
    declarations["builder_runtime"] = {
        "python": sys.version.split()[0],
        "openml": frozen_runtime["openml"],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": "0.0.0-adversarial",
        "pyarrow": frozen_runtime["pyarrow"],
        "liac-arff": frozen_runtime["liac-arff"],
    }
    declarations_path = tmp_path / "runtime-drift.json"
    declarations_path.write_text(
        json.dumps(declarations),
        encoding="utf-8",
    )

    def unexpected_fetch(_declarations):
        raise AssertionError("OpenML fetch ran before runtime validation")

    monkeypatch.setattr(
        "benchmarks.build_ctr23_contamination_registry.fetch_registry_inputs",
        unexpected_fetch,
    )
    with pytest.raises(ValueError, match="runtime differs"):
        registry_main(
            [
                "--declarations",
                str(declarations_path),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def _copy_manual_evidence_tree(tmp_path):
    root = tmp_path / "fresh-tree"
    (root / "benchmarks").mkdir(parents=True)
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    catalog_path = BENCHMARKS / "ctr23_manual_evidence_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    paths = {
        "benchmarks/ctr23_contamination_sources.json",
        "benchmarks/ctr23_manual_evidence_catalog.json",
    }
    for entry in catalog["entries"]:
        if entry["kind"] == "repository_file":
            paths.add(entry["path"])
        for source in entry.get("sources", []):
            paths.add(source["artifact_path"])
            if "proof_path" in source:
                paths.add(source["proof_path"])
    for relative in sorted(paths):
        source = Path(__file__).parents[1] / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    suite = json.loads(
        (BENCHMARKS / "ctr23_suite_snapshot.json").read_text(encoding="utf-8")
    )
    return root, declarations, catalog, suite["spent_source_tasks"]


def _rebind_manual_catalog(root, declarations, catalog):
    catalog_path = root / declarations["manual_evidence_catalog"]["path"]
    catalog_path.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    rebound = copy.deepcopy(declarations)
    rebound["manual_evidence_catalog"]["sha256"] = sha256_file(catalog_path)
    return rebound


def _first_historical_source(catalog):
    return next(
        source
        for entry in catalog["entries"]
        for source in entry.get("sources", [])
        if source["artifact_kind"] == "git_blob"
    )


def test_manual_evidence_resolves_from_fresh_tree_without_git(tmp_path):
    root, declarations, _, source_tasks = _copy_manual_evidence_tree(tmp_path)
    resolved, digest = _resolve_manual_evidence(
        declarations, source_tasks, repository_root=root
    )

    assert len(resolved) == 15
    assert digest == sha256_json(resolved)
    assert {record["kind"] for record in resolved} == {
        "historical_git_snapshot",
        "openml_task_snapshot",
        "repository_file",
    }
    source_by_id = {
        int(task["openml_task_id"]): task for task in source_tasks
    }
    for record in resolved:
        if record["kind"] == "openml_task_snapshot":
            assert record["source_task_record_sha256"] == sha256_json(
                source_by_id[record["openml_task_id"]]
            )


@pytest.mark.parametrize("mutation", ["historical_blob", "proof", "repo_file"])
def test_manual_evidence_rejects_mutated_bound_artifacts(tmp_path, mutation):
    root, declarations, catalog, source_tasks = _copy_manual_evidence_tree(
        tmp_path
    )
    if mutation == "historical_blob":
        relative = next(
            source["artifact_path"]
            for entry in catalog["entries"]
            for source in entry.get("sources", [])
            if source["artifact_kind"] == "git_blob"
        )
    elif mutation == "proof":
        relative = next(
            source["proof_path"]
            for entry in catalog["entries"]
            for source in entry.get("sources", [])
            if "proof_path" in source
        )
    else:
        relative = next(
            entry["path"]
            for entry in catalog["entries"]
            if entry["kind"] == "repository_file"
        )
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\nmutation\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        _resolve_manual_evidence(
            declarations, source_tasks, repository_root=root
        )


def test_manual_evidence_rejects_unknown_ids_and_unsafe_paths(tmp_path):
    root, declarations, catalog, source_tasks = _copy_manual_evidence_tree(
        tmp_path
    )
    unknown = copy.deepcopy(declarations)
    unknown["manual_task_exclusions"][0]["evidence_id"] = "not-in-catalog"
    with pytest.raises(ValueError, match="defined and used"):
        _resolve_manual_evidence(unknown, source_tasks, repository_root=root)

    unsafe_catalog = copy.deepcopy(catalog)
    repo_entry = next(
        entry
        for entry in unsafe_catalog["entries"]
        if entry["kind"] == "repository_file"
    )
    repo_entry["path"] = "../outside.txt"
    catalog_path = root / declarations["manual_evidence_catalog"]["path"]
    catalog_path.write_text(json.dumps(unsafe_catalog), encoding="utf-8")
    unsafe_declarations = copy.deepcopy(declarations)
    unsafe_declarations["manual_evidence_catalog"]["sha256"] = sha256_file(
        catalog_path
    )
    with pytest.raises(ValueError, match="escapes repository"):
        _resolve_manual_evidence(
            unsafe_declarations, source_tasks, repository_root=root
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("invalid_object", "invalid commit object"),
        ("duplicate_object", "duplicate Git proof object"),
        ("unused_object", "unused objects"),
    ],
)
def test_manual_evidence_reaches_inner_git_proof_validation(
    tmp_path, mutation, message
):
    root, declarations, catalog, source_tasks = _copy_manual_evidence_tree(
        tmp_path
    )
    source = _first_historical_source(catalog)
    proof_path = root / source["proof_path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if mutation == "invalid_object":
        proof["objects"][0]["payload_base64"] = base64.b64encode(
            b"corrupted commit payload"
        ).decode("ascii")
    elif mutation == "duplicate_object":
        proof["objects"].append(copy.deepcopy(proof["objects"][0]))
    else:
        payload = b"valid but unreachable proof object"
        proof["objects"].append(
            {
                "type": "blob",
                "sha1": _git_object_sha1("blob", payload),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            }
        )
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    proof_sha256 = sha256_file(proof_path)
    for entry in catalog["entries"]:
        for candidate in entry.get("sources", []):
            if candidate.get("proof_path") == source["proof_path"]:
                candidate["proof_sha256"] = proof_sha256
    rebound = _rebind_manual_catalog(root, declarations, catalog)

    with pytest.raises(ValueError, match=message):
        _resolve_manual_evidence(rebound, source_tasks, repository_root=root)


@pytest.mark.parametrize(
    ("source_path", "message"),
    [
        ("benchmarks/not-present.py", "proof path is absent"),
        ("../outside.py", "unsafe historical source path"),
        ("/absolute/outside.py", "unsafe historical source path"),
    ],
)
def test_manual_evidence_rejects_rebound_historical_paths(
    tmp_path, source_path, message
):
    root, declarations, catalog, source_tasks = _copy_manual_evidence_tree(
        tmp_path
    )
    _first_historical_source(catalog)["source_path"] = source_path
    rebound = _rebind_manual_catalog(root, declarations, catalog)

    with pytest.raises(ValueError, match=message):
        _resolve_manual_evidence(rebound, source_tasks, repository_root=root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("commit", "proof commit mismatch"),
        ("blob", "Git blob identity mismatch"),
        ("kind", "unsupported manual evidence kind"),
        ("unused_id", "defined and used"),
    ],
)
def test_manual_evidence_rejects_rebound_catalog_identity_drift(
    tmp_path, mutation, message
):
    root, declarations, catalog, source_tasks = _copy_manual_evidence_tree(
        tmp_path
    )
    source = _first_historical_source(catalog)
    if mutation == "commit":
        source["source_commit_sha1"] = "0" * 40
    elif mutation == "blob":
        source["source_git_blob_sha1"] = "0" * 40
    elif mutation == "kind":
        catalog["entries"][0]["kind"] = "unknown_kind"
    else:
        catalog["entries"].append(
            {
                "evidence_id": "unused_evidence_id",
                "kind": "openml_task_snapshot",
                "openml_task_id": 363612,
            }
        )
    rebound = _rebind_manual_catalog(root, declarations, catalog)

    with pytest.raises(ValueError, match=message):
        _resolve_manual_evidence(rebound, source_tasks, repository_root=root)


def test_fingerprint_exposes_no_target_marginals():
    fingerprint = dataset_fingerprint(
        pd.DataFrame({"a": [1, 2, 3]}), pd.Series([10.0, 20.0, 30.0])
    )
    rendered = json.dumps(fingerprint, sort_keys=True)

    for forbidden in ("target_mean", "target_std", "target_skew", "target_summary"):
        assert forbidden not in rendered


def test_committed_artifacts_bind_sources_statuses_and_allocation():
    suite = json.loads(
        (BENCHMARKS / "ctr23_suite_snapshot.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (BENCHMARKS / "ctr23_contamination_registry.json").read_text(
            encoding="utf-8"
        )
    )
    partition = json.loads(
        (BENCHMARKS / "ctr23_partition.json").read_text(encoding="utf-8")
    )
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))

    suite_unsigned = dict(suite)
    suite_digest = suite_unsigned.pop("suite_snapshot_sha256")
    assert sha256_json(suite_unsigned) == suite_digest

    registry_unsigned = dict(registry)
    registry_digest = registry_unsigned.pop("contamination_registry_sha256")
    assert sha256_json(registry_unsigned) == registry_digest

    partition_unsigned = dict(partition)
    bundle_digest = partition_unsigned.pop("registry_bundle_sha256")
    partition_digest = partition_unsigned.pop("partition_sha256")
    assert sha256_json(partition_unsigned) == partition_digest
    assert bundle_digest == sha256_json(
        {
            "suite_snapshot_sha256": suite_digest,
            "contamination_registry_sha256": registry_digest,
            "partition_sha256": partition_digest,
            "builder_source_sha256": suite["builder_source_sha256"],
            "manual_evidence_sha256": suite["manual_evidence_sha256"],
        }
    )

    assert suite["builder_source_sha256"] == sha256_file(
        BENCHMARKS / "build_ctr23_contamination_registry.py"
    )
    assert suite["declarations_sha256"] == sha256_json(declarations)
    assert suite["runtime"] == declarations["builder_runtime"]
    catalog_binding = declarations["manual_evidence_catalog"]
    assert sha256_file(Path(__file__).parents[1] / catalog_binding["path"]) == (
        catalog_binding["sha256"]
    )
    assert suite["manual_evidence_sha256"] == registry[
        "manual_evidence_sha256"
    ] == partition["manual_evidence_sha256"]
    assert registry["manual_evidence_sha256"] == sha256_json(
        registry["manual_evidence"]
    )
    resolved_evidence_ids = {
        entry["evidence_id"] for entry in registry["manual_evidence"]
    }
    for row in registry["tasks"]:
        for reason in row["exclusion_reasons"]:
            source = reason["source"]
            if source.startswith("manual_evidence:"):
                assert source.removeprefix("manual_evidence:") in (
                    resolved_evidence_ids
                )
    assert registry["counts"] == {
        "excluded": 16,
        "ambiguous": 1,
        "eligible": 18,
    }
    confirmation = set(partition["confirmation_task_ids"])
    lockbox = set(partition["lockbox_task_ids"])
    assert len(confirmation) == len(lockbox) == 9
    assert confirmation.isdisjoint(lockbox)
    assert confirmation | lockbox == set(registry["eligible_task_ids"])
    assert partition["confirmation_coordinate_count"] == 270
    assert partition["lockbox_coordinate_count"] == 270
