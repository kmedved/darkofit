from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_panel3_confirmation as analyzer
from benchmarks import build_panel3_registry as registry
from benchmarks import confirmation_target_preflight as target_check
from benchmarks import panel3_registry_common as common
from benchmarks import preflight_panel3_registry as preflight
from benchmarks import run_panel3_confirmation as runner


ROOT = Path(__file__).resolve().parents[1]


def test_atomic_create_rejects_symlink_ancestor(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink ancestor"):
        common.atomic_create(
            allowed / "redirect" / "artifact.json",
            b"{}\n",
            allowed_root=allowed,
        )

    assert not (outside / "artifact.json").exists()


def test_atomic_create_rejects_ancestor_swapped_during_traversal(
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    nested = allowed / "nested"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    original_open = common.os.open
    swapped = False

    def racing_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            nested.rename(allowed / "moved")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(
            path,
            flags,
            *args,
            dir_fd=dir_fd,
            **kwargs,
        )

    monkeypatch.setattr(common.os, "open", racing_open)

    with pytest.raises(RuntimeError, match="symlink ancestor"):
        common.atomic_create(
            nested / "artifact.json",
            b"{}\n",
            allowed_root=allowed,
        )

    assert swapped is True
    assert not (outside / "artifact.json").exists()


def test_atomic_create_uses_create_only_directory_handle(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    output = allowed / "nested" / "artifact.json"

    common.atomic_create(output, b"first\n", allowed_root=allowed)

    assert output.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        common.atomic_create(output, b"second\n", allowed_root=allowed)
    assert output.read_bytes() == b"first\n"


def test_secure_read_rejects_ancestor_swapped_during_traversal(
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    nested = allowed / "nested"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    (nested / "artifact.json").write_bytes(b"trusted\n")
    (outside / "artifact.json").write_bytes(b"redirected\n")
    original_open = common.os.open
    swapped = False

    def racing_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            nested.rename(allowed / "moved")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(
            path,
            flags,
            *args,
            dir_fd=dir_fd,
            **kwargs,
        )

    monkeypatch.setattr(common.os, "open", racing_open)

    with pytest.raises(RuntimeError, match="symlink ancestor"):
        common.secure_read_bytes(
            nested / "artifact.json",
            allowed_root=allowed,
        )

    assert swapped is True


def test_deterministic_registry_validation_covers_task_rows(monkeypatch):
    expected = {
        "created_from_clean_sources": False,
        "sources": {
            "darkofit_registry_head": "a" * 40,
            "darkofit_model_head": "b" * 40,
        },
        "tasks": [
            {
                "task_id": 1,
                "status": "selected",
                "split_policy": {"kind": "openml_official"},
            }
        ],
        "coordinates": [
            {"task_id": 1, "repeat": 0, "fold": 0, "sample": 0}
        ],
        "registry_sha256": "1" * 64,
    }
    observed = copy.deepcopy(expected)
    observed["created_from_clean_sources"] = True
    observed["sources"]["darkofit_registry_head"] = "c" * 40
    observed["registry_sha256"] = "2" * 64
    monkeypatch.setattr(
        registry,
        "build",
        lambda **_kwargs: copy.deepcopy(expected),
    )

    registry.validate_deterministic_registry_output(observed)

    observed["tasks"][0]["task_id"] = 2
    with pytest.raises(
        RuntimeError,
        match="differs from deterministic builder output",
    ):
        registry.validate_deterministic_registry_output(observed)


def test_declarations_freeze_three_ordered_strata_and_reserves():
    declarations = common.validate_declarations()
    rows = declarations["candidates"]

    assert declarations["required_per_stratum"] == 4
    assert declarations["coordinate_folds"] == [0, 1, 2]
    assert len({row["task_id"] for row in rows}) == len(rows)
    assert len({row["dataset_id"] for row in rows}) == len(rows)
    assert len({row["lineage_cluster"] for row in rows}) == len(rows)
    counts = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in common.STRATA
    }
    assert counts == {
        "smooth_numeric": 6,
        "mixed_categorical": 4,
        "applied_noisy": 7,
    }
    assert {row["origin"] for row in rows} <= {
        "ctr23_sealed_lockbox",
        "new_source_reviewed",
    }


def test_pre_h1_target_statistic_exclusions_are_permanent_and_replaced():
    declarations = common.validate_declarations()
    exclusions = declarations["pre_h1_target_statistic_exclusions"]
    rows = declarations["candidates"]

    assert exclusions == common.PRE_H1_TARGET_STATISTIC_EXCLUSIONS
    assert {
        row["task_id"]: row["replacement_task_id"] for row in exclusions
    } == {
        363370: 359931,
        363377: 360993,
        363495: 4851,
    }
    excluded = {row["task_id"] for row in exclusions}
    eligible = {row["task_id"] for row in rows}
    related = {
        task_id for row in rows for task_id in row["related_task_ids"]
    }
    assert excluded.isdisjoint(eligible)
    assert excluded.isdisjoint(related)
    assert excluded.isdisjoint(
        map(int, declarations["ordinal_features_by_task"])
    )
    selected = {
        row["task_id"]: row
        for row in rows
        if row["selection_role"] == "selected"
    }
    replacements = {row["replacement_task_id"] for row in exclusions}
    assert replacements <= selected.keys()
    assert all(
        selected[row["replacement_task_id"]]["stratum"] == row["stratum"]
        for row in exclusions
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__("unexpected", True),
            "top-level schema changed",
        ),
        (
            lambda payload: payload["candidates"][0].__setitem__(
                "unexpected", True
            ),
            "candidate declaration schema changed",
        ),
        (
            lambda payload: payload["ctr23_lockbox_policy"][
                "selected_task_ids"
            ].reverse(),
            "lockbox policy changed",
        ),
        (
            lambda payload: payload[
                "pre_h1_target_statistic_exclusions"
            ].reverse(),
            "target-statistic exclusion ledger changed",
        ),
        (
            lambda payload: payload.__setitem__(
                "panel_split_dimensions",
                {"repeats": 1, "folds": 4, "samples": 1},
            ),
            "split dimensions changed",
        ),
    ],
)
def test_declaration_schema_fails_closed(mutation, message):
    declarations = copy.deepcopy(common.validate_declarations())
    mutation(declarations)

    with pytest.raises(RuntimeError, match=message):
        common.validate_declarations(declarations)


@pytest.mark.parametrize(
    "encoded",
    [
        '{"key": 1, "key": 2}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": 9223372036854775808}',
    ],
)
def test_campaign_json_loader_rejects_ambiguous_numbers_and_keys(
    tmp_path,
    encoded,
):
    path = tmp_path / "invalid.json"
    path.write_text(encoded)

    with pytest.raises(RuntimeError, match="invalid JSON artifact"):
        common.load_json(path)


def test_provisional_power_is_exact_and_blocks_optimistic_authorization():
    result = common.power_analysis()

    assert result == common.power_analysis()
    composite = result["candidates"]["t5_composite_policy"]
    cross = result["candidates"]["guarded_cross_features_policy"]
    assert composite["effect_pool_sha256"] == (
        "76737ab531b139e1ab2db7ae717bbd65409251e0a8773db3e2994951e52e4e0a"
    )
    assert cross["effect_pool_sha256"] == (
        "06247628ee290d61cd1b9c1efbca81e49ddc76a12177f51e8f9f3af27edc761e"
    )
    assert result["simulations_per_candidate"] == 200_000
    assert result["simulated_lineages"] == 12
    assert result["bonferroni_one_sided_percentile"] == 97.5
    assert composite["component_passing_simulations"] == {
        "point": 95955,
        "bonferroni_upper": 158276,
        "leave_one_favorable_out": 113424,
        "worst_dataset": 200000,
    }
    assert composite["sampled_effect_lineages"] == 5
    assert composite["fixed_tie_lineages"] == 7
    assert composite["marginal_passing_simulations"] == 58893
    assert composite["marginal_pass_probability"] == 0.294465
    assert cross["component_passing_simulations"] == {
        "point": 139729,
        "bonferroni_upper": 127171,
        "leave_one_favorable_out": 144772,
        "worst_dataset": 200000,
    }
    assert cross["sampled_effect_lineages"] == 12
    assert cross["fixed_tie_lineages"] == 0
    assert cross["marginal_passing_simulations"] == 94588
    assert cross["marginal_pass_probability"] == 0.47294
    assert (
        result["dependence_agnostic_joint_probability_lower_bound"]
        == 0.0
    )
    assert result["authorization_blocked"] is True
    assert result["passes"] is False


def test_power_fails_closed_for_a_null_effect_pool(monkeypatch):
    profiles = [
        {"source": "null", "lineage": str(index), "ratio": 1.0}
        for index in range(15)
    ]
    monkeypatch.setattr(
        common,
        "_composite_effect_profiles",
        lambda: profiles,
    )
    monkeypatch.setattr(
        common,
        "_provisional_cross_sensitivity_profiles",
        lambda: [
            {
                "source": "optimistic_safe_non_nominee_decline_sensitivity",
                "lineage": str(index),
                "ratio": 1.0,
            }
            for index in range(15)
        ],
    )

    result = common.power_analysis()

    assert (
        result["candidates"]["t5_composite_policy"][
            "marginal_pass_probability"
        ]
        == 0.0
    )
    assert result["dependence_agnostic_joint_probability_lower_bound"] == 0.0
    assert result["passes"] is False


def _fake_record(declaration):
    return {
        "openml_task_id": declaration["task_id"],
        "openml_dataset_id": declaration["dataset_id"],
        "openml_task_type_id": 2,
        "normalized_name": declaration["expected_normalized_name"],
        "target_name": declaration["expected_target_name"],
        "fingerprint": {
            "n_rows": 3,
            "canonicalization_ambiguous": False,
        },
    }


def _fake_attestation(record):
    return {
        "policy": target_check.TARGET_POLICY,
        "checked": True,
        "passed": True,
        "target_outcome_statistics_computed": False,
        "target_values_persisted": False,
        "binding": {
            "openml_task_id": record["openml_task_id"],
            "openml_dataset_id": record["openml_dataset_id"],
            "target_name": record["target_name"],
            "dataset_fingerprint_sha256": registry.ctr.sha256_json(
                record["fingerprint"]
            ),
        },
    }


def _fake_power_decision():
    return {
        "decision_sha256": "d" * 64,
        "source_sha256": {
            str(path.relative_to(ROOT)): common.sha256_file(path)
            for path in common.PANEL3_SOURCE_PATHS
        },
        "target_preflight_authorized": True,
        "retained_candidates": [
            "t5_composite_policy",
            "guarded_cross_features_policy",
        ],
        "prospective_panel": {
            "slots": copy.deepcopy(
                common.load_json(common.POWER_DESIGN_CONTRACT)[
                    "prospective_panel"
                ]["slots"]
            )
        },
    }


def _fake_split_applicability_binding(decision):
    declarations = common.validate_declarations()
    by_task = {
        int(row["task_id"]): row
        for row in declarations["candidates"]
        if row["selection_role"] == "selected"
    }
    minimum = common.t5_minimum_outer_training_rows()
    rows = []
    for slot in decision["prospective_panel"]["slots"]:
        declaration = by_task[int(slot["task_id"])]
        applicability = slot["t5_size_gate_applicability"]
        if declaration["split_policy"] == {"kind": "openml_official"}:
            evidence_kind = "exact_openml_official_training_rows"
        elif any(applicability):
            evidence_kind = "exact_target_free_constructed_training_rows"
        else:
            evidence_kind = "dataset_row_upper_bound_below_gate"
        rows.append(
            {
                "task_id": int(slot["task_id"]),
                "dataset_id": int(declaration["dataset_id"]),
                "lineage_cluster": slot["lineage_cluster"],
                "stratum": slot["stratum"],
                "coordinate_folds": list(preflight.COORDINATE_FOLDS),
                "minimum_outer_training_rows": minimum,
                "evidence_kind": evidence_kind,
                "outer_training_rows_or_upper_bound": [
                    minimum + 100 if value else minimum - 100
                    for value in applicability
                ],
                "t5_size_gate_applicability": list(applicability),
            }
        )
    return {
        "kind": "panel3_pre_target_split_applicability_v1",
        "verified_before_target_materialization_or_inspection": True,
        "target_values_materialized_or_inspected": False,
        "target_bearing_openml_container_may_be_cached": True,
        "target_column_excluded_from_projection": True,
        "minimum_outer_training_rows": minimum,
        "attestations": rows,
    }


def _mock_power_decision(monkeypatch):
    decision = _fake_power_decision()
    monkeypatch.setattr(
        preflight.power_design,
        "load_decision_snapshot",
        lambda **_kwargs: (decision, "e" * 64),
    )
    monkeypatch.setattr(
        preflight,
        "_build_split_applicability_binding",
        lambda observed, _declarations, **_kwargs: (
            _fake_split_applicability_binding(observed)
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_recheck_authorization_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        registry.power_design,
        "validate_decision",
        lambda artifact, **_kwargs: artifact,
    )
    original_sha256 = common.sha256_file
    monkeypatch.setattr(
        common,
        "sha256_file",
        lambda path: (
            "e" * 64
            if Path(path) == common.POWER_DESIGN_DECISION
            else original_sha256(path)
        ),
    )
    return decision


def test_preflight_failure_leaves_no_output(tmp_path, monkeypatch):
    _mock_power_decision(monkeypatch)
    declarations = common.validate_declarations()
    smooth_ids = {
        row["task_id"]
        for row in declarations["candidates"]
        if row["stratum"] == "smooth_numeric"
    }
    by_id = {
        row["task_id"]: row for row in declarations["candidates"]
    }
    monkeypatch.setattr(
        preflight,
        "_require_frozen_clean_source",
        lambda _declarations, _sources: "a" * 40,
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: _fake_record(by_id[task_id]),
    )

    def reject_smooth(record):
        if record["openml_task_id"] in smooth_ids:
            raise target_check.TargetPreflightError(
                "target must contain only finite values"
            )
        return _fake_attestation(record)

    monkeypatch.setattr(
        preflight.target_check,
        "attest_openml_target",
        reject_smooth,
    )
    monkeypatch.setattr(
        common,
        "validate_create_path",
        lambda path: Path(path),
    )
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(preflight, "DEFAULT_OUTPUT", output)

    with pytest.raises(RuntimeError, match="lacks four eligible"):
        preflight.main(["--output", str(output)])

    assert not output.exists()
    assert not output.is_symlink()


def test_preflight_cli_rejects_noncanonical_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "build",
        lambda: pytest.fail("preflight must not run"),
    )

    with pytest.raises(RuntimeError, match="output path changed"):
        preflight.main(["--output", str(tmp_path / "preflight.json")])


def test_registry_cli_rejects_noncanonical_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        registry,
        "build",
        lambda **_kwargs: pytest.fail("registry build must not run"),
    )

    with pytest.raises(RuntimeError, match="registry path changed"):
        registry.main(["--preflight", str(tmp_path / "preflight.json")])
    with pytest.raises(RuntimeError, match="registry path changed"):
        registry.main(["--output", str(tmp_path / "registry.json")])


def test_preflight_source_map_is_exact_and_revalidated(monkeypatch):
    _mock_power_decision(monkeypatch)
    declarations = common.validate_declarations()
    by_id = {
        row["task_id"]: row for row in declarations["candidates"]
    }
    monkeypatch.setattr(
        preflight,
        "_require_frozen_clean_source",
        lambda _declarations, _sources: "a" * 40,
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: _fake_record(by_id[task_id]),
    )
    monkeypatch.setattr(
        preflight.target_check,
        "attest_openml_target",
        _fake_attestation,
    )

    artifact = preflight.build()
    registry._validate_preflight(artifact, declarations)
    assert set(artifact["source_sha256"]) == {
        str(path.relative_to(ROOT)) for path in common.PANEL3_SOURCE_PATHS
    }

    changed = copy.deepcopy(artifact)
    changed["source_sha256"].pop(next(iter(changed["source_sha256"])))
    changed = common.bind_artifact_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "target_preflight_sha256"
        },
        "target_preflight_sha256",
    )
    with pytest.raises(RuntimeError, match="source map changed"):
        registry._validate_preflight(changed, declarations)


def test_preflight_recomputes_power_before_any_target_access(monkeypatch):
    accessed = []
    monkeypatch.setattr(
        preflight.power_design,
        "load_decision_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError(f"blocked:{kwargs['recompute']}")
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: accessed.append(task_id),
    )

    with pytest.raises(RuntimeError, match="blocked:True"):
        preflight.build(require_clean_source=False)

    assert accessed == []


def test_preflight_rejects_candidate_contract_snapshot_mismatch_before_target(
    monkeypatch,
):
    decision = _fake_power_decision()
    candidate_relative = str(
        common.CANDIDATE_CONTRACT.relative_to(ROOT)
    )
    decision["source_sha256"][candidate_relative] = "0" * 64
    accessed = []
    monkeypatch.setattr(
        preflight.power_design,
        "load_decision_snapshot",
        lambda **_kwargs: (decision, "e" * 64),
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: accessed.append(task_id),
    )

    with pytest.raises(
        RuntimeError,
        match="source snapshot differs from the power decision",
    ):
        preflight.build(require_clean_source=False)

    assert accessed == []


def test_preflight_publication_recheck_rejects_power_decision_swap(
    monkeypatch,
):
    decision = _fake_power_decision()
    monkeypatch.setattr(
        common,
        "secure_load_json",
        lambda _path: ({"changed": True}, "f" * 64),
    )

    with pytest.raises(RuntimeError, match="power decision changed"):
        preflight._recheck_authorization_inputs(
            decision=decision,
            decision_file_sha256="e" * 64,
            source_sha256={},
        )


def test_split_applicability_drift_blocks_before_any_target_access(
    monkeypatch,
):
    decision = _fake_power_decision()
    accessed = []
    monkeypatch.setattr(
        preflight.power_design,
        "load_decision_snapshot",
        lambda **_kwargs: (decision, "e" * 64),
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: accessed.append(("record", task_id)),
    )
    monkeypatch.setattr(
        preflight.target_check,
        "attest_openml_target",
        lambda record: accessed.append(("target", record)),
    )
    fake_binding = _fake_split_applicability_binding(decision)
    first = fake_binding["attestations"][0]
    first["outer_training_rows_or_upper_bound"] = [2_100, 2_100, 2_100]
    monkeypatch.setattr(
        preflight,
        "_split_applicability_attestation",
        lambda declaration, slot, **_kwargs: copy.deepcopy(
            next(
                row
                for row in fake_binding["attestations"]
                if row["task_id"] == int(slot["task_id"])
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="pre-target split applicability binding changed",
    ):
        preflight.build(require_clean_source=False)

    assert accessed == []


def test_split_applicability_binding_rejects_wrong_evidence_provenance():
    decision = _fake_power_decision()
    declarations = common.validate_declarations()
    binding = _fake_split_applicability_binding(decision)
    custom_task_ids = {
        int(row["task_id"])
        for row in declarations["candidates"]
        if row["selection_role"] == "selected"
        and row["split_policy"]["kind"]
        == "target_free_split_construction_v1"
    }
    changed = copy.deepcopy(binding)
    row = next(
        item
        for item in changed["attestations"]
        if item["task_id"] in custom_task_ids
    )
    row["evidence_kind"] = "exact_openml_official_training_rows"

    with pytest.raises(
        RuntimeError,
        match="pre-target split applicability binding changed",
    ):
        preflight._validate_split_applicability_binding(
            changed,
            decision,
            declarations,
        )


def test_primary_target_ineligibility_blocks_registry_but_not_attestation(
    monkeypatch,
):
    _mock_power_decision(monkeypatch)
    declarations = common.validate_declarations()
    by_id = {
        row["task_id"]: row for row in declarations["candidates"]
    }
    rejected = next(
        row["task_id"]
        for row in declarations["candidates"]
        if row["stratum"] == "smooth_numeric"
        and row["selection_role"] == "selected"
    )
    monkeypatch.setattr(
        preflight,
        "_require_frozen_clean_source",
        lambda _declarations, _sources: "a" * 40,
    )
    monkeypatch.setattr(
        preflight,
        "_load_task_record",
        lambda task_id: _fake_record(by_id[task_id]),
    )

    def attest(record):
        if record["openml_task_id"] == rejected:
            raise target_check.TargetPreflightError(
                "target must contain only finite values"
            )
        return _fake_attestation(record)

    monkeypatch.setattr(
        preflight.target_check,
        "attest_openml_target",
        attest,
    )

    artifact = preflight.build()

    assert artifact[
        "exact_power_authorized_primary_tasks_target_eligible"
    ] is False
    assert artifact["registry_build_authorized"] is False
    with pytest.raises(RuntimeError, match="preflight boundary is invalid"):
        registry._validate_preflight(artifact, declarations)


def test_first_four_eligible_selection_uses_frozen_reserve_order():
    records = []
    for stratum in common.STRATA:
        for priority in range(1, 7):
            records.append(
                {
                    "task_id": len(records) + 1,
                    "stratum": stratum,
                    "priority": priority,
                    "status": (
                        "excluded" if priority in {2, 4} else "eligible"
                    ),
                }
            )
    original_order = [row["task_id"] for row in records]

    selected = registry.select_first_four(records)

    assert [row["task_id"] for row in records] == original_order
    for stratum in common.STRATA:
        assert [
            row["priority"]
            for row in selected
            if row["stratum"] == stratum
        ] == [1, 3, 5, 6]
        assert [
            row["priority"]
            for row in records
            if row["stratum"] == stratum
            and row["status"] == "selected"
        ] == [1, 3, 5, 6]


def test_custom_shuffled_split_is_deterministic_and_partitions_rows():
    constructor = {
        "kind": "shuffled_kfold_v1",
        "n_splits": 3,
        "shuffle": True,
        "random_state": 20260717,
        "allow_unused_rows": False,
    }

    first = registry._shuffled_kfold_indices(17, constructor)
    second = registry._shuffled_kfold_indices(17, constructor)

    assert [
        (train.tolist(), test.tolist()) for train, test in first
    ] == [
        (train.tolist(), test.tolist()) for train, test in second
    ]
    assert np.array_equal(
        np.sort(np.concatenate([test for _train, test in first])),
        np.arange(17),
    )
    assert all(
        not np.intersect1d(train, test).size for train, test in first
    )


def test_custom_group_split_never_crosses_a_group_and_balances_ties():
    X = pd.DataFrame(
        {
            "entity": ["A", "a", "B", "B", "C", "D", "E", "F"],
            "value": np.arange(8),
        }
    )
    constructor = {
        "kind": "size_balanced_group_kfold_v1",
        "n_splits": 3,
        "group_key": {
            "kind": "length_prefixed_nfkc_casefold_sha256_v1",
            "source_columns": ["entity"],
            "missing": "reject",
            "whitespace": "collapse",
        },
        "group_order": "descending_row_count_then_group_sha256",
        "fold_assignment": "minimum_row_count_then_lowest_fold",
        "allow_unused_rows": False,
    }

    pairs = registry._group_kfold_indices(X, constructor)

    assert np.array_equal(
        np.sort(np.concatenate([test for _train, test in pairs])),
        np.arange(len(X)),
    )
    for train, test in pairs:
        assert set(X.iloc[train]["entity"].str.casefold()).isdisjoint(
            set(X.iloc[test]["entity"].str.casefold())
        )
    assert max(len(test) for _train, test in pairs) - min(
        len(test) for _train, test in pairs
    ) <= 1


def test_custom_chronological_split_is_expanding_and_date_disjoint():
    X = pd.DataFrame(
        {
            "date": [
                "01/01/2020",
                "01/01/2020",
                "01/02/2020",
                "01/03/2020",
                "01/04/2020",
                "01/05/2020",
                "01/06/2020",
                "01/07/2020",
            ]
        }
    )
    constructor = {
        "kind": "expanding_unique_datetime_blocks_v1",
        "source_column": "date",
        "format": "%m/%d/%Y",
        "utc": False,
        "block_count": 4,
        "never_split_equal_values": True,
        "folds": [
            {"fold": 0, "train_blocks": [0], "test_blocks": [1]},
            {
                "fold": 1,
                "train_blocks": [0, 1],
                "test_blocks": [2],
            },
            {
                "fold": 2,
                "train_blocks": [0, 1, 2],
                "test_blocks": [3],
            },
        ],
        "allow_unused_rows": True,
    }

    pairs = registry._chronological_indices(X, constructor)

    assert [len(train) for train, _test in pairs] == sorted(
        len(train) for train, _test in pairs
    )
    parsed = pd.to_datetime(X["date"], format="%m/%d/%Y")
    for train, test in pairs:
        assert parsed.iloc[train].max() < parsed.iloc[test].min()
        assert set(parsed.iloc[train]).isdisjoint(parsed.iloc[test])


def test_manual_colleges_exception_suppresses_only_exact_name_alarms():
    declaration = next(
        row
        for row in common.validate_declarations()["candidates"]
        if row["task_id"] == 5166
    )
    record = {
        "openml_dataset_id": 538,
        "normalized_name": "colleges_usnews",
    }
    reasons = [
        {"kind": "known_name", "match": "colleges"},
        {"kind": "chimeraboost_known_name", "match": "colleges"},
    ]
    exposure = {"openml_dataset_ids": [42727]}

    remaining, applied = registry._manual_contamination_adjudication(
        declaration,
        record,
        reasons,
        exposure=exposure,
    )

    assert remaining == []
    assert applied["suppressed_exact_alarm_kinds"] == [
        "chimeraboost_known_name",
        "known_name",
    ]
    assert applied[
        "declared_colliding_dataset_ids_not_independently_observed"
    ] == [42159]

    changed = copy.deepcopy(reasons)
    changed.append({"kind": "spent_near_lineage_alarm", "matches": []})
    remaining, applied = registry._manual_contamination_adjudication(
        declaration,
        record,
        changed,
        exposure=exposure,
    )
    assert applied is None
    assert any(
        row["kind"] == "manual_contamination_adjudication_failed"
        for row in remaining
    )


def test_target_attestation_schema_and_bindings_fail_closed():
    declaration = common.validate_declarations()["candidates"][0]
    record = _fake_record(declaration)
    attestation = _fake_attestation(record)

    preflight._validate_target_attestation(attestation, record)

    changed = copy.deepcopy(attestation)
    changed["unexpected"] = True
    with pytest.raises(RuntimeError, match="malformed"):
        preflight._validate_target_attestation(changed, record)
    changed = copy.deepcopy(attestation)
    changed["binding"]["openml_dataset_id"] += 1
    with pytest.raises(RuntimeError, match="malformed"):
        preflight._validate_target_attestation(changed, record)
    changed = copy.deepcopy(attestation)
    changed["target_outcome_statistics_computed"] = True
    with pytest.raises(RuntimeError, match="malformed"):
        preflight._validate_target_attestation(changed, record)


def test_related_task_id_walker_handles_list_values_everywhere():
    payload = {
        "nested": [
            {"related_task_ids": [11, 12]},
            {"task_id": 13},
            {"related_task_ids": [14, "not-an-id", -1]},
        ]
    }

    assert registry._integer_values(
        payload, {"task_id", "related_task_ids"}
    ) == {11, 12, 13, 14}


def test_lockbox_repository_exception_is_exact_path_only(monkeypatch):
    declaration = next(
        row
        for row in common.validate_declarations()["candidates"]
        if row["task_id"] == 361247
    )
    records = registry._task_records(
        common.load_json(registry.CTR_SNAPSHOT)
    )
    record = next(
        row for row in records if row["openml_task_id"] == 361247
    )
    thresholds = common.load_json(registry.CTR_DECLARATIONS)[
        "near_match_thresholds"
    ]
    preflight_row = {
        "status": "target_eligible",
        "task_record": record,
    }
    spent = {
        "openml_task_ids": [],
        "openml_dataset_ids": [],
        "task_records": [record],
    }
    exposure = {"openml_dataset_ids": [], "normalized_names": []}

    def allowed_only(repository, revision, _literal):
        if repository == registry.CHIMERA_ROOT:
            return []
        return [
            f"{revision}:{path}"
            for path in sorted(
                registry.LOCKBOX_DARKOFIT_REFERENCE_ALLOWLIST
            )
        ]

    monkeypatch.setattr(registry.fresh, "_git_grep", allowed_only)
    reasons = registry._base_exclusion_reasons(
        declaration,
        preflight_row,
        spent=spent,
        exposure=exposure,
        known_names=[],
        thresholds=thresholds,
        prefreeze=common.DARKOFIT_PREFREEZE_HEAD,
        chimera_head=common.CHIMERABOOST_HEAD,
    )
    assert not any(
        reason["kind"] == "repository_reference"
        for reason in reasons
    )

    def with_unexpected(repository, revision, literal):
        values = allowed_only(repository, revision, literal)
        if repository == registry.ROOT:
            values.append(f"{revision}:benchmarks/unexpected.py")
        return values

    monkeypatch.setattr(registry.fresh, "_git_grep", with_unexpected)
    reasons = registry._base_exclusion_reasons(
        declaration,
        preflight_row,
        spent=spent,
        exposure=exposure,
        known_names=[],
        thresholds=thresholds,
        prefreeze=common.DARKOFIT_PREFREEZE_HEAD,
        chimera_head=common.CHIMERABOOST_HEAD,
    )
    repository_alarm = next(
        reason
        for reason in reasons
        if reason["kind"] == "repository_reference"
    )
    assert repository_alarm["paths"] == [
        f"{common.DARKOFIT_PREFREEZE_HEAD}:benchmarks/unexpected.py"
    ]


def test_lockbox_own_record_exception_does_not_hide_other_lineages(
    monkeypatch,
):
    declaration = next(
        row
        for row in common.validate_declarations()["candidates"]
        if row["task_id"] == 361247
    )
    record = next(
        row
        for row in registry._task_records(
            common.load_json(registry.CTR_SNAPSHOT)
        )
        if row["openml_task_id"] == 361247
    )
    other = copy.deepcopy(record)
    other["openml_task_id"] = 361253
    monkeypatch.setattr(
        registry.fresh,
        "_git_grep",
        lambda *_args: [],
    )

    reasons = registry._base_exclusion_reasons(
        declaration,
        {"status": "target_eligible", "task_record": record},
        spent={
            "openml_task_ids": [],
            "openml_dataset_ids": [],
            "task_records": [record, other],
        },
        exposure={"openml_dataset_ids": [], "normalized_names": []},
        known_names=[],
        thresholds=common.load_json(registry.CTR_DECLARATIONS)[
            "near_match_thresholds"
        ],
        prefreeze=common.DARKOFIT_PREFREEZE_HEAD,
        chimera_head=common.CHIMERABOOST_HEAD,
    )

    alarm = next(
        reason
        for reason in reasons
        if reason["kind"] == "spent_near_lineage_alarm"
    )
    assert [row["source_task_id"] for row in alarm["matches"]] == [
        361253
    ]


def test_pairwise_near_match_checks_all_prior_prospective_records():
    record = next(
        row
        for row in registry._task_records(
            common.load_json(registry.CTR_SNAPSHOT)
        )
        if row["openml_task_id"] == 361247
    )

    matches = registry._prospective_near_matches(
        record["fingerprint"],
        [(99, copy.deepcopy(record["fingerprint"]))],
        common.load_json(registry.CTR_DECLARATIONS)[
            "near_match_thresholds"
        ],
    )

    assert len(matches) == 1
    assert matches[0]["earlier_task_id"] == 99
    assert matches[0]["ambiguous"] is True


def test_post_preflight_head_diff_allows_only_preflight_artifact(
    monkeypatch,
):
    artifact = {
        "sources": {"darkofit_execution_head": "a" * 40}
    }
    monkeypatch.setattr(registry, "_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        registry,
        "_git",
        lambda *_args: "benchmarks/panel3_target_preflight.json",
    )

    registry._validate_post_preflight_boundary(
        artifact,
        registry.DEFAULT_PREFLIGHT,
        "b" * 40,
        require_clean_source=True,
    )

    monkeypatch.setattr(
        registry,
        "_git",
        lambda *_args: (
            "benchmarks/panel3_target_preflight.json\n"
            "darkofit/booster.py"
        ),
    )
    with pytest.raises(RuntimeError, match="source boundary changed"):
        registry._validate_post_preflight_boundary(
            artifact,
            registry.DEFAULT_PREFLIGHT,
            "b" * 40,
            require_clean_source=True,
        )


def test_artifact_hash_binding_rejects_mutation():
    artifact = common.bind_artifact_sha256(
        {"schema_version": 1, "value": [1, 2, 3]},
        "artifact_sha256",
    )
    common.verify_artifact_sha256(artifact, "artifact_sha256")
    changed = copy.deepcopy(artifact)
    changed["value"].append(4)

    with pytest.raises(RuntimeError, match="binding changed"):
        common.verify_artifact_sha256(changed, "artifact_sha256")


def test_design_helpers_do_not_mutate_frozen_campaign_artifacts():
    paths = (
        registry.CTR_SNAPSHOT,
        registry.CTR_PARTITION,
        registry.T5_REGISTRY,
        registry.T7_RAW,
        registry.T8_RAW,
        registry.SMOOTH_CROSS_RAW,
    )
    before = {path: common.sha256_file(path) for path in paths}

    common.validate_declarations()
    common.power_analysis()
    registry.spent_evidence()

    assert {path: common.sha256_file(path) for path in paths} == before


def test_spent_evidence_explicitly_covers_old_registry_gaps():
    spent = registry.spent_evidence()

    assert registry.EXPLICIT_SPENT_TASK_IDS <= set(
        spent["openml_task_ids"]
    )
    assert registry.EXPLICIT_SPENT_DATASET_IDS <= set(
        spent["openml_dataset_ids"]
    )
    t5 = json.loads(registry.T5_DECLARATIONS.read_text())
    assert {row["task_id"] for row in t5["candidates"]} <= set(
        spent["openml_task_ids"]
    )
    assert registry.LATER_INVALIDATED_LOCKBOX_TASK_IDS.isdisjoint(
        registry.AUTHORIZED_LOCKBOX_TASK_IDS
    )
    assert len(registry.AUTHORIZED_LOCKBOX_TASK_IDS) == 6
    assert 361264 not in registry.AUTHORIZED_LOCKBOX_TASK_IDS


def test_spent_evidence_rejects_lockbox_snapshot_drift():
    payloads = {
        path: common.load_json(path)
        for path in registry.SPENT_JSON_PATHS
    }
    payloads[registry.CTR_PARTITION] = copy.deepcopy(
        payloads[registry.CTR_PARTITION]
    )
    payloads[registry.CTR_PARTITION]["lockbox_task_ids"].pop()

    with pytest.raises(RuntimeError, match="lockbox task authorization"):
        registry.spent_evidence(
            payloads=payloads,
            source_sha256={
                str(path.relative_to(ROOT)): common.sha256_file(path)
                for path in registry.FROZEN_EVIDENCE
            },
        )


def test_captured_chimera_exposure_matches_frozen_head():
    snapshots, source_sha256 = common.secure_snapshot_files(
        list(registry.CHIMERA_EXPOSURE_PATHS),
        allowed_root=registry.CHIMERA_ROOT,
    )
    registry._validate_chimera_snapshot_at_head(
        source_sha256,
        head=common.CHIMERABOOST_HEAD,
    )

    observed = registry._chimera_exposure_catalog_from_snapshots(
        snapshots,
        source_sha256,
    )
    expected = registry.fresh._chimera_exposure_catalog()

    assert observed == expected


def test_registry_publication_recheck_rejects_preflight_swap(
    monkeypatch,
):
    artifact = {
        "target_preflight_file_sha256": "a" * 64,
        "target_preflight_sha256": "b" * 64,
        "power_design_decision": {"decision_sha256": "c" * 64},
        "power_design_file_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        common,
        "secure_load_json",
        lambda _path: ({"target_preflight_sha256": "changed"}, "e" * 64),
    )

    with pytest.raises(RuntimeError, match="authorization artifact changed"):
        registry._recheck_registry_inputs(
            artifact,
            preflight_path=registry.DEFAULT_PREFLIGHT,
        )


def test_candidate_contract_separates_nominees_and_delegates_multiplicity():
    contract = common.load_json(common.CANDIDATE_CONTRACT)
    candidates = contract["candidates"]

    assert [row["name"] for row in candidates] == [
        "t5_composite_policy",
        "guarded_cross_features_policy",
    ]
    assert contract["decision"]["post_outcome_winner_selection_allowed"] is False
    assert contract["decision"]["frozen_candidate_hypothesis_count"] == 2
    assert contract["decision"]["multiplicity_source"] == (
        "benchmarks/panel3_power_design_decision.json"
    )
    rules = contract["decision"]["retained_candidate_multiplicity"]
    assert rules["two"]["per_candidate_one_sided_alpha"] == 0.025
    assert rules["two"]["bootstrap_percentile"] == 97.5
    assert rules["one"]["per_candidate_one_sided_alpha"] == 0.05
    assert rules["one"]["bootstrap_percentile"] == 95.0
    assert rules["zero"]["execution_authorized"] is False
    assert all(
        row["decision_role"] == "descriptive_only"
        for row in contract["comparators"]
    )


def test_analyzer_adjudicates_candidates_independently():
    passing = analyzer.adjudicate_candidate(
        {f"task-{index}": 0.99 for index in range(12)},
        bonferroni_bootstrap_upper=1.0,
        equal_dataset_fit_seconds_ratio=2.0,
        worst_dataset_fit_seconds_ratio=3.0,
        equal_dataset_predict_seconds_ratio=1.1,
        equal_dataset_peak_rss_ratio=1.2,
        complete=True,
        integrity_ok=True,
        deviations=[],
    )
    failing = copy.deepcopy(passing)
    failing["passes"] = False

    result = analyzer.adjudicate_two_candidates(
        {
            "t5_composite_policy": passing,
            "guarded_cross_features_policy": failing,
        }
    )

    assert result["post_outcome_winner_selection_used"] is False
    assert result["shipping_candidates"] == ["t5_composite_policy"]
    assert result["familywise_one_sided_alpha"] == 0.05


def test_categorical_resolver_unions_flags_and_nonnumeric_dtypes():
    X = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "object_but_unflagged": ["a", "b"],
            "flagged_integer": [1, 2],
        }
    )

    assert runner.categorical_column_indices(
        X,
        [False, False, True],
    ) == (1, 2)
    assert runner.categorical_column_indices(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        [False, True],
    ) == (1,)
