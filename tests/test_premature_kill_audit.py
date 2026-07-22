from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "benchmarks/premature_kill_audit_20260722.json"
NOTE_PATH = ROOT / "benchmarks/premature_kill_audit_20260722.md"

REQUIRED_IDS = {
    "tabarena_cap_horizon_10k",
    "automatic_tree_mode_screen",
    "target_statistics_four_permutations",
    "safe_ordinal_column_transform",
    "safe_one_hot_transform",
    "global_linear_residual",
    "a10_minimal_confirmation",
    "auto_lr_early_stop_exact_refit",
    "oob5_stable_confirmation",
    "quantile_conformal_offset",
    "binary_temperature_scaling",
    "gaussian_scalar_calibration",
    "initial_random_split_linear_leaves",
    "numeric_cross_feature_donor",
    "categorical_combinations",
    "packed_prediction_router",
    "s1_robust_heads",
    "s2_entity_ensemble",
    "random_strength_1",
    "random_strength_0_5_s4",
    "e1_subset_fusion",
    "p2_prediction_certification",
    "e2_large_n_certification",
    "e3_float32_histograms",
    "smooth_linear_leaf_selector_3pct",
    "c2_native_ordinal",
    "t5_composite_confirmation",
    "panel3_t5_power",
    "panel3_guarded_cross_power",
    "t10_oob5_automatic",
    "t7b_onehot255",
    "m3a_existing_ensemble_policies",
    "q_quantization_funding",
    "b1_sampling_alone",
    "b2_member_policy_alone",
    "b1_b2_combined_archive_gate",
    "barchive_exact_factoring",
    "b3_parallel_members",
    "fused_lane_dispatch_calibration",
    "m6_successor_infrastructure_failures",
    "m6_v2_selector_tripwire",
}

FROZEN_V2_HASHES = {
    "benchmarks/m6_quality_rule_v2.py": (
        "b80520a77f3b99f14209a89535b32ca3437141d9251353618db7f1151484cb55"
    ),
    "benchmarks/m6_quality_successor_v2_contract.md": (
        "9458997b392ec9b560aca70f1dc7e3be8897c67d1145795a3e9f907923e35884"
    ),
    "benchmarks/run_m6_quality_successor_v2.py": (
        "3acdc64c7b8563def0fe01a3d4b14b65985a0390ad5fae9a9e37a07ba00061c2"
    ),
    "benchmarks/m6_quality_successor_v2_backtest_result.json": (
        "6880c679cd5f16aa61d13c2e57282e3f162769be87e478a6ddf18d8958c9cf57"
    ),
}


def _audit():
    return json.loads(AUDIT_PATH.read_text())


def test_phase_f_audit_is_complete_unique_and_evidence_backed():
    audit = _audit()
    entries = audit["entries"]
    ids = [entry["id"] for entry in entries]
    assert audit["name"] == "darkofit_phase_f_premature_kill_audit"
    assert len(entries) == len(set(ids)) == audit["summary"]["entry_count"] == 41
    assert set(ids) == REQUIRED_IDS
    for entry in entries:
        assert entry["historical_rule"]
        assert entry["observed"]
        assert entry["backlog_action"]
        assert (ROOT / entry["evidence"]).is_file()


def test_every_abolished_victim_has_a_forward_disposition():
    audit = _audit()
    allowed = {
        "healed_or_superseded",
        "readjudication_active",
        "readjudication_backlog",
    }
    abolished = [
        entry for entry in audit["entries"]
        if entry["current_rule_status"] == "abolished"
    ]
    assert abolished
    assert all(entry["current_disposition"] in allowed for entry in abolished)
    assert audit["summary"]["frozen_records_edited"] == 0
    assert "unfinished retraction" in audit["standing_rule"]


def test_readjudication_backlog_is_exact_and_requires_new_identities():
    audit = _audit()
    backlog = audit["re_adjudication_backlog"]
    assert [item["id"] for item in backlog] == [
        "smooth_linear_leaf_selector_3pct",
        "b3_parallel_members",
        "q_quantization_funding",
    ]
    by_id = {entry["id"]: entry for entry in audit["entries"]}
    assert all(by_id[item["id"]]["new_campaign_required"] for item in backlog)
    assert audit["summary"]["abolished_or_reformed_victims_pending"] == 3


def test_m6_selection_uses_only_surviving_verdicts_and_marks_old_selector():
    audit = _audit()
    assert audit["m6_v3_backtest_selection"] == {
        "known_advance": "b1_b2_combined_archive_gate",
        "known_kill": "c2_native_ordinal",
        "retired_verdict_tripwire": "smooth_linear_leaf_selector_3pct",
    }
    by_id = {entry["id"]: entry for entry in audit["entries"]}
    assert by_id["c2_native_ordinal"]["current_disposition"] == "valid_closure"
    assert by_id["smooth_linear_leaf_selector_3pct"]["current_rule_status"] == (
        "abolished"
    )
    assert by_id["m6_v2_selector_tripwire"]["current_disposition"] == (
        "readjudication_active"
    )
    assert audit["summary"]["m6_v2_forward_authority"] is False


def test_frozen_m6_v2_evidence_was_not_rewritten():
    for relative, expected in FROZEN_V2_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_human_note_states_nonretroactivity_and_evidence_boundary():
    text = NOTE_PATH.read_text()
    for token in (
        "41 historical terminal dispositions",
        "retroactively promotes nothing",
        "No fresh confirmation, TabArena",
        "A retracted rule with unexamined victims is an unfinished retraction",
        "native ordinal C2",
        "old selector as an explicit tripwire that must now advance",
    ):
        assert token in text
