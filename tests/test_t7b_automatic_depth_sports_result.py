from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from benchmarks import analyze_t7b_automatic_depth_sports_v1 as analyzer
from benchmarks import run_t7b_automatic_depth_sports_v1 as campaign


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

ARTIFACTS = {
    "contract": (
        BENCH / "t7b_automatic_depth_sports_v1_contract.json",
        "ac5a745378a086ed119af1d55a68e961ea95e1f74c4f307dce72ad9b6717fe1b",
    ),
    "launch": (
        BENCH
        / "t7b_automatic_depth_sports_v1_inspection1_launch_manifest_20260722.json",
        "07567f2585df0183bbd0f6dee9b3c18d678e28b3280ddd41c21331a23439bac1",
    ),
    "raw": (
        BENCH / "t7b_automatic_depth_sports_v1_inspection1_raw_20260722.json",
        "31b4d18576ed35efae3fe89e07375f18b82c02668586a562aa9969d1c9f0830d",
    ),
    "result": (
        BENCH / "t7b_automatic_depth_sports_v1_inspection1_result_20260722.json",
        "1ec0d2d37ef75195b66b779ec94920e05f5047147538de6eb17622947fd1a0da",
    ),
    "terminal": (
        BENCH
        / "t7b_automatic_depth_sports_v1_inspection1_terminal_attestation_20260722.json",
        "180e7ea418b4a5e53c0672c2c5b5c1672824dc83fc3f9b3e279bca0cd19d9644",
    ),
}


def _load(name: str):
    return json.loads(ARTIFACTS[name][0].read_text())


def _assert_nested_approx(actual, expected):
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_approx(actual[key], expected[key])
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence) and not isinstance(actual, (str, bytes))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_approx(actual_item, expected_item)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-14, abs=1e-15)
    else:
        assert actual == expected


def test_t7b_depth_sports_artifacts_retain_create_only_hashes():
    for path, expected in ARTIFACTS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_t7b_depth_sports_result_recomputes_from_all_raw_rows():
    contract = campaign.load_contract()
    raw = _load("raw")
    result = _load("result")
    rows = analyzer.validate_raw(raw, contract)
    recomputed = analyzer.analyze_rows(rows)

    _assert_nested_approx(recomputed, result["analysis"])
    assert len(rows) == 18
    assert recomputed["disposition"] == "eligible_for_fresh_tier_d_design"
    assert recomputed["gates"] == {
        "cluster_bootstrap_p95_at_most_1_010": True,
        "cold_player_aggregate_at_most_1_000": True,
        "held_team_aggregate_at_most_1_010": True,
        "worst_lineage_at_most_1_030": True,
        "worst_loo_at_most_1_003": True,
        "worst_season_at_most_1_020": True,
    }
    cold = recomputed["cold_player"]
    assert cold["point_ratio"] == pytest.approx(0.9502662058671157)
    assert cold["cluster_bootstrap"]["percentiles"]["p95"] == pytest.approx(
        0.9665910782546924
    )
    assert cold["worst_lineage_ratio"] == pytest.approx(0.9971997454168474)
    assert cold["worst_loo_ratio"] == pytest.approx(0.9638842337377516)
    assert all(value < 1.0 for value in cold["per_lineage_ratio"].values())
    assert result["shipping_or_default_claim_eligible"] is False
    assert recomputed["fresh_confirmation_authorized"] is False


def test_t7b_depth_sports_rows_bind_mechanism_scope_and_invariants():
    rows = _load("raw")["rows"]
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["arm"]] = row
    assert len(pairs) == 9
    for pair in pairs.values():
        control = pair[campaign.CONTROL]
        candidate = pair[campaign.CANDIDATE]
        assert control["resolved_depth"] == 6
        assert candidate["resolved_depth"] == 4
        assert candidate["auto_structure"]["candidates"]["depth"]["rule"] == (
            campaign.DEPTH_RULE
        )
        assert candidate["auto_structure"]["candidates"]["depth"]["branch"] == (
            "low_density"
        )
        assert candidate["l2_leaf_reg"] == control["l2_leaf_reg"] == 3.0
        assert control["safe_roundtrip_exact"] is True
        assert candidate["safe_roundtrip_exact"] is True
        assert control["ambient_thread_count_after_predict"] == campaign.THREADS
        assert candidate["ambient_thread_count_after_predict"] == campaign.THREADS
        assert control["fitted_thread_counts"] == [campaign.THREADS]
        assert candidate["fitted_thread_counts"] == [campaign.THREADS]


def test_t7b_depth_sports_terminal_attestation_binds_every_output():
    launch = _load("launch")
    terminal = _load("terminal")

    assert launch["inspection_spent_on_manifest_creation"] is True
    assert launch["rerun_authorized"] is False
    assert launch["contract_sha256"] == ARTIFACTS["contract"][1]
    assert terminal["disposition"] == "eligible_for_fresh_tier_d_design"
    assert terminal["launch_manifest_sha256"] == ARTIFACTS["launch"][1]
    assert terminal["raw_sha256"] == ARTIFACTS["raw"][1]
    assert terminal["result_sha256"] == ARTIFACTS["result"][1]
    assert terminal["rerun_authorized"] is False
    assert terminal["shipping_or_default_claim_eligible"] is False
    assert terminal["fresh_confirmation_authorized"] is False
