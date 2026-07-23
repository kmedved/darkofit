from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
EXPECTED_HASHES = {
    "automatic_linear_selector_v2_protein_attribution_attempt2_20260722_manifest.json": "d6cbee2249046bc8eca05080eea38457c21d0d130076a59839b313c52c8b54b7",
    "automatic_linear_selector_v2_protein_attribution_attempt2_20260722_raw.json": "0caaa2f97fd527976233f6511267c3df2b6487bc8d5a665d87c9fad2c3b11be7",
    "automatic_linear_selector_v2_protein_attribution_attempt2_20260722_result.json": "4b75f4ae048e926ec07bf3a17c4a9e9356b52a7adfd869409594cc3878f7e61c",
    "automatic_linear_selector_v2_guardrail_replay_20260722.json": "1d0ac7eedbcc86dd83b47f826e77efb70071381d88a27a68c6dc61d31e707122",
    "automatic_linear_selector_v2_terminal_attestation_20260722.json": "e35b6907ce1872a9f01ce5359d2b49064ef2bcd112b5952fcdd53db6a166387a",
}


def _load(name: str) -> dict:
    return json.loads((BENCHMARKS / name).read_text(encoding="utf-8"))


def test_selector_terminal_artifacts_are_immutable() -> None:
    for name, expected in EXPECTED_HASHES.items():
        observed = hashlib.sha256((BENCHMARKS / name).read_bytes()).hexdigest()
        assert observed == expected, name


def test_protein_failure_is_terminal_despite_aggregate_gain() -> None:
    result = _load(
        "automatic_linear_selector_v2_protein_attribution_attempt2_20260722_result.json"
    )["analysis"]

    assert result["aggregate_automatic_over_constant_rmse"] < 1.0
    assert result["gates"]["aggregate_ratio_at_most_1_02"] is True
    assert result["gates"]["every_coordinate_ratio_at_most_1_02"] is True
    assert result["gates"]["all_selector_and_exactness_invariants"] is False
    assert result["coordinates"][1]["selector_margin"] < 0.03
    assert result["coordinates"][1]["invariants"]["prediction_exact"] is False
    assert result["disposition"] == "terminal_close"


def test_historical_replay_cannot_rescue_terminal_close() -> None:
    replay = _load("automatic_linear_selector_v2_guardrail_replay_20260722.json")
    attestation = _load(
        "automatic_linear_selector_v2_terminal_attestation_20260722.json"
    )

    assert replay["analysis"]["combined"]["worst_lineage_ratio"] == 1.0
    assert replay["analysis"]["combined"]["worst_split_ratio"] == 1.0
    assert replay["limitations"]["fresh_evidence"] is False
    assert replay["limitations"]["can_reverse_protein_terminal_close"] is False
    assert attestation["disposition"] == "killed"
    assert attestation["rerun_authorized"] is False
    assert attestation["merge_authorized"] is False
    assert attestation["fresh_confirmation_authorized"] is False
