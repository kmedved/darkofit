from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import extract_smooth_selector_engagement as engagement  # noqa: E402


RESULT_PATH = BENCH_DIR / "smooth_selector_engagement_check_20260722.json"


def test_engagement_extractor_binds_current_inputs_and_exact_reference_rows():
    assert engagement.file_sha256(engagement.CURRENT_RAW_PATH) == (
        engagement.CURRENT_RAW_SHA256
    )
    assert engagement.file_sha256(engagement.CURRENT_MANIFEST_PATH) == (
        engagement.CURRENT_MANIFEST_SHA256
    )
    assert engagement.file_sha256(engagement.CURRENT_PER_DATASET_PATH) == (
        engagement.CURRENT_PER_DATASET_SHA256
    )
    airfoil = engagement._quality_ratios(
        engagement.CURRENT_PER_DATASET_PATH, "airfoil_self_noise"
    )
    protein = engagement._quality_ratios(
        engagement.CURRENT_PER_DATASET_PATH, "physiochemical_protein"
    )
    assert airfoil["test_rmse_ratio"] == 0.9533474242274157
    assert airfoil["prediction_seconds_per_call_ratio"] == 2.9627250840409163
    assert protein["test_rmse_ratio"] == 1.0679029961960846
    assert protein["fit_seconds_ratio"] == 0.23315393097367512


def test_current_v020_metadata_reproduces_the_engagement_asymmetry():
    raw = json.loads(engagement.CURRENT_RAW_PATH.read_text())
    airfoil = engagement.summarize_current_dataset(
        raw, dataset="airfoil_self_noise"
    )
    protein = engagement.summarize_current_dataset(
        raw, dataset="physiochemical_protein"
    )
    assert airfoil["member_count"] == protein["member_count"] == 3
    assert airfoil["linear_leaves_selected"] == {
        "true": 0,
        "false": 0,
        "null": 3,
    }
    assert protein["linear_leaves_selected"] == {
        "true": 3,
        "false": 0,
        "null": 0,
    }
    assert protein["cross_features_selected"] == {
        "true": 1,
        "false": 2,
        "null": 0,
    }


def test_create_only_engagement_record_has_the_declared_narrow_decision():
    payload = json.loads(RESULT_PATH.read_text())
    assert payload["kind"] == "spent_selector_engagement_verification"
    assert payload["fresh_or_lockbox_accessed"] is False
    airfoil = payload["m2_v018_outer_bagged"]["airfoil_self_noise"]
    protein = payload["m2_v018_outer_bagged"]["physiochemical_protein"]
    assert airfoil["child_fit_count"] == protein["child_fit_count"] == 24
    assert airfoil["selected_lane_counts"] == {"constant": 24}
    assert protein["selected_lane_counts"] == {"linear": 24}
    assert airfoil["linear_selection_performed"]["false"] == 24
    assert protein["linear_selection_performed"]["true"] == 24
    assert airfoil["resolved_cross_features_nonnull_count"] == 0
    assert protein["resolved_cross_features_nonnull_count"] == 0
    assert airfoil["resolved_cat_combinations"]["false"] == 24
    assert protein["resolved_cat_combinations"]["false"] == 24
    decision = payload["decision"]
    assert decision["two_dataset_selector_hypothesis_confirmed"] is False
    assert decision["protein_selector_signature_confirmed"] is True
    assert decision["airfoil_selector_signature_confirmed"] is False
    assert decision["airfoil_is_current_default_deficit"] is False


def test_engagement_record_is_create_only_and_hash_attested():
    payload = json.loads(RESULT_PATH.read_text())
    for dataset in engagement.DATASETS:
        for artifact in payload["m2_v018_outer_bagged"][dataset]["artifacts"]:
            assert len(artifact["sha256"]) == 64
            assert artifact["size_bytes"] > 0
    assert hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest() == (
        "878ffdc0bfb615714b5acd0ea0c1d09f63604d4d423d57ab0898f9bd377ab3d1"
    )
