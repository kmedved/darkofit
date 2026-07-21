"""Lineage and correction tests for B-archive attempt 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import analyze_barchive_v2 as analyzer
from benchmarks import run_barchive_v1 as v1_runner
from benchmarks import run_barchive_v2 as runner
from benchmarks.analyze_ensemble_archive_components import analyze_archive
from darkofit import DarkoRegressor
from darkofit.sklearn_api import _fit_private_ensemble_v3


ROOT = Path(__file__).resolve().parents[1]


def test_barchive_v2_is_new_identity_with_exact_terminal_lineage():
    lineage = runner.v1_attempt_lineage()

    assert runner.CONTRACT_NAME == "wave3_barchive_v2_20260721"
    assert runner.CONTRACT_PATH != v1_runner.CONTRACT_PATH
    assert runner.RAW_PATH != v1_runner.RAW_PATH
    assert runner.TERMINAL_PATH != v1_runner.TERMINAL_PATH
    assert lineage["lineage_valid"] is True
    assert lineage["v1_completed_rows_discarded"] == 0
    assert lineage["v1_partial_rows_published"] is False
    assert lineage["v1_size_outcomes_published"] is False
    assert lineage["v1_rerun"] is False
    assert lineage["source_changed"] is False
    assert lineage["cases_changed"] is False
    assert lineage["threshold_changed"] is False
    assert not v1_runner.RAW_PATH.exists()


def test_barchive_v2_changes_only_the_numpy_header_presence_rule():
    v1_contract = json.loads(v1_runner.CONTRACT_PATH.read_text(encoding="utf-8"))
    v2_rules = runner.decision_rules()
    v1_rules = v1_contract["decision_rules"]

    assert runner.MODEL_SOURCE_HEAD == v1_runner.MODEL_SOURCE_HEAD
    assert runner.ALLOWED_CANONICAL_ARRAYS == v1_runner.ALLOWED_CANONICAL_ARRAYS
    assert runner.ALLOWED_CANONICAL_HEADER_FIELDS == {
        "n_input_features",
        "prep",
        "random_state",
    }
    assert v2_rules["allowed_canonical_header_fields"] == [
        "n_input_features",
        "prep",
        "random_state",
    ]
    assert v2_rules["median_effective_archive_to_single_at_most"] == (
        v1_rules["median_effective_archive_to_single_at_most"]
    )
    for name in set(v1_rules).difference({"allowed_canonical_header_fields"}):
        assert v2_rules[name] == v1_rules[name]
    assert runner.expected_shared_preprocessing() == (
        v1_contract["expected_shared_preprocessing"]
    )
    assert runner.execution_contract() == v1_contract["execution"]
    assert runner.claim_contract() == v1_contract["claims"]


def test_barchive_v2_numpy_synthetic_has_exact_complete_header(tmp_path):
    rng = np.random.default_rng(20260721)
    X = rng.normal(size=(140, 8))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.2, size=140)
    model = _fit_private_ensemble_v3(
        DarkoRegressor(
            iterations=5,
            depth=3,
            early_stopping_rounds=2,
            random_state=4,
            n_ensembles=3,
            diagnostic_warnings="never",
        ),
        X,
        y,
        sampling="without_replacement",
        sampling_unit="rows",
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
    )
    archive = tmp_path / "ensemble.npz"
    model.save_model(archive)

    canonical = analyze_archive(archive)["canonical_preprocessor"]

    assert canonical["eligible"] is True
    assert canonical["header_fields"] == sorted(runner.ALLOWED_CANONICAL_HEADER_FIELDS)
    assert "feature_names_in" not in canonical["header_fields"]


def test_barchive_v2_analyzer_is_routed_to_v2_identity():
    assert analyzer.runner is runner
    assert analyzer.RESULT_PATH.name == "barchive_v2_result.json"
    assert analyzer.NOTE_PATH.name == "barchive_v2_result.md"
    assert analyzer._foundation.runner is runner


@pytest.mark.campaign
@pytest.mark.skipif(
    not runner.CONTRACT_PATH.exists(), reason="B-archive v2 contract not frozen yet"
)
def test_barchive_v2_frozen_contract_is_exact():
    contract = runner.load_contract()

    assert contract["name"] == runner.CONTRACT_NAME
    assert contract["sources"]["darkofit"] == runner.MODEL_SOURCE_HEAD
    assert contract["attempt_lineage"] == runner.v1_attempt_lineage()
    assert contract["decision_rules"] == runner.decision_rules()
    assert (
        contract["cases"]
        == json.loads(v1_runner.CONTRACT_PATH.read_text(encoding="utf-8"))["cases"]
    )


@pytest.mark.campaign
@pytest.mark.skipif(
    not analyzer.RESULT_PATH.exists(),
    reason="B-archive v2 result not published yet",
)
def test_barchive_v2_published_result_regenerates_exactly():
    stored = json.loads(analyzer.RESULT_PATH.read_text(encoding="utf-8"))
    regenerated = analyzer.build_result()
    regenerated["analyzed_at"] = stored["analyzed_at"]

    assert regenerated == stored
    assert analyzer.NOTE_PATH.read_text(encoding="utf-8") == (
        analyzer.render_note(stored)
    )
    assert stored["m3b_r3_amended"] is False
    assert stored["serializer_retention_authorized"] is False
