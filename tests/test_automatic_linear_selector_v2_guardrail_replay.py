from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "analyze_automatic_linear_selector_v2_guardrail_replay.py"
SPEC = importlib.util.spec_from_file_location("selector_guardrail_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_replay_recomputes_all_historical_strata() -> None:
    payload = replay.build_replay()

    combined = payload["analysis"]["combined"]
    assert combined["lineage_count"] == 21
    assert combined["split_count"] == 71
    assert set(payload["analysis"]["by_stratum"]) == {
        "categorical",
        "group_safe_sports",
        "noisy_tabular",
        "smooth_process",
    }
    assert payload["limitations"] == {
        "dependent_on_prior_artifacts": True,
        "prior_outcomes_known": True,
        "fresh_evidence": False,
        "candidate_code_executed": False,
        "can_reverse_protein_terminal_close": False,
        "note": (
            "This replay is a consistency check over spent evidence. It is not "
            "independent confirmation and cannot rescue a terminal Protein failure."
        ),
    }


def test_replay_matches_immutable_fresh_analysis() -> None:
    payload = replay.build_replay()
    source = json.loads(replay.FRESH_RESULT.read_text(encoding="utf-8"))
    expected_names = {
        "smooth_process": "smooth_process_selector_over_default",
        "categorical": "categorical_selector_over_default",
        "noisy_tabular": "noisy_tabular_selector_over_default",
    }
    for stratum, contrast_name in expected_names.items():
        observed = payload["analysis"]["by_stratum"][stratum]
        expected = source["analysis"]["contrasts"][contrast_name]
        assert observed["equal_lineage_geomean_ratio"] == pytest.approx(
            expected["equal_lineage_geomean_ratio"], abs=1e-15
        )
        assert observed["worst_lineage_ratio"] == pytest.approx(
            expected["worst_lineage_ratio"], abs=1e-15
        )
        assert observed["worst_split_ratio"] == pytest.approx(
            expected["worst_split_ratio"], abs=1e-15
        )


def test_replay_binds_historical_artifact_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(replay, "EXPECTED_SHA256", {changed: "0" * 64})

    with pytest.raises(RuntimeError, match="historical artifact hash changed"):
        replay._read_bound_json(changed)


def test_output_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"
    replay._write_create_only(output, replay.build_replay())
    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    with pytest.raises(FileExistsError):
        replay._write_create_only(output, replay.build_replay())
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_hash
