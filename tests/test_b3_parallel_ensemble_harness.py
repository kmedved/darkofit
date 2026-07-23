import json
from pathlib import Path

import pytest

from benchmarks import run_b3_parallel_ensemble_v1 as campaign


def _record(fit, peak, prediction="p", probability=None):
    return {
        "mode": None,
        "fit_seconds": fit,
        "predict_seconds": 0.1,
        "fit_rss": {"peak_bytes": peak},
        "archive_bytes": 10,
        "prediction_sha256": prediction,
        "probability_sha256": probability,
        "model": {
            "member_seeds": list(range(8)),
            "sampled_indices_sha256": [f"s{i}" for i in range(8)],
            "oob_indices_sha256": [f"o{i}" for i in range(8)],
            "best_iterations": [10] * 8,
            "fitted_thread_counts": [14] * 8,
            "prediction_thread_counts": [14] * 8,
            "schedule": None,
            "sequential": True,
        },
    }


def _rows(candidate_ratio=0.8, candidate_peak=2 * campaign.GIB):
    rows = []
    for case in campaign.CASES:
        for block in range(campaign.BLOCKS):
            control_records = []
            candidate_records = []
            for mode in campaign.MODES:
                control = _record(10.0, campaign.GIB)
                control["mode"] = mode
                candidate = _record(10.0 * candidate_ratio, candidate_peak)
                candidate["mode"] = mode
                candidate["model"]["fitted_thread_counts"] = [2] * 8
                candidate["model"]["prediction_thread_counts"] = [2] * 8
                candidate["model"]["schedule"] = {
                    "contract": campaign.CONTRACT_ID,
                    "mode": "private_process_workers",
                    "workers": 7,
                    "member_threads": 2,
                    "total_thread_budget": 14,
                    "maximum_model_threads": 14,
                    "result_order": "member_index",
                }
                candidate["model"]["sequential"] = False
                control_records.append(control)
                candidate_records.append(candidate)
            common = {
                "case_id": case,
                "block": block,
                "fingerprints": {"split_sha256": case},
            }
            rows.append({**common, "arm": campaign.ARMS[0], "records": control_records})
            rows.append({**common, "arm": campaign.ARMS[1], "records": candidate_records})
    return rows


def test_b3_analysis_advances_only_stable_exact_bounded_speedup():
    result = campaign.analyze(_rows())
    assert result["disposition"] == "advance"
    assert all(result["gates"].values())
    assert result["speed"]["cold_executor"]["equal_case_geometric_mean_ratio"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "ratio,peak,failed_gate",
    [
        (1.01, 2 * campaign.GIB, "cold_speed_stable"),
        (0.8, 7 * campaign.GIB, "hybrid_rss"),
    ],
)
def test_b3_analysis_kills_speed_or_absolute_memory_failure(ratio, peak, failed_gate):
    result = campaign.analyze(_rows(candidate_ratio=ratio, candidate_peak=peak))
    assert result["disposition"] == "kill"
    assert result["gates"][failed_gate] is False


def test_b3_hybrid_memory_requires_ratio_and_delta_when_below_ceiling():
    peak = 3.5 * campaign.GIB
    result = campaign.analyze(_rows(candidate_peak=peak))
    assert result["gates"]["hybrid_rss"] is True


def test_b3_hybrid_memory_fails_when_ratio_and_delta_both_bind():
    result = campaign.analyze(_rows(candidate_peak=5.5 * campaign.GIB))
    assert result["gates"]["hybrid_rss"] is False
    assert result["disposition"] == "kill"


def test_b3_analysis_kills_output_mismatch():
    rows = _rows()
    rows[1]["records"][0]["prediction_sha256"] = "different"
    result = campaign.analyze(rows)
    assert result["gates"]["behavior_exact"] is False
    assert result["disposition"] == "kill"


def test_b3_outputs_are_create_only(tmp_path):
    path = tmp_path / "artifact.json"
    campaign.write_create_only(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        campaign.write_create_only(path, {"ok": False})


def test_b3_contract_constants_match_frozen_document():
    text = (Path(__file__).resolve().parents[1] / "benchmarks/b3_parallel_ensemble_v1_contract.md").read_text()
    assert campaign.CONTRACT_ID in text
    assert campaign.CONTROL_HEAD.startswith("c4dae58")
    assert campaign.CANDIDATE_BASE_HEAD.startswith("4073bb9")
    assert campaign.CANDIDATE_HEAD.startswith("5116470")
    assert (campaign.WORKERS, campaign.MEMBER_THREADS, campaign.THREADS) == (7, 2, 14)
