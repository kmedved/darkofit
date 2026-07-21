import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERRATUM = ROOT / "benchmarks" / "barchive_v2_result_heading_erratum_20260721.json"


def test_barchive_v2_heading_erratum_is_narrow_and_hash_bound():
    erratum = json.loads(ERRATUM.read_text(encoding="utf-8"))
    result = erratum["authoritative_result"]
    note = erratum["original_note"]

    for record in (result, note):
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    payload = json.loads((ROOT / result["path"]).read_text(encoding="utf-8"))
    assert payload["name"] == erratum["result_identity"]
    assert payload["gate"]["median_effective_archive_to_single"] == (4.152524742578921)
    assert payload["disposition"] == ("close_barchive_nominate_fused_lane_dispatch")
    assert erratum["correction"]["scope"] == "markdown_heading_only"
    assert erratum["frozen_evidence_changed"] is False
    assert erratum["disposition_changed"] is False


def test_barchive_v2_closeout_docs_preserve_the_frozen_decision():
    plan = (ROOT / "COUNTERPUNCH_PLAN.md").read_text(encoding="utf-8")
    log = (ROOT / "benchmarks" / "TESTING_LOG.md").read_text(encoding="utf-8")

    for document in (plan, log):
        assert "4.152525" in document
        assert "4.0" in document
        assert "no serializer" in document.lower()
        assert "fused-lane dispatch" in document
    assert "Wave 3 B-archive feasibility completed and closed" in plan
    assert "close_barchive_nominate_fused_lane_dispatch" in log
    assert "v1_size_outcomes_published" not in plan
