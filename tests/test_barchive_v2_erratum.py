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
