"""Contract tests for the frozen basketball input-validation runner."""

import argparse
import json

import pytest

from benchmarks import run_basketball_input_validation as runner


def _summary(value, iqr_fraction=0.05):
    return {
        "values_seconds": [value] * runner.TIMING_BLOCKS,
        "minimum_seconds": value,
        "median_seconds": value,
        "maximum_seconds": value,
        "iqr_seconds": value * iqr_fraction,
        "iqr_fraction": iqr_fraction,
    }


def _passing_summaries():
    return {
        "first_fit": {
            runner.VALIDATED: _summary(1.6),
            runner.ASSUME_FINITE: _summary(1.6),
        },
        "first_predict": {
            runner.VALIDATED: _summary(0.0041),
            runner.ASSUME_FINITE: _summary(0.0040),
        },
        "validation_only": {
            runner.VALIDATED: _summary(0.0002),
            runner.ASSUME_FINITE: _summary(0.0001),
        },
        "warmup": {
            runner.VALIDATED: _summary(4.7),
            runner.ASSUME_FINITE: _summary(4.7),
        },
        "import": {
            runner.VALIDATED: _summary(0.15),
            runner.ASSUME_FINITE: _summary(0.15),
        },
    }


def test_schedule_is_reciprocal_and_position_balanced():
    schedule = runner.schedule()
    assert len(schedule) == runner.TIMING_BLOCKS
    assert schedule[0] == (runner.VALIDATED, runner.ASSUME_FINITE)
    assert schedule[1] == (runner.ASSUME_FINITE, runner.VALIDATED)
    assert sum(order[0] == runner.VALIDATED for order in schedule) == 3


def test_timing_summary_requires_six_positive_finite_values():
    summary = runner.timing_summary([1, 2, 3, 4, 5, 6])
    assert summary["median_seconds"] == 3.5
    with pytest.raises(RuntimeError, match="exactly"):
        runner.timing_summary([1])
    with pytest.raises(RuntimeError, match="positive"):
        runner.timing_summary([1, 1, 1, 1, 1, 0])


def test_analyze_accepts_only_a_complete_passing_campaign():
    decision = runner.analyze(
        summaries=_passing_summaries(),
        paired_predict_ratio=_summary(1.025),
        outputs_exact=True,
        behavior_fingerprints={"one"},
        archives_exact=True,
        metadata_exact=True,
        workers_clean=True,
        caches_isolated=True,
    )
    assert decision["passed"]
    assert decision["recommendation"] == "ship_input_validation_layer"
    assert not decision["model_default_change_authorized"]
    assert not decision["broad_quality_claim_authorized"]


@pytest.mark.parametrize(
    "mutation",
    [
        "fit_budget",
        "predict_budget",
        "fit_stability",
        "predict_stability",
        "paired_stability",
        "outputs",
        "fingerprints",
        "archives",
        "metadata",
        "workers",
        "caches",
    ],
)
def test_analyze_fails_closed_for_each_gate(mutation):
    summaries = _passing_summaries()
    paired = _summary(1.025)
    kwargs = {
        "outputs_exact": True,
        "behavior_fingerprints": {"one"},
        "archives_exact": True,
        "metadata_exact": True,
        "workers_clean": True,
        "caches_isolated": True,
    }
    if mutation == "fit_budget":
        summaries["first_fit"][runner.VALIDATED] = _summary(2.0)
    elif mutation == "predict_budget":
        summaries["first_predict"][runner.VALIDATED] = _summary(0.005)
    elif mutation == "fit_stability":
        summaries["first_fit"][runner.VALIDATED]["iqr_fraction"] = 0.3
    elif mutation == "predict_stability":
        summaries["first_predict"][runner.VALIDATED]["iqr_fraction"] = 0.6
    elif mutation == "paired_stability":
        paired["iqr_fraction"] = 0.3
    elif mutation == "outputs":
        kwargs["outputs_exact"] = False
    elif mutation == "fingerprints":
        kwargs["behavior_fingerprints"] = {"one", "two"}
    elif mutation == "archives":
        kwargs["archives_exact"] = False
    elif mutation == "metadata":
        kwargs["metadata_exact"] = False
    elif mutation == "workers":
        kwargs["workers_clean"] = False
    else:
        kwargs["caches_isolated"] = False
    decision = runner.analyze(
        summaries=summaries,
        paired_predict_ratio=paired,
        **kwargs,
    )
    assert not decision["passed"]
    assert decision["recommendation"].startswith("close_")


def test_worker_result_decoder_rejects_chatter_ambiguity():
    payload = {"arm": runner.VALIDATED}
    encoded = runner.WORKER_RESULT_PREFIX + json.dumps(payload)
    decoded, chatter = runner._decode_worker_result("before\n" + encoded)
    assert decoded == payload
    assert chatter == "before"
    with pytest.raises(RuntimeError, match="exactly one"):
        runner._decode_worker_result(encoded + "\n" + encoded)


def test_create_only_writer_refuses_existing_output(tmp_path):
    output = tmp_path / "result.json"
    runner._write_create_only(output, {"ok": True})
    assert json.loads(output.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        runner._write_create_only(output, {"ok": False})


def test_source_gate_requires_clean_pushed_main(monkeypatch):
    base = {
        "repository": str(runner.ROOT),
        "head": "same",
        "branch": "main",
        "origin_main": "same",
        "status_porcelain": "",
        "package_manifest_sha256": runner.EXPECTED_PACKAGE_MANIFEST,
        "support_sha256": runner.EXPECTED_SUPPORT_SHA256,
    }
    monkeypatch.setattr(runner, "_source_state", lambda: dict(base))
    monkeypatch.setattr(
        runner,
        "_sha256_file",
        lambda path: runner.EXPECTED_PROTOCOL_SHA256,
    )
    assert runner.require_clean_frozen_source()["branch"] == "main"
    base["status_porcelain"] = " M darkofit/sklearn_api.py"
    with pytest.raises(RuntimeError, match="clean"):
        runner.require_clean_frozen_source()


def test_formal_run_rejects_wrong_threads_and_output(tmp_path):
    args = argparse.Namespace(
        threads=1,
        output=runner.DEFAULT_OUTPUT,
        data_cache=runner.harness.DEFAULT_CACHE,
        temp_root=tmp_path,
    )
    with pytest.raises(ValueError, match="18 threads"):
        runner.run(args)

    args.threads = runner.EXPECTED_THREADS
    args.output = tmp_path / "wrong.json"
    with pytest.raises(ValueError, match="formal output"):
        runner.run(args)
