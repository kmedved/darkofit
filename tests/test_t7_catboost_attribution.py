import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import run_t7_catboost_attribution as runner
from benchmarks import analyze_t7_catboost_attribution as analyzer
from benchmarks.analyze_t7_catboost_attribution import depth_policy_arm


def test_arm_order_is_a_balanced_rotation():
    orders = [runner._arm_order(index) for index in range(len(runner.ARMS))]
    assert all(set(order) == set(runner.ARMS) for order in orders)
    for position in range(len(runner.ARMS)):
        assert {
            order[position] for order in orders
        } == set(runner.ARMS)


def test_spool_records_are_normalized_to_frozen_coordinate_order():
    coordinates = [(101, 0), (101, 1), (102, 0)]
    records = [
        {"task_id": 102, "fold": 0},
        {"task_id": 101, "fold": 1},
        {"task_id": 101, "fold": 0},
    ]
    assert runner._ordered_spool_records(records, coordinates) == [
        {"task_id": 101, "fold": 0},
        {"task_id": 101, "fold": 1},
        {"task_id": 102, "fold": 0},
    ]


def test_catboost_frame_preserves_numeric_and_canonicalizes_categories():
    X = pd.DataFrame(
        {"number": [1.0, 2.0], "category": ["x", None]}
    )
    result = runner._catboost_frame(X, [1])
    assert result["number"].tolist() == [1.0, 2.0]
    assert result["category"].tolist() == [
        "str:x",
        "__DARKOFIT_MISSING_CATEGORY__",
    ]


def test_t7_uses_only_spent_development_rows():
    registry, rows = runner._rows()
    assert len(rows) == 8
    assert {
        int(row["task_id"]) for row in registry["development_tasks"]
    } == set(rows)
    assert not (
        set(rows)
        & {
            int(row["task_id"])
            for row in registry["confirmation_tasks"]
        }
    )


def test_depth_policy_is_fixed_by_samples_per_feature():
    assert depth_policy_arm(99, 1) == "depth_4"
    assert depth_policy_arm(100, 1) == "default"
    assert depth_policy_arm(2_499, 1) == "default"
    assert depth_policy_arm(2_500, 1) == "depth_8"
    with pytest.raises(ValueError, match="positive integer"):
        depth_policy_arm(True, 1)
    with pytest.raises(ValueError, match="positive integer"):
        depth_policy_arm(100, "1")


def test_t7_worker_coordinate_must_match_frozen_schedule():
    registry, _rows = runner._rows()
    task_id = int(registry["development_tasks"][0]["task_id"])
    with pytest.raises(ValueError, match="frozen schedule"):
        runner.run_worker(task_id, runner.FOLDS[0], 1)


def test_catboost_warmup_target_is_nonconstant():
    pytest.importorskip("catboost")
    assert runner._warmup() > 0


def _raw():
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "benchmarks" / "t7_catboost_attribution_raw.json").read_text()
    )


def _rehash(raw):
    unsigned = {key: value for key, value in raw.items() if key != "raw_sha256"}
    raw["raw_sha256"] = runner._json_sha256(unsigned)


def _rebind_result(raw, result):
    result["behavior_sha256"] = runner._json_sha256(
        analyzer._behavior(result)
    )
    spool = next(
        row
        for row in raw["spool_records"]
        if int(row["task_id"]) == int(result["task_id"])
        and int(row["fold"]) == int(result["fold"])
    )
    payload = {
        "binding": raw["protocol"],
        "task_id": result["task_id"],
        "fold": result["fold"],
        "result_sha256": runner._json_sha256(result),
        "result": result,
    }
    spool["sha256"] = runner._json_sha256(payload)
    _rehash(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runner_sha256", "0" * 64),
        ("registry_sha256", "0" * 64),
        ("c2_raw_sha256", "0" * 64),
        ("source_head", "0" * 40),
        ("catboost_version", "0.0.0"),
    ],
)
def test_t7_raw_rejects_forged_protocol_binding(field, value):
    raw = _raw()
    raw["protocol"][field] = value
    _rehash(raw)
    with pytest.raises(RuntimeError, match="protocol"):
        analyzer._validate(raw)


def test_t7_raw_rejects_forged_schedule_and_spool_matrix():
    scheduled = _raw()
    scheduled["results"][0]["coordinate_index"] = 999
    scheduled["results"][0]["arm_order"] = list(runner._arm_order(999))
    _rehash(scheduled)
    with pytest.raises(RuntimeError, match="coordinate"):
        analyzer._validate(scheduled)

    duplicate_spool = _raw()
    duplicate_spool["spool_records"] = [
        copy.deepcopy(duplicate_spool["spool_records"][0])
        for _ in duplicate_spool["spool_records"]
    ]
    _rehash(duplicate_spool)
    with pytest.raises(RuntimeError, match="spool matrix"):
        analyzer._validate(duplicate_spool)


def test_t7_raw_rejects_forged_spool_digest():
    raw = _raw()
    raw["spool_records"][0]["sha256"] = "0" * 64
    _rehash(raw)
    with pytest.raises(RuntimeError, match="spool matrix"):
        analyzer._validate(raw)


def test_t7_raw_rejects_forged_inner_split_hash():
    raw = _raw()
    raw["results"][0]["inner_split"]["fit_index_sha256"] = "0" * 64
    _rehash(raw)
    with pytest.raises(RuntimeError, match="coordinate binding"):
        analyzer._validate(raw)


def test_t7_raw_rejects_forged_default_resolution():
    raw = _raw()
    result = raw["results"][0]
    default = next(
        arm for arm in result["arms"] if arm["arm"] == "default"
    )
    default["resolved_params"]["grow_policy"] = "Lossguide"
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="default resolution"):
        analyzer._validate(raw)


def test_t7_raw_rejects_forged_equivalent_arm_behavior():
    raw = _raw()
    result = raw["results"][0]
    plain = next(
        arm for arm in result["arms"] if arm["arm"] == "plain"
    )
    plain["test"]["rmse"] *= 0.5
    plain["test"]["prediction_sha256"] = "0" * 64
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="equivalent arm behavior"):
        analyzer._validate(raw)


def test_t7_raw_rejects_resigned_positive_metric_tampering():
    raw = _raw()
    result = raw["results"][0]
    result["arms"][0]["fit_seconds"] *= 2.0
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="frozen raw content"):
        analyzer._validate(raw)


def test_t7_raw_rejects_numeric_type_alias_in_overrides():
    raw = _raw()
    result = raw["results"][0]
    leaf = next(
        arm
        for arm in result["arms"]
        if arm["arm"] == "leaf10_no_backtracking"
    )
    leaf["overrides"]["leaf_estimation_iterations"] = 10.0
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="arm binding"):
        analyzer._validate(raw)


def test_t7_raw_rejects_coerced_schedule_and_model_scalars():
    raw = _raw()
    raw["schema_version"] = True
    _rehash(raw)
    with pytest.raises(RuntimeError, match="artifact name"):
        analyzer._validate(raw)

    raw = _raw()
    raw["results"][0]["coordinate_index"] = "0"
    _rehash(raw)
    with pytest.raises(RuntimeError, match="coordinate matrix"):
        analyzer._validate(raw)

    raw = _raw()
    result = raw["results"][0]
    result["arms"][0]["test"]["rows"] = str(
        result["arms"][0]["test"]["rows"]
    )
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="metric"):
        analyzer._validate(raw)

    raw = _raw()
    result = raw["results"][0]
    result["arms"][0]["resolved_params"]["iterations"] = 1000.0
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="default resolution"):
        analyzer._validate(raw)

    raw = _raw()
    raw["spool_records"][0]["task_id"] = str(
        raw["spool_records"][0]["task_id"]
    )
    _rehash(raw)
    with pytest.raises(RuntimeError, match="spool matrix"):
        analyzer._validate(raw)


def test_t7_raw_requires_complete_typed_runtime_and_source_schema():
    raw = _raw()
    raw["runtime"]["machine"]["logical_cpu_count"] = True
    _rehash(raw)
    with pytest.raises(RuntimeError, match="protocol structure"):
        analyzer._validate(raw)

    raw = _raw()
    raw["source"]["remotes"]["origin"] = 7
    _rehash(raw)
    with pytest.raises(RuntimeError, match="protocol structure"):
        analyzer._validate(raw)


def test_t7_raw_rejects_impossible_prediction_timing():
    raw = _raw()
    result = raw["results"][0]
    timing = result["arms"][0]["prediction_timing"]
    timing["median_seconds"] = 1.0
    timing["total_seconds"] = 1.0
    _rebind_result(raw, result)
    with pytest.raises(RuntimeError, match="timing"):
        analyzer._validate(raw)


def test_t7_json_requires_utf8_and_bounded_integers():
    with pytest.raises(RuntimeError, match="invalid T7 JSON"):
        runner._json_loads('{"task_id":1}'.encode("utf-16"), "T7")
    with pytest.raises(RuntimeError, match="invalid T7 JSON"):
        runner._json_loads('{"task_id":9223372036854775808}', "T7")
    with pytest.raises(RuntimeError, match="invalid T7 JSON"):
        analyzer._json_loads('{"task_id":1}'.encode("utf-16"), "T7")
    with pytest.raises(RuntimeError, match="invalid T7 JSON"):
        analyzer._json_loads('{"task_id":9223372036854775808}', "T7")


def test_t7_analysis_requires_the_frozen_c2_helper():
    snapshot = analyzer._dependency_snapshot()
    snapshot["c2_helper_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="C2 split helper"):
        analyzer._validate(_raw(), snapshot)


def test_t7_analysis_reads_frozen_dependencies_once(monkeypatch):
    original = Path.read_bytes
    read_counts = {
        runner.REGISTRY: 0,
        runner.C2_RAW: 0,
    }

    def tracked(path):
        if path in read_counts:
            read_counts[path] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    analyzer.analyze(_raw())
    assert read_counts == {
        runner.REGISTRY: 1,
        runner.C2_RAW: 1,
    }


def test_t7_cli_defaults_are_repository_anchored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_args = analyzer.parse_args([])
    assert analysis_args.input == analyzer.DEFAULT_INPUT
    assert analysis_args.output == analyzer.DEFAULT_OUTPUT
    assert analysis_args.markdown == analyzer.DEFAULT_MARKDOWN
    runner_args = runner.parse_args([])
    assert runner_args.output == runner.DEFAULT_OUTPUT
    assert runner_args.spool == runner.DEFAULT_SPOOL


def test_t7_spool_publish_is_create_only_and_cleans_failed_publish(
    tmp_path, monkeypatch
):
    binding = {"runner_sha256": "a" * 64}
    result = {"task_id": 123, "fold": 0}
    path = runner._spool_path(tmp_path, 123, 0)
    created, digest = runner._create_spool(path, binding, result)
    loaded, loaded_digest = runner._load_spool(
        path, binding, 123, 0
    )
    assert created == loaded == result
    assert digest == loaded_digest

    failed = runner._spool_path(tmp_path, 124, 0)

    def fail_link(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(runner.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        runner._create_spool(
            failed, binding, {"task_id": 124, "fold": 0}
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(f".{failed.name}.*.tmp"))

    symlink = runner._spool_path(tmp_path, 125, 0)
    symlink.symlink_to(path)
    with pytest.raises(RuntimeError, match="invalid T7 spool"):
        runner._load_spool(symlink, binding, 123, 0)


def test_t7_spool_loader_rejects_coerced_payload_identity(tmp_path):
    binding = {"runner_sha256": "a" * 64}
    result = {"task_id": 123, "fold": 0}
    path = runner._spool_path(tmp_path, 123, 0)
    runner._create_spool(path, binding, result)
    payload = json.loads(path.read_text())
    payload["task_id"] = "123"
    payload["spool_sha256"] = runner._json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "spool_sha256"
        }
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="spool binding"):
        runner._load_spool(path, binding, 123, 0)


def test_t7_spool_loader_uses_type_exact_binding_and_strict_json(tmp_path):
    binding = {"folds": [0, 1, 2]}
    result = {"task_id": 123, "fold": 0}
    path = runner._spool_path(tmp_path, 123, 0)
    runner._create_spool(path, binding, result)
    payload = json.loads(path.read_text())
    payload["binding"]["folds"][0] = False
    payload["spool_sha256"] = runner._json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "spool_sha256"
        }
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="spool binding"):
        runner._load_spool(path, binding, 123, 0)

    path.write_text('{"spool_sha256":"a","spool_sha256":"b"}')
    with pytest.raises(RuntimeError, match="invalid T7 spool record JSON"):
        runner._load_spool(path, binding, 123, 0)


def test_t7_spool_publish_reports_a_lost_race_as_resumed(
    tmp_path, monkeypatch
):
    binding = {"runner_sha256": "a" * 64}
    result = {"task_id": 123, "fold": 0}
    template = tmp_path / "template.json"
    runner._create_spool(template, binding, result)
    competing_bytes = template.read_bytes()
    output = runner._spool_path(tmp_path, 123, 0)

    def lose_publish_race(_source, destination):
        Path(destination).write_bytes(competing_bytes)
        raise FileExistsError("injected publish race")

    monkeypatch.setattr(runner.os, "link", lose_publish_race)
    loaded, _digest, published = runner._create_spool(
        output,
        binding,
        result,
        return_publish_state=True,
    )
    assert loaded == result
    assert published is False


def test_t7_runner_publish_rolls_back_if_temp_cleanup_fails(
    tmp_path, monkeypatch
):
    binding = {"runner_sha256": "a" * 64}
    original = Path.unlink

    def fail_temporary(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("injected temp cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary)
    spool = runner._spool_path(tmp_path, 123, 0)
    with pytest.raises(OSError, match="temp cleanup"):
        runner._create_spool(
            spool, binding, {"task_id": 123, "fold": 0}
        )
    assert not spool.exists()

    output = tmp_path / "raw.json"
    with pytest.raises(OSError, match="temp cleanup"):
        runner._create_output(output, b"{}\n")
    assert not output.exists()


@pytest.mark.parametrize("publisher", ["spool", "output"])
def test_t7_runner_rejects_substituted_temporary_inode(
    tmp_path, monkeypatch, publisher
):
    original = runner.os.link

    def substitute_temporary(source, destination):
        Path(source).unlink()
        Path(source).write_bytes(b"foreign\n")
        original(source, destination)

    monkeypatch.setattr(runner.os, "link", substitute_temporary)
    if publisher == "spool":
        output = runner._spool_path(tmp_path, 123, 0)
        with pytest.raises(RuntimeError, match="publish identity changed"):
            runner._create_spool(
                output,
                {"runner_sha256": "a" * 64},
                {"task_id": 123, "fold": 0},
            )
    else:
        output = tmp_path / "raw.json"
        with pytest.raises(RuntimeError, match="publish identity changed"):
            runner._create_output(output, b"ours\n")
    assert output.read_bytes() == b"foreign\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("publisher", ["spool", "output"])
def test_t7_runner_publish_removes_only_new_directories_on_failure(
    tmp_path, monkeypatch, publisher
):
    created_root = tmp_path / "new" / "nested"

    def fail_publish(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(runner.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        if publisher == "spool":
            runner._create_spool(
                runner._spool_path(created_root, 123, 0),
                {"runner_sha256": "a" * 64},
                {"task_id": 123, "fold": 0},
            )
        else:
            runner._create_output(created_root / "raw.json", b"{}\n")
    assert not (tmp_path / "new").exists()
    assert tmp_path.exists()


def test_t7_spool_loader_rejects_symlink_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    binding = {"runner_sha256": "a" * 64}
    result = {"task_id": 123, "fold": 0}
    path = runner._spool_path(real, 123, 0)
    runner._create_spool(path, binding, result)
    with pytest.raises(RuntimeError, match="symlink T7 spool directory"):
        runner._load_spool(
            runner._spool_path(alias, 123, 0),
            binding,
            123,
            0,
        )


def test_t7_publish_rejects_nested_symlink_ancestor_before_mkdir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    path = alias / "nested" / "raw.json"
    with pytest.raises(RuntimeError, match="symlink T7 output directory"):
        runner._create_output(path, b"{}\n")
    assert not (real / "nested").exists()


def test_t7_analysis_pair_publish_rolls_back_first_output(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    original = analyzer._atomic_create

    def fail_markdown(path, value):
        if path == markdown:
            raise OSError("injected second-output failure")
        return original(path, value)

    monkeypatch.setattr(analyzer, "_atomic_create", fail_markdown)
    with pytest.raises(OSError, match="second-output"):
        analyzer._atomic_create_pair(
            summary, b"{}\n", markdown, b"# result\n"
        )
    assert not summary.exists()
    assert not markdown.exists()


def test_t7_analysis_publish_rolls_back_if_temp_cleanup_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.json"
    original = Path.unlink

    def fail_temporary(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("injected temp cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary)
    with pytest.raises(OSError, match="temp cleanup"):
        analyzer._atomic_create(output, b"{}\n")
    assert not output.exists()


def test_t7_analysis_publish_removes_only_new_directories_on_failure(
    tmp_path, monkeypatch
):
    created_root = tmp_path / "new" / "nested"
    output = created_root / "summary.json"

    def fail_publish(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(analyzer.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        analyzer._atomic_create(output, b"{}\n")
    assert not (tmp_path / "new").exists()
    assert tmp_path.exists()


def test_t7_analysis_rejects_substituted_temporary_inode(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.json"
    original = analyzer.os.link

    def substitute_temporary(source, destination):
        Path(source).unlink()
        Path(source).write_bytes(b"foreign\n")
        original(source, destination)

    monkeypatch.setattr(analyzer.os, "link", substitute_temporary)
    with pytest.raises(RuntimeError, match="publish identity changed"):
        analyzer._atomic_create(output, b"ours\n")
    assert output.read_bytes() == b"foreign\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_t7_create_only_publish_preserves_competing_writer(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.json"

    def lose_publish_race(_source, destination):
        Path(destination).write_bytes(b"other writer\n")
        raise FileExistsError("injected publish race")

    monkeypatch.setattr(analyzer.os, "link", lose_publish_race)
    with pytest.raises(RuntimeError, match="existing output"):
        analyzer._atomic_create(output, b"ours\n")
    assert output.read_bytes() == b"other writer\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_t7_raw_output_publish_is_create_only(tmp_path):
    output = tmp_path / "raw.json"
    output.write_bytes(b"existing\n")
    with pytest.raises(RuntimeError, match="existing output"):
        runner._create_output(output, b"replacement\n")
    assert output.read_bytes() == b"existing\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_t7_pair_rollback_preserves_replacement_inode(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    original = analyzer._atomic_create

    def replace_then_fail(path, value):
        if path == markdown:
            summary.unlink()
            summary.write_bytes(b"other writer\n")
            raise OSError("injected second-output failure")
        return original(path, value)

    monkeypatch.setattr(analyzer, "_atomic_create", replace_then_fail)
    with pytest.raises(OSError, match="second-output"):
        analyzer._atomic_create_pair(
            summary, b"ours\n", markdown, b"# result\n"
        )
    assert summary.read_bytes() == b"other writer\n"
    assert not markdown.exists()


def test_t7_artifacts_are_hash_bound_and_nonpromotional(
    assert_analysis_equal,
):
    root = Path(__file__).resolve().parents[1]
    raw = _raw()
    raw_hash = raw.pop("raw_sha256")
    assert runner._json_sha256(raw) == raw_hash
    summary = json.loads(
        (
            root / "benchmarks" / "t7_catboost_attribution_summary.json"
        ).read_text()
    )
    summary_hash = summary.pop("summary_sha256")
    assert analyzer._json_sha256(summary) == summary_hash
    assert summary["frozen_research_candidates"] == ["depth_by_n_p"]
    assert set(summary["darkofit_all_arm_anchor_ratios"]) == {
        *runner.ARM_NAMES,
        "depth_by_n_p",
    }
    assert summary["default_change_authorized"] is False
    assert summary["analysis_evidence"] == {
        "raw_file_sha256": analyzer.FROZEN_RAW_FILE_SHA256,
        "raw_canonical_sha256": analyzer.FROZEN_RAW_CANONICAL_SHA256,
        "frozen_protocol_sha256": analyzer.FROZEN_PROTOCOL_SHA256,
        "frozen_runner_sha256": analyzer.FROZEN_RUNNER_SHA256,
        "current_runner_sha256": runner._sha256(
            Path(runner.__file__).resolve()
        ),
        "frozen_c2_helper_sha256": analyzer.FROZEN_C2_HELPER_SHA256,
        "current_c2_helper_sha256": runner._sha256(
            Path(analyzer.c2.__file__).resolve()
        ),
    }
    stored = {**summary, "summary_sha256": summary_hash}
    assert_analysis_equal(stored, analyzer.analyze(_raw()))
    assert (
        analyzer._markdown(stored)
        == (root / "benchmarks" / "t7_catboost_attribution_result.md").read_text()
    )
