import csv
import math
import os
import statistics
from pathlib import Path

import pytest

from benchmarks import analyze_t8_distributional_flagship as analysis


def test_t8_raw_boundary_and_shared_fingerprints_reproduce():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    rows = analysis._load(raw)
    assert len(rows) == 75
    assert {
        (row["dataset"], row["model"], int(row["seed"]))
        for row in rows
    } == {
        (dataset, model, seed)
        for dataset in analysis.DATASETS
        for model in analysis.MODELS
        for seed in analysis.SEEDS
    }


def _mutated_csv(tmp_path, predicate, updates):
    source = Path("benchmarks/t8_distributional_flagship_raw.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    row = next(item for item in rows if predicate(item))
    row.update(updates)
    output = tmp_path / "mutated.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"interval90_coverage": "1.5"}, "coverage"),
        ({"interval90_width": "-1"}, "interval width"),
        ({"fit_seconds": "-1"}, "timing"),
        ({"nll": "", "crps": ""}, "missing"),
        ({"n_train": "1"}, "data shape"),
    ],
)
def test_t8_rejects_invalid_distribution_rows(tmp_path, updates, message):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "ngboost",
        updates,
    )
    with pytest.raises(ValueError, match=message):
        analysis._load(path)


def test_t8_rejects_changed_csv_schema(tmp_path):
    source = Path("benchmarks/t8_distributional_flagship_raw.csv")
    text = source.read_text(encoding="utf-8")
    header, newline, body = text.partition("\n")
    path = tmp_path / "changed-schema.csv"
    path.write_text(f"{header},forged{newline}{body}", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV schema"):
        analysis._load(path)


def test_t8_rejects_non_utf8_csv(tmp_path):
    path = tmp_path / "utf16.csv"
    path.write_bytes(
        Path("benchmarks/t8_distributional_flagship_raw.csv")
        .read_text(encoding="utf-8")
        .encode("utf-16")
    )
    with pytest.raises(UnicodeDecodeError):
        analysis._load(path)


def test_t8_rejects_extra_csv_row_fields(tmp_path):
    source = Path("benchmarks/t8_distributional_flagship_raw.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1].append("forged-extra")
    path = tmp_path / "extra-field.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="row shape"):
        analysis._load(path)


def test_t8_rejects_distribution_metrics_on_interval_only_rows(tmp_path):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "lightgbm_quantile_pair",
        {"rmse_mu": "1.0", "nll": "1.0", "crps": "1.0"},
    )
    with pytest.raises(ValueError, match="interval-only"):
        analysis._load(path)


@pytest.mark.parametrize(
    ("model", "updates", "message"),
    [
        ("darkofit_gaussian_es_calibrated", {"best_iteration": ""}, "best iteration"),
        ("darkofit_gaussian_es_conformal", {"best_iteration": "0"}, "best iteration"),
        ("catboost_uncertainty", {"best_iteration": "1.5"}, "best iteration"),
        ("ngboost", {"best_iteration": "1"}, "unexpected best iteration"),
        ("lightgbm_quantile_pair", {"best_iteration": "1"}, "unexpected best iteration"),
        ("ngboost", {"reason": "fabricated"}, "unexpected reason"),
        ("lightgbm_quantile_pair", {"reason": ""}, "unexpected reason"),
    ],
)
def test_t8_rejects_wrong_iteration_or_reason_semantics(
    tmp_path,
    model,
    updates,
    message,
):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == model,
        updates,
    )
    with pytest.raises(ValueError, match=message):
        analysis._load(path)


def test_t8_rejects_wrong_iteration_budget_or_calibration_size(tmp_path):
    iteration = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "catboost_uncertainty",
        {"best_iteration": "119"},
    )
    with pytest.raises(ValueError, match="best iteration"):
        analysis._load(iteration)

    calibration = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == analysis.CONFORMAL_MODEL,
        {"calibration_n": "1"},
    )
    with pytest.raises(ValueError, match="calibration size"):
        analysis._load(calibration)


def test_t8_rejects_inconsistent_conditional_coverage(tmp_path):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "ngboost",
        {"cov90_by_sigma": "0/0/0/0/0"},
    )
    with pytest.raises(ValueError, match="conditional coverage"):
        analysis._load(path)


def test_t8_rejects_resigned_positive_metric_tampering(tmp_path):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "ngboost",
        {"fit_seconds": "1.0"},
    )
    with pytest.raises(ValueError, match="frozen raw content"):
        analysis._load(path)


def test_t8_sha256_validator_is_type_safe():
    assert not analysis._is_sha256(None)
    assert not analysis._is_sha256(0)
    assert analysis._is_sha256("a" * 64)


def test_t8_geomean_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="positive values"):
        analysis._geomean([1.0, math.nan])


def test_t8_integer_fields_reject_noncanonical_numeric_strings(tmp_path):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "ngboost",
        {"seed": "0.0"},
    )
    with pytest.raises(ValueError, match="integer"):
        analysis._load(path)


def test_t8_integer_fields_reject_signed_64_bit_overflow(tmp_path):
    path = _mutated_csv(
        tmp_path,
        lambda row: row["model"] == "ngboost",
        {"seed": "9223372036854775808"},
    )
    with pytest.raises(ValueError, match="out-of-range integer"):
        analysis._load(path)


def test_t8_rejects_noncanonical_campaign_row_order(tmp_path):
    source = Path("benchmarks/t8_distributional_flagship_raw.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1], rows[2] = rows[2], rows[1]
    path = tmp_path / "reordered.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="row order"):
        analysis._load(path)


def test_t8_output_publish_is_create_only_and_race_safe(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.md"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing T8 output"):
        analysis._atomic_write_text(output, "replacement\n")
    assert output.read_text(encoding="utf-8") == "existing\n"

    raced = tmp_path / "raced.md"

    def lose_publish_race(_source, destination):
        Path(destination).write_text("other writer\n", encoding="utf-8")
        raise FileExistsError("injected publish race")

    monkeypatch.setattr(analysis.os, "link", lose_publish_race)
    with pytest.raises(RuntimeError, match="existing T8 output"):
        analysis._atomic_write_text(raced, "ours\n")
    assert raced.read_text(encoding="utf-8") == "other writer\n"
    assert not list(tmp_path.glob(f".{raced.name}.*.tmp"))


def test_t8_output_rolls_back_if_temp_cleanup_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.md"
    original = Path.unlink

    def fail_temporary(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("injected temp cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary)
    with pytest.raises(OSError, match="temp cleanup"):
        analysis._atomic_write_text(output, "result\n")
    assert not output.exists()


def test_t8_output_removes_only_new_directories_on_failure(
    tmp_path, monkeypatch
):
    created_root = tmp_path / "new" / "nested"
    output = created_root / "result.md"

    def fail_publish(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(analysis.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        analysis._atomic_write_text(output, "result\n")
    assert not (tmp_path / "new").exists()
    assert tmp_path.exists()


def test_t8_output_rejects_substituted_temporary_inode(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.md"
    original = analysis.os.link
    original_fdopen = analysis.os.fdopen
    foreign_temporaries = []
    handles = []

    def track_fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def substitute_temporary(source, destination):
        source = Path(source)
        source.unlink()
        source.write_text("foreign\n", encoding="utf-8")
        foreign_temporaries.append(source)
        original(source, destination)

    monkeypatch.setattr(analysis.os, "fdopen", track_fdopen)
    monkeypatch.setattr(analysis.os, "link", substitute_temporary)
    with pytest.raises(RuntimeError, match="publish identity changed"):
        analysis._atomic_write_text(output, "ours\n")
    assert output.read_text(encoding="utf-8") == "foreign\n"
    assert len(foreign_temporaries) == 1
    assert foreign_temporaries[0].read_text(encoding="utf-8") == "foreign\n"
    assert os.path.samefile(foreign_temporaries[0], output)
    assert handles and all(handle.closed for handle in handles)


def test_t8_output_ignores_post_commit_close_error(tmp_path, monkeypatch):
    output = tmp_path / "result.md"
    original_fdopen = analysis.os.fdopen
    wrapped = []

    class CloseError:
        def __init__(self, handle):
            self.handle = handle
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            self.close_calls += 1
            self.handle.close()
            raise OSError("injected close failure")

    def wrap_fdopen(*args, **kwargs):
        handle = CloseError(original_fdopen(*args, **kwargs))
        wrapped.append(handle)
        return handle

    monkeypatch.setattr(analysis.os, "fdopen", wrap_fdopen)
    analysis._atomic_write_text(output, "result\n")
    assert output.read_text(encoding="utf-8") == "result\n"
    assert len(wrapped) == 1
    assert wrapped[0].handle.closed
    assert wrapped[0].close_calls == 1


def test_t8_output_rejects_nested_symlink_ancestor_before_mkdir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "nested" / "result.md"
    with pytest.raises(RuntimeError, match="symlink T8 output directory"):
        analysis._atomic_write_text(output, "result\n")
    assert not (real / "nested").exists()


def test_t8_evidence_snapshot_binds_protocol_and_raw_report():
    _analyzer, runner, protocol, raw_markdown = analysis._evidence_snapshot()
    assert runner == analysis.FROZEN_RUNNER_SHA256
    assert protocol == analysis.FROZEN_PROTOCOL_SHA256
    assert raw_markdown == analysis.FROZEN_RAW_MARKDOWN_SHA256


def test_t8_output_cannot_alias_frozen_evidence(tmp_path):
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    protocol = Path("benchmarks/t8_distributional_flagship_protocol.md")
    raw_hash = analysis._sha256(raw)
    protocol_hash = analysis._sha256(protocol)

    with pytest.raises(RuntimeError, match="protected T8 output"):
        analysis._validate_output_path(raw, raw)
    with pytest.raises(RuntimeError, match="protected T8 output"):
        analysis._validate_output_path(protocol, raw)
    with pytest.raises(RuntimeError, match="protected T8 output"):
        analysis._validate_output_path(analysis.RUNNER, raw)

    symlink = tmp_path / "result.md"
    symlink.symlink_to(raw.resolve())
    with pytest.raises(RuntimeError, match="symlink T8 output"):
        analysis._validate_output_path(symlink, raw)

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    alias_directory = tmp_path / "alias"
    alias_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink T8 output directory"):
        analysis._validate_output_path(alias_directory / "result.md", raw)

    assert analysis._sha256(raw) == raw_hash
    assert analysis._sha256(protocol) == protocol_hash


def test_t8_cli_defaults_are_repository_anchored():
    args = analysis.parse_args([])
    assert args.csv == analysis.FROZEN_RAW
    assert args.output == analysis.DEFAULT_OUTPUT
    assert args.csv.is_absolute()
    assert args.output.is_absolute()


def test_t8_result_is_deterministically_regenerated():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    result = Path("benchmarks/t8_distributional_flagship_result.md")
    rows, raw_sha256 = analysis._load(raw, with_sha256=True)
    assert raw_sha256 == analysis._sha256(raw)
    assert result.read_text(encoding="utf-8") == analysis._render(
        rows, raw_sha256=raw_sha256
    )
    report = result.read_text(encoding="utf-8")
    assert "| Dataset | Model | Coverage | Width | Absolute gap |" in report
    assert (
        "| Model | Mean coverage | Geomean width / conformal | "
        "Mean absolute coverage gap |"
    ) in report


def test_t8_conformal_coverage_width_summary_is_exact():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    summaries = analysis._summarize(analysis._load(raw))
    conformal = [
        summaries[(dataset, analysis.CONFORMAL_MODEL)]
        for dataset in analysis.DATASETS
    ]
    parametric = [
        summaries[(dataset, "darkofit_gaussian_es_calibrated")]
        for dataset in analysis.DATASETS
    ]
    conformal_gap = statistics.mean(
        abs(row["coverage"] - 0.9) for row in conformal
    )
    parametric_gap = statistics.mean(
        abs(row["coverage"] - 0.9) for row in parametric
    )
    width_ratio = analysis._geomean(
        left["width"] / right["width"]
        for left, right in zip(conformal, parametric)
    )
    assert conformal_gap == pytest.approx(0.010966164458409079)
    assert parametric_gap == pytest.approx(0.012935432181253769)
    assert width_ratio == pytest.approx(0.9830526057278705)
