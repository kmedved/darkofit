#!/usr/bin/env python3
"""Run the frozen Wave-3 B-archive canonical-section feasibility screen."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import numpy as np

try:
    from . import paired_evidence_contract as paired
    from . import run_m3b_ensemble_v3 as m3b
    from .analyze_ensemble_archive_components import analyze_archive
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3b_ensemble_v3 as m3b
    from analyze_ensemble_archive_components import analyze_archive


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
PROTOCOL_PATH = BENCH_DIR / "barchive_v1_protocol.md"
CONTRACT_PATH = BENCH_DIR / "barchive_v1_contract.json"
RAW_PATH = BENCH_DIR / "barchive_v1_raw.json"
TERMINAL_PATH = BENCH_DIR / "barchive_v1_terminal.json"
ANALYZER_PATH = BENCH_DIR / "analyze_barchive_v1.py"
FREEZER_PATH = BENCH_DIR / "freeze_barchive_v1.py"
DEFAULT_PANEL_CACHE = m3b.DEFAULT_PANEL_CACHE
M3B_R3_CONTRACT_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_contract.json"
M3B_R3_QUALITY_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_quality.json"
M3B_R3_RESULT_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_result.json"
M3B_VS_SINGLE_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"

CONTRACT_NAME = "wave3_barchive_v1_20260721"
SCHEMA_VERSION = 1
MODEL_SOURCE_HEAD = "858ac14c30e280491d7bd5232da56f7050561782"
FROZEN_RUNTIME = {
    "python": "3.11.8",
    "numpy": "2.2.6",
    "scikit_learn": "1.7.1",
    "numba": "0.61.2",
    "pandas": "2.2.3",
    "scipy": "1.15.1",
}
THREADS = paired.CONTRACT_THREADS
WORKER_PREFIX = "BARCHIVE_V1_RESULT="
SINGLE = m3b.SINGLE
COMBINED = m3b.COMBINED
ALLOWED_CANONICAL_ARRAYS = frozenset(
    {
        "bin__border_offsets",
        "bin__borders_flat",
        "bin__block_widths",
        "bin__n_bins",
        "prep__cat_features",
        "prep__feature_map",
        "prep__num_features",
    }
)
ALLOWED_CANONICAL_HEADER_FIELDS = frozenset(
    {
        "feature_names_in",
        "n_input_features",
        "prep",
        "random_state",
    }
)

BOUND_PATHS = {
    "protocol": "benchmarks/barchive_v1_protocol.md",
    "runner": "benchmarks/run_barchive_v1.py",
    "analyzer": "benchmarks/analyze_barchive_v1.py",
    "freezer": "benchmarks/freeze_barchive_v1.py",
    "tests": "tests/test_barchive_v1.py",
    "component_analyzer": "benchmarks/analyze_ensemble_archive_components.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
    "m3b_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "m3b_r3_contract": "benchmarks/m3b_ensemble_v3_r3_contract.json",
    "m3b_r3_quality": "benchmarks/m3b_ensemble_v3_r3_quality.json",
    "m3b_r3_result": "benchmarks/m3b_ensemble_v3_r3_result.json",
    "m3b_vs_single_readout": (
        "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
    ),
    "serialization": "darkofit/serialization.py",
    "sklearn_api": "darkofit/sklearn_api.py",
    "private_ensemble_tests": "tests/test_private_ensemble_v3.py",
}
PINNED_SOURCE_PATHS = frozenset(
    {
        "darkofit/serialization.py",
        "darkofit/sklearn_api.py",
        "tests/test_private_ensemble_v3.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        m3b._to_builtin(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decision_rules() -> dict[str, Any]:
    return {
        "median_effective_archive_to_single_at_most": 4.0,
        "expected_case_count": 13,
        "expected_numeric_target_free_cases": 11,
        "allowed_canonical_arrays": sorted(ALLOWED_CANONICAL_ARRAYS),
        "allowed_canonical_header_fields": sorted(ALLOWED_CANONICAL_HEADER_FIELDS),
        "member_local_effective_size": "current_ensemble_bytes",
        "numeric_target_free_effective_size": (
            "canonical_preprocessor_simulated_bytes"
        ),
        "advance_disposition": "advance_to_canonical_serializer_prototype",
        "close_disposition": "close_barchive_nominate_fused_lane_dispatch",
    }


def expected_shared_preprocessing() -> dict[str, str]:
    return {
        spec["case_id"]: (
            "member_local"
            if spec.get("dataset") in {"categorical_reg", "categorical_multiclass"}
            else "numeric_target_free"
        )
        for spec in m3b.case_specs()
    }


def execution_contract() -> dict[str, Any]:
    return {
        "paired_execution_contract": paired.CONTRACT_VERSION,
        "fresh_worker_per_case": True,
        "failed_attempt_terminal": True,
        "partial_rows_published": False,
        "single_reference": m3b.arm_config(SINGLE),
        "combined": m3b.arm_config(COMBINED),
        "ensemble_members": m3b.MEMBERS,
        "iterations": m3b.ITERATIONS,
        "early_stopping_rounds": m3b.PATIENCE,
        "random_state": m3b.RANDOM_STATE,
        "threads": THREADS,
        "frozen_runtime": FROZEN_RUNTIME,
        "simulation_loadable": False,
    }


def runtime_versions() -> dict[str, str]:
    import numba
    import pandas
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "numba": numba.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
    }


def assert_frozen_runtime() -> dict[str, str]:
    observed = runtime_versions()
    if observed != FROZEN_RUNTIME:
        raise RuntimeError(
            "B-archive requires its frozen runtime: "
            + json.dumps(
                {"expected": FROZEN_RUNTIME, "observed": observed},
                sort_keys=True,
            )
        )
    return observed


def claim_contract() -> dict[str, Any]:
    return {
        "tier": "E",
        "spent_development_evidence": True,
        "m3b_r3_amended": False,
        "serializer_implementation_authorized_before_advance": False,
        "public_or_default_change_authorized": False,
        "b3_authorized": False,
        "m2_authorized": False,
        "tabarena_or_m4_authorized": False,
        "fresh_confirmation_authorized": False,
        "lockbox_access_authorized": False,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def m3b_r3_lineage() -> dict[str, Any]:
    result = json.loads(M3B_R3_RESULT_PATH.read_text(encoding="utf-8"))
    readout = json.loads(M3B_VS_SINGLE_PATH.read_text(encoding="utf-8"))
    return {
        "contract": _artifact_record(M3B_R3_CONTRACT_PATH),
        "quality": _artifact_record(M3B_R3_QUALITY_PATH),
        "result": _artifact_record(M3B_R3_RESULT_PATH),
        "vs_single_readout": _artifact_record(M3B_VS_SINGLE_PATH),
        "frozen_disposition": "close_b1_b2_preserve_existing_opt_in",
        "combined_beats_matched_single_cases": 13,
        "combined_case_count": 13,
        "combined_median_archive_to_single": 5.534767493867151,
        "archive_to_single_limit": 4.0,
        "serialization_authorized": False,
        "sports_primary_scope": readout.get("sports_primary_scope"),
        "lineage_valid": (
            result.get("disposition") == "close_b1_b2_preserve_existing_opt_in"
            and result.get("candidates", {})
            .get(COMBINED, {})
            .get("resources", {})
            .get("median_archive_to_single")
            == 5.534767493867151
            and readout.get("finding", {}).get("combined_beats_single_all_cases")
            is True
            and readout.get("finding", {}).get("combined_case_count") == 13
            and readout.get("finding", {}).get("serialization_authorized") is False
            and readout.get("amends_frozen_m3b_result") is False
            and str(readout.get("sports_primary_scope", "")).startswith(
                "player-disjoint"
            )
        ),
    }


def _git_blob(relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{MODEL_SOURCE_HEAD}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def _bound_file_is_exact(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    relative = record.get("path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        return False
    if relative in PINNED_SOURCE_PATHS:
        payload = _git_blob(relative)
        return (
            payload is not None
            and len(payload) == size
            and hashlib.sha256(payload).hexdigest() == digest
        )
    path = ROOT / relative
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == size
        and _sha256(path) == digest
    )


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("B-archive contract must be a regular file")
    contract = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise RuntimeError("B-archive contract must be a JSON object")
    bound = contract.get("bound_files")
    historical = json.loads(M3B_R3_CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("name") != CONTRACT_NAME
        or contract.get("contract_frozen") is not True
        or contract.get("outcomes_opened") is not False
        or contract.get("sources", {}).get("darkofit") != MODEL_SOURCE_HEAD
        or contract.get("decision_rules") != decision_rules()
        or contract.get("execution") != execution_contract()
        or contract.get("claims") != claim_contract()
        or contract.get("cases") != historical.get("cases")
        or contract.get("case_manifests") != historical.get("case_manifests")
        or contract.get("case_fingerprints") != historical.get("case_fingerprints")
        or contract.get("expected_shared_preprocessing")
        != expected_shared_preprocessing()
        or contract.get("panel_cache") != historical.get("panel_cache")
        or contract.get("m3b_r3_lineage") != m3b_r3_lineage()
        or contract["m3b_r3_lineage"].get("lineage_valid") is not True
        or not isinstance(bound, Mapping)
        or set(bound) != set(BOUND_PATHS)
        or any(
            not isinstance(record, Mapping)
            or record.get("path") != BOUND_PATHS[name]
            or not _bound_file_is_exact(record)
            for name, record in bound.items()
        )
    ):
        raise RuntimeError("B-archive frozen contract is invalid or drifted")
    return contract


def _assert_panel_cache(path: Path, contract: Mapping[str, Any]) -> None:
    expected = contract["panel_cache"]
    required = ROOT / expected["contract_path"]
    if (
        path != required.resolve()
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected["bytes"]
        or _sha256(path) != expected["sha256"]
    ):
        raise RuntimeError("B-archive sports panel cache is invalid or drifted")


def _activate_source(source: Path, expected_head: str) -> str:
    return m3b._activate_source(source, expected_head)


def _prediction_hash(value: Any) -> str:
    return m3b._prediction_sha256(value)


def _feature_schema_payload(model: Any) -> dict[str, Any]:
    prep = model.model_.prep_
    categories = [
        [(type(value).__name__, repr(value)) for value in column]
        for column in getattr(prep, "cat_categories_", [])
    ]
    return m3b._to_builtin(
        {
            "n_features_in": int(model.n_features_in_),
            "feature_names_in": getattr(model, "feature_names_in_", None),
            "ordinal_features": getattr(model, "ordinal_features_", ()),
            "classes": getattr(model, "classes_", None),
            "prep_num_features": prep.num_features_,
            "prep_cat_features": prep.cat_features_,
            "prep_feature_map": prep.feature_map_,
            "prep_n_input_features": int(prep.n_input_features_),
            "cat_categories": categories,
            "bin_borders": prep.binner_._borders_flat_,
            "bin_border_offsets": prep.binner_._border_offsets_,
            "bin_n_bins": prep.binner_.n_bins_,
            "bin_block_widths": prep.binner_._block_widths_,
        }
    )


def _metadata_payload(model: Any) -> dict[str, Any]:
    return m3b._to_builtin(
        {
            "params": model.get_params(deep=False),
            "fitted": paired.fitted_model_metadata(model),
            "ensemble": getattr(model, "ensemble_metadata_", None),
            "best_n_estimators": int(model.best_n_estimators_),
            "learning_rate": float(model.learning_rate_),
        }
    )


def _roundtrip_record(model: Any, X_test: Any, archive: Path) -> dict[str, Any]:
    prediction = np.asarray(model.predict(X_test))
    probability = (
        None
        if not hasattr(model, "predict_proba")
        else np.asarray(model.predict_proba(X_test))
    )
    schema = _feature_schema_payload(model)
    metadata = _metadata_payload(model)
    model.save_model(archive)
    original_bytes = archive.read_bytes()
    restored = model.__class__.load_model(archive)
    restored_prediction = np.asarray(restored.predict(X_test))
    restored_probability = (
        None if probability is None else np.asarray(restored.predict_proba(X_test))
    )
    resaved = archive.with_name(archive.stem + "-resaved.npz")
    restored.save_model(resaved)
    checks = {
        "prediction_exact": np.array_equal(prediction, restored_prediction),
        "probability_exact": (
            probability is None or np.array_equal(probability, restored_probability)
        ),
        "feature_schema_exact": (schema == _feature_schema_payload(restored)),
        "metadata_exact": metadata == _metadata_payload(restored),
        "resave_bytes_exact": original_bytes == resaved.read_bytes(),
    }
    if not all(checks.values()):
        raise RuntimeError("B-archive input safe-roundtrip invariant failed")
    return {
        "archive_bytes": len(original_bytes),
        "archive_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "prediction_sha256": _prediction_hash(prediction),
        "probability_sha256": (
            None if probability is None else _prediction_hash(probability)
        ),
        "feature_schema_sha256": _json_sha256(schema),
        "metadata_sha256": _json_sha256(metadata),
        "checks": checks,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    assert_frozen_runtime()
    runtime_before = paired.assert_worker_contract(THREADS)
    contract = load_contract(Path(args.contract))
    source = Path(args.source).expanduser().resolve()
    panel_cache = Path(args.panel_cache).expanduser().resolve()
    _assert_panel_cache(panel_cache, contract)
    implementation_root = _activate_source(source, contract["sources"]["darkofit"])
    specs = {spec["case_id"]: spec for spec in m3b.case_specs()}
    if args.case_id not in specs:
        raise ValueError(f"unknown B-archive case: {args.case_id}")
    spec = specs[args.case_id]
    data = m3b.build_case(spec, panel_cache)
    fingerprints = m3b.case_fingerprints(spec, data)
    if fingerprints != contract["case_fingerprints"][args.case_id]:
        raise RuntimeError("B-archive case fingerprint drifted")

    single = m3b._build_estimator(spec, SINGLE)
    combined = m3b._build_estimator(spec, COMBINED)
    m3b._fit_model(single, spec, SINGLE, data)
    m3b._fit_model(combined, spec, COMBINED, data)
    implementation_path = str(Path(inspect.getfile(combined.__class__)).resolve())
    if not Path(implementation_path).is_relative_to(Path(implementation_root)):
        raise RuntimeError("B-archive estimator imported outside pinned source")
    if paired.fitted_model_metadata(single)["resolved_thread_counts"] != [THREADS]:
        raise RuntimeError("B-archive single fitted thread count drifted")
    if paired.fitted_model_metadata(combined)["resolved_thread_counts"] != [THREADS]:
        raise RuntimeError("B-archive combined fitted thread count drifted")

    cache_dir = Path(os.environ["NUMBA_CACHE_DIR"])
    with tempfile.TemporaryDirectory(
        prefix="darkofit-barchive-v1-", dir=cache_dir
    ) as directory:
        directory = Path(directory)
        single_path = directory / "single.npz"
        ensemble_path = directory / "combined.npz"
        single_record = _roundtrip_record(single, data["X_test"], single_path)
        ensemble_record = _roundtrip_record(combined, data["X_test"], ensemble_path)
        census = analyze_archive(
            ensemble_path,
            single_path=single_path,
            gate=decision_rules()["median_effective_archive_to_single_at_most"],
        )

    canonical = census["canonical_preprocessor"]
    provenance = combined.ensemble_metadata_["shared_preprocessing"]
    eligible = bool(canonical["eligible"])
    effective_bytes = (
        int(canonical["simulated_archive_bytes"])
        if eligible
        else int(ensemble_record["archive_bytes"])
    )
    if eligible and canonical["array_names"] != sorted(ALLOWED_CANONICAL_ARRAYS):
        raise RuntimeError("B-archive canonical array set drifted")
    if eligible and canonical["header_fields"] != sorted(
        ALLOWED_CANONICAL_HEADER_FIELDS
    ):
        raise RuntimeError("B-archive canonical header set drifted")
    if provenance == "numeric_target_free" and not eligible:
        raise RuntimeError("B-archive numeric target-free section is incomplete")
    if provenance != "numeric_target_free" and eligible:
        raise RuntimeError("B-archive member-local case became canonical")

    runtime_after = paired.assert_worker_contract(THREADS)
    return m3b._to_builtin(
        {
            "case_id": args.case_id,
            "domain": spec["domain"],
            "task": spec["task"],
            **fingerprints,
            "implementation_path": implementation_path,
            "shared_preprocessing": provenance,
            "single": single_record,
            "combined": ensemble_record,
            "current_archive_to_single": (
                ensemble_record["archive_bytes"] / single_record["archive_bytes"]
            ),
            "canonical_preprocessor": canonical,
            "optimistic_all_exact_entries": census["optimistic_all_exact_entries"],
            "components": census["components"],
            "effective_candidate_archive_bytes": effective_bytes,
            "effective_candidate_archive_to_single": (
                effective_bytes / single_record["archive_bytes"]
            ),
            "effective_uses_only_canonical_preprocessor": eligible,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
        }
    )


def _parse_worker_output(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("B-archive worker did not emit exactly one result")
    return json.loads(matches[0])


def _run_worker(
    *,
    contract: Path,
    source: Path,
    panel_cache: Path,
    cache_dir: Path,
    case_id: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--contract",
        str(contract),
        "--source",
        str(source),
        "--panel-cache",
        str(panel_cache),
        "--case-id",
        case_id,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=paired.fixed_worker_environment(cache_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"B-archive worker failed for {case_id}:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return _parse_worker_output(completed.stdout)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paired.write_create_only(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _publish_terminal(
    *,
    error: BaseException,
    contract: Path,
    source_state: Mapping[str, Any],
    harness_state: Mapping[str, Any],
    completed_rows: int,
) -> NoReturn:
    if RAW_PATH.exists() or RAW_PATH.is_symlink():
        raise RuntimeError("cannot publish terminal after B-archive raw artifact")
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "name": CONTRACT_NAME,
        "status": "terminal_failure",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": _sha256(contract),
        "source_state": source_state,
        "harness_state": harness_state,
        "completed_rows_discarded": int(completed_rows),
        "partial_rows_published": False,
        "rerun_allowed": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _write_json_create_only(TERMINAL_PATH, terminal)
    raise RuntimeError(f"B-archive v1 terminated; see {TERMINAL_PATH}") from error


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    frozen_runtime = assert_frozen_runtime()
    contract_path = Path(args.contract).expanduser().resolve()
    contract = load_contract(contract_path)
    source = Path(args.source).expanduser().resolve()
    panel_cache = Path(args.panel_cache).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if contract_path != CONTRACT_PATH.resolve():
        raise RuntimeError("B-archive execution requires the canonical contract")
    if output != RAW_PATH.resolve():
        raise RuntimeError("B-archive execution requires the canonical raw path")
    _assert_panel_cache(panel_cache, contract)
    if cache_dir.is_symlink():
        raise RuntimeError("B-archive cache directory cannot be a symlink")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise RuntimeError("B-archive cache path must be a directory")
    if (
        output.exists()
        or output.is_symlink()
        or TERMINAL_PATH.exists()
        or TERMINAL_PATH.is_symlink()
    ):
        raise RuntimeError("B-archive v1 artifact or terminal already exists")
    harness_before = m3b.git_state(ROOT)
    source_before = m3b.git_state(source)
    if harness_before["status"]:
        raise RuntimeError("B-archive harness tree must be clean")
    if (
        source_before["status"]
        or source_before["head"] != contract["sources"]["darkofit"]
    ):
        raise RuntimeError("B-archive source tree must be exact and clean")
    observed = m3b.expected_case_manifests(panel_cache)
    fingerprints = {
        case_id: record["fingerprints"] for case_id, record in observed.items()
    }
    if (
        observed != contract["case_manifests"]
        or fingerprints != contract["case_fingerprints"]
    ):
        raise RuntimeError("B-archive preflight case manifests drifted")

    rows = []
    try:
        for spec in m3b.case_specs():
            rows.append(
                _run_worker(
                    contract=contract_path,
                    source=source,
                    panel_cache=panel_cache,
                    cache_dir=cache_dir,
                    case_id=spec["case_id"],
                )
            )
        if m3b.git_state(ROOT) != harness_before:
            raise RuntimeError("B-archive harness changed during execution")
        if m3b.git_state(source) != source_before:
            raise RuntimeError("B-archive source changed during execution")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "name": CONTRACT_NAME,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_path": str(contract_path.relative_to(ROOT)),
            "contract_sha256": _sha256(contract_path),
            "source_state": source_before,
            "harness_state": harness_before,
            "runtime_versions": frozen_runtime,
            "panel_cache": {
                "contract_path": str(panel_cache.relative_to(ROOT)),
                "bytes": panel_cache.stat().st_size,
                "sha256": _sha256(panel_cache),
            },
            "case_fingerprints": fingerprints,
            "rows": rows,
        }
        digest = _write_json_create_only(output, artifact)
    except BaseException as exc:
        _publish_terminal(
            error=exc,
            contract=contract_path,
            source_state=source_before,
            harness_state=harness_before,
            completed_rows=len(rows),
        )
    return {"output": str(output), "sha256": digest, "rows": len(rows)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--case-id")
    parser.add_argument("--output", type=Path, default=RAW_PATH)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/tmp/darkofit-barchive-v1-cache")
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.source is None:
        raise ValueError("--source is required")
    if args.worker:
        if not args.case_id:
            raise ValueError("--case-id is required for a worker")
        print(
            WORKER_PREFIX
            + json.dumps(run_worker(args), sort_keys=True, allow_nan=False)
        )
        return 0
    print(json.dumps(run_parent(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
