#!/usr/bin/env python3
"""Run the one-shot T7b automatic-depth spent-sports successor."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import numpy as np
from sklearn.metrics import mean_squared_error

try:
    from . import paired_evidence_contract as paired
    from . import run_m3a_wave1 as m3a
    from . import run_m3b_ensemble_v3 as m3b
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3a_wave1 as m3a
    import run_m3b_ensemble_v3 as m3b


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "t7b-automatic-depth-spent-sports-v1-20260722"
CONTRACT_PATH = ROOT / "benchmarks" / "t7b_automatic_depth_sports_v1_contract.json"
PROTOCOL_PATH = ROOT / "benchmarks" / "t7b_automatic_depth_sports_v1_protocol.md"
ANALYZER_PATH = ROOT / "benchmarks" / "analyze_t7b_automatic_depth_sports_v1.py"
FREEZER_PATH = ROOT / "benchmarks" / "freeze_t7b_automatic_depth_sports_v1.py"
THREADS = paired.CONTRACT_THREADS
ITERATIONS = 600
PATIENCE = 30
RANDOM_STATE = 4
SEASONS = (2014, 2015, 2016)
TARGETS = ("minutes_per_game", "game_score", "box_plus_minus")
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20_260_722
CONTROL = "control"
CANDIDATE = "candidate"
ARMS = (CONTROL, CANDIDATE)
CONTROL_HEAD = "e23d2b164f10374b1c0e02521c33fc96d48980da"
CONTROL_TREE = "227e80df4c2a3761927b3682cc8ee8ed1edcf471"
CANDIDATE_HEAD = "41e948f0c53b1d124e16071a7fa66eba47d084d3"
CANDIDATE_TREE = "219baa9b0e0c6cc642163e44b2fef56dd86f40b7"
DEPTH_RULE = "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8"
WORKER_PREFIX = "T7B_DEPTH_SPORTS_RESULT="
CONTRACT_RELATIVE = "benchmarks/t7b_automatic_depth_sports_v1_contract.json"
HISTORICAL_M3B_CONTRACT_PATH = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_r3_contract.json"
)
CANDIDATE_CHANGED_FILES = [
    "darkofit/booster.py",
    "tests/test_darkofit.py",
    "tests/test_t7b_automatic_depth_policy.py",
]
GENERAL_PRECONDITIONS = {
    "invariants": (
        "benchmarks/t7b_automatic_depth_v1_invariants_20260722.json",
        "02362e5d7080c155add0846a58b6960db997bd29a0374e936a16a5a5364e5aff",
    ),
    "m5": (
        "benchmarks/t7b_automatic_depth_v1_m5_20260722.json",
        "1d3eac70f81babeb628850cf19844d7b4c590c6df67ded723fcf7caba019bca1",
    ),
    "launch": (
        "benchmarks/t7b_automatic_depth_v1_m6_inspection1_launch_manifest_20260722.json",
        "7eb95710c761f0682c00cf4b5971233089c70e654c5e5adc316d5388d933dc46",
    ),
    "raw": (
        "benchmarks/t7b_automatic_depth_v1_m6_inspection1_raw_20260722.csv",
        "e8e651459fafdea7ace0d298ccedd2c8d87145b945928111d475a007b955bafe",
    ),
    "result": (
        "benchmarks/t7b_automatic_depth_v1_m6_inspection1_result_20260722.json",
        "7af0c480221b5886c7bbf41f810147663d9da6e2c4171a70bc9db3a431eebb28",
    ),
    "manifest": (
        "benchmarks/t7b_automatic_depth_v1_m6_inspection1_result_20260722.json.manifest.json",
        "dbb47702f4e7992f34e653ea1155a8638e4e1945dbda0da1eb582345c73c32c7",
    ),
    "terminal": (
        "benchmarks/t7b_automatic_depth_v1_m6_inspection1_terminal_attestation_20260722.json",
        "b925aab09fdd71ca0f8887e1d3a4023c20412b2eefc337f2a2a7c1d5a267f598",
    ),
}
BOUND_PATHS = {
    "protocol": "benchmarks/t7b_automatic_depth_sports_v1_protocol.md",
    "runner": "benchmarks/run_t7b_automatic_depth_sports_v1.py",
    "analyzer": "benchmarks/analyze_t7b_automatic_depth_sports_v1.py",
    "freezer": "benchmarks/freeze_t7b_automatic_depth_sports_v1.py",
    "tests": "tests/test_t7b_automatic_depth_sports_v1.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "m3b_loader": "benchmarks/run_m3b_ensemble_v3.py",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
    "general_result": "benchmarks/t7b_automatic_depth_v1_m6_inspection1_result_20260722.json",
    "general_terminal": "benchmarks/t7b_automatic_depth_v1_m6_inspection1_terminal_attestation_20260722.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=repository, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if root != repository or not (root / "darkofit").is_dir():
        raise RuntimeError(f"not a DarkoFit Git root: {repository}")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def case_specs() -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(spec)
        for spec in m3b.case_specs()
        if spec["domain"] == "sports"
    )


def case_manifests(panel_cache: Path) -> dict[str, dict[str, Any]]:
    all_manifests = m3b.expected_case_manifests(panel_cache)
    return {
        spec["case_id"]: all_manifests[spec["case_id"]]
        for spec in case_specs()
    }


def quality_orders() -> dict[str, list[str]]:
    return {
        spec["case_id"]: list(ARMS if index % 2 == 0 else reversed(ARMS))
        for index, spec in enumerate(case_specs())
    }


def execution_spec() -> dict[str, Any]:
    return {
        "fresh_worker_per_case_arm": True,
        "same_arm_warmup_outside_measurement": True,
        "quality_repeats": 1,
        "threads": THREADS,
        "iterations": ITERATIONS,
        "early_stopping_rounds": PATIENCE,
        "use_best_model": True,
        "refit": False,
        "validation_fraction": 0.15,
        "validation_strategy": "group",
        "random_state": RANDOM_STATE,
        "bootstrap_clusters": list(SEASONS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "create_only_no_rerun": True,
    }


def decision_rules() -> dict[str, float]:
    return {
        "cold_player_aggregate_at_most": 1.000,
        "held_team_aggregate_at_most": 1.010,
        "cluster_bootstrap_p95_at_most": 1.010,
        "worst_season_at_most": 1.020,
        "worst_lineage_at_most": 1.030,
        "worst_loo_at_most": 1.003,
    }


def claim_spec() -> dict[str, Any]:
    return {
        "tier": "E",
        "spent_player_disjoint_sports_development": True,
        "timing_and_resources_are_single_run_telemetry": True,
        "shipping_or_default_change_authorized": False,
        "fresh_confirmation_authorized": False,
        "tabarena_or_m2_authorized": False,
        "release_authorized": False,
        "lockbox_access_authorized": False,
    }


def _bound_file_ok(record: Mapping[str, Any]) -> bool:
    try:
        path = ROOT / str(record["path"])
        return (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(record["bytes"])
            and file_sha256(path) == record["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def validate_general_preconditions() -> dict[str, Any]:
    observed = {}
    for name, (relative, expected) in GENERAL_PRECONDITIONS.items():
        path = ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"general-development precondition drifted: {name}")
        observed[name] = {"path": relative, "sha256": actual}
    result = json.loads((ROOT / GENERAL_PRECONDITIONS["result"][0]).read_text())
    terminal = json.loads((ROOT / GENERAL_PRECONDITIONS["terminal"][0]).read_text())
    launch = json.loads((ROOT / GENERAL_PRECONDITIONS["launch"][0]).read_text())
    if (
        result.get("mechanism_id") != "t7b_automatic_scalar_rmse_depth_v1"
        or result.get("analysis", {}).get("disposition") != "advance"
        or result.get("shipping_or_default_claim_eligible") is not False
        or launch.get("sources", {}).get("control", {}).get("head")
        != CONTROL_HEAD
        or launch.get("sources", {}).get("control", {}).get("tree")
        != CONTROL_TREE
        or launch.get("sources", {}).get("candidate", {}).get("head")
        != CANDIDATE_HEAD
        or launch.get("sources", {}).get("candidate", {}).get("tree")
        != CANDIDATE_TREE
        or launch.get("candidate_file_allowlist") != CANDIDATE_CHANGED_FILES
        or launch.get("sources", {}).get("candidate_changed_files")
        != CANDIDATE_CHANGED_FILES
        or launch.get("rerun_authorized") is not False
        or terminal.get("disposition") != "advance"
        or terminal.get("rerun_authorized") is not False
        or terminal.get("shipping_or_default_claim_eligible") is not False
    ):
        raise RuntimeError("general-development result is not the frozen advance")
    return observed


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path != CONTRACT_PATH.resolve() or not path.is_file() or path.is_symlink():
        raise RuntimeError("spent-sports execution requires the tracked contract path")
    contract = json.loads(path.read_text(encoding="utf-8"))
    historical_record = contract.get("historical_m3b_contract", {})
    historical = json.loads(
        HISTORICAL_M3B_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    historical_sports = {
        case_id: value
        for case_id, value in historical.get("case_manifests", {}).items()
        if case_id.startswith("sports_")
    }
    expected_panel = {
        "bytes": historical.get("panel_cache", {}).get("bytes"),
        "sha256": historical.get("panel_cache", {}).get("sha256"),
    }
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("contract_frozen") is not True
        or contract.get("outcomes_opened") is not False
        or contract.get("cases") != list(case_specs())
        or contract.get("quality_orders") != quality_orders()
        or contract.get("execution") != execution_spec()
        or contract.get("decision_rules") != decision_rules()
        or contract.get("claims") != claim_spec()
        or contract.get("general_preconditions") != validate_general_preconditions()
        or historical_record
        != {
            "path": str(HISTORICAL_M3B_CONTRACT_PATH.relative_to(ROOT)),
            "sha256": file_sha256(HISTORICAL_M3B_CONTRACT_PATH),
        }
        or contract.get("panel_cache") != expected_panel
        or contract.get("case_manifests") != historical_sports
        or set(contract.get("bound_files", {})) != set(BOUND_PATHS)
        or any(not _bound_file_ok(record) for record in contract["bound_files"].values())
        or contract.get("sources", {}).get(CONTROL, {}).get("head") != CONTROL_HEAD
        or contract.get("sources", {}).get(CONTROL, {}).get("tree") != CONTROL_TREE
        or contract.get("sources", {}).get(CANDIDATE, {}).get("head") != CANDIDATE_HEAD
        or contract.get("sources", {}).get(CANDIDATE, {}).get("tree") != CANDIDATE_TREE
    ):
        raise RuntimeError("T7b automatic-depth spent-sports contract is invalid")
    return contract


def validate_harness(contract: Mapping[str, Any]) -> dict[str, Any]:
    state = source_state(ROOT)
    if not state["clean"]:
        raise RuntimeError("spent-sports harness must be clean")
    parent = _git(ROOT, "rev-parse", f"{state['head']}^")
    changed = set(_git(ROOT, "diff", "--name-only", f"{parent}..{state['head']}").splitlines())
    if parent != contract["sources"]["harness"] or changed != {CONTRACT_RELATIVE}:
        raise RuntimeError("spent-sports contract must be the sole commit above its harness")
    return state


def validate_source(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    state = source_state(path)
    if (
        not state["clean"]
        or state["head"] != expected["head"]
        or state["tree"] != expected["tree"]
    ):
        raise RuntimeError(f"spent-sports source is not its exact clean pin: {path}")
    return state


def panel_record(panel_cache: Path) -> dict[str, Any]:
    panel_cache = panel_cache.expanduser().resolve()
    return {
        "bytes": panel_cache.stat().st_size,
        "sha256": file_sha256(panel_cache),
    }


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own_chain = {os.getpid()}
    ancestor = psutil.Process().parent()
    while ancestor is not None:
        own_chain.add(ancestor.pid)
        try:
            ancestor = ancestor.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_t7b_automatic_depth_sports",
        "run_t7b_automatic_depth_v1",
        "run_m6_quality_successor",
        "run_v011_",
        "run_m3",
        "run_tabarena",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own_chain and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
    }


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("spent-sports outputs must be outside the harness checkout")
    return {
        "launch": Path(str(prefix) + "_launch_manifest.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
        "terminal": Path(str(prefix) + "_terminal_attestation.json"),
    }


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    paired.write_create_only(path, payload)


def _activate_source(source: Path) -> None:
    source = source.resolve()
    sys.path[:] = [
        item
        for item in sys.path
        if Path(item or ".").resolve() != source
    ]
    sys.path.insert(0, str(source))


def _prediction_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    paired._update_array_hash(digest, "prediction", value)
    return digest.hexdigest()


def _model_params(*, iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "early_stopping": iterations > 2,
        "early_stopping_rounds": PATIENCE,
        "use_best_model": True,
        "refit": False,
        "validation_fraction": 0.15,
        "validation_strategy": "group",
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
        "diagnostic_warnings": "never",
    }


def _warmup() -> None:
    from darkofit import DarkoRegressor

    rng = np.random.default_rng(91)
    X = rng.normal(size=(100, 15))
    y = X[:, 0] - 0.5 * X[:, 1]
    groups = np.repeat(np.arange(20), 5)
    DarkoRegressor(**_model_params(iterations=2)).fit(X, y, groups=groups)
    gc.collect()


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    runtime_before = paired.assert_worker_contract(THREADS)
    contract = load_contract(Path(args.contract))
    source = Path(args.source).expanduser().resolve()
    expected_source = contract["sources"][args.arm]
    state = validate_source(source, expected_source)
    _activate_source(source)
    from darkofit import DarkoRegressor
    import numba

    spec = next(spec for spec in case_specs() if spec["case_id"] == args.case_id)
    data = m3b.build_case(spec, Path(args.panel_cache))
    manifest = {
        "fingerprints": m3b.case_fingerprints(spec, data),
        "fit_rows": int(len(data["y_fit"])),
        "test_rows": int(len(data["y_test"])),
        "primary_rows": int(np.sum(np.asarray(data["cold_test_mask"], dtype=np.bool_))),
        "feature_count": int(np.asarray(data["X_fit"]).shape[1]),
        "class_count": None,
    }
    if manifest != contract["case_manifests"][args.case_id]:
        raise RuntimeError(f"spent-sports case fingerprint drifted: {args.case_id}")

    _warmup()
    model = DarkoRegressor(**_model_params(iterations=ITERATIONS))
    ambient_before = int(numba.get_num_threads())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with m3a.AggregateRSSSampler() as rss:
            fit_start = time.perf_counter()
            model.fit(data["X_fit"], data["y_fit"], groups=data["groups_fit"])
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = np.asarray(model.predict(data["X_test"]), dtype=np.float64)
            predict_seconds = time.perf_counter() - predict_start
            with tempfile.TemporaryDirectory(
                prefix="darkofit-t7b-depth-sports-", dir=os.environ["NUMBA_CACHE_DIR"]
            ) as directory:
                archive = Path(directory) / "model.npz"
                model.save_model(archive)
                archive_bytes = archive.stat().st_size
                restored = DarkoRegressor.load_model(archive)
                restored_prediction = np.asarray(restored.predict(data["X_test"]), dtype=np.float64)
    ambient_after = int(numba.get_num_threads())
    if prediction.shape != (len(data["y_test"]),) or not np.isfinite(prediction).all():
        raise RuntimeError("spent-sports prediction is invalid")
    if not np.array_equal(prediction, restored_prediction):
        raise RuntimeError("spent-sports safe-NPZ prediction parity failed")
    if ambient_before != THREADS or ambient_after != THREADS:
        raise RuntimeError("spent-sports worker leaked its thread-local Numba mask")
    runtime_after = paired.assert_worker_contract(THREADS)
    implementation = Path(inspect.getfile(model.__class__)).resolve()
    if not implementation.is_relative_to(source):
        raise RuntimeError("spent-sports estimator imported outside its source pin")
    fitted = paired.fitted_model_metadata(model)
    if fitted["resolved_thread_counts"] != [THREADS]:
        raise RuntimeError("spent-sports fitted thread count drifted")
    cold = np.asarray(data["cold_test_mask"], dtype=np.bool_)
    y_test = np.asarray(data["y_test"], dtype=np.float64)
    primary = math.sqrt(mean_squared_error(y_test[cold], prediction[cold]))
    secondary = math.sqrt(mean_squared_error(y_test, prediction))
    structure = model.model_.auto_params_["auto_structure"]
    return m3b._to_builtin(
        {
            "case_id": args.case_id,
            "season": int(spec["season"]),
            "target": str(spec["target"]),
            "arm": args.arm,
            "source_head": state["head"],
            "source_tree": state["tree"],
            "fingerprints": manifest["fingerprints"],
            "primary_metric": "cold_player_rmse",
            "primary_loss": float(primary),
            "secondary_metric": "held_team_rmse",
            "secondary_loss": float(secondary),
            "fit_rows": manifest["fit_rows"],
            "primary_rows": manifest["primary_rows"],
            "test_rows": manifest["test_rows"],
            "feature_count": manifest["feature_count"],
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "peak_rss_bytes": int(rss.peak_bytes),
            "rss_samples": int(rss.samples),
            "rss_errors": list(rss.errors),
            "archive_bytes": int(archive_bytes),
            "prediction_sha256": _prediction_sha256(prediction),
            "safe_roundtrip_exact": True,
            "requested_depth": model.depth,
            "resolved_depth": int(model.model_.depth),
            "l2_leaf_reg": float(model.model_.l2_leaf_reg),
            "auto_structure": structure,
            "requested_threads": THREADS,
            "fitted_thread_counts": fitted["resolved_thread_counts"],
            "ambient_thread_count_before_fit": ambient_before,
            "ambient_thread_count_after_predict": ambient_after,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
            "implementation_path": str(implementation),
            "warnings": [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
    )


def _parse_worker(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("spent-sports worker did not emit exactly one result")
    return json.loads(matches[0])


def _run_one_worker(
    *, source: Path, panel_cache: Path, contract_path: Path, cache_dir: Path, case_id: str, arm: str
) -> dict[str, Any]:
    environment = paired.fixed_worker_environment(cache_dir)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--contract",
        str(contract_path),
        "--source",
        str(source),
        "--panel-cache",
        str(panel_cache),
        "--case-id",
        case_id,
        "--arm",
        arm,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(
            f"spent-sports worker failed for {case_id}/{arm}:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return _parse_worker(completed.stdout)


def _terminal_failure(
    *, paths: Mapping[str, Path], launch_sha256: str, rows: int, error: BaseException
) -> NoReturn:
    value = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disposition": "terminal_failure",
        "launch_manifest_sha256": launch_sha256,
        "completed_rows_discarded": int(rows),
        "rerun_authorized": False,
        "shipping_or_default_claim_eligible": False,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    if not paths["terminal"].exists():
        write_create_only_json(paths["terminal"], value)
    raise RuntimeError("spent-sports inspection failed terminally") from error


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    paths = output_paths(Path(args.output_prefix))
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise RuntimeError("spent-sports output path already exists")
    contract_path = Path(args.contract).expanduser().resolve()
    contract = load_contract(contract_path)
    harness_before = validate_harness(contract)
    panel_cache = Path(args.panel_cache).expanduser().resolve()
    if panel_record(panel_cache) != contract["panel_cache"]:
        raise RuntimeError("spent-sports panel cache drifted")
    observed_manifests = case_manifests(panel_cache)
    if observed_manifests != contract["case_manifests"]:
        raise RuntimeError("spent-sports case manifests drifted")
    source_paths = {
        CONTROL: Path(args.control).expanduser().resolve(),
        CANDIDATE: Path(args.candidate).expanduser().resolve(),
    }
    sources_before = {
        arm: validate_source(source_paths[arm], contract["sources"][arm])
        for arm in ARMS
    }
    preconditions = validate_general_preconditions()
    exclusive = _exclusive_machine_audit()
    launch = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspection_index": 1,
        "inspection_spent_on_manifest_creation": True,
        "rerun_authorized": False,
        "contract_sha256": file_sha256(contract_path),
        "harness": harness_before,
        "sources": sources_before,
        "panel_cache": contract["panel_cache"],
        "case_manifests_sha256": _json_sha256(observed_manifests),
        "general_preconditions": preconditions,
        "quality_orders": contract["quality_orders"],
        "exclusive_machine": exclusive,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_create_only_json(paths["launch"], launch)
    launch_sha256 = file_sha256(paths["launch"])
    rows = []
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for spec in case_specs():
            for arm in contract["quality_orders"][spec["case_id"]]:
                rows.append(
                    _run_one_worker(
                        source=source_paths[arm],
                        panel_cache=panel_cache,
                        contract_path=contract_path,
                        cache_dir=cache_dir,
                        case_id=spec["case_id"],
                        arm=arm,
                    )
                )
        if validate_harness(contract) != harness_before:
            raise RuntimeError("spent-sports harness changed during execution")
        for arm in ARMS:
            if validate_source(source_paths[arm], contract["sources"][arm]) != sources_before[arm]:
                raise RuntimeError(f"spent-sports {arm} source changed during execution")
        raw = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": file_sha256(contract_path),
            "launch_manifest_sha256": launch_sha256,
            "execution": execution_spec(),
            "claims": claim_spec(),
            "sources": contract["sources"],
            "panel_cache": contract["panel_cache"],
            "case_manifests": observed_manifests,
            "rows": rows,
        }
        write_create_only_json(paths["raw"], raw)
        raw_sha256 = file_sha256(paths["raw"])
        try:
            from . import analyze_t7b_automatic_depth_sports_v1 as analyzer
        except ImportError:
            import analyze_t7b_automatic_depth_sports_v1 as analyzer
        result = analyzer.analyze_raw_payload(raw, contract, raw_sha256=raw_sha256)
        write_create_only_json(paths["result"], result)
        result_sha256 = file_sha256(paths["result"])
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "disposition": result["analysis"]["disposition"],
            "launch_manifest_sha256": launch_sha256,
            "raw_sha256": raw_sha256,
            "result_sha256": result_sha256,
            "rerun_authorized": False,
            "shipping_or_default_claim_eligible": False,
            "fresh_confirmation_authorized": False,
        }
        write_create_only_json(paths["terminal"], terminal)
    except BaseException as exc:
        _terminal_failure(paths=paths, launch_sha256=launch_sha256, rows=len(rows), error=exc)
    return {
        "disposition": terminal["disposition"],
        "rows": len(rows),
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": {name: file_sha256(path) for name, path in paths.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--contract", default=str(CONTRACT_PATH))
    parser.add_argument("--control")
    parser.add_argument("--candidate")
    parser.add_argument("--source")
    parser.add_argument("--panel-cache")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-prefix")
    parser.add_argument("--case-id")
    parser.add_argument("--arm", choices=ARMS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if not all((args.source, args.panel_cache, args.case_id, args.arm)):
            raise RuntimeError("spent-sports worker arguments are incomplete")
        print(WORKER_PREFIX + json.dumps(run_worker(args), sort_keys=True, allow_nan=False))
        return 0
    if not all((args.control, args.candidate, args.panel_cache, args.cache_dir, args.output_prefix)):
        raise RuntimeError("spent-sports parent arguments are incomplete")
    print(json.dumps(run_parent(args), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
