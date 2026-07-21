#!/usr/bin/env python3
"""Validate, gate, and analyze the frozen private M3b attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

try:
    from . import paired_evidence_contract as paired
    from . import run_m3b_ensemble_v3 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3b_ensemble_v3 as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"M3b artifact is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"M3b artifact is not an object: {path}")
    return value


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"M3b {label} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"M3b {label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"M3b {label} timestamp has no timezone")


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"M3b {name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"M3b {name} is not positive and finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"M3b {name} is not a positive integer")
    return value


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise RuntimeError("M3b geometric mean received invalid values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _specs_by_id() -> dict[str, dict[str, Any]]:
    return {spec["case_id"]: spec for spec in runner.case_specs()}


def _validate_runtime(runtime: Any, label: str) -> None:
    environment = runtime.get("environment") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("ceiling") != runner.THREADS
        or runtime.get("current") != runner.THREADS
        or runtime.get("threading_layer") not in {"omp", "tbb", "workqueue"}
        or not isinstance(environment, Mapping)
        or set(environment) != set(paired.CONTRACT_ENV_KEYS)
    ):
        raise RuntimeError(f"M3b {label} runtime attestation is invalid")
    for name, expected in paired._expected_environment(runner.THREADS).items():
        if environment.get(name) != expected:
            raise RuntimeError(f"M3b {label} runtime environment drifted: {name}")
    cache_dir = environment.get("NUMBA_CACHE_DIR")
    if (
        not isinstance(cache_dir, str)
        or not cache_dir
        or not Path(cache_dir).is_absolute()
    ):
        raise RuntimeError(f"M3b {label} Numba cache attestation is invalid")


def _validate_git_state(
    state: Any,
    *,
    label: str,
    expected_head: str | None = None,
) -> Path:
    if (
        not isinstance(state, Mapping)
        or not isinstance(state.get("path"), str)
        or not Path(state["path"]).is_absolute()
        or not runner._is_hex_digest(state.get("head"), 40)
        or state.get("status") != ""
        or (expected_head is not None and state["head"] != expected_head)
    ):
        raise RuntimeError(f"M3b {label} git-state attestation is invalid")
    return Path(state["path"]).resolve()


def _validate_row(
    row: Mapping[str, Any],
    *,
    phase: str,
    repeat: int,
    case_id: str,
    arm: str,
    contract: Mapping[str, Any],
    source_root: Path,
) -> None:
    spec = _specs_by_id()[case_id]
    if (
        row.get("phase") != phase
        or row.get("repeat") != repeat
        or row.get("case_id") != case_id
        or row.get("domain") != spec["domain"]
        or row.get("task") != spec["task"]
        or row.get("arm") != arm
        or row.get("arm_config") != runner.arm_config(arm)
    ):
        raise RuntimeError(f"M3b row identity changed: {case_id}/{arm}")
    expected_fingerprints = contract["case_fingerprints"][case_id]
    observed_fingerprints = {
        name: row.get(name)
        for name in (
            "case_sha256",
            "dataset_sha256",
            "split_sha256",
            "weight_sha256",
        )
    }
    if observed_fingerprints != expected_fingerprints:
        raise RuntimeError(f"M3b row fingerprint changed: {case_id}/{arm}")
    case_manifest = contract["case_manifests"][case_id]
    if (
        row.get("fit_rows") != case_manifest["fit_rows"]
        or row.get("test_rows") != case_manifest["test_rows"]
        or row.get("primary_rows") != case_manifest["primary_rows"]
        or row.get("feature_count") != case_manifest["feature_count"]
        or row.get("class_count") != case_manifest["class_count"]
    ):
        raise RuntimeError(f"M3b row shape changed: {case_id}/{arm}")
    expected_metrics = (
        ("cold_player_rmse", "held_team_rmse")
        if spec["domain"] == "sports"
        else (
            ("weighted_rmse", "rmse")
            if spec["task"] == "regression"
            else ("weighted_log_loss", "log_loss")
        )
    )
    if (
        row.get("primary_metric"),
        row.get("secondary_metric"),
    ) != expected_metrics:
        raise RuntimeError(f"M3b row metrics changed: {case_id}/{arm}")
    for name in (
        "primary_loss",
        "secondary_loss",
        "fit_seconds",
        "predict_seconds",
    ):
        _finite_positive(row.get(name), f"{case_id}/{arm}/{name}")
    for name in (
        "primary_rows",
        "test_rows",
        "peak_rss_bytes",
        "rss_samples",
        "archive_bytes",
    ):
        _positive_int(row.get(name), f"{case_id}/{arm}/{name}")
    if row["primary_rows"] > row["test_rows"]:
        raise RuntimeError(f"M3b primary row count exceeds test rows: {case_id}")
    if (
        row.get("safe_roundtrip_exact") is not True
        or not runner._is_hex_digest(row.get("prediction_sha256"), 64)
        or (spec["task"] == "regression" and row.get("probability_sha256") is not None)
        or (
            spec["task"] != "regression"
            and (not runner._is_hex_digest(row.get("probability_sha256"), 64))
        )
    ):
        raise RuntimeError(f"M3b prediction provenance is invalid: {case_id}/{arm}")
    _validate_runtime(row.get("runtime_before"), "pre-fit")
    _validate_runtime(row.get("runtime_after"), "post-fit")
    fitted = row.get("fitted_model_metadata")
    expected_members = 1 if arm == runner.SINGLE else runner.MEMBERS
    if (
        not isinstance(fitted, Mapping)
        or fitted.get("member_count") != expected_members
        or fitted.get("resolved_thread_counts") != [runner.THREADS]
        or not isinstance(fitted.get("tree_counts"), list)
        or len(fitted["tree_counts"]) != expected_members
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fitted["tree_counts"]
        )
        or fitted.get("tree_count") != sum(fitted["tree_counts"])
        or not isinstance(fitted.get("tree_modes"), list)
        or not fitted["tree_modes"]
        or any(
            not isinstance(value, str) or not value for value in fitted["tree_modes"]
        )
        or not isinstance(fitted.get("best_iterations"), list)
        or len(fitted["best_iterations"]) != expected_members
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fitted["best_iterations"]
        )
    ):
        raise RuntimeError(f"M3b fitted model metadata is invalid: {case_id}/{arm}")
    implementation_path = Path(str(row.get("implementation_path", ""))).resolve()
    try:
        relative_implementation = implementation_path.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(
            f"M3b implementation path escaped source: {case_id}/{arm}"
        ) from exc
    if relative_implementation != Path("darkofit/sklearn_api.py"):
        raise RuntimeError(f"M3b implementation module changed: {case_id}/{arm}")
    warnings_value = row.get("warnings")
    if (
        row.get("rss_errors") != []
        or not isinstance(warnings_value, list)
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("category"), str)
            or not isinstance(item.get("message"), str)
            for item in warnings_value
        )
    ):
        raise RuntimeError(f"M3b warning or RSS provenance is invalid: {case_id}/{arm}")
    if (
        not isinstance(row.get("python"), str)
        or not row["python"]
        or not isinstance(row.get("numpy"), str)
        or not row["numpy"]
    ):
        raise RuntimeError(f"M3b runtime versions are invalid: {case_id}/{arm}")
    ensemble = row.get("ensemble_metadata")
    if arm == runner.SINGLE:
        if ensemble is not None or row.get("oob_member_scores") is not None:
            raise RuntimeError("M3b single reference contains ensemble/OOB metadata")
        return
    config = runner.arm_config(arm)
    if (
        not isinstance(ensemble, Mapping)
        or ensemble.get("version") != 2
        or ensemble.get("private_prototype") != "ensemble_v3_b1_b2"
        or ensemble.get("member_count") != runner.MEMBERS
        or ensemble.get("sampling") != config["sampling"]
        or ensemble.get("sampling_unit") != spec["sampling_unit"]
        or ensemble.get("sample_fraction") != config["sample_fraction"]
        or ensemble.get("member_policy") != config["member_policy"]
        or ensemble.get("claim_tier") != "E"
        or ensemble.get("default_changed") is not False
        or ensemble.get("sequential") is not True
        or ensemble.get("public_fit_surface") is not False
        or not isinstance(ensemble.get("members"), list)
        or len(ensemble["members"]) != runner.MEMBERS
        or any(not isinstance(member, Mapping) for member in ensemble["members"])
        or ensemble.get("input_row_count") != case_manifest["fit_rows"]
        or ensemble.get("input_feature_count") != case_manifest["feature_count"]
        or ensemble.get("fit_random_state_seed") != runner.RANDOM_STATE
        or ensemble.get("member_seeds")
        != [member.get("seed") for member in ensemble["members"]]
        or ensemble.get("oob_early_stopping") is not True
    ):
        raise RuntimeError(f"M3b ensemble metadata is invalid: {case_id}/{arm}")
    oob_scores = row.get("oob_member_scores")
    if (
        not isinstance(oob_scores, list)
        or len(oob_scores) != runner.MEMBERS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in oob_scores
        )
    ):
        raise RuntimeError(f"M3b OOB score telemetry is invalid: {case_id}/{arm}")
    expected_policy_source = (
        "member_policy" if config["member_policy"] == "donor_balanced_v1" else "base"
    )
    policy_resolutions = ensemble.get("policy_resolutions")
    if (
        ensemble.get("explicit_user_params") != []
        or not isinstance(policy_resolutions, Mapping)
        or set(policy_resolutions) != {"learning_rate", "colsample"}
        or any(
            not isinstance(record, Mapping)
            or record.get("source") != expected_policy_source
            or (
                expected_policy_source == "base"
                and record.get("resolved") != record.get("base")
            )
            for record in policy_resolutions.values()
        )
        or (
            expected_policy_source == "member_policy"
            and (
                policy_resolutions["learning_rate"].get("resolved") != 0.15
                or policy_resolutions["colsample"].get("resolved") != 0.85
            )
        )
        or ensemble.get("aggregation")
        != ("mean" if spec["task"] == "regression" else "soft_vote")
    ):
        raise RuntimeError(f"M3b member policy metadata is invalid: {case_id}/{arm}")
    for member_index, member in enumerate(ensemble["members"]):
        if (
            member.get("member") != member_index
            or member.get("fitted_thread_count") != runner.THREADS
            or member.get("validation_source") != "explicit_eval_set"
            or member.get("policy_resolutions") != policy_resolutions
            or member.get("sampled_rows", 0) < 1
            or member.get("oob_rows", 0) < 1
            or not runner._is_hex_digest(member.get("sampled_indices_sha256"), 64)
            or not runner._is_hex_digest(member.get("oob_indices_sha256"), 64)
        ):
            raise RuntimeError(
                f"M3b ensemble member metadata is invalid: {case_id}/{arm}"
            )
        if config["sampling"] == "without_replacement":
            sampling_valid = (
                member.get("sampled_rows") == member.get("sampled_unique_rows")
                and member["sampled_rows"] + member["oob_rows"]
                == case_manifest["fit_rows"]
                and member.get("requested_sample_fraction") == 0.8
            )
        else:
            sampling_valid = (
                member.get("sampled_rows") == case_manifest["fit_rows"]
                and member.get("sampled_unique_rows", 0) + member["oob_rows"]
                == case_manifest["fit_rows"]
                and member.get("requested_sample_fraction") is None
            )
        if spec["sampling_unit"] == "groups":
            partition_valid = (
                member.get("group_disjoint") is True
                and member.get("sampled_unique_groups", 0) > 0
                and member.get("oob_groups", 0) > 0
            )
        else:
            partition_valid = (
                member.get("group_disjoint") is None
                and member.get("sampled_group_draws") is None
                and member.get("sampled_unique_groups") is None
                and member.get("oob_groups") is None
            )
        if not sampling_valid or not partition_valid:
            raise RuntimeError(f"M3b sampling metadata is invalid: {case_id}/{arm}")


def validate_artifact(
    path: Path,
    contract_path: Path,
    *,
    phase: str,
    gate: Mapping[str, Any] | None = None,
    gate_path: Path | None = None,
) -> dict[str, Any]:
    contract = runner.load_contract(contract_path)
    contract_sha256 = _sha256(contract_path)
    artifact = _load_json(path)
    _validate_timestamp(artifact.get("created_at"), phase)
    rows = artifact.get("rows")
    source_root = _validate_git_state(
        artifact.get("source_state"),
        label="source",
        expected_head=contract["sources"]["darkofit"],
    )
    _validate_git_state(artifact.get("harness_state"), label="harness")
    panel_cache = artifact.get("panel_cache")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("name") != runner.CONTRACT_NAME
        or artifact.get("phase") != phase
        or artifact.get("status") != "complete"
        or artifact.get("contract_sha256") != contract_sha256
        or artifact.get("case_fingerprints") != contract["case_fingerprints"]
        or not isinstance(panel_cache, Mapping)
        or panel_cache.get("bytes") != contract["panel_cache"]["bytes"]
        or panel_cache.get("sha256") != contract["panel_cache"]["sha256"]
        or not isinstance(rows, list)
    ):
        raise RuntimeError(f"M3b {phase} artifact header is invalid")
    expected = []
    if phase == "quality":
        for spec in runner.case_specs():
            expected.extend(
                (0, spec["case_id"], arm)
                for arm in contract["quality_orders"][spec["case_id"]]
            )
        if artifact.get("quality_artifact_sha256") is not None:
            raise RuntimeError("M3b quality artifact has a parent quality hash")
        if artifact.get("gate_sha256") is not None:
            raise RuntimeError("M3b quality artifact has a timing gate hash")
    elif phase == "timing":
        if gate is None or gate_path is None:
            raise RuntimeError("M3b timing validation requires a gate artifact")
        eligible = tuple(gate["eligible_candidates"])
        timing_arms = (runner.SINGLE, runner.CONTROL, *eligible)
        if eligible:
            for repeat in contract["decision_rules"]["timing_repeats"]:
                for spec in runner.case_specs():
                    expected.extend(
                        (repeat, spec["case_id"], arm)
                        for arm in runner.timing_order(
                            spec["case_id"], repeat, timing_arms
                        )
                    )
        if artifact.get("quality_artifact_sha256") != gate[
            "quality_artifact_sha256"
        ] or artifact.get("gate_sha256") != _sha256(gate_path):
            raise RuntimeError("M3b timing artifact is not bound to its gate")
    else:
        raise ValueError(f"unknown M3b phase: {phase}")
    observed = [
        (row.get("repeat"), row.get("case_id"), row.get("arm"))
        for row in rows
        if isinstance(row, Mapping)
    ]
    if observed != expected or len(observed) != len(rows):
        raise RuntimeError(f"M3b {phase} artifact grid is incomplete or reordered")
    for row, (repeat, case_id, arm) in zip(rows, expected):
        _validate_row(
            row,
            phase=phase,
            repeat=repeat,
            case_id=case_id,
            arm=arm,
            contract=contract,
            source_root=source_root,
        )
    versions = {(row["python"], row["numpy"]) for row in rows}
    if len(versions) > 1:
        raise RuntimeError(f"M3b {phase} runtime versions changed between rows")
    return artifact


def _quality_summaries(
    quality: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    by_key = {(row["case_id"], row["arm"]): row for row in quality["rows"]}
    summaries = {}
    rules = contract["decision_rules"]["quality"]
    specs = _specs_by_id()
    for arm in runner.CANDIDATE_ARMS:
        primary_ratios = {}
        secondary_sports = []
        for case_id, spec in specs.items():
            candidate = by_key[(case_id, arm)]
            control = by_key[(case_id, runner.CONTROL)]
            primary_ratios[case_id] = (
                candidate["primary_loss"] / control["primary_loss"]
            )
            if spec["domain"] == "sports":
                secondary_sports.append(
                    candidate["secondary_loss"] / control["secondary_loss"]
                )
        general = [
            ratio
            for case_id, ratio in primary_ratios.items()
            if specs[case_id]["domain"] == "general"
        ]
        sports = [
            ratio
            for case_id, ratio in primary_ratios.items()
            if specs[case_id]["domain"] == "sports"
        ]
        values = {
            "all_primary_geomean": _geomean(list(primary_ratios.values())),
            "general_primary_geomean": _geomean(general),
            "sports_cold_geomean": _geomean(sports),
            "sports_held_geomean": _geomean(secondary_sports),
            "worst_primary": max(primary_ratios.values()),
            "per_case_primary": primary_ratios,
        }
        checks = {
            "all_primary_geomean": (
                values["all_primary_geomean"] <= rules["all_primary_geomean_at_most"]
            ),
            "general_primary_geomean": (
                values["general_primary_geomean"]
                <= rules["general_primary_geomean_at_most"]
            ),
            "sports_cold_geomean": (
                values["sports_cold_geomean"] <= rules["sports_cold_geomean_at_most"]
            ),
            "sports_held_geomean": (
                values["sports_held_geomean"] <= rules["sports_held_geomean_at_most"]
            ),
            "worst_primary": (
                values["worst_primary"] <= rules["worst_primary_at_most"]
            ),
        }
        summaries[arm] = {
            **values,
            "checks": checks,
            "eligible": all(checks.values()),
        }
    return summaries


def build_gate(
    quality_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    quality = validate_artifact(
        quality_path,
        contract_path,
        phase="quality",
    )
    contract = runner.load_contract(contract_path)
    summaries = _quality_summaries(quality, contract)
    eligible = [arm for arm in runner.CANDIDATE_ARMS if summaries[arm]["eligible"]]
    return {
        "schema_version": 1,
        "name": runner.CONTRACT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": _sha256(contract_path),
        "quality_artifact_sha256": _sha256(quality_path),
        "eligible_candidates": eligible,
        "timing_required": bool(eligible),
        "quality_summaries": summaries,
    }


def _median_by_case_arm(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], row["arm"]), []).append(float(row[field]))
    return {key: float(median(values)) for key, values in grouped.items()}


def _resource_summary(
    quality: Mapping[str, Any],
    timing: Mapping[str, Any],
    arm: str,
) -> dict[str, float]:
    rows = [
        row
        for row in [*quality["rows"], *timing["rows"]]
        if row["arm"] in {arm, runner.CONTROL, runner.SINGLE}
    ]
    medians = {
        field: _median_by_case_arm(rows, field)
        for field in (
            "fit_seconds",
            "predict_seconds",
            "archive_bytes",
            "peak_rss_bytes",
        )
    }
    case_ids = [spec["case_id"] for spec in runner.case_specs()]

    def ratios(field: str, denominator: str) -> list[float]:
        return [
            medians[field][(case_id, arm)] / medians[field][(case_id, denominator)]
            for case_id in case_ids
        ]

    return {
        "fit_to_control_geomean": _geomean(ratios("fit_seconds", runner.CONTROL)),
        "predict_to_control_geomean": _geomean(
            ratios("predict_seconds", runner.CONTROL)
        ),
        "archive_to_control_geomean": _geomean(ratios("archive_bytes", runner.CONTROL)),
        "rss_to_control_geomean": _geomean(ratios("peak_rss_bytes", runner.CONTROL)),
        "median_archive_to_single": float(
            median(ratios("archive_bytes", runner.SINGLE))
        ),
        "median_rss_to_single": float(median(ratios("peak_rss_bytes", runner.SINGLE))),
    }


def build_final_result(
    quality_path: Path,
    gate_path: Path,
    timing_path: Path | None,
    contract_path: Path,
) -> dict[str, Any]:
    gate = _load_json(gate_path)
    _validate_timestamp(gate.get("created_at"), "gate")
    expected_gate = build_gate(quality_path, contract_path)
    for key in (
        "schema_version",
        "name",
        "contract_sha256",
        "quality_artifact_sha256",
        "eligible_candidates",
        "timing_required",
        "quality_summaries",
    ):
        if gate.get(key) != expected_gate.get(key):
            raise RuntimeError(f"M3b gate differs from frozen analysis: {key}")
    quality = validate_artifact(
        quality_path,
        contract_path,
        phase="quality",
    )
    if gate["eligible_candidates"]:
        if timing_path is None:
            raise RuntimeError("M3b eligible candidates require timing evidence")
        timing = validate_artifact(
            timing_path,
            contract_path,
            phase="timing",
            gate=gate,
            gate_path=gate_path,
        )
        timing_sha256 = _sha256(timing_path)
    else:
        if timing_path is not None:
            raise RuntimeError(
                "M3b timing must remain skipped when no candidate is eligible"
            )
        timing = {"rows": []}
        timing_sha256 = None
    timing_arms = {row["arm"] for row in timing["rows"]}
    for arm in timing_arms:
        for case_id in _specs_by_id():
            hashes = {
                row["prediction_sha256"]
                for row in [*quality["rows"], *timing["rows"]]
                if row["arm"] == arm and row["case_id"] == case_id
            }
            if len(hashes) != 1:
                raise RuntimeError(f"M3b repeated prediction changed: {case_id}/{arm}")
    contract = runner.load_contract(contract_path)
    common_rules = contract["decision_rules"]["common_final"]
    value_rules = contract["decision_rules"]["value"]
    candidates = {}
    for arm in runner.CANDIDATE_ARMS:
        quality_summary = gate["quality_summaries"][arm]
        if not quality_summary["eligible"]:
            candidates[arm] = {
                "quality": quality_summary,
                "resources": None,
                "checks": {"quality_eligible": False},
                "survives": False,
            }
            continue
        resources = _resource_summary(quality, timing, arm)
        checks = {
            "quality_eligible": True,
            "predict": (
                resources["predict_to_control_geomean"]
                <= common_rules["predict_ratio_at_most"]
            ),
            "archive_to_control": (
                resources["archive_to_control_geomean"]
                <= common_rules["archive_to_control_at_most"]
            ),
            "rss_to_control": (
                resources["rss_to_control_geomean"]
                <= common_rules["rss_to_control_at_most"]
            ),
            "archive_to_single": (
                resources["median_archive_to_single"]
                <= common_rules["median_archive_to_single_at_most"]
            ),
            "rss_to_single": (
                resources["median_rss_to_single"]
                <= common_rules["median_rss_to_single_at_most"]
            ),
        }
        primary = quality_summary["all_primary_geomean"]
        fit_ratio = resources["fit_to_control_geomean"]
        if arm == runner.COMBINED:
            checks["value"] = primary <= value_rules[arm]["quality_route_at_most"] or (
                primary <= value_rules[arm]["pareto_primary_at_most"]
                and fit_ratio <= value_rules[arm]["pareto_fit_at_most"]
            )
        else:
            checks["value"] = (
                primary <= value_rules[arm]["all_primary_at_most"]
                and fit_ratio <= value_rules[arm]["fit_ratio_at_most"]
            )
        candidates[arm] = {
            "quality": quality_summary,
            "resources": resources,
            "checks": checks,
            "survives": all(checks.values()),
        }
    if candidates[runner.COMBINED]["survives"]:
        disposition = "continue_private_combined"
        retained = [runner.COMBINED]
    else:
        retained = [
            arm for arm in (runner.B1, runner.B2) if candidates[arm]["survives"]
        ]
        disposition = (
            "continue_private_components"
            if retained
            else "close_b1_b2_preserve_existing_opt_in"
        )
    return {
        "schema_version": 1,
        "name": runner.CONTRACT_NAME,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": _sha256(contract_path),
        "quality_artifact_sha256": _sha256(quality_path),
        "gate_sha256": _sha256(gate_path),
        "timing_artifact_sha256": timing_sha256,
        "candidates": candidates,
        "disposition": disposition,
        "retained_private_arms": retained,
        "public_or_default_change_authorized": False,
        "b3_authorized": False,
        "fresh_confirmation_authorized": False,
    }


def render_note(result: Mapping[str, Any]) -> str:
    lines = [
        "# Wave 2 M3b private ensemble-v3 result",
        "",
        "## Outcome",
        "",
        f"Private disposition: **{result['disposition']}**.",
        "",
        "No public/default change, B3, fresh confirmation, TabArena, or "
        "lockbox access is authorized.",
        "",
        "## Frozen attribution",
        "",
        "| Arm | All loss | General | Sports cold | Fit | Archive/single | "
        "RSS/single | Survives |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for arm in runner.CANDIDATE_ARMS:
        record = result["candidates"][arm]
        quality = record["quality"]
        resources = record["resources"]
        lines.append(
            "| "
            + arm
            + f" | {quality['all_primary_geomean']:.6f}"
            + f" | {quality['general_primary_geomean']:.6f}"
            + f" | {quality['sports_cold_geomean']:.6f}"
            + (
                " | — | — | —"
                if resources is None
                else f" | {resources['fit_to_control_geomean']:.6f}"
                f" | {resources['median_archive_to_single']:.6f}"
                f" | {resources['median_rss_to_single']:.6f}"
            )
            + f" | {'yes' if record['survives'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All ratios are paired candidate/control unless the column names the "
            "single reference. This is spent private development evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_final_pair(
    result_path: Path, note_path: Path, result: dict[str, Any]
) -> None:
    if result_path.exists() or result_path.is_symlink():
        raise RuntimeError(f"refusing existing M3b result: {result_path}")
    if note_path.exists() or note_path.is_symlink():
        raise RuntimeError(f"refusing existing M3b note: {note_path}")
    payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    note = render_note(result).encode("utf-8")
    created = []
    try:
        paired.write_create_only(result_path, payload)
        created.append(result_path)
        paired.write_create_only(note_path, note)
        created.append(note_path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "final"), required=True)
    parser.add_argument("--contract", default=str(runner.CONTRACT_PATH))
    parser.add_argument("--quality", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--timing")
    parser.add_argument("--result")
    parser.add_argument("--note")
    return parser


def main() -> int:
    args = _parser().parse_args()
    contract = Path(args.contract).expanduser().resolve()
    quality = Path(args.quality).expanduser().resolve()
    gate = Path(args.gate).expanduser().resolve()
    if args.mode == "gate":
        payload = build_gate(quality, contract)
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        paired.write_create_only(gate, encoded)
        print(
            json.dumps(
                {
                    "gate": str(gate),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "eligible_candidates": payload["eligible_candidates"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.result or not args.note:
        raise RuntimeError("M3b final analysis requires result and note paths")
    result = build_final_result(
        quality,
        gate,
        (None if not args.timing else Path(args.timing).expanduser().resolve()),
        contract,
    )
    result_path = Path(args.result).expanduser().resolve()
    note_path = Path(args.note).expanduser().resolve()
    _write_final_pair(result_path, note_path, result)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "note": str(note_path),
                "disposition": result["disposition"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
