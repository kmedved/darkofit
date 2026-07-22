import json
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import run_automatic_linear_selector_v2_protein_attribution as protein


def _rows(*, ratios=(0.95, 0.96, 0.97), margin=0.04):
    rows = []
    for cell in protein.expected_ordered_grid():
        coordinate = cell["coordinate"]
        arm = cell["arm"]
        constant_rmse = 10.0 + coordinate
        if arm == "constant":
            rmse = constant_rmse
        else:
            rmse = constant_rmse * ratios[coordinate]
        shared_hash = f"{coordinate:064x}"
        rows.append(
            {
                **cell,
                "schema_version": 1,
                "kind": "automatic_linear_selector_v2_protein_worker",
                "dataset": protein.DATASET,
                "task_id": protein.TASK_ID,
                "test_rows": 100,
                "fingerprints": {"combined_sha256": f"split-{coordinate}"},
                "test_rmse": rmse,
                "prediction_sha256": (
                    shared_hash if arm != "constant" else "c" * 64
                ),
                "core_booster_state_sha256": (
                    shared_hash if arm != "constant" else "d" * 64
                ),
                "fit_seconds": 1.0 + coordinate + 0.1 * cell["position"],
                "prediction": {
                    "rows": 100,
                    "pilots_seconds": [0.01, 0.01, 0.01],
                    "calls": 100,
                    "interval_seconds": 1.0,
                    "seconds_per_call": 0.01 + coordinate * 0.001,
                    "prediction_sha256": (
                        shared_hash if arm != "constant" else "c" * 64
                    ),
                },
                "fit_rss": {
                    "scope": "worker_plus_recursive_children",
                    "peak_bytes": 100_000_000 + coordinate,
                    "peak_delta_bytes": 10_000_000 + coordinate,
                    "errors": [],
                    "interval_seconds": protein.RSS_INTERVAL_SECONDS,
                },
                "selector": (
                    {
                        "eligible": True,
                        "reason": "selected_linear",
                        "fit_random_state_seed": cell["seed"],
                        "relative_validation_improvement": margin,
                        "split": {
                            "source": "automatic_holdout",
                            "policy": "weighted_target_stratified",
                            "rows_disjoint": True,
                        },
                        "final_booster_linear_leaves": True,
                        "final_linear_leaves_active": True,
                    }
                    if arm == "automatic"
                    else None
                ),
                "model": {
                    "requested_linear_leaves": protein.ARMS[arm],
                    "selected_linear_leaves": arm != "constant",
                    "linear_leaves_active": arm != "constant",
                },
                "environment": protein.WORKER_ENVIRONMENT,
                "numba_threads_before_fit": protein.THREADS,
                "numba_threads_after_fit": protein.THREADS,
                "numba_threads_after_predict": protein.THREADS,
                "numba_threads_after_timing": protein.THREADS,
            }
        )
    return rows


def _row(rows, coordinate, arm):
    return next(
        item
        for item in rows
        if item["coordinate"] == coordinate and item["arm"] == arm
    )


def test_frozen_grid_uses_exact_release_coordinates_and_latin_rotation():
    grid = protein.expected_ordered_grid()
    assert len(grid) == 9
    assert [
        (item["repeat"], item["fold"], item["seed"])
        for item in protein.COORDINATES
    ] == [(0, 0, 0), (1, 1, 1001), (2, 2, 2002)]
    assert [item["arm"] for item in grid] == [
        "constant",
        "automatic",
        "explicit_linear",
        "automatic",
        "explicit_linear",
        "constant",
        "explicit_linear",
        "constant",
        "automatic",
    ]
    assert all(
        [item["arm"] for item in grid].count(arm) == 3
        for arm in protein.ARMS
    )
    assert len(protein.ordered_grid_sha256()) == 64


def test_bound_spent_evidence_and_release_coordinates_are_current():
    bindings = protein.validate_bound_evidence()
    assert bindings[
        "benchmarks/automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json"
    ]["sha256"] == protein.EXPECTED_HASHES[protein.M6_RESULT_PATH]


def test_fresh_runner_import_does_not_preload_product_packages():
    script = (
        "import json,sys; "
        "from benchmarks import "
        "run_automatic_linear_selector_v2_protein_attribution; "
        "print(json.dumps({'darkofit':'darkofit' in sys.modules,"
        "'tabarena':'tabarena' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=protein.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "darkofit": False,
        "tabarena": False,
    }


def test_protocol_freezes_scope_harm_rule_and_nonshipping_disposition():
    text = protein.PROTOCOL_PATH.read_text()
    for phrase in (
        "already-spent development evidence",
        "automatic RMSE / constant RMSE",
        "There is no minimum improvement gate",
        "ready_for_powered_fresh_design",
        "launch spends attempt 1",
    ):
        assert phrase in text


def test_analyzer_advances_only_to_powered_fresh_design():
    result = protein.analyze_rows(_rows())
    assert result["disposition"] == "ready_for_powered_fresh_design"
    assert result["all_conditions_pass"] is True
    assert all(result["gates"].values())
    assert result["aggregate_automatic_over_constant_rmse"] == pytest.approx(
        (0.95 * 0.96 * 0.97) ** (1 / 3)
    )
    assert result["worst_coordinate"] == 2
    assert result["worst_coordinate_ratio"] == pytest.approx(0.97)
    assert result["minimum_selector_margin"] == pytest.approx(0.04)
    assert set(result["cost_summary"]) == set(protein.ARMS)


def test_coordinate_harm_closes_even_when_aggregate_passes():
    result = protein.analyze_rows(_rows(ratios=(1.021, 0.90, 0.90)))
    assert result["aggregate_automatic_over_constant_rmse"] < 1.02
    assert result["gates"]["aggregate_ratio_at_most_1_02"] is True
    assert result["gates"]["every_coordinate_ratio_at_most_1_02"] is False
    assert result["disposition"] == "terminal_close"


def test_aggregate_harm_closes_candidate():
    result = protein.analyze_rows(_rows(ratios=(1.03, 1.03, 1.03)))
    assert result["gates"]["aggregate_ratio_at_most_1_02"] is False
    assert result["disposition"] == "terminal_close"


@pytest.mark.parametrize(
    "mutation",
    [
        "not_eligible",
        "wrong_reason",
        "low_margin",
        "wrong_provenance",
        "overlap",
        "inactive_final",
        "prediction_mismatch",
        "state_mismatch",
        "fingerprint_mismatch",
    ],
)
def test_each_selector_or_exactness_invariant_is_binding(mutation):
    rows = _rows()
    automatic = _row(rows, 0, "automatic")
    if mutation == "not_eligible":
        automatic["selector"]["eligible"] = False
    elif mutation == "wrong_reason":
        automatic["selector"]["reason"] = "margin_below_threshold"
        automatic["model"]["selected_linear_leaves"] = False
    elif mutation == "low_margin":
        automatic["selector"]["relative_validation_improvement"] = 0.029999
    elif mutation == "wrong_provenance":
        automatic["selector"]["fit_random_state_seed"] = 99
    elif mutation == "overlap":
        automatic["selector"]["split"]["rows_disjoint"] = False
    elif mutation == "inactive_final":
        automatic["selector"]["final_linear_leaves_active"] = False
    elif mutation == "prediction_mismatch":
        automatic["prediction_sha256"] = "e" * 64
        automatic["prediction"]["prediction_sha256"] = "e" * 64
    elif mutation == "state_mismatch":
        automatic["core_booster_state_sha256"] = "e" * 64
    else:
        automatic["fingerprints"] = {"combined_sha256": "changed"}
    result = protein.analyze_rows(rows)
    assert result["gates"]["all_selector_and_exactness_invariants"] is False
    assert result["disposition"] == "terminal_close"


def test_analyzer_rejects_missing_and_duplicate_rows():
    rows = _rows()
    with pytest.raises(RuntimeError, match="row count"):
        protein.analyze_rows(rows[:-1])
    rows[-1] = dict(rows[0])
    with pytest.raises(RuntimeError, match="duplicate"):
        protein.analyze_rows(rows)


def test_output_paths_must_be_external_and_create_only(tmp_path):
    with pytest.raises(RuntimeError, match="external"):
        protein._output_paths(protein.ROOT / "benchmarks/forbidden")
    paths = protein._output_paths(tmp_path / "protein")
    assert paths["manifest"].name == "protein_manifest.json"
    protein._write_create_only_json(paths["manifest"], {"status": "launched"})
    assert json.loads(paths["manifest"].read_text()) == {"status": "launched"}
    with pytest.raises(FileExistsError):
        protein._write_create_only_json(paths["manifest"], {"status": "changed"})


def test_core_booster_digest_is_content_based(tmp_path):
    class Booster:
        def __init__(self, value):
            self.value = value

        def save_model(self, path):
            np.savez_compressed(
                path,
                header=np.array(json.dumps({"auto_params": {}})),
                values=np.array([self.value], dtype=np.float64),
            )

    class Model:
        def __init__(self, value):
            self.model_ = Booster(value)

    del tmp_path
    assert protein._core_booster_state_sha256(Model(1.0)) == (
        protein._core_booster_state_sha256(Model(1.0))
    )
    assert protein._core_booster_state_sha256(Model(1.0)) != (
        protein._core_booster_state_sha256(Model(2.0))
    )


def test_core_digest_normalizes_only_selector_provenance():
    class Booster:
        def __init__(self, *, selector=False, policy=3):
            self.selector = selector
            self.policy = policy

        def save_model(self, path):
            auto_params = {"policy": self.policy}
            if self.selector:
                auto_params["automatic_linear_selector"] = {"margin": 0.04}
                auto_params["diagnostics"] = {
                    "automatic_linear_selector": {"margin": 0.04}
                }
            np.savez_compressed(
                path,
                header=np.array(json.dumps({"auto_params": auto_params})),
                values=np.array([1.0]),
            )

    class Model:
        def __init__(self, **kwargs):
            self.model_ = Booster(**kwargs)

    automatic = protein._core_booster_state_sha256(Model(selector=True))
    explicit = protein._core_booster_state_sha256(Model(selector=False))
    changed_policy = protein._core_booster_state_sha256(
        Model(selector=False, policy=4)
    )
    assert automatic == explicit
    assert automatic != changed_policy


@pytest.mark.parametrize("requested", [False, True])
def test_explicit_model_metadata_does_not_require_selector_wrapper_state(requested):
    class Booster:
        linear_leaves = requested
        linear_leaves_active_ = requested
        n_threads_ = 14
        trees_ = [object(), object()]

    class Model:
        linear_leaves = requested
        best_n_estimators_ = 2
        learning_rate_ = 0.1
        model_ = Booster()

    metadata = protein._fitted_model_metadata(Model())
    assert metadata["selected_linear_leaves"] is requested
    assert metadata["linear_leaves_active"] is requested


def test_internal_worker_arguments_are_all_or_none(tmp_path):
    base = [
        "--candidate-source",
        str(tmp_path / "candidate"),
        "--tabarena-source",
        str(tmp_path / "tabarena"),
        "--output-prefix",
        str(tmp_path / "output"),
    ]
    assert protein.parse_args(base).worker_index is None
    with pytest.raises(SystemExit):
        protein.parse_args([*base, "--worker-index", "0"])
    complete = [
        *base,
        "--worker-index",
        "0",
        "--arm",
        "constant",
        "--parent-pid",
        "123",
        "--worker-started-at",
        "2026-07-22T00:00:00+00:00",
    ]
    assert protein.parse_args(complete).arm == "constant"


def test_worker_environment_and_policy_are_frozen():
    assert protein.THREADS == 14
    assert protein.WARMUP_ROWS == 1_400
    assert protein.HARM_BOUND == 1.02
    assert protein.ATTEMPT_INDEX == 1
    assert protein.ARMS == {
        "constant": False,
        "automatic": "auto",
        "explicit_linear": True,
    }
    assert protein.WORKER_ENVIRONMENT["NUMBA_NUM_THREADS"] == "14"
    assert protein.WORKER_ENVIRONMENT["PYTHONNOUSERSITE"] == "1"
