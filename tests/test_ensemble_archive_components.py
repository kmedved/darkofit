import io
import json

import numpy as np
import pytest

from benchmarks.analyze_ensemble_archive_components import (
    R3_REQUIRED_SHARE_PER_DUPLICATE,
    analyze_archive,
)
from darkofit import DarkoRegressor
from darkofit.sklearn_api import _fit_private_ensemble_v3


def _data(seed=20260721, n=120, p=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 4,
        "depth": 3,
        "early_stopping_rounds": 2,
        "random_state": 17,
        "diagnostic_warnings": "never",
    }
    params.update(extra)
    return params


def _archives(tmp_path):
    X, y = _data()
    single = DarkoRegressor(**_params(n_ensembles=1)).fit(X, y)
    ensemble = _fit_private_ensemble_v3(
        DarkoRegressor(**_params(n_ensembles=3)),
        X,
        y,
        sampling="without_replacement",
        sampling_unit="rows",
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
    )
    single_path = tmp_path / "single.npz"
    ensemble_path = tmp_path / "ensemble.npz"
    single.save_model(single_path)
    ensemble.save_model(ensemble_path)
    return ensemble_path, single_path


def test_archive_component_census_measures_complete_exact_preprocessor(tmp_path):
    ensemble, single = _archives(tmp_path)
    result = analyze_archive(
        ensemble,
        single_path=single,
        reference_ratio=5.534767493867151,
        reference_member_count=8,
    )

    canonical = result["canonical_preprocessor"]
    assert canonical["eligible"] is True
    assert canonical["array_schema_identical"] is True
    assert canonical["arrays_byte_identical"] is True
    assert canonical["headers_byte_identical"] is True
    assert set(canonical["array_names"]) == {
        "bin__border_offsets",
        "bin__borders_flat",
        "bin__block_widths",
        "bin__n_bins",
        "prep__cat_features",
        "prep__feature_map",
        "prep__num_features",
    }
    assert canonical["simulated_archive_bytes"] < result["ensemble"]["bytes"]
    assert (
        result["optimistic_all_exact_entries"]["simulated_archive_bytes"]
        <= result["ensemble"]["bytes"]
    )
    assert result["gate"]["required_savings_per_duplicate_bytes"] is None or (
        result["gate"]["required_savings_per_duplicate_bytes"] > 0.0
    )
    assert R3_REQUIRED_SHARE_PER_DUPLICATE == pytest.approx(0.2192524991)
    assert result["gate"]["reference_screen"][
        "required_share_per_duplicate_to_single"
    ] == pytest.approx(R3_REQUIRED_SHARE_PER_DUPLICATE)
    assert result["gate"]["reference_screen"]["verdict"] in {
        "advance_reference_canonical_plausible",
        "kill_reference_all_exact_insufficient",
        "kill_reference_requires_out_of_scope_sections",
    }


def test_archive_component_census_refuses_nonidentical_preprocessor(tmp_path):
    ensemble, single = _archives(tmp_path)
    with np.load(ensemble, allow_pickle=False) as archive:
        outer = {name: archive[name].copy() for name in archive.files}
    payload = io.BytesIO(np.asarray(outer["member_0001"], dtype=np.uint8).tobytes())
    with np.load(payload, allow_pickle=False) as archive:
        member = {name: archive[name].copy() for name in archive.files}
    feature_map = member["prep__feature_map"]
    feature_map.flat[0] = feature_map.flat[0] + 1
    output = io.BytesIO()
    np.savez_compressed(output, **member)
    outer["member_0001"] = np.frombuffer(
        output.getvalue(), dtype=np.uint8
    ).copy()
    forged = tmp_path / "nonidentical-preprocessor.npz"
    np.savez_compressed(forged, **outer)

    result = analyze_archive(forged, single_path=single)
    assert result["canonical_preprocessor"]["arrays_byte_identical"] is False
    assert result["canonical_preprocessor"]["eligible"] is False
    assert result["canonical_preprocessor"]["simulated_archive_bytes"] is None
    assert result["gate"]["canonical_preprocessor_to_single"] is None


def test_archive_component_census_output_is_json_serializable(tmp_path):
    ensemble, single = _archives(tmp_path)
    result = analyze_archive(ensemble, single_path=single)
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert restored["analysis"] == "ensemble_archive_component_census"
    assert restored["non_loadable_size_simulation"] is True
