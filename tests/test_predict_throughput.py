from benchmarks import run_predict_throughput as experiment


def _case(public, binning, core):
    return {
        "exactness": {
            "cold_equals_warm": True,
            "public_equals_packed_core": True,
        },
        "warm_public": {"median_seconds": public},
        "binning": {"median_seconds": binning},
        "packed_core": {"median_seconds": core},
    }


def _worker(arm, public, binning, core, fingerprint):
    return {
        "arm": arm,
        "behavior_fingerprint_sha256": fingerprint,
        "cases": {
            dataset: {
                str(rows): _case(public, binning, core)
                for rows in experiment.BATCH_SIZES
            }
            for dataset in experiment.DATASETS
        },
    }


def test_analysis_uses_paired_ratios_and_selects_excess_component():
    results = []
    for _ in range(3):
        results.extend(
            [
                _worker(experiment.DARKOFIT, 1.2, 0.8, 0.4, "dark"),
                _worker(experiment.CHIMERABOOST, 1.0, 0.4, 0.5, "chimera"),
            ]
        )
    canonical = {
        experiment.DARKOFIT: results[0],
        experiment.CHIMERABOOST: results[1],
    }

    analysis = experiment.analyze(canonical, results)

    assert analysis["meets_public_target"] is True
    assert analysis["largest_excess_component"] == "binning"
    summary = analysis["paired_ratios"]["basketball_numeric"]["8192"][
        "warm_public"
    ]
    assert summary["paired_ratios"] == [1.2, 1.2, 1.2]
    assert summary["stable"] is True


def test_analysis_fails_target_or_nondeterministic_behavior():
    results = []
    for block in range(3):
        results.extend(
            [
                _worker(
                    experiment.DARKOFIT,
                    1.4,
                    0.5,
                    0.9,
                    f"dark-{block}",
                ),
                _worker(experiment.CHIMERABOOST, 1.0, 0.5, 0.5, "chimera"),
            ]
        )
    canonical = {
        experiment.DARKOFIT: results[0],
        experiment.CHIMERABOOST: results[1],
    }

    analysis = experiment.analyze(canonical, results)

    assert analysis["meets_public_target"] is False
    assert (
        analysis["target_gates"]["behavior_fingerprints_stable"] is False
    )
    assert analysis["largest_excess_component"] == "packed_core"
    assert analysis["recommendation"] == "start_p2_with_packed_core"


def test_block_orders_are_reciprocal_and_batch_repeats_are_frozen():
    assert experiment.BLOCK_ORDERS == (
        (experiment.DARKOFIT, experiment.CHIMERABOOST),
        (experiment.CHIMERABOOST, experiment.DARKOFIT),
        (experiment.DARKOFIT, experiment.CHIMERABOOST),
    )
    assert set(experiment.WARM_REPEATS) == set(experiment.BATCH_SIZES)
    assert all(value >= 2 for value in experiment.WARM_REPEATS.values())
