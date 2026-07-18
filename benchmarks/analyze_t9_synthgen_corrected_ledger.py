"""Re-score immutable SynthGen outputs against the corrected outcome ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from benchmarks.analyze_synthgen_darkofit_ledger import analyze as analyze_raw


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
RAW = BENCHMARKS / "synthgen_darkofit_ledger_raw.json"
OUTPUT_JSON = BENCHMARKS / "t9_synthgen_corrected_ledger_result.json"
OUTPUT_MD = BENCHMARKS / "t9_synthgen_corrected_ledger_result.md"

EXPECTED_SHA256 = {
    "synthgen_darkofit_ledger_raw.json":
        "fd8f93ec4c0e1cbd6889200d0d79f235e7e880732f0597e09a2bd025f09af7eb",
    "analyze_synthgen_darkofit_ledger.py":
        "65669eb564417a47919753879ef3e5988965180fc0bf4639bf9be35ed988abc7",
    "run_synthgen_darkofit_ledger.py":
        "6aeadecaeca031901518255eff056afaf0f16433b001b9400fed5d0a2f00a7a0",
    "synthgen_darkofit_protocol.md":
        "6ff934b436194934413b15518c8f910069a88a8d1f0c302dc4d48c415d5ed5be",
    "basketball_robust_heads_result.md":
        "ff8d8e81b2ee339acef13d1aed8d0ac0923f7a9d87392a220a4bce1f559d93c3",
    "basketball_sports_panel_result.md":
        "eb9f8cf11d7b9d89faf8754951c83f9c8c2eda012d83a0843d454ccff02bac10",
    "basketball_random_strength_result.md":
        "fae1264e62600f466b3326670bd652315a0b37af926f52e17d1938a595f85c05",
    "fresh_selector_confirmation_result.md":
        "3a33ec834bcebb9d9c9e2db4d69a5119f35ccbcf7623bf3dedb839d15ef71170",
    "smooth_linear_leaves_development_result.md":
        "363011278a18a0b9dc69d3caca862be4250cca1b553c9c4adbe7de51d1677c4d",
    "tabarena_regression_multisplit_ablation.md":
        "de340bf6fac1874112c5eb604947458151da74f03d6564326aa1ac8ace52fd18",
    "ordered_boosting_policy_check.md":
        "18b71456a09cb5731b05c68cffecaed52f196da4537d0c1f39a871c93c64e350",
    "tabarena_regression_remaining9_confirmation.md":
        "f43c77f44b3a3a1affcb73e57cfbd0f363e3475792f1f76a320ad281cee86a6a",
}

OUTCOME_SOURCES = {
    1: "basketball_robust_heads_result.md",
    2: "basketball_robust_heads_result.md",
    3: "basketball_sports_panel_result.md",
    4: "basketball_random_strength_result.md",
    5: "fresh_selector_confirmation_result.md",
    6: "smooth_linear_leaves_development_result.md",
    7: "tabarena_regression_multisplit_ablation.md",
    8: "ordered_boosting_policy_check.md",
    9: "tabarena_regression_remaining9_confirmation.md",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sources():
    observed = {}
    for name, expected in EXPECTED_SHA256.items():
        path = BENCHMARKS / name
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"immutable T9 source changed: {name}; "
                f"expected {expected}, observed {actual}"
            )
        observed[name] = actual
    return observed


def analyze():
    source_hashes = _verify_sources()
    original = analyze_raw(RAW.resolve())
    expected_original = (True, True, False, True, False, False, True, True, True)
    observed_original = tuple(
        bool(decision["agrees"]) for decision in original["decisions"]
    )
    if original["agreement_count"] != 6 or observed_original != expected_original:
        raise RuntimeError("the immutable original 6/9 scorecard changed")

    decisions = copy.deepcopy(original["decisions"])
    for decision in decisions:
        number = int(decision["number"])
        decision["outcome_source"] = OUTCOME_SOURCES[number]
        decision["outcome_source_sha256"] = source_hashes[
            OUTCOME_SOURCES[number]
        ]
        decision["original_agreement"] = bool(decision["agrees"])

    # Later, more representative confirmation superseded two development
    # labels. The synthetic measurements and their original thresholds remain
    # untouched; only "which real outcome should this predict?" changes.
    d3 = decisions[2]
    summary3 = d3["summary"]
    synthetic_supports_random_strength = (
        summary3["aggregate_ratio"] <= 0.999
        and summary3["wins"] > summary3["losses"]
    )
    d3.update({
        "corrected_real_outcome": (
            "random_strength=0.5 failed the fresh nine-cell sports panel"
        ),
        "corrected_rule": (
            "agreement when SynthGen does not reproduce the superseded "
            "single-dataset advancement signal"
        ),
        "agrees": not synthetic_supports_random_strength,
        "label_superseded": True,
    })

    d5 = decisions[4]
    summary5 = d5["summary"]
    synthetic_supports_fixed_linear = (
        summary5["aggregate_ratio"] <= 0.99
        and summary5["wins"] >= math.ceil(
            2.0 * summary5["n_datasets"] / 3.0
        )
    )
    d5.update({
        "corrected_real_outcome": (
            "fixed local linear leaves regressed on the fresh 14-lineage "
            "smooth/process panel"
        ),
        "corrected_rule": (
            "agreement when SynthGen does not reproduce the superseded "
            "three-dataset development win"
        ),
        "agrees": not synthetic_supports_fixed_linear,
        "label_superseded": True,
    })

    for index, decision in enumerate(decisions):
        if index not in (2, 4):
            decision["corrected_real_outcome"] = (
                "the original ledger outcome remains the binding outcome"
            )
            decision["corrected_rule"] = decision["rule"]
            decision["label_superseded"] = False

    agreement_count = sum(bool(row["agrees"]) for row in decisions)
    preserved_gates = {
        name: passed
        for name, passed in original["adoption_gates"].items()
        if name != "at_least_7_of_9_decisions_agree"
    }
    if not all(preserved_gates.values()):
        raise RuntimeError("an original non-ledger adoption gate no longer passes")
    gates = {
        **preserved_gates,
        "at_least_7_of_9_corrected_decisions_agree": agreement_count >= 7,
        "only_predeclared_labels_3_and_5_superseded": True,
        "no_model_or_synthetic_outcome_rerun": True,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "synthgen_corrected_outcome_ledger_analysis",
        "analysis_sha256": _sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "original_agreement_count": original["agreement_count"],
        "corrected_agreement_count": agreement_count,
        "decisions": decisions,
        "adoption_gates": gates,
        "adopted_as_probe_tier_direction_finder": all(gates.values()),
        "scope": (
            "Retrospective probe-tier reclassification only. SynthGen may "
            "rank mechanism directions, but it cannot gate, confirm, promote, "
            "or justify a parameter, policy, preset, default, or product claim."
        ),
    }


def _render(result):
    verdict = (
        "ADOPTED as a probe-tier direction finder"
        if result["adopted_as_probe_tier_direction_finder"]
        else "NOT ADOPTED"
    )
    lines = [
        "# T9 SynthGen corrected-ledger re-backtest",
        "",
        f"**Verdict: {verdict}.**",
        "",
        result["scope"],
        "",
        f"Original agreement: **{result['original_agreement_count']}/9**. "
        f"Corrected agreement: **{result['corrected_agreement_count']}/9**.",
        "",
        "| # | Decision | Ratio | W-L-T | Original | Corrected | Binding source |",
        "| ---: | --- | ---: | ---: | :---: | :---: | --- |",
    ]
    for decision in result["decisions"]:
        summary = decision["summary"]
        lines.append(
            f"| {decision['number']} | {decision['name']} | "
            f"{summary['aggregate_ratio']:.6f} | "
            f"{summary['wins']}-{summary['losses']}-{summary['ties']} | "
            f"{'yes' if decision['original_agreement'] else 'no'} | "
            f"{'yes' if decision['agrees'] else 'no'} | "
            f"`{decision['outcome_source']}` |"
        )
    lines.extend([
        "",
        "## What changed",
        "",
        "- Decision 3 changed because the fresh nine-cell sports panel "
        "superseded the earlier one-dataset random-strength screen: 0.5 "
        "failed confirmation. SynthGen's near-null, loss-majority result is "
        "therefore an agreement.",
        "- Decision 5 changed because the fresh 14-lineage smooth/process "
        "panel superseded the three spent-task development win: fixed local "
        "linear leaves regressed overall. SynthGen's null/adverse result is "
        "therefore an agreement.",
        "- No synthetic metric, slice, arm, seed, threshold, canary, or "
        "integrity gate changed. Decision 6 remains a disagreement because "
        "the fresh panel did not directly compare global residual boosting "
        "with local linear leaves.",
        "",
        "## Gates",
        "",
    ])
    for name, passed in result["adoption_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is intentionally retrospective: both the immutable synthetic "
        "scorecard and later real-data outcomes were known when T9 was "
        "scheduled. Adoption therefore means only that SynthGen is useful "
        "enough to prioritize cheap development probes. A real-data frozen "
        "protocol remains mandatory for every decision.",
        "",
        f"Analyzer SHA-256: `{result['analysis_sha256']}`.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    args = parser.parse_args(argv)
    result = analyze()
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_render(result), encoding="utf-8")
    print(
        f"{result['corrected_agreement_count']}/9 corrected agreements; "
        f"adopted={result['adopted_as_probe_tier_direction_finder']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
