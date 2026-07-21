"""Frozen-shape contract for the M5 sentinels and M6 development slice."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

try:
    from benchmark_adapters import DATASETS, SIZE_SAMPLES
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import DATASETS, SIZE_SAMPLES


CONTRACT_VERSION = "standing-evidence-v3"
M6_CONTRACT_FROZEN = True
M6_BACKTEST_COMPLETE = False
M6_BACKTEST_TERMINAL = True
M6_RELEASE_ANCHOR_EVIDENCE_PATH = "benchmarks/m6_release_anchors.json"
M6_RELEASE_ANCHOR_EVIDENCE_SHA256 = (
    "59747bc08d48a2ddad9b3cec05c965ecbd9edf21025c537f17dc58d816385409"
)
M6_BACKTEST_EVIDENCE_PATH = ""
M6_BACKTEST_EVIDENCE_SHA256 = ""
M6_BACKTEST_FAILURE_EVIDENCE_PATH = (
    "benchmarks/m6_historical_backtest_failure.json"
)
M6_BACKTEST_FAILURE_EVIDENCE_SHA256 = (
    "18b902e6099a4686b8eda71fac9ac327a0b5243872b80b5da79c5e01e5e2c201"
)
M5_CONTRACT_VERSION = "m5-sentinels-v1"
M5_CONTRACT_FROZEN = True
M5_BASELINE_EVIDENCE_PATH = "benchmarks/m5_sentinel_baseline.json"
M5_BASELINE_EVIDENCE_SHA256 = (
    "0971e06d4ed307d352d75e1e6400b849c0001b5e11f40243173d7080b6c5859d"
)
M5_CONTROL_SOURCE = "726e5d8e6131c580bce948db833a5007d0692dca"
M5_THREADS = 4
M5_ARMS = ("control", "candidate")


@dataclass(frozen=True)
class SentinelDomain:
    id: str
    task: str
    source: str
    weighted: bool = False


@dataclass(frozen=True)
class SentinelCase:
    domain_id: str
    dataset_key: str
    seeds: tuple[int, ...]
    model_profile: str
    expected_normalized_loss_max: float
    known_floor: str = ""
    dataset_sha256: str = ""


@dataclass(frozen=True)
class ReleaseAnchor:
    id: str
    version: str
    source_pin: str


@dataclass(frozen=True)
class BacktestVerdict:
    mechanism_id: str
    expected_disposition: str
    primary_axis: str
    control_source: str
    candidate_source: str
    replay_adapter: str
    historical_result: str
    historical_result_sha256: str
    replay_cases: tuple[str, ...]
    advance_rule: str
    max_stability_iqr_fraction: float


M5_SENTINEL_DOMAINS = (
    SentinelDomain(
        "grouped_entity_regression",
        "regression",
        "generic_group_generator",
    ),
    SentinelDomain(
        "smooth_numeric_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "noisy_numeric_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "categorical_missing_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "high_row_numeric",
        "regression",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "binary_classification",
        "binary",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "multiclass_classification",
        "multiclass",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "weighted_regression",
        "regression",
        "benchmark_adapters",
        weighted=True,
    ),
    SentinelDomain(
        "weighted_classification",
        "binary",
        "benchmark_adapters",
        weighted=True,
    ),
)

M5_SENTINEL_CASES = (
    SentinelCase(
        "grouped_entity_regression",
        "generic:grouped-entity-v1",
        (0, 1),
        "group_ensemble",
        1.10,
    ),
    SentinelCase(
        "smooth_numeric_regression",
        "syn:df1/311",
        (0, 1),
        "standard",
        1.10,
        dataset_sha256=(
            "a323d3f5ddaaf570ecc765b7810d7415344d81de7e3d01bd44d3cacda52822ae"
        ),
    ),
    SentinelCase(
        "noisy_numeric_regression",
        "syn:df1/241",
        (0, 1),
        "standard",
        1.10,
        dataset_sha256=(
            "99d3068ded75bc7704c689a751d4582c33ff500b72da840f328bfbba733bcbb0"
        ),
    ),
    SentinelCase(
        "categorical_missing_regression",
        "syn:df1/234",
        (0, 1),
        "standard",
        1.10,
        dataset_sha256=(
            "cf660a560e6ec769fb10ec259863156d22dadec2fa64036dc3558403a86902dc"
        ),
    ),
    SentinelCase(
        "high_row_numeric",
        "adapter:friedman_numeric:large",
        (0,),
        "high_row",
        1.10,
    ),
    SentinelCase(
        "binary_classification",
        "syn:df1/647",
        (0, 1, 2),
        "canary",
        1.10,
        known_floor="excess_brier_mean<=0.005;worst<=0.01",
        dataset_sha256=(
            "35df5d94bdd4aaa96924ea7ed1d06cef964e780720065b969939e61dd3bd0fda"
        ),
    ),
    SentinelCase(
        "multiclass_classification",
        "syn:df1/077",
        (0, 1, 2),
        "canary",
        1.10,
        known_floor="excess_brier_mean<=0.005;worst<=0.01",
        dataset_sha256=(
            "fe0a27a60186de31e22197b853966ceb9b857ea5bc91af9dff4f061c4f4fa658"
        ),
    ),
    SentinelCase(
        "weighted_regression",
        "adapter:wide_numeric_reg:medium",
        (0, 1),
        "standard",
        1.10,
    ),
    SentinelCase(
        "weighted_classification",
        "adapter:numeric_binary:medium",
        (0, 1),
        "standard",
        1.10,
    ),
)

M6_DATASETS = (
    "diabetes_resampled",
    "friedman_numeric",
    "wide_numeric_reg",
    "categorical_reg",
    "breast_cancer_resampled",
    "numeric_binary",
    "wine_resampled",
    "numeric_multiclass",
    "categorical_binary",
    "categorical_multiclass",
)
M6_SMOKE_DATASETS = (
    "friedman_numeric",
    "numeric_binary",
    "categorical_binary",
)
M6_MODELS = ("control_default", "candidate_default")
M6_SIZES = ("small", "medium")
M6_REQUIRED_FREEZE_SIZES = ("small", "medium")
M6_REQUIRED_RELEASE_ANCHORS = ("chimeraboost", "catboost")
M6_RELEASE_ANCHORS = (
    ReleaseAnchor(
        "chimeraboost",
        "0.18.0.dev6",
        "git:f14be606b641f1bf0dc92bb14b3951f1fe631c6b",
    ),
    ReleaseAnchor(
        "catboost",
        "1.2.10",
        (
            "record-sha256:"
            "9c20fb35750d9ff814309323b225e836b538c1496745f357c8fd50187e7824ed"
        ),
    ),
)
M6_SEED_COUNT = 3
M6_WEIGHT_MODES = ("none", "stress")
M6_REPEAT = 1
M6_THREADS = 4
_REPO_ROOT = Path(__file__).resolve().parents[1]

M6_BACKTEST_VERDICTS = (
    BacktestVerdict(
        mechanism_id="fused_variable_hessian",
        expected_disposition="advance",
        primary_axis="fit_speed",
        control_source="7097e7ac6125cb260ae67ee353458a2cb12fe2e1",
        candidate_source="1016e7e8d70c403a70feab7762de8837ea8fd09c",
        replay_adapter="exact_historical_internal_toggle_runner",
        historical_result="benchmarks/fused_variable_hessian_result.md",
        historical_result_sha256=(
            "d22337f4bab69bba7a13b9d3bca583a41aaa873a10a1974ca11d996244febac3"
        ),
        replay_cases=("binary_logloss_50k", "weighted_rmse_50k"),
        advance_rule=(
            "advance iff behavior is exact, candidate engagement is positive, "
            "reference engagement is zero, the fit-ratio geometric mean is "
            "<=0.90, and every paired fit series has IQR/median <=0.10"
        ),
        max_stability_iqr_fraction=0.10,
    ),
    BacktestVerdict(
        mechanism_id="forest_work_packed_router",
        expected_disposition="kill",
        primary_axis="predict_speed",
        control_source="e0899435e166f8c4856e5f8f77db1e0fa71c322f",
        candidate_source="e961bcc2ea64706169641722b5935f9f31402fa3",
        replay_adapter="exact_historical_candidate_vs_legacy_runner",
        historical_result="benchmarks/basketball_packed_prediction_result.md",
        historical_result_sha256=(
            "9c8d636f467fab118a492ef64194ec48eb6c800f24d8f31bb73f42039296a7f4"
        ),
        replay_cases=(
            "repeated_127",
            "repeated_525",
            "repeated_585",
            "repeated_2409",
            "repeated_8192",
            "repeated_100000",
        ),
        advance_rule=(
            "advance iff predictions are exact, the candidate route engages, "
            "all timing series have IQR/median <=0.30, the 525- and 585-row "
            "candidate cores are >=2x faster than legacy, and the 8192- and "
            "100000-row candidate/legacy core ratios are <=1.10"
        ),
        max_stability_iqr_fraction=0.30,
    ),
    BacktestVerdict(
        mechanism_id="linear_leaf_selector_3pct",
        expected_disposition="kill",
        primary_axis="quality",
        control_source="29bd30cdcf476139c30efe4e09773ca812ba443f",
        candidate_source="29bd30cdcf476139c30efe4e09773ca812ba443f",
        replay_adapter="source_pinned_m6_selector_3pct",
        historical_result="benchmarks/fresh_selector_confirmation_result.md",
        historical_result_sha256=(
            "3a33ec834bcebb9d9c9e2db4d69a5119f35ccbcf7623bf3dedb839d15ef71170"
        ),
        replay_cases=(
            "friedman_numeric_small",
            "friedman_numeric_medium",
            "wide_numeric_reg_small",
            "wide_numeric_reg_medium",
            "categorical_reg_small",
            "categorical_reg_medium",
        ),
        advance_rule=(
            "advance iff the selector/default geometric-mean RMSE ratio is "
            "<=0.98, it wins at least 4 of 6 cells, no cell ratio exceeds "
            "1.02, and every selected arm is chosen by the frozen 3% internal "
            "validation-margin policy"
        ),
        max_stability_iqr_fraction=0.0,
    ),
)


def m6_freeze_blockers() -> tuple[str, ...]:
    """Return concrete reasons the draft cannot yet be marked frozen."""
    blockers = []
    missing_sizes = sorted(set(M6_REQUIRED_FREEZE_SIZES) - set(M6_SIZES))
    if missing_sizes:
        blockers.append(f"missing required M6 sizes: {missing_sizes}")
    anchor_ids = {anchor.id for anchor in M6_RELEASE_ANCHORS}
    missing_anchors = sorted(
        set(M6_REQUIRED_RELEASE_ANCHORS) - anchor_ids
    )
    if missing_anchors:
        blockers.append(f"missing pinned M6 release anchors: {missing_anchors}")
    if (
        not M6_RELEASE_ANCHOR_EVIDENCE_PATH
        or not M6_RELEASE_ANCHOR_EVIDENCE_SHA256
    ):
        blockers.append("missing hash-bound M6 release-anchor evidence")
    return tuple(blockers)


def m5_freeze_blockers() -> tuple[str, ...]:
    blockers = []
    if not M5_BASELINE_EVIDENCE_PATH or not M5_BASELINE_EVIDENCE_SHA256:
        blockers.append("missing hash-bound M5 baseline evidence")
    return tuple(blockers)


def _validate_evidence_binding(path: str, expected_sha256: str, label: str):
    if not path and not expected_sha256:
        return
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise RuntimeError(f"{label} evidence path is invalid")
    if (
        len(expected_sha256) != 64
        or set(expected_sha256) - set("0123456789abcdef")
    ):
        raise RuntimeError(f"{label} evidence hash is invalid")
    evidence_path = _REPO_ROOT / path
    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise RuntimeError(f"{label} evidence is missing: {path}")
    actual = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} evidence hash drifted")


def validate_contract():
    """Fail closed if the standing contract drifts out of adapter coverage."""
    domain_ids = [domain.id for domain in M5_SENTINEL_DOMAINS]
    if len(domain_ids) != len(set(domain_ids)):
        raise RuntimeError("M5 sentinel domain ids must be unique")
    if not any(domain.task == "binary" for domain in M5_SENTINEL_DOMAINS):
        raise RuntimeError("M5 must cover binary classification")
    if not any(domain.task == "multiclass" for domain in M5_SENTINEL_DOMAINS):
        raise RuntimeError("M5 must cover multiclass classification")
    weighted_tasks = {
        domain.task for domain in M5_SENTINEL_DOMAINS if domain.weighted
    }
    if not {"regression", "binary"}.issubset(weighted_tasks):
        raise RuntimeError("M5 must cover weighted regression and classification")
    case_domains = [case.domain_id for case in M5_SENTINEL_CASES]
    if len(case_domains) != len(set(case_domains)):
        raise RuntimeError("M5 sentinel case domains must be unique")
    if set(case_domains) != set(domain_ids):
        raise RuntimeError("M5 sentinel cases do not cover the domain registry")
    for case in M5_SENTINEL_CASES:
        if (
            not case.seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in case.seeds
            )
            or len(case.seeds) != len(set(case.seeds))
        ):
            raise RuntimeError(
                f"M5 sentinel {case.domain_id!r} has invalid seeds"
            )
        if (
            not math.isfinite(case.expected_normalized_loss_max)
            or case.expected_normalized_loss_max <= 0.0
        ):
            raise RuntimeError(
                f"M5 sentinel {case.domain_id!r} has an invalid quality range"
            )
        if case.dataset_sha256 and (
            len(case.dataset_sha256) != 64
            or set(case.dataset_sha256) - set("0123456789abcdef")
        ):
            raise RuntimeError(
                f"M5 sentinel {case.domain_id!r} has an invalid dataset hash"
            )
    if M5_CONTRACT_FROZEN:
        blockers = m5_freeze_blockers()
        if blockers:
            raise RuntimeError(
                "M5 contract cannot be frozen: " + "; ".join(blockers)
            )
        _validate_evidence_binding(
            M5_BASELINE_EVIDENCE_PATH,
            M5_BASELINE_EVIDENCE_SHA256,
            "M5 baseline",
        )

    unknown_datasets = sorted(set(M6_DATASETS) - set(DATASETS))
    if unknown_datasets:
        raise RuntimeError(f"unknown M6 datasets: {unknown_datasets}")
    unknown_sizes = sorted(set(M6_SIZES) - set(SIZE_SAMPLES))
    if unknown_sizes:
        raise RuntimeError(f"unknown M6 sizes: {unknown_sizes}")
    anchor_ids = [anchor.id for anchor in M6_RELEASE_ANCHORS]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise RuntimeError("M6 release anchor ids must be unique")
    if set(anchor_ids) != set(M6_REQUIRED_RELEASE_ANCHORS):
        raise RuntimeError("M6 release anchors must exactly match the required set")
    for anchor in M6_RELEASE_ANCHORS:
        if not anchor.version or not anchor.source_pin:
            raise RuntimeError(
                f"M6 release anchor {anchor.id!r} is not fully pinned"
            )
        if ":" not in anchor.source_pin:
            raise RuntimeError(
                f"M6 release anchor {anchor.id!r} has an untyped source pin"
            )
    if not set(M6_SMOKE_DATASETS).issubset(M6_DATASETS):
        raise RuntimeError("M6 smoke datasets must be a subset of the full slice")
    tasks = {DATASETS[name].task for name in M6_DATASETS}
    if tasks != {"regression", "binary", "multiclass"}:
        raise RuntimeError(f"M6 task coverage drifted: {sorted(tasks)}")

    verdict_ids = [
        verdict.mechanism_id for verdict in M6_BACKTEST_VERDICTS
    ]
    if len(verdict_ids) != len(set(verdict_ids)):
        raise RuntimeError("M6 backtest mechanism ids must be unique")
    dispositions = {
        verdict.expected_disposition for verdict in M6_BACKTEST_VERDICTS
    }
    if not {"advance", "kill"}.issubset(dispositions):
        raise RuntimeError(
            "M6 backtest subset must include positive and negative verdicts"
        )
    if dispositions - {"advance", "kill"}:
        raise RuntimeError(
            f"unknown M6 backtest dispositions: "
            f"{sorted(dispositions - {'advance', 'kill'})}"
        )
    for verdict in M6_BACKTEST_VERDICTS:
        if not all(
            (
                verdict.primary_axis,
                verdict.control_source,
                verdict.candidate_source,
                verdict.replay_adapter,
                verdict.historical_result,
                verdict.historical_result_sha256,
                verdict.replay_cases,
                verdict.advance_rule,
            )
        ):
            raise RuntimeError(
                f"M6 backtest verdict {verdict.mechanism_id!r} is incomplete"
            )
        if len(verdict.historical_result_sha256) != 64:
            raise RuntimeError(
                f"M6 backtest verdict {verdict.mechanism_id!r} has an "
                "invalid result hash"
            )
        if (
            not math.isfinite(verdict.max_stability_iqr_fraction)
            or verdict.max_stability_iqr_fraction < 0.0
        ):
            raise RuntimeError(
                f"M6 backtest verdict {verdict.mechanism_id!r} has an "
                "invalid stability limit"
            )
        for source_pin in (verdict.control_source, verdict.candidate_source):
            if len(source_pin) != 40 or set(source_pin) - set(
                "0123456789abcdef"
            ):
                raise RuntimeError(
                    f"M6 backtest verdict {verdict.mechanism_id!r} has an "
                    "invalid source pin"
                )
        result_path = _REPO_ROOT / verdict.historical_result
        if not result_path.is_file():
            raise RuntimeError(
                f"M6 backtest result is missing: {verdict.historical_result}"
            )
        actual_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        if actual_hash != verdict.historical_result_sha256:
            raise RuntimeError(
                f"M6 backtest result hash drifted for "
                f"{verdict.mechanism_id!r}"
            )

    blockers = m6_freeze_blockers()
    if M6_CONTRACT_FROZEN and blockers:
        raise RuntimeError(
            "M6 contract cannot be frozen: " + "; ".join(blockers)
        )
    _validate_evidence_binding(
        M6_RELEASE_ANCHOR_EVIDENCE_PATH,
        M6_RELEASE_ANCHOR_EVIDENCE_SHA256,
        "M6 release-anchor",
    )
    if M6_BACKTEST_COMPLETE:
        if not M6_CONTRACT_FROZEN:
            raise RuntimeError("M6 backtest cannot complete before contract freeze")
        if not M6_BACKTEST_EVIDENCE_PATH or not M6_BACKTEST_EVIDENCE_SHA256:
            raise RuntimeError(
                "M6 backtest cannot complete without hash-bound evidence"
            )
        _validate_evidence_binding(
            M6_BACKTEST_EVIDENCE_PATH,
            M6_BACKTEST_EVIDENCE_SHA256,
            "M6 backtest",
        )
    if M6_BACKTEST_TERMINAL:
        if M6_BACKTEST_COMPLETE:
            raise RuntimeError(
                "M6 backtest cannot be both complete and terminal-failed"
            )
        if (
            not M6_BACKTEST_FAILURE_EVIDENCE_PATH
            or not M6_BACKTEST_FAILURE_EVIDENCE_SHA256
        ):
            raise RuntimeError(
                "terminal M6 backtest requires hash-bound failure evidence"
            )
        _validate_evidence_binding(
            M6_BACKTEST_FAILURE_EVIDENCE_PATH,
            M6_BACKTEST_FAILURE_EVIDENCE_SHA256,
            "M6 backtest failure",
        )


def m6_expected_grid(*, smoke=False):
    """Return the exact expected M6 row identities for validation and tests."""
    datasets = M6_SMOKE_DATASETS if smoke else M6_DATASETS
    seed_count = 1 if smoke else M6_SEED_COUNT
    return tuple(
        product(
            M6_MODELS,
            datasets,
            M6_SIZES,
            range(seed_count),
            M6_WEIGHT_MODES,
        )
    )


def m5_expected_grid():
    return tuple(
        (arm, case.domain_id, seed)
        for case in M5_SENTINEL_CASES
        for seed in case.seeds
        for arm in M5_ARMS
    )


def contract_payload():
    """Return a JSON-ready representation suitable for provenance hashing."""
    return {
        "contract_version": CONTRACT_VERSION,
        "m5": {
            "contract_version": M5_CONTRACT_VERSION,
            "contract_frozen": M5_CONTRACT_FROZEN,
            "freeze_blockers": list(m5_freeze_blockers()),
            "control_source": M5_CONTROL_SOURCE,
            "threads": M5_THREADS,
            "arms": list(M5_ARMS),
            "sentinel_domains": [
                asdict(domain) for domain in M5_SENTINEL_DOMAINS
            ],
            "sentinel_cases": [
                asdict(case) for case in M5_SENTINEL_CASES
            ],
            "baseline_evidence": {
                "path": M5_BASELINE_EVIDENCE_PATH,
                "sha256": M5_BASELINE_EVIDENCE_SHA256,
            },
        },
        "m6": {
            "contract_frozen": M6_CONTRACT_FROZEN,
            "backtest_complete": M6_BACKTEST_COMPLETE,
            "backtest_terminal": M6_BACKTEST_TERMINAL,
            "freeze_blockers": list(m6_freeze_blockers()),
            "datasets": list(M6_DATASETS),
            "smoke_datasets": list(M6_SMOKE_DATASETS),
            "models": list(M6_MODELS),
            "sizes": list(M6_SIZES),
            "required_freeze_sizes": list(M6_REQUIRED_FREEZE_SIZES),
            "required_release_anchors": list(M6_REQUIRED_RELEASE_ANCHORS),
            "release_anchors": [
                asdict(anchor) for anchor in M6_RELEASE_ANCHORS
            ],
            "release_anchor_evidence": {
                "path": M6_RELEASE_ANCHOR_EVIDENCE_PATH,
                "sha256": M6_RELEASE_ANCHOR_EVIDENCE_SHA256,
            },
            "seed_count": M6_SEED_COUNT,
            "weight_modes": list(M6_WEIGHT_MODES),
            "repeat": M6_REPEAT,
            "threads": M6_THREADS,
            "backtest_verdicts": [
                asdict(verdict) for verdict in M6_BACKTEST_VERDICTS
            ],
            "backtest_evidence": {
                "path": M6_BACKTEST_EVIDENCE_PATH,
                "sha256": M6_BACKTEST_EVIDENCE_SHA256,
            },
            "backtest_failure_evidence": {
                "path": M6_BACKTEST_FAILURE_EVIDENCE_PATH,
                "sha256": M6_BACKTEST_FAILURE_EVIDENCE_SHA256,
            },
        },
    }


validate_contract()
