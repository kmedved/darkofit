"""Run the frozen remaining-nine TabArena regression confirmation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path


TASK_SPLIT_COUNTS = {
    "Another-Dataset-on-used-Fiat-500": (363615, 30),
    "concrete_compressive_strength": (363625, 30),
    "Food_Delivery_Time": (363672, 9),
    "healthcare_insurance_expenses": (363675, 30),
    "houses": (363678, 9),
    "miami_housing": (363686, 9),
    "QSAR-TID-11": (363697, 9),
    "QSAR_fish_toxicity": (363698, 30),
    "wine_quality": (363708, 9),
}
TASK_IDS = [task_id for task_id, _ in TASK_SPLIT_COUNTS.values()]
SPLIT_INDICES = [f"r{repeat}f{fold}" for repeat in range(10) for fold in range(3)]
EXPECTED_DATASET_SPLITS = sum(count for _, count in TASK_SPLIT_COUNTS.values())
EXPECTED_JOBS = 2 * EXPECTED_DATASET_SPLITS
FROZEN_CANDIDATE = {
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
}


def validate_chimera_coverage(results) -> None:
    """Require one non-imputed CHIMERA default row per frozen dataset-split."""
    selected = results[
        results["dataset"].isin(TASK_SPLIT_COUNTS)
        & (results["method"] == "CHIMERA (default)")
    ]
    if selected["imputed"].astype(bool).any():
        raise RuntimeError("registered CHIMERA coverage contains imputed rows")
    duplicates = selected.duplicated(["dataset", "fold"], keep=False)
    if duplicates.any():
        raise RuntimeError("registered CHIMERA coverage contains duplicate rows")
    actual = selected.groupby("dataset")["fold"].agg(["count", "min", "max"])
    for dataset, (_, expected_count) in TASK_SPLIT_COUNTS.items():
        if dataset not in actual.index:
            raise RuntimeError(f"missing registered CHIMERA coverage for {dataset}")
        row = actual.loc[dataset]
        if (
            int(row["count"]) != expected_count
            or int(row["min"]) != 0
            or int(row["max"]) != expected_count - 1
        ):
            raise RuntimeError(
                f"unexpected registered CHIMERA folds for {dataset}: "
                f"count={int(row['count'])}, min={int(row['min'])}, "
                f"max={int(row['max'])}; expected 0..{expected_count - 1}"
            )
    if len(selected) != EXPECTED_DATASET_SPLITS:
        raise RuntimeError(
            f"expected {EXPECTED_DATASET_SPLITS} CHIMERA rows, got {len(selected)}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/tabarena-regression-remaining9-0.9.0-20260712"),
    )
    parser.add_argument("--time-limit", type=int, default=3_600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.time_limit < 1:
        parser.error("--time-limit must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    from benchmarks.tabarena_adapter import DarkoFitModel

    context = TabArenaContext()
    validate_chimera_coverage(context.load_results(methods=["ChimeraBoost"]))

    generator = ConfigGenerator(
        model_cls=DarkoFitModel,
        manual_configs=[{}, dict(FROZEN_CANDIDATE)],
        search_space={},
    )
    experiments = generator.generate_all_bag_experiments(
        num_random_configs=0,
        name_id_suffix="_remaining9_confirm",
        add_seed="fold-wise",
        fold_fitting_strategy="sequential_local",
        time_limit=args.time_limit,
    )
    jobs = context.build_jobs(
        experiments,
        task_ids=TASK_IDS,
        split_indices=SPLIT_INDICES,
    )
    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, built {len(jobs)}")

    print(
        f"validated {EXPECTED_DATASET_SPLITS} registered CHIMERA rows; "
        f"built {len(jobs)} DarkoFit jobs"
    )
    if args.dry_run:
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = context.run_jobs(
        jobs,
        expname=str(output_dir / "experiments"),
        new_result_prefix="[DarkoFit remaining-nine confirmation] ",
        debug_mode=True,
    )
    print(f"REMAINING9_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
