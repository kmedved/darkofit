"""Run DarkoFit on TabArena's official three-dataset Lite quickstart scope."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

from benchmarks.tabarena_adapter import DarkoFitModel


DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-configs",
        type=int,
        default=1,
        help="Random HPO configurations in addition to DarkoFit's default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/tabarena-smoke"),
        help="Resumable experiment and leaderboard output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_configs < 0:
        raise ValueError("--n-configs must be nonnegative")

    output_dir = args.output_dir.resolve()
    experiment_dir = output_dir / "experiments"
    evaluation_dir = output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(DarkoFitModel.config_generator(), args.n_configs)],
    ).build_experiments()

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(experiment_dir),
        subset="lite",
        build_kwargs={"dataset_names": DATASETS},
        new_result_prefix="[DarkoFit smoke] ",
        debug_mode=True,
    )

    leaderboard = context.compare(output_dir=evaluation_dir)
    website_leaderboard = context.leaderboard_to_website_format(leaderboard=leaderboard)
    darkofit_mask = website_leaderboard["Model"].str.contains(
        "DarkoFit", case=False, na=False
    )
    # This file is a local comparison artifact, not an official TabArena result.
    website_leaderboard.loc[darkofit_mask, "Verified"] = "➖"
    leaderboard_path = output_dir / "website_leaderboard.csv"
    website_leaderboard.to_csv(leaderboard_path, index=False)

    darkofit_rows = website_leaderboard[darkofit_mask]
    if darkofit_rows.empty:
        raise RuntimeError("DarkoFit was not found in the generated leaderboard")

    print("\n=== DarkoFit three-dataset TabArena-Lite smoke ===")
    print(darkofit_rows.to_string(index=False))
    print(f"\nSaved leaderboard: {leaderboard_path}")
    print(f"Saved evaluation artifacts: {evaluation_dir}")


if __name__ == "__main__":
    main()
