from pathlib import Path

import benchmarks


def test_local_benchmark_package_wins_import_resolution():
    expected = Path(__file__).resolve().parents[1] / "benchmarks"

    assert Path(benchmarks.__file__).resolve().parent == expected
