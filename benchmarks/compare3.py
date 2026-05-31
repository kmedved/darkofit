"""N-corner comparison for the autonomous experiment session.

Loads up to several result JSONs, restricts every one to the set of models they
ALL contain (so "% vs best" is computed against an identical field -- otherwise a
run that happened to include ChimeraBoostEns10 has a different "best" column and
the comparison is apples-to-oranges), prints each corner's headline table, then
the ChimeraBoost deltas between consecutive corners.

Usage:
    python benchmarks/compare3.py BASE.json EXP.json [EXP2.json ...] \
        --labels "clf OB on" "clf OB off" "depth 8"
"""
import argparse
import copy

import summarize


def restrict_models(data, keep):
    d = copy.deepcopy(data)
    d["records"] = [r for r in d["records"] if r["model"] in keep]
    return d


def common_models(datas):
    sets = [{r["model"] for r in d["records"]} for d in datas]
    return set.intersection(*sets) if sets else set()


def _delta_line(col, bv, nv):
    if bv is None or nv is None:
        return None
    if col == "Bin Calib":
        dd = (bv - nv) * 1000
        return (f"  {col:<12} {bv*1000:.2f}m -> {nv*1000:.2f}m  "
                f"({dd:+.2f}m {'better' if dd > 0 else 'worse'})")
    if col == "Speed":
        dd = bv - nv
        return (f"  {col:<12} {bv:.1f}x -> {nv:.1f}x  "
                f"({dd:+.1f}x {'faster' if dd > 0 else 'slower'})")
    dd = nv - bv
    return f"  {col:<12} {bv:.1f}% -> {nv:.1f}%  ({dd:+.1f}pp)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--focus", default="ChimeraBoost")
    args = ap.parse_args()

    datas = [summarize.load(p) for p in args.jsons]
    keep = common_models(datas)
    datas = [restrict_models(d, keep) for d in datas]
    labels = args.labels or [f"corner{i}" for i in range(len(datas))]

    for d, lab in zip(datas, labels):
        print(summarize.format_table(d, f"=== {lab} ==="))
        print()

    aggs = [summarize.aggregate(d)[0] for d in datas]
    for i in range(1, len(datas)):
        a, b = aggs[i - 1], aggs[i]
        print(f"=== {args.focus} delta: {labels[i]} vs {labels[i-1]} ===")
        for c in summarize.COLS:
            line = _delta_line(c, a[c].get(args.focus), b[c].get(args.focus))
            if line:
                print(line)
        print()


if __name__ == "__main__":
    main()
