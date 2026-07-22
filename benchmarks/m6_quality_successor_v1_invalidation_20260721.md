# M6 quality successor v1 binding invalidation

The one-shot `m6-quality-successor-v1` calculation completed from clean
checkpoint `8abf0b8` and reproduced both declared historical dispositions, but
it does **not** activate candidate ranking.

Before the result-binding checkpoint, a contract audit found three structural
problems:

1. The result bound the whole-file SHA-256 of `m6_quality_successor.py`, while
   that same file stored `BACKTEST_COMPLETE=False`. Changing the flag to enable
   ranking would change the analyzer hash the result claimed to bind.
2. The future analyzer accepted strict `paired-evidence-v1` rows but those rows
   do not record the requested repeat count. A one-repeat CSV could therefore
   satisfy a contract that claimed three repeats.
3. The documented `--datasets all` command was not an exact frozen coordinate
   list if new adapters were added later.

These are evidence-integrity bugs, not changes to either historical verdict.
The create-only v1 output remains at
[`m6_quality_successor_backtest_result.json`](m6_quality_successor_backtest_result.json),
SHA-256 `360a60130c99220a3466ff0fab40b54ead99a2d0a29a2bde3a33a12e38500baa`,
and is labeled a completed calculation with no forward authority. It will not
be edited, rebound, or rerun.

A new v2 identity must keep the quality rule and declared subset unchanged,
separate immutable decision code from activation state, invoke the paired
runner through an exact committed command wrapper, record its repeat count,
and pass its own one-shot backtest. Until then M6 remains non-ranking.
