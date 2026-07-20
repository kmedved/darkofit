# Invalid M6 historical-backtest launch

On 2026-07-20, the first formal backtest command stopped in source preflight
before creating its temporary execution directory or launching any fused,
packed, or selector replay.

The executor built a source map with key `chimeraboost_015` and passed it by
keyword to a validator whose corresponding parameter was named
`chimera_015`. Python raised `TypeError` before any historical outcome was
accessed. No raw or combined artifact was created.

The repair changes only the validator parameter name and converts the source
validation test from positional to the exact keyword call used by the parent.
The declared subset, source pins, cases, thresholds, analyzers, and model
policies are unchanged. Because the failure occurred before replay execution,
the same frozen backtest may be relaunched after the repair commit.
