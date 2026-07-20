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

## Second no-outcome launch

The next launch passed source validation but stopped while importing the
fused historical runner, again before any model fit or replay outcome. The
parent had inherited a workstation `PYTHONPATH` containing another
repository's `benchmarks` package. Because the exact historical source root
also appeared later on that path, the historical runner did not move it to
the front and imported the wrong package.

The repair removes inherited `PYTHONPATH` from every replay-worker
environment. Each exact historical runner inserts its own repository root,
and the selector worker explicitly prepends its pinned source. A named test
now enforces the isolated environment. The declared evidence contract remains
unchanged, and no raw or combined artifact was created.

An outcome-free import probe before the next launch then established why
removing the variable alone was insufficient on this workstation: the
historical `benchmarks/` directories are namespace-only, while an unrelated
installed checkout exposes a regular `benchmarks` package and therefore wins
Python's package resolution. The executor now creates a temporary regular
package shim whose only search path is the exact historical source's
`benchmarks/` directory, then places that shim and source first for the fused
and packed subprocess trees. The historical clones remain byte-unchanged and
clean. A regression test pins the shim contents and path order.
