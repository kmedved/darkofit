# Wave 2 M3b private ensemble-v3 attribution protocol, attempt 2

_Prospective successor amendment. It becomes frozen only when the create-only
attempt-2 machine contract binds this file, the unchanged attempt-1 protocol,
implementation, base harness, successor wrappers, source pin, case
fingerprints, and decision rules._

## Why a successor identity exists

Attempt 1 (`wave2_m3b_ensemble_v3_20260720`) passed data/source preflight but
terminated on entry to the first worker's RSS sampler, before any model fit.
The inherited M3a sampler called `psutil.Process.children(recursive=True)`,
which macOS implements through `sysctl(KERN_PROC_ALL)`; the managed execution
sandbox denied that operation. The create-only failure artifact discarded
zero completed rows. Its immutable digest and disposition are preserved in
`m3b_ensemble_v3_attempt1_failure_record.json`. Attempt 1 remains terminal and
must not be rerun.

This is an evidence-harness capability defect, not a model result. Attempt 2
therefore receives a new contract identity. No attempt-1 model outcome exists
or informed this amendment.

## Sole methodological amendment

Attempt 2 replaces process-tree RSS with peak RSS of the fresh worker process
itself. This is the correct measurement boundary for the funded prototype:
all eight members fit sequentially in one worker, B3/parallel members are
excluded, and the model path launches no child model workers. Numba/OpenMP
threads remain part of that process's RSS. The sampler otherwise retains the
same 10 ms interval, entry/exit samples, peak calculation, sample count, and
fail-closed error recording.

Before contract construction and again before a parent or worker may consume
a formal attempt, a capability probe must obtain a positive finite RSS value
from `psutil.Process().memory_info().rss`. A failed capability probe is a
preflight failure and creates no model evidence. Once that probe and the
unchanged source/data preflight pass, the base protocol's create-only terminal
failure rule applies.

The memory rules remain unchanged:

- candidate/control peak worker-RSS geometric mean at most `1.10`; and
- median candidate/single peak worker-RSS ratio at most `2.0`.

The new artifact labels the field `peak_rss_bytes` consistently with attempt
1, while its contract records `rss_scope: self_worker_process`. Attempt-1
process-tree values do not exist and are not mixed with attempt-2 values.

## Everything else remains fixed

The complete arm definitions, source implementation, 13 cases and splits,
weights, quality metrics, arm order, four-thread environment, two-round
warmup, 600-round/30-patience fit, OOB telemetry, quality-first eligibility
thresholds, selective timing repeats, resource/value gates, deterministic
disposition, and forbidden public/B3/fresh/TabArena/lockbox claims remain
exactly as declared in `m3b_ensemble_v3_protocol.md` and the attempt-1 machine
contract. The attempt-2 machine contract rebinds all of them prospectively.
