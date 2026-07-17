# Invalid group-linear-selector attempt

Two formal invocations completed all six workers but failed closed before
writing a result artifact:

```text
RuntimeError: selector behavior changed across repeats
```

The first invocation's fingerprint included raw NPZ archive byte length. The
archive serializes `verbose_timing` phase durations, whose decimal string
lengths can differ between otherwise identical fits. The binding canonical
model-state hash already replaces only that observational timing field with
`null`; raw container length is therefore not model behavior and was removed
from the fingerprint payload.

The second invocation proved a second observational leak: process peak RSS
was included in the behavior fingerprint. A direct structural comparison
between independent candidate workers found no difference in predictions,
selection decisions, split identities, fitted metadata, or canonical model
payloads; only peak RSS differed. Peak RSS remains in the reciprocal resource
gate but is excluded from the behavior fingerprint.

No block was dropped, no result file existed, no quality or timing decision
threshold changed, and neither invalid attempt can support a result claim.
The protocol is unchanged. The corrected runner must be committed and the
complete reciprocal campaign rerun from clean source.
