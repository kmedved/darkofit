# v0.11 M2 broad-panel v2 preflight failure

Status: **terminal before worker execution; zero fits and zero outcome artifacts.**

On 2026-07-22, the published v2 contract at `c5dad48` passed its dry run and
was invoked once for formal execution from its exact clean source and pinned
framework environment. The parent wrote the run manifest, then stopped before
warmup because `tabarena_comparator_warmup.py` still required its historical
18-thread constant while the v2 protocol correctly supplied 14.

The harness wrote `terminal.json` with worker index `-1`, return code `-3`, and
zero completed workers. No warmup history, worker attestation, result,
completion, analysis payload, or analysis output exists. The only campaign
artifacts are:

- `run_manifest.json`: 179400 bytes,
  SHA-256 `01a92754893a915bb49891e352f7901f2f9af02c1c0fd5e0a49c00db11fa6efa`;
- `terminal.json`: 258 bytes,
  SHA-256 `bd79fa163f11f595e428177a14ae72c44196bf456c122385c39eec27f89b316b`.

The v2 identity will not be retried. Its successor may retain the complete v2
scientific protocol unchanged and make only the harness correction needed to
bind the warmup module's thread constant to the already-frozen 14-core budget.
