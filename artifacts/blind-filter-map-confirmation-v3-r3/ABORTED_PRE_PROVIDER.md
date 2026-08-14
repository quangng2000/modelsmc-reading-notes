# Blind filter-map confirmation v3-r3: aborted pre-provider attempt

The first frozen arm (`blind-v3-01`, `grammar-random`) exited with status 1 on
2026-08-13 before any program execution or provider call. The fail-closed
validation in `research.evidence_shortlist_smc` required per-run `epsilon` and
`evidence_scale` fields that the frozen launcher did not place in its derived
run record.

The attempt is invalid and will not be resumed, repaired, retried, backfilled,
or included in any outcome analysis. A successor method revision must be
frozen before generating a new secret and a new blind task suite.

Preserved remote evidence:

- driver log SHA-256: `2ecaeba87692ae854f89c5e4addc94b795f389b55a0872b67632afdb6d952f0a`
- derived invocation SHA-256: `083af5dca78e4548f857e4fc07856d246e35be6dffa10c5abca6a4d5d669d52b`
- preflight record SHA-256: `af625bac56e0995ea8651a98be6f8f2d6a1d484f6040a11cda10108c379e9ebe`
- result files: 0
- execution files: 0
- provider request/response artifacts: 0

The private custody boundary remained intact: `/root` was traversal-only
(`0711`) for the frozen Python runtime, while the custody and private-suite
directories remained `0700` and the reveal remained `0600`.
