# Blind filter-map confirmation v3: r5 supersession record

R5 supersedes, but does not repair or resume, r4. R4 remains an immutable
post-analysis, pre-unblinding infrastructure/method-integrity abort because
its frozen public sealer rejected the legitimate four round ledgers in every
run directory. R4's reveal-free diagnostics are not a confirmatory outcome,
and r4 must never be unblinded.

R5 was frozen before any r5 secret, suite, provider seal, or provider call:

- Protocol SHA-256:
  `36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d`
- Method-seal SHA-256:
  `5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228`
- Corrected public-sealer SHA-256:
  `c1003027cd853e53e3c3a40591d79f4d3dc00468956e81e8c24785b9cfff0220`
- Focused public-sealer test SHA-256:
  `195593d0db0a0837ae130f2b2bbba39a0c4bcce64b168f10915ac285285583a2`
- Complete bound suite: 35 tests passed.

The recursive protocol-body comparison is identical outside
`freeze_requirements`. Within `freeze_requirements`, only the public-sealer
hash and the focused sealer-test hash/bundle changed. The runner, analyzer,
harness, generator, common code, dependency bundle, ModelSMC Python tree,
prompt binding, task distribution, run order, run seeds, search settings, and
scientific analysis are unchanged. Therefore r4 to r5 has zero semantic
execution change and no data-dependent method change.

Before r5 secret preparation, the frozen r5 sealer was exercised against the
immutable r4 public runs and reveal-free analysis. It accepted the actual
24-arm, four-round layout and produced the separately labeled descriptive
abort-evidence bundle SHA-256
`84ef009c3b64aa79d9bd7df054c3685b7585c1166872ea71fb67539b29e3ee6b`.
The 385-file r4 source-tree digest was identical before and after:
`4459c1bba65a6afdfc0f9fdbd60fd75e88a857f18d857bb7e5798aa015e2383e`.
This posthoc evidence does not rescue r4, and no r4 private reveal was read.

R5 must use new absent paths
`/root/blind-filter-map-confirmation-v3-r5-custody` and
`artifacts/blind-filter-map-confirmation-v3-r5`. Its secret commitment must
differ from both r3
`10e3e473540d07378c37f833a9a8faf6e8de8cd0a3c0083fa53d05f2340f6a61`
and r4
`a3440d0d8ec1c88e7d57870966a14fa40a7b4cf165ba809cadc2e0386892e8b7`.
Only after those gates and an independent frozen-boundary signoff may r5
custody preparation begin.
