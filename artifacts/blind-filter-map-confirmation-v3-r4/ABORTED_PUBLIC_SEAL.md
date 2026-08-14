# Blind filter-map confirmation r4: aborted before public sealing and unblinding

R4 completed all 24 frozen task-arm runs and its reveal-free analysis, but the
frozen public sealer rejected the legitimate run layout before creating a
public bundle.  The exact integration defect is that every harness run writes
`round-01.json` through `round-04.json`, while the frozen sealer's run-root
allowlist admitted only `protocol.json`, `executions.json`,
`provider-seal.json`, `result.json`, and the optional `provider/` directory.
It therefore classified the four required round ledgers as unexpected.

This is a post-analysis, pre-unblinding infrastructure/method-integrity abort.
It is not a confirmatory scientific outcome.  The partial attempt is preserved
without repair, retry, resume, task replacement, or private-reveal access.

After the abort was declared, the frozen r5 corrected sealer was run solely
to preserve the complete reveal-free r4 evidence tree.  This post-outcome seal
does not rescue or change r4's frozen decision:

- Corrected sealer source SHA-256:
  `c1003027cd853e53e3c3a40591d79f4d3dc00468956e81e8c24785b9cfff0220`
- Posthoc abort-evidence bundle SHA-256:
  `84ef009c3b64aa79d9bd7df054c3685b7585c1166872ea71fb67539b29e3ee6b`
- Reveal-free files inventoried: 385.
- Evidence path, deliberately outside the r4 study tree:
  `artifacts/blind-filter-map-confirmation-v3-r4-posthoc-abort-evidence-r5/`
- R4 runs-and-analysis tree SHA-256 before and after the read-only pass:
  `4459c1bba65a6afdfc0f9fdbd60fd75e88a857f18d857bb7e5798aa015e2383e`.
- R5 protocol and method-seal SHA-256:
  `36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d`
  and `5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228`.

Reveal-free descriptive output before the sealing failure was:

- LLM exact discovery by slot 29: 11/12 tasks.
- Grammar-random exact discovery by slot 29: 2/12 tasks.
- Discordant pairs: 9 LLM-only and 0 grammar-only.
- Exact one-sided paired sign-test tail: 0.001953125.
- LLM/grammar exact discoveries by slot 21: 10/12 and 0/12.

These values must remain labeled as invalidated, reveal-free diagnostics.  A
clean successor must freeze the corrected exact-round allowlist before a new
secret and fresh suite are generated.

Key frozen and output hashes:

- R4 protocol: `9c44d1391413163d93dd538288b470e63ea3aedb1616b77dd82f65fac4d95990`
- R4 method seal: `a0dba72f8d2df04668fab9799d2a357bf47b389421854949d4fdc73c4056f9dd`
- Frozen public sealer: `851f22c7dff76c9aafe1a6961d661e0475466e942ca215f0c119151a9f516413`
- Reveal-free analysis: `dc80265f2a72f365c701c64d936f25609edd009da6074b3c2ad21f1c0ac0783e`
- Driver log: `bd8d6d402bc6ccb8329236ff57a60dd47afe554eb04c04a5aeedf92418c0f9f5`
