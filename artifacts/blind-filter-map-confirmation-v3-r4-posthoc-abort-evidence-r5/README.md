# R4 posthoc abort evidence produced by the frozen r5 sealer

This directory is descriptive abort evidence only. It is not the frozen r4
public-bundle gate, does not rescue r4, and must not be cited as an r4
confirmatory outcome. R4 remains an immutable post-analysis, pre-unblinding
infrastructure/method-integrity abort.

After r5 was frozen and before any r5 secret, suite, provider seal, or provider
call existed, the frozen r5 public sealer was run on only the immutable r4
public runs and reveal-free analysis. The output was deliberately written
outside the r4 study tree. The r4 private reveal remained unreadable to the
`nobody` identity and was not read.

Bindings and evidence:

- R5 protocol SHA-256:
  `36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d`
- R5 method-seal SHA-256:
  `5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228`
- Corrected frozen sealer SHA-256:
  `c1003027cd853e53e3c3a40591d79f4d3dc00468956e81e8c24785b9cfff0220`
- Preflight and actual bundle SHA-256:
  `84ef009c3b64aa79d9bd7df054c3685b7585c1166872ea71fb67539b29e3ee6b`
- Reveal-free source files inventoried: 385.
- Source-tree SHA-256 before and after sealing:
  `4459c1bba65a6afdfc0f9fdbd60fd75e88a857f18d857bb7e5798aa015e2383e`
- `BUNDLE_SHA256` file SHA-256:
  `25e4d2c08b4824c468c7f2dfaa0a6453eef5f2e446541dd5fe2c2f70593b9c83`
- `SHA256SUMS` file SHA-256:
  `84ef009c3b64aa79d9bd7df054c3685b7585c1166872ea71fb67539b29e3ee6b`

The pass demonstrates that the frozen r5 sealer accepts the actual 24-arm,
four-round harness layout. Missing and extra round-ledger rejection is covered
by the frozen focused tests; the complete bound suite passed 35 tests.
