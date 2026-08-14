# Calibrated Program Inference V3 Fresh R1 Supersession

The first V3 fresh protocol freeze was superseded on 2026-08-14 before any
secret, custody seal, task suite, Monte Carlo draw, replay receipt, unblinding,
or provider call existed.

- Protocol SHA-256: `efbbdf4b4a2e03d68fe60c51ba801bae5384c4e11819cee0a10bceeadc1c74ea`
- Method-seal SHA-256: `9222c7fd000dfcaab56a5ca1e33521c3ad53ebd01dc7aa133d4191c36c990669`
- Scientific or numerical outcome: none

The sole issue was textual: the R1 protocol described the central 90% bias
interval as lying within `[-0.03,+0.03]`, while the source-bound validator
already required both endpoints to lie strictly inside the equivalence bounds.
R2 changes that wording to `(-0.03,+0.03)` and preserves every R1 byte and
hash as a pre-secret superseded attempt.
