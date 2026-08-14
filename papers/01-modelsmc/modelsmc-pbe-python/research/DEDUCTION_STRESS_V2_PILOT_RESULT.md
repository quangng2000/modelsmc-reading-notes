# Corrected deduction-stress v2 provider-backed scoring pilot

## Scope and provenance

The corrected one-seed paired pilot ran on 2026-08-11 UTC. It reran only the
failed deduction-stress comparison: D on the local controller and QD configured
to query a RunPod proxy for the pinned 32B Qwen scorer, both at seed 101. No
unrelated size-study cells or additional seeds are present in this matrix.

The run used:

- protocol `modelsmc-pbe-deduction-stress-v2`, SHA-256
  `6bb7ce19c83b50b0b347c9c086a52a1f863ca1810b552e12e61a47bc0b665085`;
- implementation commit `ba04543`;
- post-run auditor `research/audit_math.py`, SHA-256
  `bb063c5de9ed15ba91a5caf35f03f7dd6e7103770cab456c273e7b0665f54af0`;
- hash-bound provider-free target-audit certificate SHA-256
  `f8160fba4691b1efa1c9bdc6413fbe57e8c5ec145fd24236c4dbf1960a0fa301`;
- configured `Qwen/Qwen2.5-Coder-32B-Instruct` model and tokenizer revision
  `381fc969f78efac66bc87ff7ddeadb7e73c218a7`; the manifest declares vLLM
  `0.11.0` with processed prompt log-probabilities; and
- four particles, one guided iteration, family deduction mix `0.75`, and hole
  deduction mix `0.0`.

The executed protocol snapshot retains its frozen pre-run `provider-unrun`
text. That string is not a current status claim, and the snapshot is not edited
after execution because doing so would change the protocol hash to which the
certificate and result are bound.

## Outcome

| Arm | Status | Exact | Best state | Loss | Held-out | Scored candidates | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| D | completed | no | 8,890 | 9 | 25/96 | 737 | 9.469 s |
| QD | completed | no | 8,890 | 9 | 25/96 | 737 | 136.934 s |

The paired common random numbers produced the same discrete sampled path. The
four ancestor and final state IDs were:

```text
ancestors: 27,222; 36,049; 17; 14
final:      8,890;     15;  4; 36,122
```

All ten score-ledger requests also recorded the same selected indices. Their
proposal log probabilities, weights, and log-normalizer estimates differed, so
the Qwen component was active; the close categorical distributions merely
mapped the shared random draws to the same choices. The paired run found no
exact terminal training program. The selected best final program was not
held-out exact in either arm (`25/96` each). This one exploratory seed cannot
establish an improvement or regression between D and QD.

The corrected target does rank both exact states above every inexact state by
`4.0` log-target units, and the exact states carry `0.9207000842` of the
enumerated terminal target mass. The miss occurred before target ranking:
neither arm proposed an exact terminal construction. D's exact proposal mass
is `4.0312369971e-5` per guided draw, so its four guided draws have only
`0.01612%` probability of a hit; about 17,195 independent guided draws would be
needed for a 50% D hit probability. The D miss is unsurprising at this budget.
No corresponding QD hit probability is available because its exact-path
proposal probability is not identified.

## Trace diagnosis and prefix-consistent QD audit

The strict audit rebuilt every finite candidate catalog, deduction guide, and
ancestor/previous-filling prompt prefix; replayed every D and QD categorical
ledger; and checked the terminal importance identity
`log(q) + log(weight increment) = log(target)`, normalized particle weights,
finite-reference target mass, paired paths, and telemetry with zero violations.
The recorded arithmetic and importance denominator are working. This is a
deterministic artifact replay, not independent attestation that the remote
provider produced the stored token log probabilities. Stored token IDs are
type/count checked but are not independently retokenized against the remote
tokenizer.

The weak point on the visited prefixes is score discrimination. Mean-full-prompt
normalization averages over all scored positions in the shared prompt plus
candidate, which can dilute candidate-specific differences. Across the ten
visited QD prefixes, normalized categorical entropy `H/log(K)` was `0.999783`
to `0.999989`. Qwen changed the probabilities, but not enough to change the
discrete choices at this seed.

In the single visited 600-way predicate ledger for ancestor 27,222, the two
exact predicates ranked 17 and 53 under both Qwen and the final proposal;
neither was sampled. Consequently the provider made zero mapper requests
conditioned on an exact predicate prefix.

In the one 60-way mapper ledger that was reached under a sampled wrong
predicate, `item * item` ranked 51 of 60. That rank is useful evidence of weak
discrimination on the visited branch, but it is not a valid mapper factor for
either exact path.

The QD exact-path proposal probability remains `NOT_IDENTIFIED`: multiplying an
exact-predicate probability by a mapper probability scored under the sampled
wrong predicate would again be a cross-prefix splice. No discovery probability
or N50 is reported from that non-identifying proxy. A future preregistered
sensitivity run could score only the tokenizer-observed candidate continuation
or use a semantic verdict signal; the frozen pilot is not rescored after seeing
its outcome. A narrow diagnosis of the visited slot-0 branch needs two 60-way
mapper requests, one under each exact predicate prefix. Identifying the full
four-particle QD hit probability also requires the 600-way predicate catalog
for the three ancestors that selected other families, plus both 60-way mapper
prefixes for all four ancestors: 11 counterfactual requests and 2,280 finite
candidates. That would identify the two exact-path probabilities per realized
ancestor and the conditional four-draw hit probability for this paired run.
Estimating an unconditional future-run discovery probability or general
particle budget would also require preregistered sampling over ancestors.

## Telemetry and archive

QD telemetry reconciles exactly:

```text
10 logical provider score requests
3 provider invocations
29 HTTP batches
737 provider candidates
447,816 provider-scored token positions
0 cache-served token positions
0 provider invocation failures
0 HTTP failures
125.392 s provider-await wall time
```

The audited pilot archive contains 21 run files, including the strict math-audit
JSON, plus its internal `SHA256SUMS` inventory. A separate reference archive
preserves the original target-audit matrix and certificate:

```text
research/archives/deduction-stress-v2-paired-seed101-audited.tar.gz
SHA-256 1663cad63a888767fb31638a2e198a9253a2e71d74a44d065dae8e5c9f2a33bb

research/archives/deduction-stress-v2-reference-audit.tar.gz
SHA-256 f0fd7ee687010c0b95205c0258608e39b26b03624f67dd479cfcb5e3bafe796e
```

The pilot archive includes the run and audit artifacts, but not the referenced
certificate or auditor source. Strict math replay requires the bound
implementation/task checkout; certificate-binding verification additionally
requires the certificate bytes in the reference archive. Resuming the original
matrix harness also requires its recorded certificate path. The archives record
run integrity; they do not independently attest remote GPU identity.

Research outputs and archives are intentionally git-ignored. The RunPod was
stopped after the audit.
