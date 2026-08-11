# Deduction-stress end-to-end math audit

This report corrects the interpretation of the sealed
`deduction-stress-paired-v1` artifacts without modifying them. The executable
audit is `research/audit_math.py`; `--strict` fails on any seal, replay,
probability, target, pairing, visit-count, or telemetry invariant violation.

## Result

The sealed matrix passes all implemented invariants: 10 planned and 10 observed
cells, five prefix-consistent D/QD pairs, and zero violations. This means the
recorded sampler arithmetic is internally consistent. It does **not** validate
the old post-run interpretation.

The terminal target is

```text
log target(trace) = log prior(trace) - beta * lossScale * loss(trace)
```

under an equal-family, within-family Occam prior. At the v1 terminal settings
(`beta=1`, `lossScale=0.75`), each exact trace has log target
`-11.5993154069`, while the loss-6 expression `[1]` has log target
`-8.4935083270`. Therefore the intended exact program is not the global MAP.
The minimum loss scale for an exact MAP tie is `1.2676345133`.

There are two exact traces among 36,198 states. Their support fraction is
`2 / 36,198 = 5.5251671363e-5`, but that is not their prior probability because
the target prior is not uniform over states. Their exact combined prior mass is
`1.8344729851e-5`.

The catalog/deduction D proposal is fully identifiable from local factors. Its
combined probability of either exact trace is `4.0042653715e-5` per guided draw;
20 independent guided draws give `0.0008005485` probability of at least one hit,
and the 50% draw count is 17,310.

The claimed QD value `4.0069301386e-5` is **not** an exact-path probability.
Every mapper score in the sealed QD ledgers was conditioned on the predicate
actually sampled, and none of those predicates was exact. Multiplying exact
predicate factors by mapper factors from those sibling prefixes creates a
non-identifying splice proxy. The audit labels it
`NONIDENTIFYING_SPLICE_PROXY` and refuses discovery-probability or N50 math.
The true sealed-run QD exact-path probability is `NOT_IDENTIFIED`; determining
it would require counterfactual mapper score requests under both exact predicate
prefixes.

The learned finite-choice scores were nearly flat. Across the sealed QD cells,
the exact mapper's Qwen rank is 49--56 of 60 and its final proposal rank is
58--60. With `proposal_epsilon=0.05` and `deduction_mix=0.75`, the learned
coefficient is only `(1 - 0.05) * (1 - 0.75) = 0.2375`. Identical sampled paths
under common random numbers are consequently compatible with slightly different
categorical distributions; they do not show that Qwen was bypassed.

Telemetry also reconciles once units are kept distinct:

```text
4,715,582 provider-scored token positions
+ 1,870,206 cache-served token positions
= 6,585,788 ledger token positions
```

There were 43 logical provider score requests and 272 HTTP batches; those are
different units. Seeds 101, 211, 307, and 401 evaluated eight unique programs
per arm. Seed 503 evaluated seven because its four initial particles contained
only three unique traces before four new proposals.

## Corrected experiment contract

`examples/foldr-sparse-bounded-square-v2.json` changes only `lossScale` from
`0.75` to `2.0`. Exhaustive enumeration confirms two exact states in the same
36,198-state support and puts the worst exact trace above the best inexact trace
by `4.0` log-target units.

`protocol-deduction-stress-v2.json` retains useful deduction at family selection
while removing the evidence-free hole-level deduction/Occam mixture:

```text
family_deduction_mix = 0.75
hole_deduction_mix   = 0.0
```

The exact D proposal mass under that split is `4.0312369971e-5`: family mass
`0.7256226595`, uniform predicate mass `1/600`, uniform mapper mass `1/60`, and
two exact traces. Finite and lazy kernels agree on both exact construction
densities.

The legacy `deduction_mix` remains a fallback, so existing protocols and callers
are unchanged. The local provider-free v2 gate completed successfully on
2026-08-10 with one completed cell, no process failures, and a provider score
cap of zero. Its independently recomputed, hash-bound certificate records two
exact states and a `4.0` dominance margin. No v2 Qwen/RunPod pilot was executed
as part of this audit.

## Reproduction

```bash
uv run python -m research.audit_math \
  research/outputs/deduction-stress-paired-v1 --strict

uv run pytest -q \
  research/tests/test_audit_math.py \
  research/tests/test_deduction_stress_v2.py

uv run python -m research.run_matrix \
  --protocol research/protocol-deduction-stress-v2.json \
  --output /tmp/modelsmc-deduction-v2-reference-audit \
  --stage provider-free-reference-audit
```
