# ModelSMC-PBE

ModelSMC-PBE is a standalone Python implementation of probability-accountable
SMC for bounded programming by example. The final publication path studies a
typed Filter-then-Map grammar, an execution-evidence LLM shortlist, and a
matched grammar-random acquisition arm. The implementation records the finite
proposal law actually used by the application rather than interpreting opaque
model scores as posterior probabilities.

The concise manuscript is available as
[paper/main.pdf](paper/main.pdf). The final evidence and claim policy are in
[research/PUBLICATION_EXPERIMENT_PLAN.md](research/PUBLICATION_EXPERIMENT_PLAN.md).

## Final method

For each scheduled acquisition slot, the system:

1. builds typed local repair choices from execution evidence;
2. obtains a finite LLM-ranked shortlist or a matched grammar-random draw;
3. maps invalid, duplicate, no-op, and failed acquisitions into an explicit
   four-slot totalized proposal;
4. evaluates the resulting proposal probability on the declared bounded
   grammar; and
5. applies importance weighting, ESS-based resampling, and the frozen stopping
   rule to the particle population.

The two r5 arms share the interpreter, evidence, grammar, slot budget,
weighting, resampling, and stopping mechanics. The contrast is therefore a
system-level acquisition comparison, not an isolated estimate of an LLM
component.

See [DESIGN.md](DESIGN.md) for the mathematical and implementation boundary.

## Completed evidence ledger

### Fresh-blind r5: primary bounded-discovery evidence

- 12 freshly generated finite-domain Filter-then-Map tasks, paired arms, and 29
  logical complete-program slots per arm.
- Exact discovery: 10/12 for LLM-shortlist SMC versus 2/12 for grammar-random;
  eight LLM-only discordances and none in the reverse direction.
- One-sided exact paired sign-test value: 1/256 (0.00390625).
- A post hoc sensitivity treating the observed-but-invalid r4 diagnostic as an
  additional numerical look gives a two-look Bonferroni value of 0.0078125.
  This is not the frozen r5 analysis; r3 had no numerical outcome and r4
  remains invalid and excluded.
- Scope: bounded system-level discovery on the frozen generator and domain;
  neither out-of-domain generalization nor compute superiority.

Protocol, method seal, and runbook:
[research/protocol-blind-filter-map-confirmation-v3-r5.json](research/protocol-blind-filter-map-confirmation-v3-r5.json),
[research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json](research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json), and
[research/BLIND_FILTER_MAP_CONFIRMATION_V3_R5_RUNBOOK.md](research/BLIND_FILTER_MAP_CONFIRMATION_V3_R5_RUNBOOK.md).
The imported evidence is at
[../../../artifacts/blind-filter-map-confirmation-v3-r5](../../../artifacts/blind-filter-map-confirmation-v3-r5).

### Calibration V1: frozen negative gate

- Four exactly enumerable developmental supports, five particle counts, and 32
  repetitions per task-particle-count cell.
- At 256 particles, exact-program-mass RMSE was 0.2395265 and signed bias was
  +0.1130184; both frozen gate components failed.
- Every 256-particle run happened to discover an exact program, but discovery
  is not target-mass calibration and was not a gate component.

The protocol and implementation are
[research/protocol-particle-calibration-v1.json](research/protocol-particle-calibration-v1.json)
and
[research/particle_calibration_study_v1.py](research/particle_calibration_study_v1.py).
The verified analysis is at
[../../../artifacts/particle-calibration-v1](../../../artifacts/particle-calibration-v1).

### Terminal diagnostic V2: mechanism evidence only

- Six arms over four tasks at 256 and 512 particles, with 128 repetitions per
  task-arm-particle-count cell.
- Finite-state terminal importance identities held to maximum absolute error
  2.22e-16.
- A global exhaustive top-64, `epsilon=0.50` positive control reduced pooled 512-particle
  exact-mass RMSE from 0.2792 for the sticky V1 terminal proposal to 0.0626.
- The control is not a search algorithm. V2 diagnoses poor terminal overlap;
  it does not repair the failed V1 gate or validate the four-stage recurrence.

The protocol, implementation, and analysis are
[research/protocol-particle-calibration-terminal-diagnostic-v2.json](research/protocol-particle-calibration-terminal-diagnostic-v2.json),
[research/particle_calibration_terminal_diagnostic_v2.py](research/particle_calibration_terminal_diagnostic_v2.py), and
[../../../artifacts/particle-calibration-terminal-diagnostic-v2](../../../artifacts/particle-calibration-terminal-diagnostic-v2).

### ExeDec V2: released-data debug evidence

- 64/64 completed runs in 32 paired seed blocks over four deduplicated public
  targets; 1,856 logical slots.
- Hidden-probe exactness: 17/32 for LLM-SMC and 3/32 for grammar-random, with
  16 LLM-only, 2 grammar-only, 1 both, and 13 neither blocks.
- Public-example exactness: 21/32 versus 5/32. LLM-SMC made 188 provider calls
  under a cap of 224; grammar-random made none. The study recorded 1,040
  online physical scorer calls before reference enumeration; exhaustive
  public-reference evaluations are excluded from that count.
- The exact two-sided conditional sign-test value, 0.001312255859375, is
  descriptive. Repeated seeds are nested within only four public targets and
  do not justify a population-level significance or confirmation claim.

This is an unofficial strict Filter-then-Map adapter, not the full DeepCoder
DSL or an official ExeDec evaluation. The released targets are potentially
contaminated, finite stored probes do not establish semantic equivalence, and
shared-FUSE staging makes the result integrity-limited. R5 and calibration
V1/V2 use score scales `(0.75, 0.02, 2)`; ExeDec uses the frozen scorer defaults
`(2.0, 0.15, 2)`, so it is not a scale-matched external confirmation.

The canonical 1,143-file study import is
[../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2](../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2).
Its archive SHA-256 is
`5e97831610852264f686d0f37d6f9c4aef8d783e45c29e6cc3fd5023dbc7e46c`;
the authoritative analysis SHA-256 is
`5bc9a74a39add1a60d8d5adedad89beb9dad389bd4e6cbdb4447b45e1888c2bb`.

The separate original operations-evidence export failed its frozen safe-mode
verifier on a shared-FUSE tar mode. All 149 content hashes and cross-bindings
validated, but the failed transfer remains preserved at
[../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2-operations-evidence-failed-v1](../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2-operations-evidence-failed-v1).
A locally mode-header-normalized derivative validates but is noncanonical,
unimported, and not a second successful import.

## Claim boundaries

This project supports claims about a fixed bounded grammar and the complete r5
system. It does not support claims of:

- a calibrated LLM posterior or calibrated finite-particle target mass;
- a causal LLM-only contribution;
- arbitrary-task, out-of-domain, or full-DeepCoder generalization;
- semantic equivalence inferred from finite examples or probes; or
- wall-clock, provider-compute, or exhaustive-search superiority.

## Install and verify

Python 3.12 is pinned in `.python-version`. From this directory:

```bash
uv sync --dev
uv run pytest -q
uv run ruff check src tests research
uv run mypy --strict src/modelsmc_pbe
```

The provider-free suite does not require a model server. A minimal deterministic
CLI smoke test is:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode paper-search --proposal catalog \
  --skeleton map-arithmetic --particles 8 --iterations 6 \
  --device cpu
```

Do not use this smoke command as a substitute for a frozen study. Completed
artifact trees are read-only evidence and should not be retried, resumed,
backfilled, or rewritten.

## Reader paths

- [paper/README.md](paper/README.md): manuscript build and submission notes.
- [paper/AUTHOR_REVIEW_GUIDE.md](paper/AUTHOR_REVIEW_GUIDE.md): paragraph-level
  argument map.
- [research/README.md](research/README.md): current research index and retained
  failure history.
- [NOTICE.md](NOTICE.md): upstream provenance and licensing notes.

Superseded Qwen family/hole studies, joint-semantic slates, Gate-2 work, blind
evidence-frontier iterations, and developmental matrices remain in frozen files
and version history for auditability. They are not part of the reader-facing
publication narrative.
