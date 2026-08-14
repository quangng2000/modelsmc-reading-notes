# Research evidence index

This directory contains the experiment code and frozen protocols supporting the
r5-centered ModelSMC-PBE paper and its provider-free calibration evidence
chain. The concise publication plan is
[PUBLICATION_EXPERIMENT_PLAN.md](PUBLICATION_EXPERIMENT_PLAN.md). It is the
authoritative map from completed studies to allowed manuscript claims.

The research tree also retains superseded protocols, failed gates, aborted
runs, and developmental utilities. Retention is intentional: those files are
audit history, not competing versions of the final method.

## Evidence-handling rule

Treat frozen protocols, seals, and imported artifact directories as read-only.
Do not retry, resume, backfill, rename, or rewrite a completed or failed study.
New work must use a new protocol identity and output directory. Local tests may
exercise code against temporary fixtures; they must not mutate archived
evidence.

## Publication evidence

### 1. Fresh-blind r5

This is the primary bounded-discovery study.

- Protocol:
  [protocol-blind-filter-map-confirmation-v3-r5.json](protocol-blind-filter-map-confirmation-v3-r5.json)
- Method seal:
  [protocol-blind-filter-map-confirmation-v3-r5.method-seal.json](protocol-blind-filter-map-confirmation-v3-r5.method-seal.json)
- Frozen runbook:
  [BLIND_FILTER_MAP_CONFIRMATION_V3_R5_RUNBOOK.md](BLIND_FILTER_MAP_CONFIRMATION_V3_R5_RUNBOOK.md)
- Core method and runner:
  [evidence_shortlist_smc.py](evidence_shortlist_smc.py) and
  [run_blind_filter_map_confirmation_v3.py](run_blind_filter_map_confirmation_v3.py)
- Analyzer:
  [analyze_blind_filter_map_confirmation_v3.py](analyze_blind_filter_map_confirmation_v3.py)
- Imported evidence:
  [../../../../artifacts/blind-filter-map-confirmation-v3-r5](../../../../artifacts/blind-filter-map-confirmation-v3-r5)

Result: 10/12 exact discoveries for LLM-shortlist SMC versus 2/12 for the
matched grammar-random acquisition arm. This is a system-level result on the
frozen finite-domain generator, not an isolated LLM effect. The primary exact
value is 1/256. A post hoc sensitivity treating observed-but-invalid r4 as an
additional look gives 0.0078125; this is not the frozen r5 analysis. R3 had no
numerical outcome, and r4 remains invalid and excluded.

### 2. Provider-free calibration V1

This is a frozen negative gate, not a successful validation study.

- Protocol:
  [protocol-particle-calibration-v1.json](protocol-particle-calibration-v1.json)
- Implementation:
  [particle_calibration_study_v1.py](particle_calibration_study_v1.py)
- Findings and math audit:
  [PARTICLE_CALIBRATION_V1_FINDINGS.md](PARTICLE_CALIBRATION_V1_FINDINGS.md) and
  [PARTICLE_CALIBRATION_V1_MATH_AUDIT.md](PARTICLE_CALIBRATION_V1_MATH_AUDIT.md)
- Imported evidence:
  [../../../../artifacts/particle-calibration-v1](../../../../artifacts/particle-calibration-v1)

At 256 particles, exact-program-mass RMSE was 0.2395265 and signed bias was
+0.1130184, so both frozen gate components failed. Exact-program
discovery in every 256-particle run does not change that calibration outcome.

### 3. Terminal diagnostic V2

This post-failure study diagnoses a terminal mechanism without reopening V1.

- Protocol:
  [protocol-particle-calibration-terminal-diagnostic-v2.json](protocol-particle-calibration-terminal-diagnostic-v2.json)
- Implementation:
  [particle_calibration_terminal_diagnostic_v2.py](particle_calibration_terminal_diagnostic_v2.py)
- Imported evidence:
  [../../../../artifacts/particle-calibration-terminal-diagnostic-v2](../../../../artifacts/particle-calibration-terminal-diagnostic-v2)

Finite-state importance identities held to maximum absolute error 2.22e-16.
The six-arm grid used 256 and 512 particles with 128 repetitions per
task-arm-particle-count cell. The exhaustive global top-64, `epsilon=0.50`
positive control reduced pooled 512-particle exact-mass
RMSE from 0.2792 to 0.0626 relative to the sticky V1 terminal proposal. This
supports poor proposal-target overlap as a material terminal mechanism; the
exhaustive control is not a search algorithm and V2 does not rescue V1.

### 4. Fresh calibration V2

This is a second frozen negative calibration gate on 20 freshly generated
provider-free tasks.

- Protocol:
  [protocol-calibrated-program-inference-v2-fresh.json](protocol-calibrated-program-inference-v2-fresh.json)
- Method seal:
  [protocol-calibrated-program-inference-v2-fresh.method-seal.json](protocol-calibrated-program-inference-v2-fresh.method-seal.json)
- Imported evidence:
  [../../../../artifacts/calibrated-program-inference-v2-fresh](../../../../artifacts/calibrated-program-inference-v2-fresh)

At 256 particles, exact-mass RMSE was 0.2612355 and bias was -0.0748415;
the task-first bootstrap upper-95 RMSE was 0.3767653. The frozen gate failed
and remains failed.

### 5. Factorized V3 reused-task diagnostic

This post-failure diagnostic changed mode acquisition and reused the 20 fresh
V2 tasks, particle counts, and seeds.

- Protocol:
  [protocol-calibrated-program-inference-v3-factorized-diagnostic.json](protocol-calibrated-program-inference-v3-factorized-diagnostic.json)
- Imported evidence:
  [../../../../artifacts/calibrated-program-inference-v3-factorized-diagnostic](../../../../artifacts/calibrated-program-inference-v3-factorized-diagnostic)

At 256 particles, exact-mass RMSE was 0.0221995, bootstrap upper-95 RMSE was
0.0245201, and the central 90% bias interval was
[-0.0081112, -0.0057479]. Because the method was designed after seeing V2 and
the tasks were reused, this is mechanism evidence rather than confirmation.

### 6. Fresh V3 R2 terminal finite-support confirmation

This is the second fresh provider-free suite and the final calibration result
in the current evidence chain.

- R2 protocol:
  [protocol-calibrated-program-inference-v3-fresh-r2.json](protocol-calibrated-program-inference-v3-fresh-r2.json)
- R2 method seal:
  [protocol-calibrated-program-inference-v3-fresh-r2.method-seal.json](protocol-calibrated-program-inference-v3-fresh-r2.method-seal.json)
- R1 pre-secret supersession record:
  [CALIBRATED_PROGRAM_INFERENCE_V3_FRESH_R1_SUPERSESSION.md](CALIBRATED_PROGRAM_INFERENCE_V3_FRESH_R1_SUPERSESSION.md)
- Imported evidence:
  [../../../../artifacts/calibrated-program-inference-v3-fresh](../../../../artifacts/calibrated-program-inference-v3-fresh)
- Deterministic replay receipt:
  [../../../../artifacts/calibrated-program-inference-v3-fresh-replay-verification.json](../../../../artifacts/calibrated-program-inference-v3-fresh-replay-verification.json)
- Unblind verification:
  [../../../../artifacts/calibrated-program-inference-v3-fresh-unblind-verification.json](../../../../artifacts/calibrated-program-inference-v3-fresh-unblind-verification.json)

The frozen gate passed. At 256 particles, exact-mass RMSE was 0.0398662,
bias was -0.0109472, bootstrap upper-95 RMSE was 0.0494323, and the central
90% bias interval was [-0.0149663, -0.00763969]. Point RMSE decreased to
0.0304513 and 0.0258555 at 512 and 1,024 particles. The result is limited to
provider-free terminal exact-program mass on the declared singleton-complete
synthetic law. It does not calibrate the LLM, the four-stage recurrence, the
full PBE system, a large DSL, or target mean loss (whose 256-particle RMSE was
0.2991).

The R2 protocol, method, custody, and replay SHA-256 values are respectively
`36f4a6c5930c2da2c89c1515d44267b6cc9bab94e7a000352f4343fd97fc97d6`,
`788a44cc32ed617aa853a9e7ac97ab2236844382825149eba0073cc6d0a31dc4`,
`32631befddaecd28aa4008c73232475b3676b26a7e1e0fe457dac6995ab0932f`,
and `c7b08c0385867436e3e2d7a08cfe5a506ed8dfaea702defcabeff1e88b1859a7`.

### 7. ExeDec V2 released-data debug

This is descriptive, non-confirmatory adapter evidence.

- Frozen protocol:
  [protocol-exedec-deepcoder-ho-debug-benchmark-v2.json](protocol-exedec-deepcoder-ho-debug-benchmark-v2.json)
- Adapter, runner, and analyzer:
  [exedec_deepcoder_ho_adapter_v1.py](exedec_deepcoder_ho_adapter_v1.py),
  [run_exedec_deepcoder_ho_debug_benchmark_v2.py](run_exedec_deepcoder_ho_debug_benchmark_v2.py), and
  [analyze_exedec_deepcoder_ho_debug_benchmark_v2.py](analyze_exedec_deepcoder_ho_debug_benchmark_v2.py)
- Canonical imported study:
  [../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2](../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2)
- Post-import audit:
  [../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2.POST_IMPORT_AUDIT.md](../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2.POST_IMPORT_AUDIT.md)

All 64 runs and 32 paired blocks completed. Private-debug-probe exactness was 17/32
for LLM-SMC and 3/32 for grammar-random; the exact two-sided conditional
sign-test value was 0.001312255859375. The blocks are repeated seeds nested in
only four public and potentially contaminated targets, so the value is
descriptive rather than a significance or confirmation claim. The strict
Filter-then-Map adapter is unofficial, and finite probes are not semantic
equivalence. R5 and calibration V1/V2 use score scales `(0.75, 0.02, 2)`;
ExeDec uses frozen scorer defaults `(2.0, 0.15, 2)`, so the studies are not
scale-matched.

The canonical study import contains 1,143 files. Its archive SHA-256 is
`5e97831610852264f686d0f37d6f9c4aef8d783e45c29e6cc3fd5023dbc7e46c`,
its analysis SHA-256 is
`5bc9a74a39add1a60d8d5adedad89beb9dad389bd4e6cbdb4447b45e1888c2bb`,
and its analysis-inventory SHA-256 is
`848e204685c8b4459bb51c7471d5c644464f3183af142f6cecc60feb85a65b25`.

The separate original operations-evidence archive failed its frozen safe-mode
verifier on `files/operations/bin/exedec_v2_driver.py`. All 149 content hashes
and cross-bindings validated, but the failure remains authoritative at
[../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2-operations-evidence-failed-v1](../../../../artifacts/exedec-deepcoder-ho-debug-benchmark-v2-operations-evidence-failed-v1).
The locally mode-header-normalized derivative is validated but explicitly
noncanonical and unimported; it is not a second successful import.

## Preserved history outside the main narrative

- Blind r1/r2 were superseded before secret preparation.
- R3 aborted before execution or provider calls because required record fields
  were missing.
- R4 completed public execution but failed its frozen sealing gate before
  unblinding and remains blinded and excluded.
- Fresh V3 R1 was superseded before secret creation or any numerical work to
  align bias-interval wording with the already strict validator; it had no
  outcome.
- ExeDec V1 was superseded before provider calls because source closure was
  incomplete.
- Developmental matrices, blind evidence-frontier studies, Gate-2 transport
  work, staged family/hole proposals, joint-semantic slates, and exploratory
  repair studies remain available for historical audit only.

The frozen files stay in this directory and the root `artifacts/` tree. Their
presence does not authorize pooling their outcomes into the final r5 result.

## Local verification

From the package root:

```bash
uv sync --dev
uv run pytest -q research/tests
uv run ruff check research
```

Targeted tests for the final studies include
`test_evidence_shortlist_smc.py`,
`test_analyze_blind_filter_map_confirmation_v3.py`,
`test_particle_calibration_study_v1.py`,
`test_particle_calibration_terminal_diagnostic_v2.py`,
`test_calibrated_program_inference_v2_fresh.py`,
`test_calibrated_program_inference_v3_factorized_diagnostic.py`,
`test_calibrated_program_inference_v3_fresh.py`,
`test_calibrated_program_inference_v3_replay.py`,
`test_calibrated_program_inference_v3_validation.py`, and
`test_exedec_deepcoder_ho_debug_benchmark_v2.py`.

These checks are provider-free. They validate code and fixtures; they do not
recreate custody, rerun a frozen experiment, or upgrade descriptive evidence
to confirmation.

## Manuscript handoff

- [../paper/main.tex](../paper/main.tex): concise publisher-facing source.
- [../paper/AUTHOR_REVIEW_GUIDE.md](../paper/AUTHOR_REVIEW_GUIDE.md):
  paragraph-level claim map.
- [PUBLICATION_EXPERIMENT_PLAN.md](PUBLICATION_EXPERIMENT_PLAN.md): final
  evidence register, exclusions, future roadmap, and submission blockers.
