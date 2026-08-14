---
license: cc-by-4.0
pretty_name: ModelSMC-PBE Auditable Program Inference Artifacts
tags:
- program-synthesis
- programming-by-example
- sequential-monte-carlo
- large-language-models
- reproducibility
---

# ModelSMC-PBE research artifacts

This dataset is the public data-and-paper companion for **Deduce, Propose,
Correct: Probability-Accountable LLM Shortlists for Typed Program Synthesis**
by **Tri Nguyen** and **Thanh-Dat Nguyen**. The verified manuscript is 16 pages.

- Dataset: [hackerprofile1/modelsmc-pbe-research](https://huggingface.co/datasets/hackerprofile1/modelsmc-pbe-research)
- Implementation source and tests: [GitHub commit 50369a6b0e264eda9cb6e1aab45949cb3be11cb6](https://github.com/quangng2000/modelsmc-reading-notes/tree/50369a6b0e264eda9cb6e1aab45949cb3be11cb6)
- Manuscript: [`paper/main.pdf`](paper/main.pdf)

The archive preserves successful, failed, diagnostic, superseded, and
integrity-limited records separately. A later pass never rewrites an earlier
failed gate.

## Main evidence

### Fresh-blind r5 bounded discovery

On 12 freshly generated finite-domain Filter-then-Map tasks, the complete
LLM-shortlist SMC system found an exact program on 10 tasks versus 2 for matched
grammar-random acquisition. All eight discordant pairs favored the LLM-guided
system (`p = 1/256`, one-sided exact sign test). This is a bounded system-level
discovery result, not an isolated LLM effect, a speed claim, or broad
generalization evidence.

### Calibration chronology

The provider-free calibration evidence must be read in this order:

1. **Calibration V1 failed.** At 256 particles, exact-program-mass RMSE was
   0.2395265 and bias was +0.1130184. Exact-program discovery in every primary
   run did not change the failed gate.
2. **Terminal diagnostic V2 was diagnostic only.** On the same developmental
   supports, an exhaustive positive control supported poor terminal
   proposal--target overlap as a material mechanism. It did not rescue V1 or
   validate the four-stage recurrence.
3. **Fresh calibration V2 failed.** On 20 newly generated tasks, 256-particle
   RMSE was 0.2612355, bias was -0.0748415, and the task-first bootstrap
   upper-95 RMSE was 0.3767653. No difficult task was replaced.
4. **The factorized V3 diagnostic reused the V2 tasks.** Its 256-particle RMSE
   was 0.0221995, but the method was designed after the V2 failure and reused
   the same tasks and seeds. It was mechanism evidence, not confirmation.
5. **Fresh V3 R2 passed its narrow frozen gate.** A new secret generated 32
   independent tasks. At 256 particles, exact-mass RMSE was 0.0398662, bias was
   -0.0109472, the bootstrap upper-95 RMSE was 0.0494323, and the central 90%
   bias interval was [-0.0149663, -0.00763969]. Point RMSE decreased to
   0.0304513 and 0.0258555 at 512 and 1,024 particles.

Fresh V3 R2 is a provider-free confirmation for **terminal exact-program target
mass on the declared singleton-complete finite synthetic law**. It does not
calibrate an LLM posterior, LLM mode banks, the four-stage SMC recurrence, the
full PBE system, a large DSL, or external tasks. Target-mean-loss was not the
primary endpoint and remained much less accurate (256-particle RMSE 0.2991), so
the pass does not extend to that functional.

### ExeDec V2 released-data debug

ExeDec V2 observed private-debug-probe exactness in 17/32 paired seed blocks for
LLM-SMC versus 3/32 for grammar-random on four public released targets. The
strict Filter-then-Map adapter is unofficial, repeated seeds are nested within
only four potentially contaminated targets, finite probes are not semantic
equivalence, and nonexclusive staging limits integrity. The result is
descriptive debug evidence, not a confirmatory ExeDec benchmark.

## Artifact locations

- r5 evidence: `artifacts/blind-filter-map-confirmation-v3-r5/`
- Calibration V1: `artifacts/particle-calibration-v1/`
- Terminal diagnostic V2:
  `artifacts/particle-calibration-terminal-diagnostic-v2/`
- Failed fresh V2:
  `calibration-v3-r2/artifacts/calibrated-program-inference-v2-fresh/`
- Reused-task factorized V3 diagnostic:
  `calibration-v3-r2/artifacts/calibrated-program-inference-v3-factorized-diagnostic/`
- Fresh V3 R2 protocol: [frozen source record](https://github.com/quangng2000/modelsmc-reading-notes/blob/50369a6b0e264eda9cb6e1aab45949cb3be11cb6/papers/01-modelsmc/modelsmc-pbe-python/research/protocol-calibrated-program-inference-v3-fresh-r2.json)
- Fresh V3 R2 analysis:
  `calibration-v3-r2/artifacts/calibrated-program-inference-v3-fresh/analysis.json`
- Fresh V3 R2 deterministic replay receipt:
  `calibration-v3-r2/artifacts/calibrated-program-inference-v3-fresh-replay-verification.json`
- Fresh V3 R2 unblind verification:
  `calibration-v3-r2/artifacts/calibrated-program-inference-v3-fresh-unblind-verification.json`
- Fresh V3 R2 release checksums: `calibration-v3-r2/SHA256SUMS`
- ExeDec V2 canonical study:
  `artifacts/exedec-deepcoder-ho-debug-benchmark-v2/`
- Preserved failed ExeDec operations transfer:
  `artifacts/exedec-deepcoder-ho-debug-benchmark-v2-operations-evidence-failed-v1/`

The Fresh V3 R2 protocol, method seal, custody seal, and deterministic replay
receipt have SHA-256 values
`36f4a6c5930c2da2c89c1515d44267b6cc9bab94e7a000352f4343fd97fc97d6`,
`788a44cc32ed617aa853a9e7ac97ab2236844382825149eba0073cc6d0a31dc4`,
`32631befddaecd28aa4008c73232475b3676b26a7e1e0fe457dac6995ab0932f`,
and `c7b08c0385867436e3e2d7a08cfe5a506ed8dfaea702defcabeff1e88b1859a7`,
respectively.

## Intended use and replay boundary

Use this dataset to audit manuscript claims, inspect frozen analyses and
failure histories, reproduce provider-free validation with the linked source,
or verify checksum bindings. The publication copy excludes credentials and
ephemeral provider endpoints. Frozen studies must not be retried, resumed,
backfilled, or rewritten to obtain a different outcome.

Do not treat these bounded tasks as evidence of general-purpose program
synthesis, formal verification, a Bayesian posterior over unbounded programs,
wall-clock superiority, or calibration beyond the explicitly gated Fresh V3
terminal exact-mass endpoint.

## Licensing

This dataset card, experiment metadata, and sanitized run outputs are released
under CC BY 4.0. Third-party model and paper references retain their original
licenses. Source code remains under its repository license and is distributed
through the linked GitHub revision, not this dataset.
