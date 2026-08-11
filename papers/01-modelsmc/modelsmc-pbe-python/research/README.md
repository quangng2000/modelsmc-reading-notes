# Reproducible U/D/Q/QD benchmark

This directory is the experiment layer for the ModelSMC-PBE prototype. It is
kept outside `src/modelsmc_pbe`: the synthesizer produces observations, while
this package fixes the comparison, generates held-out cases, retains failed
runs, aggregates metrics, and creates verifiable archives.

## Evidence labels

`protocol.json` is currently marked `draft-preregistration-not-yet-frozen`.
Runs made under a draft are exploratory regardless of the task label. Before a
confirmatory run, commit the protocol and implementation, change the status to
`frozen-before-confirmatory-execution`, and do not edit either until the matrix
is complete. Every matrix records the exact SHA-256 of the protocol bytes.

All task families in this repository were visible during development. The
`confirmatory` label therefore means a fresh paired replication over frozen
seeds and arms; it does **not** mean an unseen-task test. The exploratory tasks
may guide engineering. They must not be pooled into the primary confirmatory
claim after inspection.

The primary endpoint is intention-to-treat exact discovery. A timeout, provider
failure, malformed model response, missing artifact, or valid but inexact
program remains a row with `success=false`. `--resume` retains terminal cells;
it does not rerun failures until they become successes.

## Arms

The four arms isolate the two proposed search aids under identical particle,
iteration, support, and scoring caps:

| Arm | Finite scorer | Deduction guide | Interpretation |
| --- | --- | --- | --- |
| U | Uniform catalog | No | Search control |
| D | Uniform catalog | Yes | Symbolic deduction only |
| Q | Qwen | No | Learned ranking only |
| QD | Qwen | Yes | Combined method |

Qwen scores canonical finite choices; it does not emit an unrestricted program
in this experiment. Importance correction and the uniform support floor remain
part of every arm's declared proposal distribution.

## Model-size ladder and cost gates

Schema v2 declares an ordered, within-family ladder of dense
Qwen2.5-Coder-Instruct checkpoints: 3B, 7B, 14B, and 32B. Each entry records a
stable study id, the served alias, Hugging Face repository, nominal total and
active parameter counts, architecture, dtype, quantization, immutable model and
tokenizer commits, and an optional model-specific endpoint environment name.
This supports a within-family checkpoint-size comparison; it does not identify
a causal parameter-count effect because training details also differ by
checkpoint.

U and D do not consult Qwen. They are therefore one size-invariant control cell
per task and seed, even when several models are selected. Q and QD cross every
selected checkpoint. The model id is part of each learned-arm cell id, cell
manifest, and aggregate row.

The protocol contains three ordered exploratory allocations:

| Stage | Selection | Per-run caps | Protocol-wide provider-score cap |
| --- | --- | --- | ---: |
| `gate-1-score-smoke` | sign/Q/seed 101, selected models | 1 particle, 1 stage, 128 scores | 512 |
| `gate-2-size-pilot` | map + signed-window, Q/QD, seed 101 | 4 particles, 1 stage, 4,000 scores | 64,000 |
| `gate-3-size-replication` | Gate-2 tasks/arms, four remaining seeds | 4 particles, 1 stage, 4,000 scores | 256,000 |

A stage may be narrowed with `--models`, `--tasks`, `--arms`, or `--seeds`, but
those selectors cannot expand beyond it. The stage's provider-score ceiling is
enforced before execution. `--max-provider-cells` and
`--max-provider-score-cap` add stricter operator-side gates.

### Exploratory Gate-2 transport amendment

The original protocol at SHA-256
`095f7ad83eaea20168a14764ec501fb916b1c9677ed3d988998ed34402b9b0e2`
used `candidate_batch_size=128`. During the 32B Gate-2 preflight, map Q/QD
completed, signed-window Q encountered an HTTP 524 in a large hole-scoring
request, and signed-window QD was interrupted. Those artifacts remain retained
as exploratory preflight evidence; they are not pooled with amended outcomes.

The frozen exploratory amendment is
`protocol-size-study-transport32.json`, SHA-256
`56c590864d456f884c82bf62ada3c11c1c2504d21b021e650bc323007553fc60`.
Its sole computational change is `candidate_batch_size: 128 -> 32`. Candidate
support, prompts, finite energies, seeds, particles, stages, and the 4,000-score
per-cell budget are unchanged. The amendment was made after a transport failure,
so every outcome under it remains exploratory even when a task formerly carried
a confirmatory label.

All four 32B Gate-2 task-by-arm cells must be rerun in a new output directory
under the amended hash, including the two map cells that completed during
preflight. Every later size-study allocation must also name the amended protocol;
do not resume or append to an output directory created under the old hash.

Dry-run the four-cell 32B rerun with an explicit 16,000-score ceiling:

```bash
uv run python -m research.run_matrix \
  --protocol research/protocol-size-study-transport32.json \
  --output research/outputs/size-gate2-32b-transport32-rerun \
  --stage gate-2-size-pilot \
  --models qwen25-coder-32b \
  --max-provider-cells 4 \
  --max-provider-score-cap 16000 \
  --dry-run
```

After verifying that plan, remove `--dry-run` and add
`--base-url "$MODELSMC_VLLM_BASE_URL"`. This harness does not execute that paid
rerun as part of local verification.

## Deduction-underconstrained stress test

`protocol-deduction-stress-v1.json` is a separate exploratory D-versus-QD
experiment. It must not be pooled with the Gate-2 size study. Its sparse
bounded-square task retains the target program in a 36,198-trace auto-family
support, including two exact construction traces, but none of the nonempty
training inputs has an observed suffix. Consequently, sound deduction derives
no examples for either the filter predicate or mapped-value hole. Deduction
therefore supplies no discriminating evidence inside those catalogs; the Occam
component still distinguishes programs by cost.

The task was deliberately designed and screened after the original pilot. D
failed all five frozen seeds before any Qwen score was observed. This makes the
experiment a transparent stress test of whether Qwen finite-choice scores can
add discovery signal where deduction is underconstrained, not an unbiased task
sample or confirmatory benchmark.

The exact per-cell score ceiling is
`4 * (3 family choices + 600 predicates + 60 mapped values) = 2,652`.
Run the provider-free preflight first:

```bash
uv run python -m research.run_matrix \
  --protocol research/protocol-deduction-stress-v1.json \
  --output research/outputs/deduction-stress-d-v1 \
  --stage local-d-preflight \
  --arms D
```

Dry-run the first paid QD gate without contacting a provider:

```bash
uv run python -m research.run_matrix \
  --protocol research/protocol-deduction-stress-v1.json \
  --output research/outputs/deduction-stress-qd32-v1 \
  --stage paired-d-qd-32b \
  --arms QD \
  --seeds 101 \
  --models qwen25-coder-32b \
  --max-provider-cells 1 \
  --max-provider-score-cap 2652 \
  --dry-run
```

After the pinned 32B vLLM processed-logprob contract succeeds, remove
`--dry-run` and add `--base-url "$MODELSMC_VLLM_32B_BASE_URL"`. Continue with
the remaining four seeds if and only if that provider contract succeeds,
regardless of whether seed 101 finds an exact program. The five QD cells have a
hard combined ceiling of 13,260 scored candidates and at most 440 HTTP batches
at batch size 32. The final self-contained paired matrix reruns D locally with
QD rather than joining observations from separate output directories.

### V1 math audit and corrected v2 protocol

The sealed v1 artifacts remain immutable, but their earlier interpretation is
corrected by `research.audit_math`. Run the end-to-end replay with:

```bash
uv run python -m research.audit_math \
  research/outputs/deduction-stress-paired-v1 \
  --strict --output /tmp/deduction-stress-v1-math-audit.json
```

The audit checks the matrix seal, replays every finite categorical ledger,
reconciles paired paths and telemetry, materializes all 36,198 target states,
and tracks whether each derived probability is identified by a prefix-consistent
chain. Its findings are recorded in
`research/DEDUCTION_STRESS_MATH_AUDIT.md`.

`protocol-deduction-stress-v2.json` is the corrected exploratory follow-up. It
changes only the task's `lossScale` from `0.75` to `2.0`, preserves the original
task ID and held-out corpus, and uses family/hole deduction mixes of `0.75/0.0`
for D and QD. Run its provider-free reference gate first:

```bash
uv run python -m research.run_matrix \
  --protocol research/protocol-deduction-stress-v2.json \
  --output research/outputs/deduction-stress-v2-reference \
  --stage provider-free-reference-audit
```

That stage writes `target_audit_certificate.json` only after the exhaustive
target and its artifact hashes pass. The paid pilot requires that certificate:

```bash
uv run python -m research.run_matrix \
  --protocol research/protocol-deduction-stress-v2.json \
  --output research/outputs/deduction-stress-v2-pilot \
  --stage paired-d-qd-32b-pilot \
  --audit-certificate \
    research/outputs/deduction-stress-v2-reference/target_audit_certificate.json \
  --base-url "$MODELSMC_VLLM_32B_BASE_URL"
```

The one-seed D/QD provider pilot remains explicitly unrun. It must not reuse
the v1 splice proxy as a QD exact-path probability.

## Dry-run and execution

Run commands from `papers/01-modelsmc/modelsmc-pbe-python`. This is the exact
one-cell Gate-1 dry run for the pinned 3B server. It makes no provider calls,
does not require an endpoint, and writes no artifacts:

```bash
uv run python -m research.run_matrix \
  --output research/outputs/size-gate1-3b-q-sign-seed101 \
  --stage gate-1-score-smoke \
  --models qwen25-coder-3b \
  --max-provider-cells 1 \
  --max-provider-score-cap 128 \
  --dry-run
```

Run that exact gated cell against the live server:

```bash
export MODELSMC_VLLM_BASE_URL='https://<POD_ID>-8000.proxy.runpod.net/v1'
uv run python -m research.run_matrix \
  --output research/outputs/size-gate1-3b-q-sign-seed101 \
  --stage gate-1-score-smoke \
  --models qwen25-coder-3b \
  --max-provider-cells 1 \
  --max-provider-score-cap 128 \
  --base-url "$MODELSMC_VLLM_BASE_URL"
```

Without `--base-url`, each model first uses its own declared endpoint variable
(`MODELSMC_VLLM_3B_BASE_URL`, and analogously 7B/14B/32B), then the shared
`MODELSMC_VLLM_BASE_URL`. A single `--base-url` override is appropriate only
when all selected aliases really are served at that endpoint.

Use a restricted selection only for an explicitly exploratory smoke test:

```bash
uv run python -m research.run_matrix \
  --output research/outputs/smoke-u-d \
  --tasks map-increment \
  --arms U,D \
  --seeds 101
```

If the driver is interrupted, rerun the same command with `--resume`. Existing
cell manifests are immutable observations and are retained. Configuration
errors before a cell can be materialized abort the driver rather than being
misreported as an algorithmic failure.

Learned arms use `research/cache/candidate-scores` as a content-addressed
read-write score cache. Its identity includes repository, immutable revisions,
served alias, processed-logprob server fingerprint, and the explicit
`mean-full-prompt-conditional-logprob` energy. `research/cache/` is ignored by
Git and excluded from release bundles: it is an execution optimization, not
evidence. Reproducible replay evidence remains in each run's checksummed score
ledger, manifest, events, and result artifacts.

## Held-out evaluation

Held-out inputs come from SHA-256 counter streams and exclude all training
inputs. Their expected outputs come from version-controlled task oracles, never
from the synthesized program. Scalar tasks use a frozen integer domain; list
tasks use frozen domains, length caps, counts, and seeds. Every prediction is
stored in `heldout.json` for independent checking.

To re-evaluate a single core artifact:

```bash
uv run python -m research.heldout \
  --task foldr-signed-window \
  --result runs/RUN_ID/result.json \
  --output heldout.json
```

## Aggregation and archive

Aggregation walks the matrix plan, not merely the successful artifact
directories. It reads each cell manifest plus the core `manifest.json`,
`result.json`, and `events.jsonl`, then writes equivalent tidy CSV and JSON:

```bash
uv run python -m research.aggregate research/outputs/confirmatory-v1
```

The stable columns include full model identity and size, ever-visited training
success, held-out correctness and program provenance, loss, cost, final/minimum
ESS, total-variation distance, signed/absolute log-normalizer error, finite
candidates scored, wall time, support size, and failure reason.
Missing diagnostics stay null; they are never silently replaced with favorable
values.

## Publication figures

`research.figures` consumes one or more tidy `metrics.json` or `metrics.csv`
files. It rejects mixed protocol hashes, conflicting duplicate cells, an
existing destination, and malformed rows. A successful build contains the
normalized plotted rows, an input/checksum manifest, and every generated chart
as vector PDF, editable SVG, and 450-DPI PNG.

The default figure grid is amended Gate 2: four Qwen2.5-Coder checkpoints,
map-increment and signed-window tasks, Q and QD arms, and seed 101. The smallest
defensible chart set is:

- `outcome_matrix`: raw training-exact and held-out-exact cells by
  checkpoint/task/arm;
- `provider_work`: raw provider-scored token positions, provider wait for cache
  misses, and cold/mixed/warm cache disposition;
- `paired_q_vs_qd`: raw seed-paired Q-to-QD transitions, generated only when a
  checkpoint/task has at least two paired seeds.

With one seed, the figures say “exploratory,” show raw cells, and explicitly say
that no confidence intervals are present. The pipeline never treats particles
as replicates and never manufactures an interval from `n=1`. Even when repeated
seeds exist, the paired chart shows raw seed pairs without inferential bars.

First aggregate every amended per-model matrix, then provide all four tidy files
to one build. For example:

```bash
uv run python -m research.figures \
  research/outputs/size-gate2-3b-transport32/analysis/metrics.json \
  research/outputs/size-gate2-7b-transport32/analysis/metrics.json \
  research/outputs/size-gate2-14b-transport32/analysis/metrics.json \
  research/outputs/size-gate2-32b-transport32-rerun/analysis/metrics.json \
  --output paper/generated/figures/gate2-size-transport32 \
  --models qwen25-coder-3b,qwen25-coder-7b,qwen25-coder-14b,qwen25-coder-32b \
  --tasks map-increment,foldr-signed-window \
  --arms Q,QD \
  --seeds 101 \
  --dpi 450 \
  --latex-prefix generated/figures/gate2-size-transport32
```

The manifest sets `eligible_for_exploratory_paper_insertion=true` only when the
entire ordered four-checkpoint Gate-2 grid is terminal under one protocol hash.
It always leaves `confirmatory_claim_ready=false` for this pilot. Do not add a
preliminary subset to `paper/main.tex`; once complete, the PDF paths are already
relative to that LaTeX file, for example:

```latex
\includegraphics[width=\linewidth]{generated/figures/gate2-size-transport32/outcome_matrix.pdf}
```

The intended paragraph attachments and claim checks are recorded separately in
`paper/AUTHOR_REVIEW_GUIDE.md`. Figure generation is provider-free and never
modifies live matrix output directories.

Create a deterministic tar archive after aggregation:

```bash
uv run python -m research.bundle \
  research/outputs/confirmatory-v1 \
  --output research/archives/confirmatory-v1.tar.gz
```

The archive contains a sorted `SHA256SUMS` inventory. A sidecar
`confirmatory-v1.tar.gz.sha256` authenticates the archive itself. Provider key
values are never written by the harness; only an environment-variable name may
appear in manifests.

## Verification

The research utilities have deterministic and golden-fixture tests independent
of paid providers:

```bash
uv run pytest -q research/tests
uv run ruff check research
uv run mypy --strict research
```
