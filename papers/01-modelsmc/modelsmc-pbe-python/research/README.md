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

## Dry-run and execution

Run commands from `papers/01-modelsmc/modelsmc-pbe-python`. A dry run makes no
provider calls and writes no artifacts:

```bash
uv run python -m research.run_matrix \
  --output research/outputs/confirmatory-v1 \
  --label confirmatory \
  --dry-run \
  --base-url 'https://<POD_ID>-8000.proxy.runpod.net/v1'
```

For the real Q/QD matrix, provide the vLLM endpoint either with `--base-url` or
without putting it in the protocol:

```bash
export MODELSMC_VLLM_BASE_URL='https://<POD_ID>-8000.proxy.runpod.net/v1'
uv run python -m research.run_matrix \
  --output research/outputs/confirmatory-v1 \
  --label confirmatory
```

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

The stable columns include training success, held-out correctness, loss, cost,
final/minimum ESS, total-variation distance, signed/absolute log-normalizer
error, finite candidates scored, wall time, support size, and failure reason.
Missing diagnostics stay null; they are never silently replaced with favorable
values.

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
