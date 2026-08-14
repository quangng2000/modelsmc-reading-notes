---
license: cc-by-4.0
pretty_name: ModelSMC-PBE Gate-2 Size Grid Evidence
tags:
- program-synthesis
- programming-by-example
- sequential-monte-carlo
- large-language-models
- reproducibility
---

# ModelSMC-PBE exploratory Gate-2 evidence

This data-only release accompanies **Deduce, Propose, Correct:
Probability-Accountable LLM Guidance for Typed Program Inference**, by **Tri
Nguyen** and **Thanh-Dat Nguyen**. Version `v0.2.0` archives the complete
exploratory Gate-2 size grid: two tasks, two arms, four pinned Qwen2.5-Coder
checkpoints, and seed 101 (16/16 planned cells). All completed and inexact cells
are retained under an intention-to-treat policy.

This is exploratory evidence, not a confirmatory benchmark. It has one seed per
checkpoint-task-arm cell, every task family was visible during development, and
the 128-to-32 candidate transport amendment followed an HTTP 524. The archive
therefore supports artifact audit and descriptive comparison only. It does not
establish unseen-task generalization, statistical significance, or a monotone
model-size effect.

## Result in one sentence

Across this grid, Qwen-only (`Q`) found an exact program in 0/8 cells, whereas
Qwen plus symbolic deduction (`QD`) found an exact, held-out-correct program in
8/8 cells. This is evidence for the combined proposal on these cells; it does
not isolate a beneficial effect of Qwen, deduction, model scaling, resampling,
or cache use.

## Reproducibility contract

- Frozen amended protocol: `protocol/protocol-size-study-transport32.json`,
  SHA-256 `56c590864d456f884c82bf62ada3c11c1c2504d21b021e650bc323007553fc60`.
- Checkpoints: `Qwen/Qwen2.5-Coder-{3B,7B,14B,32B}-Instruct`; exact model and
  tokenizer revisions are recorded in `release_manifest.json`, each cell, each
  run manifest, and each score wave.
- Serving contract: vLLM 0.11.0, processed prompt log probabilities, bfloat16,
  4096-token model length, and NVIDIA A100-SXM4-80GB. The complete declared
  server configuration is preserved; ephemeral endpoints and local paths are
  redacted.
- Energy: mean conditional log probability over the scored positions of the
  complete teacher-forced prefix-plus-candidate prompt. It is a finite energy
  heuristic, not a candidate-only likelihood.
- Every score wave preserves its prompt prefix and hash, every canonical
  candidate, token IDs and processed log probabilities, total and normalized
  energies, component and mixture probabilities, selection, immutable
  revisions, and cache provenance.

Large JSON/JSONL evidence files are sanitized and deterministically gzip
compressed. `EVIDENCE_INDEX.json` maps every source-relative artifact to its
packaged path and records source, sanitized-uncompressed, and packaged hashes
and sizes. `SHA256SUMS` covers every packaged file. Decompression is ordinary
gzip; no custom software is needed.

## Layout

- `release_manifest.json`: authors, evidence label, exact grid allowlist,
  protocol/manuscript hashes, checkpoint revisions, and packaging semantics.
- `protocol/`: the frozen transport-amended protocol used by all 16 cells.
- `evidence/grid/`: four declared model groups. Each includes the matrix
  manifest, per-group metrics, cell status and held-out evaluation, and
  complete structured core run artifacts. No raw stdout/stderr or undeclared
  output directory is copied.
- `aggregate/metrics.{json,csv}`: the sanitized 16-row descriptive table.
- `figures/`: publication figures plus their data and checksum manifest.
- `paper/main.pdf`: the final checksummed manuscript PDF.
- `EVIDENCE_INDEX.json` and `SHA256SUMS`: replay map and byte-level integrity.

Score-cache blobs are deliberately excluded. Cache hits remain auditable from
the immutable cache keys, score-wave provenance, and run-level hit/miss metrics;
the archive does not duplicate the local cache store. The release also excludes
all executable source, tests, environments, and dependency lockfiles.

## Implementation reference

The implementation lives only on GitHub:
[repository](https://github.com/quangng2000/modelsmc-reading-notes), branch
[`python-modelsmc-pbe`](https://github.com/quangng2000/modelsmc-reading-notes/tree/python-modelsmc-pbe),
clean publication commit
[`{{SOURCE_REVISION}}`](https://github.com/quangng2000/modelsmc-reading-notes/commit/{{SOURCE_REVISION}}),
under `papers/01-modelsmc/modelsmc-pbe-python`.

The experimental run manifests honestly record base revision
`9c94d2906d4d89290d22b4b90ac308bbf6f6c8a0` and `dirty=true`. The clean
publication commit is therefore the exact implementation reference for the
released method and builder, not a claim that the older base commit alone
reproduces the experimental worktree. For the archived observations, the
packaged ledgers, manifests, protocol, and checksums are authoritative.

## Privacy and intended use

The publication builder removes local usernames, hostnames, absolute paths,
ephemeral RunPod endpoints, email addresses other than the verified publication
address, tokens, and secret-valued fields. It scans uncompressed content even
when packaged as gzip and rejects unknown binary files or incomplete checksum
coverage.

Use this release to audit finite proposal probabilities, compare descriptive
outcomes, inspect cache provenance, and verify the manuscript tables and
figures. Do not treat it as a calibrated posterior over unbounded programs, a
general-purpose synthesizer benchmark, or evidence that one checkpoint size is
superior.

## License

Dataset metadata and sanitized experimental outputs are released under CC BY
4.0. The manuscript, Qwen checkpoints, vLLM, and cited works retain their own
licenses.
