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

This dataset archives the evidence package for **Deduce, Propose, Correct:
Probability-Accountable LLM Guidance for Typed Program Inference**.
The manuscript authors are **Tri Nguyen** and **Dat Nguyen**.

The `exploratory-pilots-v0.1.2` release is intentionally labeled
**exploratory, not confirmatory**. It contains the eight matched seed-23 pilot
runs declared in `pilot_release.json`, including inexact outcomes rather than
only successful examples. It must not be used to infer population success
rates or statistical significance.

The implementation snapshot is pinned to Git revision
`578eba5c9c97456335bdc7dc9a51a50602b3809f`; the uploaded `SHA256SUMS` file
independently covers every artifact in this release.

## What is being studied

The system transforms input-output examples into typed program skeletons,
refutes structurally impossible families, derives specifications for finite
holes, and asks an LLM to score canonical choices. The sampled proposal mixes
LLM energies, a symbolic deduction guide, and a uniform support floor. Because
every family and hole choice has an explicit probability, the proposal can be
importance-corrected and compared with an exact finite reference on small
supports.

The LLM does **not** emit unrestricted source code in these experiments. The
initial model is `Qwen/Qwen3-Coder-30B-A3B-Instruct` at revision
`b2cff646eb4bb1d68355c01b18ae02e7cf42d120`, served through vLLM 0.11.0 with
processed prompt log probabilities.

The implementation supports two declared finite energies over each complete
teacher-forced prefix-plus-candidate prompt: the total scored log probability
(the historical default) and its mean over scored full-prompt positions. The
mean is a length-normalized energy, not a candidate-only likelihood; this
release makes no token-boundary claim for the textual prefix and candidate.

## Important negative result

The initial pilots do not show that Qwen improves synthesis. On the hardest
signed-window task, Qwen strongly disfavored the exact family and hole choices;
the symbolic deduction mixture rescued the successful particle. This is why
the archive exposes proposal components rather than presenting one passing
program as evidence of LLM value.

## Layout

- `pilot_release.json`: immutable labels for the matched exploratory subset.
- `pilots/runs/`: sanitized run manifests, events, particles, and results;
  failures are retained.
- `research/`: frozen held-out generators, matrix runner, aggregation code, and
  draft preregistration.
- `paper/`: JMLR-style manuscript source and the separate paragraph-level
  author review map; `paper/main.pdf` is the verified compiled manuscript.
- `src/`, `tests/`, and `examples/`: implementation, tests, and specifications
  in their original executable project layout.
- `pyproject.toml`, `uv.lock`, and `PROJECT_README.md`: the frozen environment
  and original project documentation needed to replay the controls.
- `SHA256SUMS`: checksums of every uploaded evidence file.

## Privacy and replay boundary

The publication copy removes local usernames, hostnames, absolute paths, and
ephemeral provider endpoints. No provider tokens are stored. Historical pilot
runs predate the complete candidate-score ledger and therefore cannot replay
every Qwen categorical exactly. Confirmatory releases will include complete
candidate strings, full-prompt token IDs, processed log probabilities, total
and configured energy, probabilities, selections, immutable model/tokenizer
revisions, and provider configuration.

## Intended use

Use this dataset to inspect implementation claims, reproduce provider-free
controls, validate result aggregation, or audit negative LLM-guidance outcomes.
Run `uv sync`, `uv run pytest -q`, and the commands in `research/README.md`
directly from the dataset root.
Do not treat the current handcrafted DSL tasks as evidence of general-purpose
program synthesis, a Bayesian posterior over unbounded programs, formal
verification, or an improvement over exhaustive enumeration.

## Licensing

This dataset card, experiment metadata, and sanitized run outputs are released
under CC BY 4.0. Third-party model and paper references retain their original
licenses. Source code is mirrored for reproducibility but does not receive a
new license through this dataset release.
