# Blind-V2 Evidence-Frontier Confirmation

Status: **PASSED**

This was a preregistered fresh blind confirmation using the frozen
GPT-OSS-120B execution-guided evidence-frontier system on 12 generated
finite-domain filter-map tasks.

## Primary result

- Exact zero-loss behavior by proposal slot 29: **12/12 tasks**.
- Exact zero-loss behavior by proposal slot 21: **9/12 tasks**.
- First-exact slots: `26, 18, 18, 18, 10, 14, 10, 18, 10, 10, 26, 22`.
- Mean first-exact slot: `16.67`; median: `18`.
- Matched-random mean task success rate at slot 29: **4.2292%**
  (`5,075 / 120,000` task-trials).
- None of 10,000 trial-index matched-random aggregate replicates reached the
  system's 12/12 result; their maximum was 5/12.
- Prespecified add-one Monte Carlo tail value:
  `p_MC = (1 + 0) / (10,000 + 1) = 0.0000999900009999`.

The preregistered success rule required at least 6/12 exact tasks and
`p_MC <= 0.05`; both conditions passed.

## Search effort

- Provider calls attempted: `58` (`53` valid, `5` invalid).
- Proposal slots consumed: `244`, including one initial state per task and
  all invalid-call slots.
- Logical candidate evaluations: `224`.
- Unique programs physically scored: `178`.

The system proposed four typed local repairs per expanded state. It did not
enumerate the full complete-program space to construct its proposals.

## Blindness and integrity

- Reveal-free analysis passed before unblinding.
- All 324 public run/analysis artifacts verify against `SHA256SUMS`.
- Reveal-free public-bundle commitment:
  `586bbddf9324502ce107fa5dfc62ea6a14b120c281892435a8b29b99062ed425`.
- Post-unblind verification passed the secret commitment, one-use receipt,
  hidden-reveal hash, deterministic byte regeneration, all 12 target
  commitments, and all oracle executions.
- Post-unblinding, every discovered program matched the hidden target's
  behavior over the complete declared domain `{-3, ..., 4}`. Only one of the
  12 recovered programs was syntactically identical; the rest were equivalent
  simplifications, reordered conjunctions, or commutative arithmetic forms.

## Supported claim

On this frozen 12-task distribution and 29-slot budget, the hybrid
LLM-plus-interpreter-plus-selector system showed a bounded exact-discovery
advantage over the matched random proposal mechanism.

This result does **not** establish hidden-AST recovery, wall-clock speedup,
superiority to exhaustive search, valid importance sampling, arbitrary-task
generalization, or scaling to million/billion-program spaces.

## Evidence

- Reveal-free summary: `analysis/SUMMARY.md`
- Full reveal-free analysis: `analysis/analysis.json`
- Immutable public inventory: `public-bundle-seal/SHA256SUMS`
- Public bundle commitment: `public-bundle-seal/BUNDLE_SHA256`
- Post-unblind verification: `post-unblind-verification.json`

