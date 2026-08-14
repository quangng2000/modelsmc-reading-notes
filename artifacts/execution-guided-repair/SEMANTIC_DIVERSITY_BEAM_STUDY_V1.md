# Semantic-Diversity Beam Study V1

## Outcome

The finite component-diversity selector fixed the observed beam-collapse bug, but
none of the three frozen GPT-OSS-120B trajectories reached an exact program.

The selector represents each state by predicate decisions and mapper values over
the sorted distinct training-input items. These signatures are finite
observational equivalence only, not proofs of global semantic equivalence. Every
selected beam contained two distinct joint semantic cells in every round; the
previous bounded-square pilot had only one semantic cell despite a syntactic
beam width of two.

## Frozen runs

### Primary exploratory task B

- Exact support after the provider seal: 2 / 36,000 programs.
- Search: 21 proposal slots, five provider calls.
- Best loss: 18 -> 10 -> 3; no exact program.
- GPT-OSS generated the correct mapper `sub(3,item)` but proposed the incomplete
  predicate `lt(item,2)`, which incorrectly retained `-3`.
- Matched random semantic beam: 6 exact successes / 10,000 trials.
- Only 0.16% of random trials achieved best loss at most 3, so this single LLM
  trajectory had lower loss than 99.84% of the matched random trials.

### Secondary exploratory task A

- Exact support after the provider seal: 6 / 36,000 programs.
- Search: 13 proposal slots, three provider calls; unsupported branches stalled
  rather than receiving invented feedback or replacement proposals.
- Best loss: 16 -> 10; no exact program.
- GPT-OSS generated a correct mapper form (`item+2`) under the wrong predicate
  `eq(item,3)`.
- Matched random semantic beam: 1 exact success / 10,000 trials.
- 5.14% of random trials achieved best loss at most 10.

### Developmental bounded-square replay

- Exact support after the provider seal: 2 / 36,000 programs.
- Search: 21 proposal slots, five provider calls.
- Best loss: 24 -> 17; no exact program.
- The selector collapsed predicate and mapper syntax aliases and maintained two
  semantic cells, but this provider seed never proposed the square mapper.
- Matched random semantic beam: 22 exact successes / 10,000 trials.
- 56.71% of random trials achieved best loss at most 17.

## Interpretation

This study establishes a narrow mechanism result: typed LLM shortlists can be
generated without enumerating or disclosing all complete programs, and finite
component signatures can prevent syntactic aliases from consuming the beam.

It does **not** establish faster exact-program discovery. The observed exact-hit
rate was 0/2 on the newly designed exploratory tasks and 0/1 on the developmental
replay. The LLM generated the correct mapper on both new tasks, but the
three-round local-feedback loop did not combine it with the exact predicate.

The two new tasks were designed after diagnosing the earlier failure. They were
frozen before provider execution and contain no reused target expressions, but
they are exploratory rather than confirmatory held-out tasks. A defensible
speedup claim needs a task generator and task seeds sealed before further prompt
or algorithm development, followed by repeated LLM seeds and equal-budget
matched baselines.

## Reproducibility

- Frozen protocol: `papers/01-modelsmc/modelsmc-pbe-python/research/protocol-semantic-diversity-beam-v1.json`
- Executed harness SHA-256:
  `1e27cef2ba45792b2249af0a65758b81347747805faa979cd6ca11001fc5a739`
- Raw artifacts:
  - `semantic-beam-opaque-b-low-k4-w2-r3-v1/`
  - `semantic-beam-opaque-a-low-k4-w2-r3-v1/`
  - `semantic-beam-bounded-low-k4-w2-r3-v1/`

All three provider inventories were independently recomputed and matched their
stored seals. Prompt audit found no full-space cardinalities, target ASTs, exact
predicate labels, or exhaustive loss tables.
