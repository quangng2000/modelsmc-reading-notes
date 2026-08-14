# Iterative Typed Beam Pilot

This is one exploratory GPT-OSS-120B trajectory on `foldr-bounded-square.json`.
It evaluates the search mechanism; it is not an importance-sampling run and is
not evidence of general speedup.

## Frozen search protocol

- Neutral initial state from seed 17: predicate
  `and(lt(item,-2),lt(item,-1))`, mapper `mul(4,item)`, loss 24.
- Generate four typed repairs per expanded state.
- Elitist beam width two for three rounds.
- Maximum budget: 21 proposal slots (1 initial + 4 + 8 + 8).
- Hole selection uses only automatic execution feedback from the current state.
- The provider receives the grammar templates but no full program list, component
  catalog cardinalities, target AST, target indices, or exhaustive loss table.
- Temperature 0, low reasoning effort, no retry or candidate backfill.

## Observed trajectory

- Round 1 repaired the predicate and improved best loss from 24 to 17.
- Round 2 generated the correct square mapper, `mul(item,item)`, on one branch.
  Because the retained predicate kept only zero, square tied at loss 17 with
  several identity-like mappers. The frozen hash tie-break removed the square
  branch.
- Round 3 improved best loss to 16 with predicate `eq(item,1)` and an identity
  mapper.
- Final result: no exact program in 21 proposal slots.

## Matched random-beam reference

Across 10,000 simulated random-beam trials using the same start, topology,
adaptive hole rule, loss selection, and budget:

- Exact successes: 1/10,000 = 0.01%.
- Wilson 95% interval: approximately 0.0018% to 0.0566%.
- Mean best loss: 17.6843; median best loss: 17.
- 11.89% of random trials achieved best loss at most 16, so this single LLM
  trajectory was strictly better in loss than 88.11% of random trials.

One failed LLM trajectory versus a Monte Carlo random reference cannot establish
an LLM speedup or disadvantage. It shows that the LLM can generate a constant-size
typed shortlist without enumerating all complete programs, but this frozen beam
did not preserve the promising square branch long enough to find the exact answer.

The raw provider inventory is sealed in `provider-seal.json`. The exact executed
harness is `executed_harness.py`; its SHA-256 matches `protocol.json`.
