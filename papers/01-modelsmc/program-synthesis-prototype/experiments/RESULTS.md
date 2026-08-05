# Results: Proposer capability, not search strategy, gates LLM-guided verified SMC synthesis

*Lab report for the verified-SMC programming-by-example prototype (papers 1 + 2). Experiment run 2026-08-04.*

## Research questions

The prototype combines three ingredients: **(i)** programming-by-example over recursive
list structures with typed program-family deduction (Paper 2), **(ii)** a
Dafny/LemmaScript-verified core that type-checks and evaluates every candidate, and
**(iii)** sequential-Monte-Carlo search with LLM proposers, soft edit-distance loss, and
cost-penalized weights (Paper 1). We ask:

- **RQ1** — At equal proposal budget, does ancestry-aware iterative SMC refinement beat
  independent one-shot sampling?
- **RQ2** — How does the answer depend on proposer capability (frontier cloud model vs.
  small cloud model vs. local open-weight model vs. offline catalog)?
- **RQ3** — What work does the verified boundary do during search?

## Setup

**Task.** `foldr-bounded-square`: from 14 input/output examples over `List<Int>`, discover

```text
λxs. foldr (λitem. λacc. if (-2 < item && item < 3) then (item*item :: acc) else acc) [] xs
```

i.e. keep elements in the open interval (−2, 3), square them, preserve order. The task
requires composing three ideas (fold traversal, a two-sided boundary predicate, and an
arithmetic transform) under a structural cost cap of 30, which rejects
memorize-the-examples enumerations. Both search arms receive **exactly 4 proposal
calls** per run:

| Arm | Configuration | Character |
|---|---|---|
| one-shot | 4 particles × 1 iteration | independent samples, no feedback |
| iterative | 2 particles × 2 iterations (relative-ESS threshold 1) | round 2 revises round-1 survivors with loss-ranked failure feedback |

**Proposers.** Claude Sonnet 5 and Claude Haiku 4.5 via the Anthropic Messages API
(forced tool-use, temperature omitted); Qwen3-Coder 30B (`qwen3-coder:30b-a3b-q8_0`)
via a local Ollama endpoint with strict JSON-schema output; and the deterministic
offline catalog as a floor. 10 SMC seeds per cell (5 for the local model). All other
knobs at repository defaults; every run logs a deterministic JSONL trace.

**Metrics.** Primary: exact solution (verified evaluator matches all 14 examples) within
budget. Secondary: final best-so-far soft loss (bounded edit distance, 0 = exact), and
the proposal call at which the first exact program appeared.

## Headline results

![Exact-solution rate by proposer and search arm](figures/fig1-success-rate.svg)

| Proposer | one-shot | iterative | Fisher p (success) | mean final loss one-shot → iterative | permutation p (loss) |
|---|---|---|---|---|---|
| Claude Sonnet 5 | **10/10** | **9/10** | 1.00 | 0.0 → 0.4 | 1.00 |
| Claude Haiku 4.5 | 0/10 | **3/10** | 0.21 | 12.2 → **4.4** | **0.011** |
| Qwen3-Coder 30B | 0/5 | 0/5 | 1.00 | 21.4 → 22.0 | 1.00 |
| Catalog (offline) | 0/10 | 0/10 | 1.00 | 22.0 → 22.0 | 1.00 |

*Wilson 95% CIs are drawn as whiskers in the figure. Fisher: two-sided exact test on
exact-solve counts. Permutation: exact two-sided test on mean final loss (all
C(20,10)=184,756 partitions enumerated; C(10,5)=252 for Qwen).*

![Final best-so-far loss per run](figures/fig2-final-loss.svg)

The answer to RQ1 is **conditional on capability** — three regimes are visible:

### Regime 1 — Above threshold, feedback is unnecessary (Sonnet 5)

Sonnet solves the task from a standing start in essentially every run (10/10 one-shot,
median first exact at proposal call 2; 9/10 iterative). At ceiling there is no headroom
for iteration to add anything: both arms are statistically indistinguishable (p = 1.0).
A frontier proposer makes the *search strategy* irrelevant on a task of this
difficulty — the interesting engineering margin is entirely in the verified rejection
loop (below).

### Regime 2 — Near threshold, feedback is decisive on loss (Haiku 4.5)

Haiku one-shot never solves the task (0/10; median loss 11, spread 4–21). With
ancestry-aware refinement at the same budget it reaches exactness in 3/10 runs and
compresses the loss distribution to median 4: mean final loss falls from 12.2 to 4.4,
significant under the exact permutation test (**p = 0.011**) despite the binary
success difference not reaching significance at this sample size (Fisher p = 0.21).
Every one of the 10 iterative lineages improves on its seed (fig. 3, middle panel).
This is the zone where the paper-1 mechanism — resample onto the best partial program,
hand the proposer its loss-ranked failures — visibly converts a non-solving model into
a near-solving one.

### Regime 3 — Below threshold, feedback cannot rescue (Qwen3-Coder 30B, catalog)

Qwen3-Coder 30B never improves on the loss-22 empty-list seed in either arm (one run
reached 19 by luck); the deterministic catalog is identically stuck. Iterative feedback
multiplies capability that must already exist — it cannot create the composite insight
(fold + two-sided boundary + squaring) from nothing.

![Champion-lineage loss along SMC ancestry](figures/fig3-lineage.svg)

## What the traces show (forensics)

### Haiku's near-misses are one bug, and feedback is what fixes it

All six loss-4 iterative champions share a **single identical defect**: lower bound
`-1 < item` instead of `-2 < item`, silently dropping `item = -1` (whose square, 1,
appears in 4 of 14 expected outputs — hence exactly loss 4). The `{0, 1, 2}` keep-range
is deceptive: since 0² = 0 and 1² = 1, identity looks correct on most kept values, and
several champions even replace `item*item` with identity plus a special case for 2.
The exact/near-miss split among iterative runs is decided entirely by what cycle 2
does with the feedback. In the solved runs the mechanism is legible in the model's own
rationale — seed 3's second cycle reads the failing-example list and states: *"examples
3, 12, 13, and 14 show that Item=-1 should produce 1 in the output (since -1 × -1 = 1)…
adjusting the condition to -2 < Item"* — and emits the target program exactly
(loss 4 → 0, 14/14). One-shot haiku stops precisely where iterative cycle 1 stops:
its best run lands on the same canonical near-miss, and with no second cycle the
boundary is never repaired (0/10, median loss 11).

### Sonnet's single miss was a wrong hypothesis plus two wasted repair calls

The one sonnet iterative failure (seed 3) locked onto a plausible-but-wrong piecewise
rule in cycle 1 — *negate negatives, map 2 → 4, drop ≥ 3* (loss 4, 11/14) — never
proposing `Multiply(item, item)`. Both cycle-2 repair calls were then lost to
non-semantic causes: one schema-envelope violation, and one cost-cap rejection at
cost 31 caused by a redundant `&& true` padding the guard — and that rejected program
was semantically identical to its parent anyway. At ceiling, the residual failure mode
is bad luck stacked on a wrong prior, not a capability gap.

### Qwen's failure is mode collapse below the composition threshold

Qwen3-Coder 30B always picks the right *family* (`foldr` with initial `[]`) — the
deduction hints do land — but never composes the three required ideas: it **never
proposes a two-sided conjunction** (only one-sided `item < 0` or per-value equality
chains), **never proposes `item*item`** (it misreads `[2] → [4]` as doubling and
`[-1] → [1]` as negation), and it misuses the accumulator, returning `[]` instead of
`acc` to "drop" an item — which actually truncates everything folded so far. Its modal
proposal scores loss 24, *worse than the loss-22 empty seed*, so 9 of 10 runs end with
the champion still being the seed. Most striking is the **mode collapse**: 23 of its 28
accepted proposals are byte-identical across seeds, arms, and feedback rounds — after
receiving "loss 22 → 24" feedback it re-proposes the same program verbatim. Its only
structurally different idea is a cost-53 per-value memorization chain, rejected by the
cost cap in 7 of 10 runs. Feedback cannot help a proposer whose proposal distribution
has collapsed.

## The verified boundary's role (RQ3)

Across all 280 proposal calls, the boundary filtered **50 (17.9%) before scoring**:

| Proposer | requested | accepted | cost-cap rejected | type-error | envelope/API failed | accepted but worse than parent |
|---|---|---|---|---|---|---|
| Claude Sonnet 5 | 80 | 53 | 21 | 0 | 6 | **0** |
| Claude Haiku 4.5 | 80 | 69 | 9 | 0 | 2 | 6 |
| Qwen3-Coder 30B | 40 | 28 | 10 | 2 | 0 | 24 |
| Catalog | 80 | 80 | 0 | 0 | 0 | 60 |

Three observations. **First**, every one of the 40 cost rejections is a
memorize-the-examples if-chain (cost 31–44 for the Claude models, a byte-identical
cost-53 chain for qwen) — the cost cap is doing exactly its Occam job, forcing
relational predicates over enumeration. **Second**, the *strongest* proposer is
rejected the *most* (33.75% for sonnet vs 13.75% for haiku): sonnet explores
aggressively at the cost boundary, and the verifier converts that aggression into
safe search pressure. Yet among its accepted proposals, sonnet never once produced a
child scoring worse than its parent (0/53; qwen: 24/28). **Third**, scope violations
and decode failures were both zero across 280 calls — the forced tool-use schema plus
prompt constraints fully prevented the malformed-AST class the decoder was built to
catch, leaving the semantic checks (types, cost) as the active defenses. On every
rejection or failure the engine fell back to the particle's ancestor; no run crashed,
and one sonnet run went exact despite losing 2 of its 4 calls to envelope failures.

An incidental grammar finding: haiku once requested an `Or` node the DSL doesn't have
(the API rejected it against the tool schema), and sonnet worked around the missing
disjunction with a De Morgan rewrite in an earlier exploratory run — evidence that the
16-node grammar's lack of `Or` is a real, if minor, expressiveness gap that models
notice.

## Threats to validity

- **Single task.** These three regimes are indexed to one task's difficulty; the
  capability threshold moves with the task. The easy tasks in `examples/` are solved
  one-shot by every LLM proposer tested, including Qwen.
- **Small N.** 10 seeds per cell (5 for Qwen) gives wide Wilson intervals; the
  haiku success-rate difference (0/10 vs 3/10) is suggestive, not significant. The
  loss-based permutation test is the powered comparison.
- **LLM nondeterminism.** Claude 5 models ignore temperature; run-to-run variation
  comes from the model itself and from SMC seed effects on feedback routing, and the
  two are not separable in this design.
- **Feedback design not ablated.** The iterative arm bundles resampling, loss-ranked
  failure feedback, sibling-avoidance lists, and champion retention; this experiment
  does not isolate which component carries the effect.
- **Budget scale.** 4 calls is a deliberately tight budget. Larger budgets may lift
  haiku into regime 1 or qwen into regime 2; the regime *structure* is the claim, not
  the specific model placements.

## Reproduction

```bash
ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts   # ~15 min, ~160 cloud calls
npx tsx experiments/analyze.ts                            # stats.json + console table
npx tsx experiments/figures.ts                            # SVG figures
```

Raw JSONL traces land in `runs/experiments/` (gitignored); parsed results in
`experiments/results/summary.json`.
