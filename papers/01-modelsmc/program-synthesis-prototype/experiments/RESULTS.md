# Results: Capability gates, refinement depth unlocks — LLM proposers in verified SMC synthesis

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
C(20,10)=184,756 partitions enumerated; C(10,5)=252 for Qwen). Eight tests are
reported in this table; the haiku loss difference is the only one clearing α = 0.05,
and its uncorrected p = 0.011 does not survive a Bonferroni correction across all
eight (adjusted ≈ 0.088). We therefore read it as strong suggestive evidence pending
a larger-seed replication, backed by the mechanistic trace evidence below rather than
by the p-value alone.*

![Final best-so-far loss per run](figures/fig2-final-loss.svg)

The answer to RQ1 is **conditional on capability** — three regimes are visible:

### Regime 1 — Above threshold, feedback is unnecessary (Sonnet 5)

Sonnet solves the task from a standing start in essentially every run (10/10 one-shot,
median first exact at proposal call 2; 9/10 iterative). Both arms are statistically
indistinguishable at this sample size (p = 1.0) — and the point estimate is even
consistent with iteration slightly *hurting* at ceiling, since the iterative arm gets
only 2 independent first draws instead of 4 and a wrong cycle-1 hypothesis consumes
both repair calls (exactly the anatomy of the single sonnet miss, below). Either way,
a frontier proposer makes the *search strategy* irrelevant on a task of this
difficulty — the interesting engineering margin is entirely in the verified rejection
loop (below).

### Regime 2 — Near threshold, feedback shifts the loss distribution (Haiku 4.5)

Haiku one-shot never solves the task (0/10; median loss 11, spread 4–21). With
ancestry-aware refinement at the same budget it reaches exactness in 3/10 runs and
shifts the bulk of the loss distribution to 4 or below (9/10 runs, vs 1/10 one-shot):
mean final loss falls from 12.2 to 4.4 (exact permutation **p = 0.011** uncorrected;
the binary success difference does not reach significance at this sample size, Fisher
p = 0.21). The mechanism is not fail-safe — one iterative run (seed 5) lost two of its
four calls to rejections and finished at loss 20, inside the one-shot range. Note that
merely improving on the seed does not discriminate the arms (every haiku one-shot run
also beats the seed); the arm difference is in *how far below* the seed runs land
(fig. 2). This is the zone where the paper-1 mechanism — resample onto the best
partial program, hand the proposer its loss-ranked failures — visibly converts a
non-solving model into a near-solving one.

### Regime 3 — Below threshold, feedback did not rescue (Qwen3-Coder 30B, catalog)

In 9 of 10 Qwen3-Coder 30B runs the final champion is still the loss-22 empty-list
seed (the best run reached loss 19); the deterministic catalog is identically stuck.
With 5 seeds per arm the interval estimates are wide (Wilson upper bound 43%), so the
claim is strictly *did not rescue here*, not *cannot* — but the trace forensics below
give a mechanistic reason to expect more seeds at this budget to behave the same way.
Iterative feedback multiplies capability that must already exist; in these runs it had
nothing to multiply.

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
of its two best runs (both loss 4), one lands on the same canonical near-miss, and
with no second cycle the boundary is never repaired (0/10, median loss 11).

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
receiving "loss 22 → 24" feedback it re-proposes the same program verbatim. Its main
structural alternative is a cost-53 per-value memorization chain, rejected by the cost
cap in 9 of 10 runs; the remainder of its repertoire is a negate/`==4`-filter variant
(which produced its best run, loss 19), one doubling variant, and a type-rejected
Bool-returning reducer. Feedback has no purchase on a proposal distribution this
collapsed — though see the harness-parity caveat under threats: constrained decoding
on a quantized local checkpoint may itself contribute to the collapse.

## The verified boundary's role (RQ3)

Across all 280 proposal calls, the verified boundary rejected **42 (15.0%) before
scoring** (40 cost-cap, 2 type errors), and another 8 calls (2.9%) failed upstream of
it (response-envelope violations, decode errors, one `max_tokens` truncation):

| Proposer | requested | accepted | cost-cap rejected | type-error | envelope/API failed | accepted but worse than parent |
|---|---|---|---|---|---|---|
| Claude Sonnet 5 | 80 | 53 | 21 | 0 | 6 | **0** |
| Claude Haiku 4.5 | 80 | 69 | 9 | 0 | 2 | 6 |
| Qwen3-Coder 30B | 40 | 28 | 10 | 2 | 0 | 24 |
| Catalog | 80 | 80 | 0 | 0 | 0 | 60 |

Three observations. **First**, the cost cap is doing its Occam job. All ten of qwen's
cost rejections are one byte-identical cost-53 memorization chain; the Claude models'
30 cost rejections (cost 31–44) mix per-value enumerations with near-correct
*relational* programs pushed just over budget — including one padded with a redundant
`&& true` that landed at cost 31. **Second**, the *strongest* proposer is rejected the
*most* by the boundary itself (cost + type: sonnet 26.25% of calls vs haiku 11.25%):
sonnet explores aggressively at the cost boundary, and the verifier converts that
aggression into safe search pressure. Yet among its accepted proposals, sonnet never
once produced a child scoring worse than its parent (0/53; qwen: 24/28). **Third**,
scope violations were zero across all 280 calls, and only two calls died at the decode
stage — a haiku proposal using an `Or` node the grammar lacks (caught by the shell's
AST decoder, surfaced as a failed call) and one JSON-payload decode error — so the
forced tool-use schema plus prompt constraints almost entirely prevented the
malformed-AST class, leaving the semantic checks (types, cost) as the active defenses.
On every rejection or failure the engine fell back to the particle's ancestor; no run
crashed, and one sonnet run went exact despite losing 2 of its 4 calls to call-level
failures (one envelope violation, one `max_tokens` truncation).

An incidental grammar finding: haiku once proposed an `Or` node the DSL doesn't have
(rejected by the shell's AST decoder), and several sonnet proposals in this experiment
encode ranges via De Morgan rewrites such as `!(item < -1) && !(2 < item)` — evidence
that the 17-constructor grammar's lack of `Or` is a real, if minor, expressiveness gap
that models notice and work around.

## Experiment 2: a task-difficulty sweep to escape the one-shot ceiling

Experiment 1 could not measure feedback value at the frontier because sonnet one-shot
solved bounded-square 10/10. We therefore designed two harder tasks, validated before
any API spend by scoring the intended target program with the verified evaluator
(`experiments/validate-task.ts`):

- **`foldr-signed-window`** (list → list): negate values in (−3, 0), square values in
  (−1, 3), drop the rest. Three branches, two distinct transforms, and a planted
  deception — on `x = −1` negation and squaring coincide, so only `x = −2 → 2`
  separates the hypotheses. Target cost 28/32.
- **`foldr-window-penalty-sum`** (list → scalar): add `x²` for the window (−2, 3),
  subtract `x` for `x ≥ 3`, ignore `x ≤ −2`. Accumulator arithmetic with two distinct
  operations. Target cost 21/30.

Each task ran {Sonnet 5, Haiku 4.5} × {one-shot, iterative} × 10 seeds. First runs
used the experiment-1 response budget (2048 tokens); both were then **rerun at 4096
tokens** after a confound surfaced (below). Deconfounded results:

![Exact-solve rate across the task-difficulty sweep](figures/fig4-difficulty-sweep.svg)

| Task | Sonnet one-shot | Sonnet iterative | Haiku one-shot | Haiku iterative | Haiku loss permutation |
|---|---|---|---|---|---|
| bounded-square | 10/10 | 9/10 | 0/10 | 3/10 | iterative better, p = 0.011 |
| signed-window (4k) | 9/10 | 8/10 | 0/10 | 0/10 | n.s. (11.0 vs 9.5) |
| penalty-sum (4k) | 10/10 | 10/10 | 0/10 | 0/10 | **one-shot better, p = 0.011** |

### Finding A — The call-attrition confound: token budget is a hidden difficulty knob

At 2048 tokens the harder tasks provoked much longer model reasoning, and call-level
failures exploded: on penalty-sum, 38 of sonnet's 80 calls died (mostly `max_tokens`
truncation of the tool call), and its apparent "ceiling break" (8/10 both arms)
vanished at 4096 tokens (10/10 both arms, median first exact at call 1). On
signed-window at 2048, sonnet iterative dropped to 6/10 with a marginal deficit
(Fisher p = 0.087) — and all four misses had the *identical* anatomy: two of four
calls lost to failures, stranding the lineage at a loss-4 near-miss. At 4096 tokens
the gap disappears (9/10 vs 8/10, p = 1.0). Two lessons: **iterative arms are
structurally more fragile to lost calls** (a failed repair call wastes a lineage
step; a failed independent draw costs nothing), and arm comparisons are meaningless
unless call-failure rates are reported alongside them. Envelope violations, unlike
truncations, persisted at 4096 (19/80 sonnet calls on signed-window) — a harness
robustness gap, not a token problem.

### Finding B — The feedback zone is narrow, and iteration has a cost below it

Haiku's two significant loss results point in *opposite directions*: on
bounded-square, iteration helps (12.2 → 4.4, p = 0.011); on penalty-sum, iteration
*hurts* (38.0 vs 42.8, p = 0.011 at 4096 tokens, replicating p = 0.003 on the
independent 2048-token run set with zero haiku call failures in the 4k run). The
mechanism is the diversity cost of coupling: below the capability threshold, revising
a bad partial program is worth less than drawing two more independent samples. The
zone where ancestry-aware feedback pays — a reachable near-miss plus interpretable
failure feedback — occupied exactly one cell of this six-cell sweep (haiku ×
bounded-square). *Experiment 3 revises this picture: at a larger budget the zone is
far wider than these B = 4 corners suggest — what looked like a narrow capability
zone was substantially a refinement-depth ceiling.*

### Finding C — Difficulty is model-relative

The two hard tasks order differently for the two models. For sonnet, signed-window is
the harder task (9/10, 8/10 — its only sub-ceiling cells at 4096 tokens; the planted
−1/−2 deception works even at the frontier occasionally), while penalty-sum is
trivial (median first exact: call 1). For haiku, penalty-sum is catastrophically hard
(median loss ≈ 40) while signed-window leaves it within sight of the answer (median
loss 10). A one-dimensional "difficulty" axis does not exist; regime membership is a
joint property of model and task.

## Experiment 3: the particles × iterations frontier — SMC's mechanism finally gets room

Both experiment-1 arms were corners of a tiny budget: 4×1 has no feedback, and 2×2
gives the SMC machinery almost nothing to work with — with two particles, resampling
can only copy one particle over the other, so "iterative" was closer to greedy
hill-climbing than to the paper's population mechanism. Experiment 3 fixes the budget
at **16 calls** and sweeps the allocation for Haiku 4.5 on bounded-square (the cell
where feedback was known to engage): 16×1, 8×2, 4×4, 2×8, ten seeds each, 4096-token
responses.

![The particles × iterations frontier at fixed budget](figures/fig5-frontier.svg)

| Config | Exact | Mean loss | vs 16×1: Fisher / loss permutation |
|---|---|---|---|
| 16×1 (pure sampling) | 0/10 | 5.7 | — |
| 8×2 | 7/10 | 1.9 | p = 0.0031 / p = 0.012 |
| 4×4 | 9/10 | 0.4 | **p = 0.00012 / p < 0.0001** |
| 2×8 (deep refinement) | 9/10 | 0.3 | **p = 0.00012 / p < 0.0001** |

This is the lab's most decisive result, and it survives Bonferroni across every test
we have run. Three conclusions:

1. **Width cannot substitute for depth.** Haiku's per-call solve rate on this task is
   effectively zero — 16 independent draws produced 0 exact programs across 160
   calls. Refinement depth converts the same budget into a 90% solve rate. First
   exact programs appear at calls 5–15: haiku needs *many* repair cycles, which is
   why the B = 4 iterative arm (two cycles) only reached 3/10.
2. **The experiment-1 "narrow zone" was substantially a budget artifact.** At B = 4
   the feedback zone occupied one cell of the sweep; at B = 16 haiku sits near
   ceiling on the same task. Regime membership is a joint property of model, task,
   *and* refinement depth — the capability threshold is real (qwen's mode collapse
   would survive any depth), but where a proposer can use feedback at all, depth is
   the operative variable.
3. **The flat 4×4-vs-2×8 top suggests depth, not population size, is the binding
   factor here** — consistent with the observation that at these scales the paper's
   resampling mechanism still has little room (weight-proportional descendants matter
   more at larger populations). Distinguishing population effects from depth effects
   needs larger budgets still.

A grammar footnote with new force: 16 of haiku's 25 call-level failures in this sweep
are attempted `Or` nodes rejected by the AST decoder — under deep refinement, the
missing disjunction becomes the model's single largest source of wasted calls.

## Threats to validity

- **Task coverage.** Experiment 1's regimes are indexed to one task; experiment 2
  widens this to three tasks but all share the single-combinator `foldr` family. (In
  unarchived probe runs, the easier `map-increment` task was solved in one call by
  both haiku and qwen.)
- **Task-design circularity.** The experiment-2 tasks were designed *after* the
  regime hypothesis was formed, by the same investigators, informed by the
  experiment-1 forensics. They test the hypothesis's predictions but are not an
  independent sample of task space.
- **Multiple comparisons and selective emphasis.** Experiment 1 ran eight hypothesis
  tests; its single significant result (haiku loss, p = 0.011) does not survive
  Bonferroni across the eight (≈ 0.088). Experiment 2 added more tests; its
  headline (haiku penalty-sum, one-shot better on loss) is partially insulated by
  direct replication in the same direction on two independent run sets (p = 0.003
  at 2048 tokens, p = 0.011 at 4096), but a pre-registered replication at larger N
  is still the standard these findings should meet.
- **Small N.** 10 seeds per cell (5 for Qwen) gives wide Wilson intervals; the
  haiku success-rate difference (0/10 vs 3/10) is suggestive, not significant, and
  qwen's 0/5 arms formally admit success rates up to 43%.
- **Harness non-parity.** The Claude models propose via Anthropic forced tool-use;
  qwen proposes via Ollama's strict JSON-schema constrained decoding on a q8_0
  quantized checkpoint. Constrained decoding is a known cause of repetitive output,
  so qwen's mode collapse may be partly a harness artifact rather than purely a
  model property.
- **LLM nondeterminism.** Claude 5 models ignore temperature; run-to-run variation
  comes from the model itself and from SMC seed effects on feedback routing, and the
  two are not separable in this design.
- **Feedback design not ablated.** The iterative arm bundles resampling, loss-ranked
  failure feedback, sibling-avoidance lists, and champion retention; this experiment
  does not isolate which component carries the effect.
- **Budget scale.** 4 calls is a deliberately tight budget, and experiment 3 confirms
  the concern: at 16 calls haiku moves to near-ceiling on bounded-square given
  sufficient refinement depth. Regime placements are budget-indexed; only the
  capability *floor* (qwen) appears budget-robust, and even that is untested beyond
  B = 4.
- **Mid-lab harness change.** Between experiments 2 and 3 the proposal-envelope
  decoder was hardened (missing `rationale` now tolerated rather than failed), in a
  parallel session, after envelope violations were identified as a failure mode.
  Experiment 3 ran with the lenient decoder; experiments 1–2 with the strict one.
  This shifts failure *classification* across experiments (E3 shows zero
  envelope-shape failures) and modestly raises E3's effective call budget relative
  to a strict-decoder counterfactual; it does not affect within-experiment
  comparisons, which are the basis of every statistical claim.

## Reproduction

```bash
# Experiment 1 (bounded-square, all four proposers)
ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts   # ~5 min, ~160 cloud calls

# Experiment 2 (hard tasks, deconfounded reruns)
ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts \
  --task examples/foldr-signed-window.json --tag signed-window-4k \
  --proposers claude-sonnet-5,claude-haiku-4-5 --max-tokens 4096
ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts \
  --task examples/foldr-window-penalty-sum.json --tag penalty-sum-4k \
  --proposers claude-sonnet-5,claude-haiku-4-5 --max-tokens 4096

# Experiment 3 (particles x iterations frontier at budget 16)
ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts \
  --task examples/foldr-bounded-square.json --tag budget16-haiku \
  --proposers claude-haiku-4-5 --arms 16x1,8x2,4x4,2x8 --max-tokens 4096

# Analysis and figures
npx tsx experiments/analyze.ts [--summary results/summary-<tag>.json]
npx tsx experiments/validate-task.ts   # verifies new-task target programs
npx tsx experiments/figures.ts && npx tsx experiments/figures-e2.ts && npx tsx experiments/figures-e3.ts
```

Raw JSONL traces land in `runs/experiments/` (gitignored); parsed results in
`experiments/results/summary*.json`.
