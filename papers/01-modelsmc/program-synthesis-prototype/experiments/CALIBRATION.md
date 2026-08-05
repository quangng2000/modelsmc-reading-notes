# Toward a calibrated posterior: exact prior, true likelihood, and the road to corrected LLM proposals

*Design note and first results for the calibration phase. Builds on the three-experiment
study in [RESULTS.md](RESULTS.md), which used an uncalibrated potential.*

## Motivation

The study's potential `log G(m) = -lambda * loss(m) - beta * cost(m)` is a Feynman-Kac
*surrogate*: structurally prior x likelihood, but neither factor is a normalized
probability, and the SMC population is a search distribution rather than a posterior
(RESULTS.md section 3, "Scope of the theory"). This phase removes the first two of the
three obstacles to a true posterior:

1. an unnormalized prior  -> **normalized exactly** (this note),
2. a surrogate likelihood -> **replaced with a proper observation model** (this note),
3. no importance correction for the LLM proposal density -> **path identified;
   requires self-served logits** (research findings below).

## 1. The prior, exactly normalized

`p(m) = exp(-beta * cost(m)) / Z` over the finite verifier-accepted space (finiteness is
Proposition 2 of RESULTS.md). `src/shell/prior/count.ts` computes N(c) — the number of
accepted programs at each cost — by dynamic programming over the typed grammar
(convolutions over subterm costs, per scope: outer/mapper/fold-initial/fold-reducer),
giving `log Z = logsumexp_c(ln N(c) - beta*c)` exactly (BigInt counts, ~1e-15 relative
log accuracy).

Guarantees, tested in `tests/prior-count.test.ts`:
- **Completeness/soundness vs the verified core**: raw *untyped* AST enumeration filtered
  through the core's `inferType` reproduces the DP counts exactly (expression, map, and
  fold families; the scope rules — no `Input` in mapper/initial/reducer, closed initial,
  scalar mapper output — are inherited from the core, not re-assumed).
- **Exact sampling**: `src/shell/prior/sample.ts` backward-samples the DP tables, drawing
  from p(m) *with known density* (cost marginal, then family, then splits by exact
  BigInt weights). Statistical tests confirm the advertised density to 5 sigma.

## 2. The likelihood, properly normalized

`src/shell/scoring/emission.ts` replaces the edit-distance surrogate with a proper
conditional distribution p(observed | predicted):

- **Int**: two-sided geometric on the integers centred at the prediction.
- **Bool**: epsilon-flip channel.
- **Lists**: a pair-HMM insertion/deletion/substitution channel — at every source
  position the actions {delete, insert, emit-match} have probabilities summing to one,
  and past the end {insert, stop} likewise, so the channel is a distribution over finite
  lists *by construction*; the likelihood sums over all alignments via the forward
  algorithm. `tests/emission.test.ts` verifies the channel against an independent
  length-marginal chain exactly, plus closed forms.

Proposition 1 (machine-checked type soundness) is what makes this likelihood total on
the accepted set: evaluation cannot crash and always yields a value of the type the
channel expects.

## 3. Ground truth: the exact posterior, and a first calibrated sampler

Because the support is finite, `experiments/exact-posterior.ts` enumerates *every*
accepted program at a small cost cap and computes the exact normalized posterior
(cross-checking enumeration counts against the DP on every run).

First result (map-increment, cost cap 9, beta = 0.15, default noise):

- support = **974,185 programs**; posterior entropy 8.55 nats;
- **posterior mass on exactly-solving programs = 0.9975** — the calibrated posterior
  concentrates almost entirely on semantically correct programs;
- the Occam prior separates syntactic variants exactly as designed: the three cost-5
  increment forms each get p = 3.55e-4, the cost-7 variants 2.63e-4, all with identical
  likelihood.

`experiments/prior-is.ts` then runs the first *fully calibrated* sampler in the
prototype: proposal = the exact prior sampler (known density), so self-normalized
importance weights reduce to the likelihood. Against the exact posterior:

| samples | ESS | exact-mass estimate | truth | per-program TV |
|---|---|---|---|---|
| 1,000 | 6.0 | 0.99784 | 0.99745 | 0.999 |
| 10,000 | 54.3 | 0.99743 | 0.99745 | 0.989 |
| 100,000 | 492.7 | 0.99729 | 0.99745 | 0.907 |

Two honest readings: **semantic functionals converge quickly** (exact-solve mass is
right to 4 decimal places by 10k samples), while **per-program TV stays high** — with
~1M atoms and max atom probability 3.6e-4, resolving individual programs needs far more
than 100k draws; this is a property of the syntax-diffuse posterior, not an estimator
defect. The ESS ratio (~0.5%) quantifies how poor the prior is as a proposal — which is
precisely the formal case for an LLM proposal with an importance correction.

## 4. Exact proposal probabilities from local serving (research findings)

Empirical probe of the installed stack (Ollama 0.31.1, qwen3-coder:30b-a3b-q8_0):

- Ollama **does** return per-token logprobs (undocumented), including alongside
  structured output — but they are provably **pre-mask** raw-softmax values: under a
  forced JSON schema, the grammar-forced token `{"` came back with logprob **-45.77**
  while the top *unconstrained* candidate (schema-illegal) sat at ~0. The masked
  normalizer is unrecoverable (`top_logprobs` capped at 20), so exact q is impossible
  through Ollama as shipped.
- **llama.cpp**: Ollama's GGUF blob is directly loadable by `llama-server`. Stock
  `post_sampling_probs` applies the grammar only on resample steps (accept-path probs
  miss the /p(legal) factor); a one-line `grammar_first` change in the server's sampler
  call makes the reported probabilities exactly the masked-renormalized sampling
  distribution.
- **Recommended path**: a small llama-cpp-python sidecar loading the same GGUF with an
  explicit grammar-mask + log-softmax + sample loop — exact q by construction (~1-2
  days), exposing an OpenAI-shaped endpoint with a per-token `q_logprobs` field; the
  TypeScript proposer then needs ~20-50 changed lines.

One structural prerequisite recorded from the trace analysis: the current prompt's
sibling-avoidance list couples same-iteration particles, which breaks the per-particle
Markov-kernel form the importance correction assumes; the corrected proposer must build
prompts from lineage-only context.

## Next steps

- **E5**: LLM-as-proposal with exact q on a small-cap task; corrected SMC/IS population
  vs the exact posterior (TV and functional error vs ESS), against the prior-proposal
  baseline above. Scope: local open-weight model only (Claude exposes no logits) — this
  demonstrates calibration; the frontier-model results in RESULTS.md remain search, not
  inference.
- Optional: semantic-class posteriors (grouping programs by behaviour on the examples)
  as the natural coarse-graining under which sampler convergence is fast.

## Reproduction

```bash
npm test                                   # includes prior-count + emission suites (55 tests)
npx tsx experiments/exact-posterior.ts examples/map-increment.json --cost-cap 9 --top 8
npx tsx experiments/prior-is.ts examples/map-increment.json --cost-cap 9 --samples 1000,10000,100000
```
