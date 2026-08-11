# ModelSMC for programming by example

This standalone Python package studies whether ModelSMC's population,
feedback, resampling, and revision loop helps synthesize a typed program from
input-output examples. It contains the bounded program language, semantic
scorer, search engines, LLM adapters, GPU-aware numerics, and structured
logging in one installable project.

The package has three deliberately different modes:

| Mode | Proposal mechanism | What the weights mean |
| --- | --- | --- |
| `paper-search` | finite catalog or black-box LLM | Heuristic allocation scores from an uncorrected proposal kernel; **not** posterior probabilities |
| `grammar-smc` | known finite skeleton prior | An SMC approximation to the declared finite-skeleton Gibbs target, checked against exact enumeration |
| `importance-smc` | finite typed holes, an LLM joint-semantic slate, or an exhaustive joint-target oracle | Importance-corrected SMC for the fixed, bounded, deduction-refuted support; exact enumeration is available only to materialized controls |

The latter two modes are calibrated to their declared computational targets.
They are not thereby Bayesian posteriors over every possible program or over a
real-world system: their supports are bounded, and the soft PBE loss is not
claimed to be a scientific observation model.

See [DESIGN.md](DESIGN.md) for the algorithms and exact assurance boundary.

## Relationship to ModelSMC

The design follows [A Probabilistic Framework for LLM-Based Model
Discovery](https://arxiv.org/abs/2602.18266) and uses the authors'
[ModelSMC repository](https://github.com/mackelab/ModelSMC) as an implementation
reference. It does not import or vendor that project. Upstream targets
scientific simulator discovery and uses a substantially larger stack including
Hydra, DSPy, Ray, SBI, neural likelihood/posterior estimation, and TabPFN. This
project keeps the population lifecycle but replaces that likelihood pipeline
with deterministic PBE scoring.

See [NOTICE.md](NOTICE.md) for provenance and licensing notes.

## Publication artifacts

The executable implementation and tests live in this GitHub repository. The
companion [Hugging Face dataset](https://huggingface.co/datasets/hackerprofile1/modelsmc-pbe-research)
contains only the sanitized experimental evidence, aggregate results,
publication figures, protocols, checksums, and manuscript PDF; it deliberately
does not mirror the source tree or dependency environment.

## Install

From the repository root:

```bash
cd papers/01-modelsmc/modelsmc-pbe-python
uv sync --dev
```

Python 3.12 is pinned by `.python-version`.

## Run

All commands below start in `papers/01-modelsmc/modelsmc-pbe-python`.

### Deterministic catalog search

This exercises the practical resample/revise lifecycle without a model server:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode paper-search \
  --proposal catalog \
  --skeleton map-arithmetic \
  --particles 8 --iterations 6 \
  --device cpu --trace
```

### Persistent finite-score cache

For paid, cross-run model comparisons, persist the raw finite scores in an
immutable content-addressed cache. A cold run uses `read-write`; an offline
replay uses `replay-only` and fails rather than contacting vLLM when evidence is
missing or corrupt:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode importance-smc --proposal vllm \
  --base-url http://127.0.0.1:18000/v1 --model qwen-coder \
  --model-repository Qwen/Qwen2.5-Coder-3B-Instruct \
  --model-revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 \
  --tokenizer-revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 \
  --vllm-server-config 'vllm=0.11.0;logprobs=processed_logprobs;dtype=bfloat16;max_model_len=4096' \
  --score-cache-dir runs/candidate-score-cache \
  --score-cache-mode read-write \
  --skeleton foldr-filter-map --particles 1 --iterations 1 \
  --alpha 0 --temperature 0.7 --device cpu --trace
```

Cache keys bind the served alias, exact repository/model/tokenizer revisions,
declared vLLM scoring configuration, energy mode, tokenization policy, prompt
prefix, candidate bytes, and expression-validation context. Entries store the
raw token IDs and token log probabilities needed to reconstruct the original
score batch. Writes are atomic and never replace an existing entry; malformed
or mismatched entries fail closed. The run manifest records cache hits/misses,
cache- and provider-served candidates/token positions, provider HTTP telemetry,
and provider wait time. `--max-scored-candidates` still counts every finite
choice requested by the algorithm, including cache hits.

### Uncorrected Qwen through Ollama (baseline only)

Start Ollama in another terminal, ensure the model is installed, then run:

```bash
ollama serve
```

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode paper-search \
  --proposal ollama \
  --model qwen3-coder:30b-a3b-q8_0 \
  --particles 2 --iterations 8 --alpha 0 \
  --ess-threshold 1 --temperature 0.7 \
  --max-tokens 4096 --timeout-seconds 300 \
  --device auto --trace
```

Ollama controls the LLM's device placement. `--device` controls Torch work in
this process; it does not move an Ollama model. Ollama is intentionally not
accepted by `importance-smc`; this command is only the uncorrected search
baseline.

### Uncorrected free-form generation through vLLM

Start vLLM with a model and stable served name, for example:

```bash
vllm serve YOUR_HUGGING_FACE_MODEL --served-model-name qwen-coder --port 8000
```

Then point the OpenAI-compatible proposal client at it:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode paper-search \
  --proposal vllm \
  --base-url http://localhost:8000/v1 \
  --model qwen-coder \
  --particles 8 --iterations 8 \
  --max-concurrency 8 --device auto --trace
```

This is still `paper-search`: it asks vLLM for a free-form complete AST and does
not know the probability of that proposal.

### Importance-corrected deduction/Qwen proposal

`importance-smc` combines Paper 2's typed generalization and deduction with an
explicit finite proposal law:

```text
examples
  -> infer the signature and structural relationships
  -> infer typed structural skeletons
  -> refute incompatible skeletons
  -> derive typed examples for each hole
  -> enumerate finite canonical choices for small holes
  -> ask Qwen to score only those candidates
  -> combine Qwen scores with exact deduction-subtree marginals
  -> sample locally from the fully known categorical q
  -> assemble, type-check, and execute complete programs
  -> apply importance correction
  -> let SMC resample promising program particles
```

The sound skeleton refutations restrict the fixed support. Derived hole
examples also define a soft, normalized deduction guide, but do not hard-delete
imperfect candidates. The target is unchanged; only the proposal improves.
This preserves generalization and a meaningful soft-loss target while stopping
Qwen's cumulative JSON token scores from overwhelming sound symbolic evidence.

Run this primary experiment against vLLM on the CUDA machine; no Ollama bridge
is involved. Start vLLM with Qwen and request `processed_logprobs`; vLLM's
[engine documentation](https://docs.vllm.ai/en/stable/configuration/engine_args/)
defines that mode as values after configured logit processors. The client
generates no output tokens and sends no top-p rule; it treats those
teacher-forced values as finite-candidate energies, then applies the positive
local categorical `--temperature` itself:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen-coder \
  --port 8000 \
  --logprobs-mode processed_logprobs
```

For the paid-GPU transport check, deliberately condition on the
`foldr-filter-map` family first. This is a controlled ablation: it tests Qwen's
two finite scoring waves and the importance denominator without paying for
family selection or irrelevant generic catalogs. Start with one particle and
one stage:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode importance-smc \
  --proposal vllm \
  --skeleton foldr-filter-map \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen-coder \
  --model-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --tokenizer-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --particles 1 --iterations 1 \
  --alpha 0 --temperature 0.7 --proposal-epsilon 0.05 \
  --deduction-mix 0.5 --deduction-strength 2 \
  --candidate-batch-size 128 --max-concurrency 8 \
  --max-scored-candidates 1000 \
  --support-limit 40000 --timeout-seconds 600 \
  --device cpu --trace
```

Port `18000` is the local end of the RunPod SSH tunnel used in this experiment.
When the CLI runs on the same machine as vLLM, use
`--base-url http://127.0.0.1:8000/v1` instead.

The conditioned skeleton is

```text
foldr(
  (item, acc) =>
    if ?predicate(item)
    then ?mapped_value(item) :: acc
    else acc,
  [],
  xs
)
```

With the example's eight constants, deduction exposes 600 canonical predicate
choices and 60 mapped-value choices. Their Cartesian product contains 36,000
complete programs and exactly two syntactic exact solutions. Qwen is not asked
to generate an AST: it ranks the 600 choices and then the 60 choices
conditioned on the selected predicate. At candidate batch size 128, that costs
at most six vLLM HTTP batches for one distinct ancestor path. Identical requests
created by resampling are deduplicated before provider I/O.

Once that smoke succeeds, exercise generalization with `--skeleton auto`:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode importance-smc \
  --proposal vllm \
  --skeleton auto \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen-coder \
  --particles 2 --iterations 1 \
  --alpha 0 --temperature 0.7 --proposal-epsilon 0.05 \
  --deduction-mix 0.5 --deduction-strength 2 \
  --candidate-batch-size 128 --max-concurrency 8 \
  --max-scored-candidates 2000 \
  --hole-max-cost 3 --support-limit 40000 \
  --timeout-seconds 600 --device cpu --trace
```

Here `auto` does **not** commit to a family from list lengths. It keeps all
type-correct hypotheses, lets deduction refute only impossible ones, and uses
the deduction/Qwen mixture to score families and holes. When both specialized
filter/map catalogs have the same abstract type, auto deterministically keeps
the smallest catalog capable of satisfying every sound mapped-value example:
simple arithmetic for this task, or the signed piecewise catalog when simple
arithmetic cannot fit. For this
task the bounded support contains 36,198 programs: 18 expression programs,
36,000 specialized filter/map folds, and 180 generic folds; `map` is refuted.
The support contains the same two exact programs. `--skeleton general` is the
legacy generic-only comparison, while an explicit named skeleton is a
single-family ablation.

The harder signed-window task requires an actual conditional mapper:

```text
filter to -2 <= item <= 2
negative item    -> -item
nonnegative item -> item * item
```

Run the provider-free accounting control first:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-signed-window.json \
  --mode importance-smc --proposal catalog --skeleton auto \
  --particles 4 --iterations 1 --alpha 0 \
  --deduction-mix 0.75 --deduction-strength 3 \
  --max-scored-candidates 3292 \
  --hole-max-cost 3 --support-limit 150000 \
  --device cpu --trace
```

With eight declared constants, the selected specialized family has 600 window
predicates and 220 canonical signed piecewise mappers, hence 132,000 complete
programs and four syntactic exact solutions. Expression and generic-fold
families bring auto support to 132,198 states; `map` is refuted and the simple
filter/map catalog is excluded because none of its 60 arithmetic mappings fits
all derived `item -> output` examples. A single distinct Qwen path scores at
most 3 family choices, 600 predicates, and 220 mappers: 823 candidate
continuations. Replace `catalog` with `vllm` and provide the same vLLM options
used above only after this local control succeeds.

`--max-scored-candidates` is a general run-wide scoring ceiling, not a
batch-size hint. With vLLM, each teacher-forced canonical candidate prompt
consumes one unit of paid model work; with `catalog`, the same ceiling limits local
control work. The shell rejects a whole scoring wave before scorer I/O if it
would cross the ceiling, and persists both usage and limit. Raising particles
or iterations therefore requires an explicit budget decision.

For a provider-free probability-accounting control, replace `--proposal vllm`
with `--proposal catalog`. The model component is then uniform, while the same
explicit deduction guide remains active; pass `--deduction-mix 0` to recover
the old fully uniform construction proposal.

`--deduction-mix` remains the backward-compatible default for every proposal
wave. `--family-deduction-mix` and `--hole-deduction-mix` can override it at
family selection and hole filling, respectively. This matters when deduction
soundly identifies a useful family but derives no discriminating examples for
the holes: the family guide can remain active without treating a hole-level
Occam distribution as if it were evidence. Score ledgers store the resolved
mix used by each wave, and results and manifests retain both resolved values.

The Qwen path does **not** sample free-form JSON. For prompt `P` and canonical
candidate `u`, it teacher-forces the complete concatenated prompt and obtains
an energy from that prompt's tokenization

$$
s_\theta(P,u)=\sum_{r=2}^{|\operatorname{tok}(P\Vert u)|}
\log p_\theta(t_r\mid t_{<r}).
$$

`--llm-energy-normalization total-full-prompt-logprob` (the default) uses
$s_\theta$ unchanged. The publication ablation
`mean-full-prompt-conditional-logprob` instead uses $s_\theta/n(P,u)$, where
$n(P,u)$ is the number of scored positions. Both are explicit finite energies
over the complete teacher-forced `P || u` token path; the mean mode is **not**
described as a candidate-only likelihood because tokenization can merge across
the text boundary. Write the configured energy as
$\widetilde{s}_\theta(P,u)\in\{s_\theta(P,u),s_\theta(P,u)/n(P,u)\}$.
Every run persists a compact score ledger containing the
prefix and its SHA-256 digest, canonical candidates, token IDs/log-probabilities,
the total and configured energy, pre-mixture Qwen probabilities, final mixture
probabilities, and selections. `--model-revision` and `--tokenizer-revision`
add archival identifiers to the manifest and vLLM ledger without changing
server behavior.

For a retained program $e$, let $D(e)$ be the number of deduplicated derived
hole examples violated by its fillings. The stage-dependent guide is

$$
r_t(e)\propto p_0(e)\exp\!\left[-\kappa_{\max}
\frac{\beta_t}{\beta_{\max}}D(e)\right],
$$

where $p_0$ is the equal-family, within-family Occam prior. Exact subtree sums
of $r_t$ give $r_{t,j}(u)$ for every currently reachable family or hole choice.
The shell samples

$$
q_j(u)=(1-\varepsilon)\left[(1-\lambda)
\operatorname{softmax}\!\left(\frac{\widetilde{s}_\theta(P,u)}{\tau}\right)
+\lambda r_{t,j}(u)\right]
+\frac{\varepsilon}{|\mathcal C_j|},
\qquad \tau>0,\ \varepsilon>0.
$$

A complete proposal therefore has

$$
q_{\mathrm{construct}}(e\mid a,D,F)
=q_H(h\mid a,D,F)
 \prod_j q_j(u_j\mid h,u_{<j},a,D,F).
$$

The family probability and every conditional hole probability are included in
`log q`. With only one surviving family, the shell sets $q_H=1$ locally and
skips the paid no-op provider request. The uniform component keeps every target
program reachable and bounds otherwise extreme importance corrections. The
clone transition is also accounted exactly:

$$
Q_\alpha(e'\mid e)
=\alpha\mathbf 1[e'=e]+(1-\alpha)q_{\mathrm{construct}}(e'\mid e,D,F).
$$

If both cloning and the guided construction can return the ancestor, both masses enter the
denominator. Provider errors abort this strict mode; there is no hidden
ancestor fallback. `alpha=1` is rejected because it destroys full support.

Canonical ASTs and a deterministic family-ownership rule remove the otherwise
intractable sum over whitespace, key order, prose, alternative serializations,
and cross-family traces. The specialized filter/map family owns an exact
canonical overlap with generic `foldr`; the discarded alias is counted, while
non-overlapping generic folds remain in support. Any other duplicate trace is
an invariant failure. Because vLLM supplies scores rather than performing the
draw, native top-p, output temperature, grammar masking, and EOS accounting do not
enter this `q`; `--temperature` is the positive local categorical temperature.
Scoring the complete prompt also avoids assuming that tokenizing `P` separately
produces a token prefix of `P || u`, an assumption that fails for Qwen at some
text boundaries.

For surviving family supports $\mathcal E_h$, the base program prior first
assigns equal mass to every family, then applies the Occam preference within
that family:

$$
p_0(e)
=\frac{1}{H}
  \frac{\exp[-\lambda_C\operatorname{cost}(e)]}
       {\sum_{u\in\mathcal E_{h(e)}}
        \exp[-\lambda_C\operatorname{cost}(u)]}.
$$

This prevents a large catalog from receiving more prior mass merely because it
contains more programs. The finite target is

$$
\widetilde\pi_\beta(e)
=p_0(e)\exp[-\beta\lambda_L\operatorname{loss}(e)].
$$

At stage `t`, the incremental potential is

$$
G_t(e_{t-1},e_t)
=\frac{\widetilde\pi_{\beta_t}(e_t)}
       {Q_\alpha(e_t\mid e_{t-1},D,F)}.
$$

The declared Feynman--Kac path target is the product of these stage targets,
so its terminal marginal is the desired finite `pi_beta`. Accordingly, the
reported accumulated log normalizer is for the **product path target**
`product_t Z_beta_t`, not just the final posterior's `Z_1`.

### LLM joint-semantic proposal

`--proposal joint-semantic` is the scalable counterpart to the exhaustive
oracle below. For a complete candidate program $e=h(p,m)$, Qwen sees the PBE
examples and the full program but never sees its execution output or loss. It
scores four teacher-forced answer paths with swapped meanings for labels `A`
and `B`:

$$
a_{\mathrm{LLM}}(e;D)
=\frac12\left[
\ell_+(A)-\ell_+(B)+\ell_-(B)-\ell_-(A)
\right].
$$

The two paths for each mapping must have equal lengths, identical prefix token
IDs, and one distinct final label token. Their raw prefix-logprob vectors are
committed separately rather than required to be bit-identical; the semantic
ledger records their maximum absolute difference as a provider-numerics
diagnostic. Only the two final-label logprobs enter each contrast. Any actual
tokenizer-boundary mismatch aborts the calibrated run. The swapped mapping
cancels a fixed preference for label `A` or `B`; this is a compatibility
log-score contrast, not an AST fluency total and not an execution-loss estimate.
Derived artifacts identify this layout as `joint-semantic-proposal-ledger-v2`.

For a deterministic scored slate $A$, the proposal is

$$
r_A(e)\propto
\mathbf1[e\in A]p_0(e)\exp[\eta a_{\mathrm{LLM}}(e;D)],
\qquad
q(e)=\epsilon p_0(e)+(1-\epsilon)r_A(e).
$$

The exact prior floor gives every bounded trace positive mass, including traces
outside a partial slate. Prefix masses of this same joint table define the
family, predicate, and mapper conditionals, so their product telescopes to the
persisted $q(e)$. Only the initial and subsequently sampled complete programs
are executed. After execution, the ordinary correction is

$$
\log w(e)=
\log p_0(e)-\beta\lambda_L\operatorname{loss}(e)-\log q(e).
$$

This is exact importance accounting for the realized semantic proposal; it
does not claim that the LLM surrogate equals the normalized target. Omit
`--semantic-slate-size` to score the full bounded support. A practical RunPod
smoke starts with 512 traces, which costs at most 2,048 raw label paths:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-sparse-bounded-square-v2.json \
  --mode importance-smc --proposal joint-semantic \
  --skeleton auto --particles 16 --iterations 1 --alpha 0 \
  --semantic-scale 1 --semantic-slate-size 512 \
  --proposal-epsilon 0.05 --candidate-batch-size 128 \
  --max-scored-candidates 2048 \
  --score-cache-mode read-write \
  --score-cache-dir artifacts/joint-semantic-score-cache \
  --base-url http://127.0.0.1:18000/v1 --model qwen-coder \
  --device cpu --trace
```

Four raw paths are charged once per unique canonical program. On the corrected
36,198-trace stress support, a full no-alias slate therefore needs a budget of
144,792 raw paths; use the persistent score cache for that run. Slate ASTs are
assembled only as prompt material and are not run through the semantic core.
Results keep the guided score ledger empty and instead persist a compact
semantic ledger with the label-boundary evidence, slate law, and every selected
conditional probability. Its replay recomputes the contrastive scores,
normalization, defensive mixture, and selected densities. The recorded
$p_0$ factors are checked against the live factorized support when the law is
built; auditing those prior factors from artifacts alone requires rebuilding
that support from the bound task configuration. Raw token paths are committed
by hash rather than copied into `result.json`; independently rechecking their
single-token boundary requires retaining the referenced content-addressed
score-cache entries, as in the command above.

### Exact joint-execution oracle

`--proposal joint-target` implements the joint execution score directly as a
materialized oracle control. For every complete program $e$ in the bounded
deduction-refuted support $\mathcal E_D$ it uses

$$
S_{\mathrm{joint},\beta}(e)
=\log p_0(e)-\beta\lambda_L\operatorname{loss}(e),
\qquad
q_{\mathrm{joint},\beta}(e)
=\frac{\exp S_{\mathrm{joint},\beta}(e)}
       {\sum_{u\in\mathcal E_D}\exp S_{\mathrm{joint},\beta}(u)}.
$$

The draw remains sequential. If $A(r)$ is the log-sum-exp target mass of all
complete programs below construction prefix $r$, then

$$
\log q(c\mid r)=A(r+c)-A(r).
$$

For the stress family’s predicate $p$ and mapper $m$, this is exactly

$$
q(p)=\sum_m q_{\mathrm{joint}}(p,m),
\qquad
q(m\mid p)=\frac{q_{\mathrm{joint}}(p,m)}{q(p)}.
$$

The implementation generalizes the same marginal/conditional construction to
the family choice and any ordered sequence of typed holes.
Family, predicate, and mapper conditionals therefore telescope to the direct
joint probability of the final program. With cloning disabled,

$$
\log\widetilde\pi_\beta(e)-\log q_{\mathrm{joint},\beta}(e)=\log Z_\beta
$$

for every sampled state. Thus all incremental importance weights are constant;
the mode is an exact end-to-end check of support construction, target math,
sequential proposal accounting, and SMC normalization.

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-sparse-bounded-square-v2.json \
  --mode importance-smc --proposal joint-target \
  --materialize-reference --skeleton auto \
  --particles 4 --iterations 1 --alpha 0 --beta-max 1 \
  --hole-max-cost 3 --support-limit 40000 \
  --device cpu --trace
```

The proposal requires no Qwen scorer or GPU and uses no candidate-score cache
or LLM score budget.
Results identify `proposal_strategy=joint-target`; deduction/Qwen-only controls
and guide masses are `null`, and the LLM score ledger is empty.
Its real cost is exhaustive: it assembles and executes every unique program in
the finite support before drawing any particle. For a one-stage diagnostic,
if the normalized target gives zero-training-loss programs total mass
$p_\star$, then $N$ independent oracle draws select at least one such program
with probability
$1-(1-p_\star)^N$. On the corrected 36,198-state stress task,
$p_\star=0.9207000842$, so four draws give `0.9999604551` oracle sampling hit
probability, conditional on this fixed support and target. The practical budget
is therefore 36,198 up-front program materializations/scores—180,990
program-example evaluations for the five training examples—plus a four-particle
one-stage diagnostic; larger populations are needed only for additional Monte
Carlo diagnostics, not this discovery criterion. This is a target-specific
training-example oracle and scalability ceiling, not a deployable learned
semantic scorer or a held-out guarantee.

### Calibrated finite-grammar control

Map increment has a small arithmetic-map support:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode grammar-smc \
  --skeleton map-arithmetic \
  --particles 2048 --iterations 12 \
  --beta-max 1 --moves-per-stage 1 \
  --device auto --trace
```

The bounded-square task needs the conditioned fold skeleton:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode grammar-smc \
  --skeleton foldr-filter-map \
  --particles 4096 --iterations 16 \
  --beta-max 1 --moves-per-stage 1 \
  --grammar-limit 250000 --device auto --trace
```

`--grammar-limit` is a safety ceiling. Enumeration either returns the complete
skeleton support or fails; it never takes an order-dependent prefix and calls
that a probability space.

A skeleton is a named, bounded hypothesis family, not a Python language
restriction or a general-purpose synthesis grammar. `expression-arithmetic`
completes one scalar arithmetic body, `map-arithmetic` completes an arithmetic
mapper, and `foldr-filter-map` completes the particular fold/filter/map shape
needed by the bounded-square control. `foldr-filter-piecewise-map` retains the
same outer fold but admits a canonical zero-sign conditional whose distinct
branches are identity, negation, squaring, or a declared constant. This
conditioning makes exact enumeration and probability accounting possible.

`importance-smc` has analogous `--hole-state-limit` and `--support-limit`
ceilings. Every attempted construction trace, including a counted alias, uses
that ceiling. `--hole-max-cost` controls generic support. Named conditioned
skeletons instead use their complete factorized catalogs; for
`foldr-filter-map`, `--support-limit` must therefore be at least 36,000 with
eight constants. Multi-family `auto` needs at least 36,198 at the default hole
cost. The signed piecewise family requires 132,000, and its auto support needs
132,198. Increasing any declared bound changes the finite target.

## What happens at startup

`paper-search` begins from one fixed model, `m0`, derived from the signature:

| Signature relation | Initial AST |
| --- | --- |
| input type equals output type | identity (`Input`) |
| output is `Int` | first declared integer constant |
| output is `Bool` | `false` |
| output is `List<Int>` or `List<Bool>` | corresponding typed empty list |

The semantic core scores `m0` once, and the search copies it into all `N`
particle slots. For revisions, each prompt receives the current AST,
counterexamples, and at most four recent ancestry snapshots. Full lineage IDs
remain in artifacts while bounded prompt ancestry prevents unbounded context
growth.

Provider failures do not become successful proposals. A failed call or a
scorer-rejected AST retains its selected ancestor, records a typed warning, and
increments separate response, provider-error, scorer-rejection, and accepted-
proposal counters. `result.json` sets `degraded: true` when every provider call
failed before returning an AST.

That fallback policy applies only to heuristic `paper-search`.
`importance-smc` never generates or parses a candidate response: it scores an
already validated finite catalog, samples locally, and aborts if any score
request fails. This is necessary because an unmeasured fallback would add
unknown probability mass to the ancestor.

## Configuration and semantics

Pydantic parses either the compact flat specification or the nested experiment
form into one immutable `ExperimentConfig`. CLI overrides are applied to that
same normalized object. `ProgramScorer` receives it directly, so examples,
signature, constants, loss scales, and AST bounds have one in-process source of
truth.

The semantic core strictly normalizes ASTs, enforces allowed constants and
structural bounds, performs scope-aware type inference, evaluates accepted
programs, computes structural cost and capped per-example loss, and decides
exact acceptance. Rejected candidates are typed values, so one bad model
proposal does not abort a population batch.

The bounded DSL supports four value types—`Int`, `Bool`, `List<Int>`, and
`List<Bool>`—and three complete program forms:

| Program form | Available expressions |
| --- | --- |
| scalar expression | input, integer/Boolean literals, arithmetic, comparisons, Boolean operations, conditionals, typed empty lists, and typed prepend |
| `map` | the same expressions with `Item` in scope; element type may change between `Int` and `Bool` |
| right fold | the same expressions with `Item` and `Accumulator` in scope; the initial value fixes the result type |

Map and fold bodies cannot capture the outer input list. This keeps their
semantics first-order and makes scope checking explicit. The named grammar-SMC
skeletons are finite subsets of this full transport language; model-backed
`paper-search` proposals may return any AST in the bounded DSL.

Integer values in specifications may be JSON integers or canonical decimal
strings such as `"-2"`; a leading `+` is rejected. The integer-constant catalog
must be nonempty. A signature can be inferred from nonempty scalar or list
examples, while all-empty list examples require an explicit signature. SMC
numeric settings are strict and finite rather than silently coercing strings or
accepting `NaN`/infinity.

## Artifacts

Every run that passes startup validation creates
`runs/<UTC timestamp>-<name>-<run id>/`:

| File | Contents |
| --- | --- |
| `manifest.json` | normalized configuration, git state, package versions, seed, requested/resolved device, and probabilistic claim |
| `events.jsonl` | append-only run, ESS, resampling, proposal, scoring, fallback, timing, and failure events |
| `result.json` | champion, generated/viable/refuted families, support and mass summaries, and degradation diagnostics |
| `final_particles.jsonl` | final ASTs, weights, scores, sources, and ancestry |

Raw prompts and free-form model rationale are omitted by default; their hashes
and operational diagnostics remain. Accepted program ASTs are retained because
they are the search states being studied. `--trace` renders the structured
event stream without changing stored records.

## Package structure

The code is organized by responsibility:

```text
src/modelsmc_pbe/
  shell/          CLI options, command assembly, provider and skeleton selection
  search/paper/   initial model, particles, objective, propagation, prompts, engine
  search/grammar_control/
                  support, target math, transitions, engine, result records
  search/importance/
                  fixed support, prompts, exact q, FK engine, reference metrics
  induction/      typed hypotheses and structural relationships
  deduction/      sound refutations and typed hole-example inference
  enumeration/    complete increasing-cost typed expression catalogs
  core/           type checking, evaluation, cost, loss, rendering, batch scorer
  proposals/      catalog and OpenAI-compatible provider adapters
  grammar/        named, complete finite skeleton enumerators
  smc/            pure population numerics
  domain/         validated PBE and AST transport types
  runtime/        device resolution and seeded generators
  observability/  structured run artifacts
```

The small `__init__.py` files mark importable packages and expose intentional
public names; they contain no algorithm. Recognizable boundaries come from
narrow modules, typed protocols, immutable records, dependency injection, and
one-way imports rather than from a large application framework.

## Assurance and reproducibility

The pure-Python semantic core is strictly validated and covered by unit,
property, integration, and golden-target tests. It is **not formally verified**.
Do not describe this package as a verified synthesizer or claim the previous
formal guarantees transfer automatically to this reimplementation.

The probabilistic shell, floating-point potentials, provider clients, and
logging are also test-backed rather than formally verified. `grammar-smc` is
calibrated because its finite target and proposal law are explicit, not because
the implementation has a machine-checked proof. `importance-smc` has the same
finite-target limitation; its extra claim is that its locally sampled proposal
law is explicit and appears in the importance denominator.

`auto` resolves Torch devices in the order CUDA, Apple MPS, then CPU. Explicitly
requesting an unavailable accelerator fails instead of silently using CPU.
Normalization, ESS, resampling, and population decisions use seeded CPU
`float64`; accelerator kernels can still differ across platforms.

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
