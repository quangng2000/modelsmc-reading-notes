# Paragraph-level author review guide

This file is deliberately separate from the manuscript. Paragraph IDs appear
as LaTeX comments in `main.tex`; they are not rendered in the paper.

Review each paragraph in this order:

1. Does the opening sentence state the topic or claim immediately?
2. Does the paragraph have exactly one primary job?
3. Does the middle develop that job with the listed evidence or dependency?
4. Does the final sentence resolve the paragraph and hand off to what follows?
5. Would a skeptical reviewer interpret it more broadly than intended?

| ID | Purpose | Evidence or dependency | Likely reviewer challenge |
|---|---|---|---|
| P-ABS-01 | State the problem, method, paired result, and boundary in under 200 words. | Entire paper; complete one-seed Gate-2 grid is exploratory. | Report the paired Q failure/Q+D success and attribute the tested difference to deduction; do not imply independent Qwen value, checkpoint scaling, speed, resampling benefit, or generality. |
| P-INT-01 | Establish PBE and the symbolic/learned search context. | Gulwani; Feser; Kalyan; HYSYNTH; Li. | Why another hybrid synthesizer? |
| P-INT-02 | Define the attribution problem that motivates the work. | Support-builder audit and matched pilots. | Is this only an evaluation hygiene issue? |
| P-INT-03 | Explain why SMC is relevant and expose the proposal-density gap. | Ellis et al.; ModelSMC; SMC background. | Prior SMC synthesis already exists. |
| P-INT-04 | Give the central design move: finite canonical energy scoring at staged-choice or deterministic complete-slate granularity. | Sections 3--5, staged score ledger, and semantic score ledger. | These are declared energy proposals with evaluable construction probabilities, not free-form AST probabilities; keep the two scoring regimes distinct. |
| P-INT-05 | Enumerate the factorized support, two accountable proposal granularities, corrected target, and bounded evaluation protocol. | Implementation, propositions, protocol. | Contributions must survive closest-work comparison without implying general synthesis or speed. |
| P-BG-01 | Formalize PBE exactness, language, cost, and held-out semantics. | DSL implementation; Feser. | Training consistency is not intended correctness. |
| P-BG-02 | Define the equal-family conditional Occam measure on construction traces. | Factorized family normalization and exact control. | It is data-dependent; call it a base measure, not a Bayesian prior, and do not silently quotient trace aliases into ASTs. |
| P-BG-03 | Define the finite Gibbs target and its interpretation. | Target implementation and exact oracle. | Soft loss is not a calibrated likelihood. |
| P-SYM-01 | Explain type-directed hypotheses with a concrete fold skeleton. | Induction engine and grammar. | Specialized families may encode task knowledge. |
| P-SYM-02 | Separate sound refutation from soft deduction guidance. | Deduction rules and tests. | Prove each rule or label its tested boundary. |
| P-SYM-03 | Define finite catalogs and distinguish online trace identity from reference alias ownership. | Lazy support, materialized support, and alias tests. | The modes define different targets when cross-family aliases are present; do not compare their oracle metrics without checking injectivity. |
| P-PROP-01 | Define what the staged family-and-hole scorer asks Qwen to score and what is archived. | vLLM scorer and staged score-ledger schema. | The implemented scores cover the complete concatenated prompt. They are neither AST probabilities nor candidate-suffix probabilities. |
| P-PROP-02 | Define the staged node proposal and defensive mixture. | Normalization tests and deduction guide. | Explain why mixture, not additive score penalties, and do not apply this law to the complete-program semantic scorer. |
| P-PROP-03 | Factorize a complete construction trace and instantiate the family--predicate--mapper conditional law. | Construction-factorization equation, filter--map skeleton, proposal trie, and forced-evaluation tests. | Verify $q(e)=q_H(h)q_P(p\mid h)q_M(m\mid h,p)$: mapper scoring must condition on both $h$ and $p$; also check the single-family shortcut and request deduplication. |
| P-PROP-04 | Bound staged scorer work for one distinct sparse bounded-square ancestry. | Three surviving families and the finite 600-predicate and 60-mapper catalogs. | $3+600+60=663$ is a per-ancestry choice bound, not a claim that all $600\times60=36{,}000$ complete programs were semantically scored or that total run-wide work is 663. |
| P-PROP-05 | Account for clone mass, failures, and budget. | `logaddexp`, failure policy, and budget tests. | Any fallback with unknown mass breaks the claim. |
| P-PROP-06 | Define the separate joint-semantic proposal over a deterministic complete-program slate and recover its sequential conditionals. | Symmetrized label-score law, prefix-mass implementation, compact semantic ledger, and replay tests. | Do not conflate complete-program compatibility with staged hole scores, execution loss, or calibrated target probabilities; verify telescoping prefix conditionals, the $p_0$ floor, and $\alpha=0$. |
| P-SMC-01 | Give the Feynman--Kac path target and potential for either accountable transition. | Staged and joint-semantic SMC engines; Del Moral. | Clarify the product-of-static-target path and which transition law supplies each denominator. |
| P-SMC-02 | Separate path normalizer and terminal marginal diagnostics. | Exact target and result schema. | Never claim unbiased log normalizer. |
| P-SMC-03 | Separate online lazy evaluation from exhaustive reference. | Lazy-mode regression test and explicit `--materialize-reference` control. | Initialization samples are executed too; count them. Matching oracle claims also require identical trace semantics. |
| P-EXP-01 | Describe the planned four-arm factorial and identify the two-arm Gate-2 slice. | `research/protocol-size-study-transport32.json`. | Gate 2 contains Q and Q+D only; it cannot identify deduction-only or resampling effects. |
| P-EXP-02 | Define recorded outcomes, cache/provider accounting, transport amendment, seed-level independence, and the exploratory boundary. | Amended protocol SHA `56c590864d456f884c82bf62ada3c11c1c2504d21b021e650bc323007553fc60`; lazy records; archive schema and provider/cache telemetry. | Particles are not independent trials; cached reuse is not new provider work; one seed and unfinished inference block scaling and confirmatory claims. |
| P-EXP-03 | Scope the benchmark and disclose external-validity limits. | Frozen task manifests. | Current task count is too small for JMLR. |
| P-EXP-04 | Separate the partial-slate preflight from the full-slate semantic-scoring diagnostic. | Completed 512-trace semantic ledger; rebuilt exact-trace identities; 36,198-trace support and 144,792-path regression tests. | The 512-trace slate omitted both exact traces and validates only the pipeline; the full slate is an exhaustive scoring diagnostic, not scalable discovery. |
| P-EXP-05 | Predeclare exact-trace ranks, mass decomposition, fixed-$q$ hit calculation, and identical-slate model comparison. | Full semantic ledgers and independent replay, once complete. | Report ties plus $p_0$, semantic-component, and defensive masses separately; the hit formula is proposal-only and excludes initialization, and all rank/mass/model claims remain pending until both ledgers verify. |
| P-RES-01 | Report the complete exploratory Gate-2 discovery and held-out matrix. | `paper/generated/figures/gate2-size-transport32/outcome_matrix.pdf`; 16 cells under one protocol hash. | One seed cannot estimate success; particles are not replicates. |
| P-RES-02 | Attribute the Q versus Q+D difference to defensive deduction under the tested conditions. | Paired raw cells across 3B, 7B, 14B, and 32B checkpoints. | No D arm means no independent Qwen-value estimate; identical columns are not evidence of a size effect or SMC benefit. |
| P-RES-03 | Treat provider work as cache-disposition accounting and bound score semantics. | `paper/generated/figures/gate2-size-transport32/provider_work.pdf`; cache and provider telemetry. | Fully warm 32B map cells and partly warm 32B hard cells preclude checkpoint timing or efficiency comparisons; mean-full-prompt energy is not a candidate-suffix or AST probability. |
| P-REL-01 | Position against symbolic and LLM-guided synthesis. | Primary related-work papers. | HYSYNTH and LLM-guided enumeration are closest. |
| P-REL-02 | Position against SMC synthesis and LLM steering. | Ellis; ModelSMC; Lew; Zhao. | Do not claim first use of SMC; restrict novelty to evaluable staged and complete-slate proposal terms. |
| P-LIM-01 | State technical and external-validity limitations. | Implementation audit and full-slate score budget. | Specialized DSL, exact enumeration, and exhaustive full-slate semantic scoring are major limitations. |
| P-LIM-02 | State resource, safety, bias, and transparency implications. | Lazy execution, uniform support, and archive design. | Avoiding unvisited executions is not evidence of lower wall time or total compute; mathematical support is not deployment safety. |
| P-CON-01 | Re-state the contribution as attribution and auditability, then name both required follow-ups. | Entire paper. | Repeated-seed D controls and replay-verified full-slate ranks/masses remain necessary; do not end with a speed or model-quality claim. |
| P-ACK-01 | Disclose current funding and compute, then flag the declarations still required for submission. | Author statements, RunPod records, and final conflict/funding review. | The final submission must name all funding, donated compute, and competing interests. |
| P-APP-01 | Sketch normalization and full-support arguments for both proposal laws. | Node, clone, and joint-semantic equations; proposal replay tests. | Check staged trie induction, joint-semantic slate normalization, defensive $p_0$ support, and prefix telescoping; expand to a formal proposition before submission. |
| P-APP-02 | Prevent oracle enumeration from being misreported as discovery. | Lazy/reference separation; Gate-2 search-success schema; full-slate retrospective rank lookup. | The Gate-2 grid is lazy, while exact identities in the full-slate diagnostic are used only after model scoring; neither reference construction nor identity lookup is discovery. |

## Opening and closing sentence plan

These entries describe the **job** of the first and final sentence, not frozen
wording. The manuscript owns the prose; this guide owns the argument structure.

| ID | Opening-sentence job | Closing-sentence job or handoff |
|---|---|---|
| P-ABS-01 | Pose the attribution problem and the free-form proposal-density gap. | End on the paired Q failure/Q+D success, its tested deduction attribution, and the paper's measurement—not performance—claim. |
| P-INT-01 | Define PBE and locate classical symbolic versus learned prioritization. | Hand learned guidance to the attribution question. |
| P-INT-02 | Challenge the naive question “did the LLM help?” | Demand isolation of model proposal, symbolic support, target, and selection. |
| P-INT-03 | Introduce SMC as a population view of synthesis. | Expose the unavailable canonical-AST proposal density, motivating the bounded construction. |
| P-INT-04 | Narrow the problem to staged finite choices and a separate deterministic complete-program slate. | Establish two evaluable energy proposals—not free-form AST probabilities—then hand off to the contributions. |
| P-INT-05 | Enumerate the four technical and evaluation contributions, including both proposal granularities. | Bound generality, calibration, and speed claims before formalization. |
| P-BG-01 | Formalize examples, exactness, the DSL, and structural cost. | Separate training consistency from intended semantics through held-out evaluation. |
| P-BG-02 | Define construction traces and the equal-family conditional Occam measure. | Label it data-conditional, not an unconditional Bayesian prior, motivating the Gibbs target. |
| P-BG-03 | Define the finite loss-tempered target. | Restrict interpretation to the declared finite target, then move to constructing its support. |
| P-SYM-01 | Map types and structural relations to skeleton hypotheses. | Use the filter–map example to expose typed holes for deduction. |
| P-SYM-02 | Split deduction into refutation and hole-example inference. | Preserve all choices not eliminated by sound invariants, handing off to finite catalogs. |
| P-SYM-03 | Define complete typed catalogs and online trace identity. | Warn that alias handling can change the target, setting the accounting requirements for proposals. |
| P-PROP-01 | Specify the staged common prompt and complete full-prompt token scoring. | Define the staged energies and explicitly deny candidate-suffix likelihood semantics. |
| P-PROP-02 | Define the staged Qwen, deduction, and uniform node mixture. | Explain why normalized mixtures, rather than additive penalties, control serialization-length domination. |
| P-PROP-03 | Factor a complete trace into family and sequential-hole probabilities, then instantiate the filter--map conditional law. | Make $q_H(h)q_P(p\mid h)q_M(m\mid h,p)$ explicit; close with the single-family shortcut and request deduplication before work accounting. |
| P-PROP-04 | Contrast $600\times60=36{,}000$ specialized leaves with at most $3+600+60=663$ choices along one distinct staged ancestry. | Restrict 663 to per-ancestry staged-scoring work—not complete-slate or run-wide work—before clone accounting. |
| P-PROP-05 | Add the clone route to the transition law. | Close unknown fallback mass and reserve the score budget before introducing the separate complete-program semantic proposal. |
| P-PROP-06 | Separate complete-program semantic-slate scoring from the staged family-and-hole scorer. | End with prefix conditionals telescoping to the persisted defensive $q(z)$ while only sampled programs are executed, then hand off to SMC correction. |
| P-SMC-01 | Define the product-of-static-target path law and incremental potential for either accountable transition. | State the terminal marginal and the resampled versus non-resampled base-weight rule. |
| P-SMC-02 | Separate path-normalizer and terminal-target references. | Bound diagnostics to a finite-particle approximation, motivating execution-boundary separation. |
| P-SMC-03 | Contrast exhaustive reference and lazy online execution. | Refuse to credit support construction as discovery, handing off to experimental design. |
| P-EXP-01 | Define the U, D, Q, and Q+D plan, then identify Gate 2 as Q/Q+D only. | Prevent the two-arm slice from implying a deduction-only or resampling comparison. |
| P-EXP-02 | Define archived outcomes, provider/cache accounting, the transport amendment, and the seed as the independent unit. | End with the one-seed and unfinished-inference boundary that blocks confirmatory or size claims. |
| P-EXP-03 | Introduce the controlled benchmark and covered DSL structures. | Disclose the small synthetic scope before presenting pilots. |
| P-EXP-04 | Open with the fixed 36,198-trace sparse bounded-square support and the completed 512-trace preflight. | Establish that the preflight omitted both exact traces and require all 36,198 scores before interpreting semantic discrimination. |
| P-EXP-05 | Define tie-aware raw semantic ranks and separate base, learned-component, and defensive exact-trace masses. | Restrict the hit budget to independent draws from fixed $q$, require an identical-slate model comparison, and leave all measured values pending replay. |
| P-RES-01 | Open with the complete exploratory Gate-2 grid and its fixed design. | Close by making seed, not particles or cells, the unit and refusing a general model effect. |
| P-RES-02 | State that checkpoint size did not separate the raw outcomes. | Attribute the paired difference to defensive deduction under test while denying independent Qwen, scaling, and resampling claims. |
| P-RES-03 | Separate discovery outcomes from cache-confounded provider telemetry. | Bound cache, latency, and mean-full-prompt semantics; deny checkpoint-efficiency, speed, and scaling conclusions. |
| P-REL-01 | Position against symbolic, neural-guided, and LLM-guided synthesis. | Differentiate this work through exact factorized proposal accounting. |
| P-REL-02 | Position against SMC synthesis, ModelSMC, and token steering. | Deny priority claims and state the narrower evaluability of staged and complete-slate proposal terms. |
| P-LIM-01 | Lead with the small DSL, specialized catalogs, data-dependent support, and exhaustive diagnostic costs. | Rule out general synthesis, scalable full-slate scoring, formal correctness, and unbounded-code posterior claims. |
| P-LIM-02 | State that lazy execution does not yet prove lower total cost. | End on sandboxing and publication of failures and ledgers, handing off to the conclusion. |
| P-CON-01 | Recast credible LLM synthesis as an attribution and accounting problem. | Leave where Qwen adds value to repeated-seed D controls and the replay-verified full-slate rank-and-mass diagnostic. |
| P-ACK-01 | State the draft's current funding status. | Require complete funding, compute, conflict, and author disclosures before submission. |
| P-APP-01 | Prove staged-node and joint-semantic normalization plus positive support. | Lift staged normalization through the trie and clone mixture, prove joint-slate normalization and prefix telescoping, then hand off to oracle boundaries. |
| P-APP-02 | State that reference enumeration itself discovers exact programs. | Permit only retrospective rank/mass lookup after model scoring; forbid crediting oracle-only discoveries to Qwen or SMC. |

## Multi-paragraph section skeletons

- **Abstract:** pose attribution gap -> state bounded method -> name the
  accountable proposal -> report paired Q failure/Q+D success -> attribute the
  tested difference to deduction -> bound the claim.
- **Introduction:** establish PBE -> expose attribution/probability gap -> state
  controlled solution -> enumerate bounded contributions.
- **Background:** define examples and DSL -> define conditional base measure ->
  define finite target.
- **Symbolic front end:** generate typed skeletons -> refute/derive evidence ->
  enumerate canonical hole choices.
- **Proposal:** score finite staged choices -> mix normalized guides ->
  instantiate the conditional construction path -> bound per-ancestry staged
  work -> add cloning and failure policy -> define the separate complete-program
  semantic slate and telescoping prefix law.
- **SMC:** define target/potential -> state measurable diagnostics -> separate
  online search from exact oracle.
- **Experiments:** isolate components -> define statistical unit/outcomes ->
  disclose benchmark scope -> separate the 512-trace preflight from the full
  36,198-trace diagnostic -> predeclare tie-aware ranks, mass decomposition,
  fixed-$q$ hit accounting, and identical-slate model comparison.
- **Results:** report the complete one-seed grid -> attribute the paired
  difference to defensive deduction -> separate cache-disposition accounting
  and score semantics from discovery outcomes.
- **Related work:** position against symbolic and learned synthesis -> position
  against SMC and steering -> isolate the finite-accounting contribution.
- **Limitations:** disclose technical/external-validity limits -> disclose
  compute, bias, safety, and transparency limits.
- **Conclusion:** restate the attribution requirement -> summarize inspectable
  components -> leave LLM value to confirmatory evidence.
- **Acknowledgments:** state present funding and compute -> require final
  disclosure review.
- **Appendices:** establish normalization/full support -> enforce the boundary
  between oracle enumeration and online discovery.

## Non-paragraph material

The author block, keywords, equations, table captions, and bibliography are not
paragraph units. Review them separately for metadata accuracy, notation,
self-contained captions, and citation completeness.

### Exploratory figure attachments

The manifest reports `complete_four_checkpoint_gate2: true`, so the outcome and
provider-work figures are attached to `main.tex`. They remain exploratory and
cannot add confidence intervals, success-rate estimates, or confirmatory
language to their host paragraphs. The paired-seed figure remains inactive
because only seed 101 is available.

| Figure | Planned paragraph | Purpose | Reviewer check |
|---|---|---|---|
| Exact/held-out outcome matrix (`paper/generated/figures/gate2-size-transport32/outcome_matrix.pdf`) | P-RES-01 | Show every raw checkpoint-task-arm outcome, including failures and held-out unavailability. | All four amended Gate-2 checkpoints must be present under one protocol hash; `n=1` is not a success-rate estimate. |
| Paired Q versus QD seeds (inactive) | P-RES-01 | Show within-seed changes attributable to adding the deduction guide. | Generate only with at least two genuinely paired seeds; show raw pairs without treating particles as replicates. |
| Provider work and cache status (`paper/generated/figures/gate2-size-transport32/provider_work.pdf`) | P-RES-03 | Separate provider-scored token positions, provider wait for cache misses, and cached reuse. | Cached scores are replayed evidence, not new provider work; neither provider seconds nor wall time alone establishes faster synthesis. |
