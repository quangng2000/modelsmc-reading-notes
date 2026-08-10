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
| P-ABS-01 | State the problem, method, result, and boundary in under 200 words. | Entire paper; pilot artifacts are exploratory. | Do not imply confirmed speed or generality. |
| P-INT-01 | Establish PBE and the symbolic/learned search context. | Gulwani; Feser; Kalyan; HYSYNTH; Li. | Why another hybrid synthesizer? |
| P-INT-02 | Define the attribution problem that motivates the work. | Support-builder audit and matched pilots. | Is this only an evaluation hygiene issue? |
| P-INT-03 | Explain why SMC is relevant and expose the proposal-density gap. | Ellis et al.; ModelSMC; SMC background. | Prior SMC synthesis already exists. |
| P-INT-04 | Give the central design move: finite canonical energy scoring. | Sections 3--5 and score ledger. | Is this still an LLM proposal probability? Answer: it is a declared energy proposal, not free-form AST probability. |
| P-INT-05 | Enumerate contributions and immediately bound them. | Implementation, propositions, protocol. | Contributions must survive closest-work comparison. |
| P-BG-01 | Formalize PBE exactness, language, cost, and held-out semantics. | DSL implementation; Feser. | Training consistency is not intended correctness. |
| P-BG-02 | Define the equal-family conditional Occam measure on construction traces. | Factorized family normalization and exact control. | It is data-dependent; call it a base measure, not a Bayesian prior, and do not silently quotient trace aliases into ASTs. |
| P-BG-03 | Define the finite Gibbs target and its interpretation. | Target implementation and exact oracle. | Soft loss is not a calibrated likelihood. |
| P-SYM-01 | Explain type-directed hypotheses with a concrete fold skeleton. | Induction engine and grammar. | Specialized families may encode task knowledge. |
| P-SYM-02 | Separate sound refutation from soft deduction guidance. | Deduction rules and tests. | Prove each rule or label its tested boundary. |
| P-SYM-03 | Define finite catalogs and distinguish online trace identity from reference alias ownership. | Lazy support, materialized support, and alias tests. | The modes define different targets when cross-family aliases are present; do not compare their oracle metrics without checking injectivity. |
| P-PROP-01 | Define what Qwen scores and what is archived. | vLLM scorer and score-ledger schema. | The implemented scores cover the complete concatenated prompt. They are neither AST probabilities nor candidate-suffix probabilities. |
| P-PROP-02 | Define the node proposal and defensive mixture. | Normalization tests and deduction guide. | Explain why mixture, not additive score penalties. |
| P-PROP-03 | Factorize complete construction probability. | Proposal trie and forced-evaluation tests. | Check sequential conditionals and single-family shortcut. |
| P-PROP-04 | Account for clone mass, failures, and budget. | `logaddexp`, failure policy, budget tests. | Any fallback with unknown mass breaks the claim. |
| P-SMC-01 | Give the Feynman--Kac path target and potential. | SMC engine and Del Moral. | Clarify this unusual product-of-static-target path. |
| P-SMC-02 | Separate path normalizer and terminal marginal diagnostics. | Exact target and result schema. | Never claim unbiased log normalizer. |
| P-SMC-03 | Separate online lazy evaluation from exhaustive reference. | Lazy-mode regression test and explicit `--materialize-reference` control. | Initialization samples are executed too; count them. Matching oracle claims also require identical trace semantics. |
| P-EXP-01 | Describe only the four arms actually present in the draft protocol. | `research/protocol.json`. | All arms share symbolic hard pruning; uncorrected and resampling ablations are not yet protocol arms. |
| P-EXP-02 | Define recorded outcomes, seed-level independence, and the unfinished analysis boundary. | Draft protocol, lazy stage schema, matrix runner, and row-level aggregator. | Particles are not independent trials; weight variance, time-to-first-success endpoints, and inferential tests are not all implemented yet. |
| P-EXP-03 | Scope the benchmark and disclose external-validity limits. | Frozen task manifests. | Current task count is too small for JMLR. |
| P-RES-01 | Report oracle-backed pilots only as proposal-accounting motivation. | Archived seed-23 results. | One seed cannot estimate success, and pre-evaluated support cannot establish online discovery or speed. |
| P-RES-02 | Attribute the hard-task success to deduction, not Qwen. | Selected component probabilities in events. | Complete score ledger must reproduce values. |
| P-RES-03 | Identify full-prompt length bias and pre-register the implemented mean-full-prompt alternative. | Qwen-only failures and energy-mode implementation. | Candidate-suffix scoring is future work until a tokenizer boundary protocol exists; avoid tuning on the test tasks. |
| P-REL-01 | Position against symbolic and LLM-guided synthesis. | Primary related-work papers. | HYSYNTH and LLM-guided enumeration are closest. |
| P-REL-02 | Position against SMC synthesis and LLM steering. | Ellis; ModelSMC; Lew; Zhao. | Do not claim first use of SMC. |
| P-LIM-01 | State technical and external-validity limitations. | Implementation audit. | Specialized DSL and enumeration are major limitations. |
| P-LIM-02 | State resource, safety, bias, and transparency implications. | Lazy execution, uniform support, and archive design. | Avoiding unvisited executions is not evidence of lower wall time or total compute; mathematical support is not deployment safety. |
| P-CON-01 | Re-state the contribution as attribution and auditability. | Entire paper. | Do not end with an unsupported speed claim. |
| P-ACK-01 | Disclose current funding and compute, then flag the declarations still required for submission. | Author statements, RunPod records, and final conflict/funding review. | The final submission must name all funding, donated compute, and competing interests. |
| P-APP-01 | Sketch normalization and full-support argument. | Equation 4 and proposal tests. | Expand to a formal proposition before submission. |
| P-APP-02 | Prevent oracle enumeration from being misreported as discovery. | Lazy/reference separation. | Current pilots are oracle-backed; any future online-efficiency table must count newly realized initialization and proposal traces separately. |

## Opening and closing sentence plan

These entries describe the **job** of the first and final sentence, not frozen
wording. The manuscript owns the prose; this guide owns the argument structure.

| ID | Opening-sentence job | Closing-sentence job or handoff |
|---|---|---|
| P-ABS-01 | Pose the attribution problem and the free-form proposal-density gap. | End on the negative Qwen result and the paper's measurement—not performance—claim. |
| P-INT-01 | Define PBE and locate classical symbolic versus learned prioritization. | Hand learned guidance to the attribution question. |
| P-INT-02 | Challenge the naive question “did the LLM help?” | Demand isolation of model proposal, symbolic support, target, and selection. |
| P-INT-03 | Introduce SMC as a population view of synthesis. | Expose the unavailable canonical-AST proposal density, motivating the bounded construction. |
| P-INT-04 | Narrow the problem to finite canonical family and hole choices. | Establish an evaluable energy proposal—not free-form AST probability—then hand off to the contributions. |
| P-INT-05 | Enumerate the four technical and evaluation contributions. | Bound generality, calibration, and speed claims before formalization. |
| P-BG-01 | Formalize examples, exactness, the DSL, and structural cost. | Separate training consistency from intended semantics through held-out evaluation. |
| P-BG-02 | Define construction traces and the equal-family conditional Occam measure. | Label it data-conditional, not an unconditional Bayesian prior, motivating the Gibbs target. |
| P-BG-03 | Define the finite loss-tempered target. | Restrict interpretation to the declared finite target, then move to constructing its support. |
| P-SYM-01 | Map types and structural relations to skeleton hypotheses. | Use the filter–map example to expose typed holes for deduction. |
| P-SYM-02 | Split deduction into refutation and hole-example inference. | Preserve all choices not eliminated by sound invariants, handing off to finite catalogs. |
| P-SYM-03 | Define complete typed catalogs and online trace identity. | Warn that alias handling can change the target, setting the accounting requirements for proposals. |
| P-PROP-01 | Specify the common prompt and complete full-prompt token scoring. | Define the implemented energies and explicitly deny candidate-suffix likelihood semantics. |
| P-PROP-02 | Define the Qwen, deduction, and uniform node mixture. | Explain why normalized mixtures, rather than additive penalties, control serialization-length domination. |
| P-PROP-03 | Factor a complete trace into family and sequential-hole probabilities. | Note the single-family shortcut and request deduplication without changing the proposal law. |
| P-PROP-04 | Add the clone route to the transition law. | Close unknown fallback mass and reserve the score budget before moving to SMC weighting. |
| P-SMC-01 | Define the product-of-static-target path law and incremental potential. | State the terminal marginal and the resampled versus non-resampled base-weight rule. |
| P-SMC-02 | Separate path-normalizer and terminal-target references. | Bound diagnostics to a finite-particle approximation, motivating execution-boundary separation. |
| P-SMC-03 | Contrast exhaustive reference and lazy online execution. | Refuse to credit support construction as discovery, handing off to experimental design. |
| P-EXP-01 | Define the U, D, Q, and Q+D arms and their shared controls. | Mark absent ablations as future preregistered matrices. |
| P-EXP-02 | Define archived outcomes and the seed as the independent unit. | List unfinished inference and endpoints that block confirmatory claims. |
| P-EXP-03 | Introduce the controlled benchmark and covered DSL structures. | Disclose the small synthetic scope before presenting pilots. |
| P-RES-01 | Label the table exploratory, oracle-backed, and non-speed evidence. | Use outcomes only to motivate repeated-seed protocol and component attribution. |
| P-RES-02 | Decompose the exact hard-task choice into Qwen and deduction masses. | Attribute success to the defensive symbolic mixture, leading to Qwen's scoring failure. |
| P-RES-03 | Diagnose short-AST bias from total full-prompt scores. | Identify mean-full-prompt energy as implemented and token-bounded scoring as future work. |
| P-REL-01 | Position against symbolic, neural-guided, and LLM-guided synthesis. | Differentiate this work through exact factorized proposal accounting. |
| P-REL-02 | Position against SMC synthesis, ModelSMC, and token steering. | Deny priority claims and state the narrower finite-evaluability contribution. |
| P-LIM-01 | Lead with the small DSL, specialized catalogs, and data-dependent support. | Rule out general synthesis, formal correctness, and unbounded-code posterior claims. |
| P-LIM-02 | State that lazy execution does not yet prove lower total cost. | End on sandboxing and publication of failures and ledgers, handing off to the conclusion. |
| P-CON-01 | Recast credible LLM synthesis as an attribution and accounting problem. | Leave the empirical question—where Qwen adds value—to confirmatory benchmarks. |
| P-ACK-01 | State the draft's current funding status. | Require complete funding, compute, conflict, and author disclosures before submission. |
| P-APP-01 | Prove node normalization and positive support from the convex mixture. | Lift normalization through the trie and clone mixture, then hand off to oracle-claim boundaries. |
| P-APP-02 | State that reference enumeration itself discovers exact programs. | Forbid crediting oracle-only discoveries to Qwen or SMC and close the paper's claim boundary. |

## Multi-paragraph section skeletons

- **Abstract:** pose attribution gap -> state bounded method -> name the
  accountable proposal -> report the negative result -> bound the claim.
- **Introduction:** establish PBE -> expose attribution/probability gap -> state
  controlled solution -> enumerate bounded contributions.
- **Background:** define examples and DSL -> define conditional base measure ->
  define finite target.
- **Symbolic front end:** generate typed skeletons -> refute/derive evidence ->
  enumerate canonical hole choices.
- **Proposal:** score finite choices -> mix normalized guides -> multiply the
  construction path -> add cloning and failure policy.
- **SMC:** define target/potential -> state measurable diagnostics -> separate
  online search from exact oracle.
- **Experiments:** isolate components -> define statistical unit/outcomes ->
  disclose benchmark scope.
- **Results:** label pilots -> attribute hard success to deduction -> motivate
  score-semantics ablation.
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
