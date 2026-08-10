# Paragraph-level author review guide

This file is deliberately separate from the manuscript. Paragraph IDs appear
as LaTeX comments in `main.tex`; they are not rendered in the paper.

Review each row in this order:

1. Does the paragraph have exactly one primary job?
2. Is its strongest claim supported by the listed evidence?
3. Does it depend on a term or result introduced earlier?
4. Would a skeptical reviewer interpret it more broadly than intended?

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
| P-APP-01 | Sketch normalization and full-support argument. | Equation 4 and proposal tests. | Expand to a formal proposition before submission. |
| P-APP-02 | Prevent oracle enumeration from being misreported as discovery. | Lazy/reference separation. | Current pilots are oracle-backed; any future online-efficiency table must count newly realized initialization and proposal traces separately. |

## Multi-paragraph section skeletons

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
- **Discussion:** distinguish closest work -> disclose limitations and safety ->
  conclude with the auditable contribution.
