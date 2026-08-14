# Author review guide

This guide is separate from the manuscript. Paragraph IDs appear as LaTeX
comments in `main.tex` and are not rendered.

For every paragraph, ask:

1. Does the opening sentence state its topic or claim immediately?
2. Does the paragraph perform one primary job?
3. Is every number traceable to a frozen, verified artifact?
4. Does the final sentence enforce the correct claim boundary or handoff?
5. Could a skeptical reader interpret it more broadly than the evidence allows?

The paper has one narrative: probability-accountable four-slot repair SMC,
fresh-blind r5 discovery, a chronological calibration chain (failed V1,
terminal diagnostic V2, failed fresh V2, reused-task factorized diagnostic, and
second-fresh V3 R2), and a separately marked ExeDec V2 debug study. Do not
restore the staged family/hole Qwen proposal, joint-semantic slate, Gate-2 grid,
blind evidence-frontier V2, developmental matrix, or unexecuted full-slate
study to the rendered paper.

## Front matter and motivation

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-ABS-01` | State the proposal, r5 result, two retained calibration failures, and narrow fresh V3 result in under 200 words. | ExeDec is not an abstract-level result; V3 does not calibrate the LLM, four-stage SMC, full PBE, or target mean loss. |
| `P-INT-01` | Locate PBE among symbolic and learned search methods. | Gulwani, Feser, Kalyan, HYSYNTH, and LLM-guided enumeration. |
| `P-INT-02` | Define the system-attribution problem. | Treat model, interpreter, selector, target, and comparator as separate components. |
| `P-INT-03` | Explain why SMC needs an evaluable proposal law. | Free-form text probability is not canonical-program probability. |
| `P-INT-04` | Introduce application totalization and auxiliary-law cancellation. | Claim an accountable implemented transition, not an inferred LLM AST probability. |
| `P-INT-05` | State the bounded methodological contribution. | Close on shortlist replayability, not empirical efficacy. |
| `P-INT-06` | State the completed evidence sequence and bounded ExeDec debug result. | Discovery and calibration remain distinct; the calibration sequence is chronological rather than a set of competing paper versions. |

## Setting, proposal, and SMC

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-BG-01` | Define PBE exactness, bounded DSL, structural cost, and semantic scope. | Example exactness is not out-of-domain correctness. |
| `P-SYM-01` | Define the strict Filter-then-Map skeleton and normalized recursive grammar. | Do not imply support enumeration is required online. |
| `P-SYM-02` | Explain sound singleton evidence and provider-visible information. | Hidden targets, secrets, catalogs, and comparator outcomes remain unavailable. |
| `P-SYM-03` | Make the application authoritative for parsing, typing, execution, and caching. | The provider neither validates programs nor supplies confidence. |
| `P-PROP-SHORT-01` | Define four-slot totalization and `q=(1-epsilon)H_S+epsilon g`. | Preserve missing, malformed, invalid, no-op, and duplicate slots in the fixed denominator. |
| `P-PROP-SHORT-02` | Establish normalization, full support, and frozen failure handling. | `epsilon=0.05` for r5, V1, ExeDec V2, and sticky V2; terminal V2 also has declared alternatives. |
| `P-PROP-SHORT-03` | Introduce the unknown acquisition law `R` on an auxiliary shortlist. | Distinguish LLM, grammar-random, and deterministic-oracle acquisition without assigning an LLM AST probability. |
| `P-PROP-SHORT-04` | Show exact cancellation of `R` and identify the implemented proposal laws. | Four-slot arms use Equation 2; the terminal V2 top-64 control uses a declared 64-slot empirical law. |
| `P-PROP-SHORT-05` | Separate proposal correctness from overlap and finite-particle accuracy. | Set up both retained failures and the proposal-specific V3 pass. |
| `P-SMC-01` | Define the four declared stage targets and frozen scales. | `L_D` is capped soft execution loss; r5/V1/V2 use `(0.75,0.02,2)`, ExeDec uses scorer defaults `(2.0,0.15,2)`; targets are not scale-matched. |
| `P-SMC-02` | Define the initial-program-conditioned extended target, proposal, and incremental potential. | The same acquisition factor must occur in numerator and denominator. |
| `P-SMC-03` | Define child weights, resampling, and terminal/path distinction. | Do not conflate the product-path normalizer with the terminal target. |
| `P-SMC-04` | Separate r5/ExeDec online execution from all calibration and ExeDec reference work. | ExeDec's online physical-call count excludes public-reference enumeration; oracle discoveries are never online synthesis. |

## Study methods

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-EXP-01` | Introduce the evidence layers in their final order. | R5, the complete calibration chronology, then ExeDec debug; no parallel paper fork. |
| `P-EXP-R5-01` | State pre-generation freeze, custodial generation, task count, and singleton-complete public evidence. | Freshness and the declared finite domain precede provider details. |
| `P-EXP-R5-02` | Bind GPT-OSS-120B settings and blind custody. | R5 and ExeDec calls share frozen settings and no retries; private material stays outside the provider environment. |
| `P-EXP-R5-03` | Define matched arms and paired seeds. | Acquisition is the only arm difference; shared sampling and resampling mechanics make the pairing explicit. |
| `P-EXP-R5-04` | Define the `1+4+3x8=29` schedule, endpoint, statistical unit, and no-early-stop rule. | Singleton completeness supports only finite-domain behavior after unblinding. |
| `P-EXP-CAL-01` | Define V1's four developmental supports and public-evidence oracle. | The ranking oracle is exhaustive and not an LLM or search baseline. |
| `P-EXP-CAL-02` | Define V1's particle grid, repetitions, schedule, and frozen gate. | Close on the two prespecified gate components. |
| `P-EXP-CAL-03` | Define terminal V2's six-arm, two-`N`, 128-repetition post-failure design. | It cannot rescue V1 or validate the four-stage recurrence. |
| `P-EXP-CAL-04` | Define fresh V2's 20-task proposal--bridge design and frozen gate. | The secret, thresholds, source, seeds, and failure policy all precede task generation. |
| `P-EXP-CAL-05` | Define the factorized mode-acquisition change. | Close by making reused tasks and non-confirmatory status explicit. |
| `P-EXP-CAL-06` | Define fresh V3 R2's second fresh suite, bridge, descriptive arms, and all gate components. | The paragraph must end with the full joint gate, not an outcome. |
| `P-EXP-EXEDEC-V2-01` | Define the four-target, eight-seed paired debug design. | Local strict Filter-then-Map adapter; not an official ExeDec split; no efficacy gate. |
| `P-EXP-EXEDEC-V2-02` | Disclose public-data, finite-probe, contamination, custody, and transfer limits. | The study archive is canonical and imported; the original operations-evidence export failed, and its derivative is noncanonical and unimported. Never say that two imports passed. |

## Results

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-RES-01` | Preserve the methods/results ordering and endpoint separation. | Never pool r5 discovery, calibration, or ExeDec debug outcomes. |
| `P-RES-R5-01` | Report 10/12 versus 2/12 and the exact paired test. | Primary `p=1/256`; complete-system comparison is not an isolated LLM effect. The post hoc r4 sensitivity belongs only in the integrity appendix. |
| `P-RES-R5-02` | Report slots, physical scoring, cache reuse, provider work, and totalized failures. | Logical and physical work are different quantities; no speed claim. |
| `P-RES-R5-03` | Report sealing and semantic-alias audit. | Finite-domain recovery is not unique-AST or out-of-domain recovery. |
| `P-RES-CAL-01` | State that V1 failed despite exact-program discovery in all 128 `N=256` runs. | Both gate components remain visible; do not generalize discovery to the full 640-run grid. |
| `P-RES-CAL-02` | Report the particle-count RMSE sequence. | Do not claim an `N^{-1/2}` rate or that more particles intrinsically worsen SMC. |
| `P-RES-CAL-03` | Report V2 identities and global top-64, `epsilon=0.50` mechanism evidence. | The exhaustive positive control is not search and does not revise V1. |
| `P-RES-CAL-04` | Report fresh V2's failed gate, uncertainty, convergence, and difficult tasks. | Close by preserving the failure and rejecting outcome-dependent replacement. |
| `P-RES-CAL-05` | Report the successful factorized calculation on reused V2 tasks. | Its close must hand off to the new fresh test, never call the diagnostic confirmation. |
| `P-RES-CAL-06` | Report the fresh V3 primary bias, RMSE, uncertainty, task maximum, and particle-count sequence. | Hand off to the table as the complete frozen gate. |
| `P-RES-CAL-07` | Interpret the descriptive matched/sticky arms and target-mean-loss miss. | Close on the exact boundary: provider-free terminal exact mass, not LLM, four-stage, or full PBE calibration. |
| `P-RES-EXEDEC-V2-01` | Report 17/32 versus 3/32, paired cells, difference, and exact two-sided value. | Eight seeds are nested within only four public targets; the exact value is descriptive, not a significance or confirmation claim. |
| `P-RES-EXEDEC-V2-02` | Report public exactness, logical slots, online scoring, and provider calls. | The 1,040 count excludes public-reference enumeration; do not convert work counts into speed. |
| `P-RES-EXEDEC-V2-03` | Report conditional first-success slots and terminal ESS. | Conditioning on success precludes a speed claim, and finite probes do not establish equivalence. |

## Interpretation and close

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-REL-01` | Position against symbolic, neural-guided, and LLM-guided PBE. | The narrow contribution is application-computed shortlist probability. |
| `P-REL-02` | Position against SMC synthesis and model discovery. | Do not claim first use of SMC; connect exact accounting to both retained failures and the narrow V3 pass. |
| `P-REL-03` | Position the ExeDec adapter against the actual ExeDec contribution. | No complete DSL, official split, or execution-decomposition claim. |
| `P-LIM-01` | Bound r5 by task count, generator, integer domain, model, and comparator. | No task-population or independent model-replicate inference. |
| `P-LIM-02` | Bound the entire calibration sequence by method, stage, support, and functional. | Preserve V1/fresh-V2 failures; V3 is terminal exact-mass evidence only. |
| `P-LIM-03` | State ExeDec target, adapter, contamination, repetition, and custody limits. | The exact test is descriptive and the local derivative does not remove the original integrity limit. |
| `P-LIM-04` | State cross-study compute, wall-time, execution-safety, and review limits. | Close by sending protocol-history detail to the appendix; make no speed claim. |
| `P-CON-01` | Conclude the positive bounded r5 result. | No LLM-only, speed, or broad-generalization conclusion. |
| `P-CON-02` | Conclude with two retained failures, the narrow V3 pass, and the next full-pipeline test. | Keep discovery, accuracy by functional, and compute separate. |
| `P-ACK-01` | Record the draft disclosure status. | Replace before submission with complete funding, compute, author, and conflict disclosures. |

## Appendices

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-APP-NORM-01` | Prove four-slot normalization and note V2's analogous 64-slot control law. | Acquisition cancellation is shared; none of this is a finite-particle calibration proof. |
| `P-APP-NORM-02` | Account for fresh V2/V3 grammar, mode, center, and alias-slice masses. | Evaluable bridge ratios and a mass ledger are inputs to, not proofs of, the fresh calibration gate. |
| `P-APP-R5-TASK-01` | Introduce all r5 task--arm records. | Totals must reproduce 10/12 versus 2/12 and the failure-accounting text. |
| `P-APP-CAL-TASK-01` | Expose V1 task heterogeneity. | Values must reproduce the pooled gate and retain discovery as descriptive. |
| `P-APP-INTEGRITY-01` | Preserve r1/r2, r3, r4, and r5 chronology. | R4 values are invalidated history, not evidence. |
| `P-APP-INTEGRITY-02` | Preserve V1, terminal V2, fresh V2, reused-task V3 diagnostic, and pre-secret R1 chronology. | R2 supplements rather than overwrites failures; R1 had no numerical outcome. |
| `P-APP-INTEGRITY-03` | Record R2 acquisition-before-reference, reveal-free replay, and final unblind binding. | Authentication supports only the narrow V3 claim. |
| `P-APP-INTEGRITY-04` | Preserve ExeDec V1/V2 study-import chronology and the original operations verifier failure. | The 1,143-file study archive and canonical analysis are distinct from operations evidence. |
| `P-APP-INTEGRITY-05` | Preserve the local derivative status. | The derivative is noncanonical and unimported and cannot erase the original failure. |
| `P-APP-ORACLE-01` | State the exact-reference claim boundary. | Exhaustive controls do not establish online discovery or speed. |
| `P-APP-ORACLE-02` | State fresh V3's acquisition-before-reference boundary. | Only the descriptive sticky baseline is reference-derived; the primary bank is not an exhaustive oracle. |
| `P-APP-AVAIL-01` | List canonical r5/calibration/ExeDec artifacts, V3 authentication records, the failed operations transfer, and derivative locators. | Hashes must match the local artifacts; never label the derivative canonical or imported. Replace local paths before publication. |

## Complete paragraph flow audit

Every current paragraph ID in `main.tex` appears below exactly once. “Close”
describes the final-sentence job and its connection to the next paragraph or
section; it is not permission to broaden the claim.

### Abstract and introduction

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-ABS-01` | “Large language models can guide…” | Compress method, r5, two failures, and fresh V3 into the bounded abstract claim. | Excludes LLM, four-stage, full-PBE, and target-mean-loss calibration; Introduction explains why accounting is needed. |
| `P-INT-01` | “Programming by example…” | Place PBE between symbolic control and learned prioritization. | Learned guidance obscures attribution, motivating the next paragraph. |
| `P-INT-02` | “That shift creates an attribution problem.” | Make system-level attribution and hidden failure handling explicit. | Commits to separating proposal, target, budget, and comparator before SMC formalizes the requirement. |
| `P-INT-03` | “Sequential Monte Carlo makes…” | Explain why canonical-program proposal probabilities are required. | The free-form-probability gap motivates the application-computed law next. |
| `P-INT-04` | “DPC addresses this gap…” | Introduce four-slot totalization and auxiliary-law cancellation. | Claims an evaluable transition, not an LLM program distribution, handing off to the contribution. |
| `P-INT-05` | “The methodological contribution…” | Name the bounded replayable synthesis mechanism and its components. | Replayability sets up the empirical questions rather than claiming efficacy. |
| `P-INT-06` | “We evaluate that proposal…” | Preview r5, the calibration chain, and ExeDec as distinct evidence layers. | Makes accuracy method-, functional-, and domain-specific before Related Work. |

### Related work and setting

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-REL-01` | “DPC lies at the intersection…” | Position the shortlist law within symbolic, learned, and LLM-guided PBE. | Narrows novelty to the sampling law, preparing the SMC comparison. |
| `P-REL-02` | “SMC and related particle methods…” | Distinguish the contribution from prior particle inference. | Links exact accounting to both retained failures and the narrow V3 pass. |
| `P-REL-03` | “ExeDec studies execution decomposition…” | Separate the local adapter from official ExeDec scope. | Calls it only a stress test, then hands off to the paper’s actual typed setting. |
| `P-BG-01` | “Let D be examples…” | Define exactness, typed DSL, structural cost, and semantic scope. | Requires every study to declare its evaluation domain before specializing the grammar. |
| `P-SYM-01` | “The confirmatory study restricts…” | Define the Filter--Map skeleton and normalized bounded support. | States that online providers need not enumerate support, motivating public evidence. |
| `P-SYM-02` | “Examples constrain individual holes…” | Define sound singleton evidence and the provider-visible record. | Limits provider input to public execution evidence, leading to application authority. |
| `P-SYM-03` | “All validity decisions remain…” | Assign parsing, typing, execution, and caching to the application. | Leaves proposal-law definition as the remaining problem for the next section. |

### Proposal and SMC

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-PROP-SHORT-01` | “Given current program…” | Define fixed four-slot totalization and the mixture law. | Equation 2 exposes the probability whose normalization is checked next. |
| `P-PROP-SHORT-02` | “Because g is normalized…” | Prove normalization/full support and freeze no-retry failure handling. | Records study-specific epsilon choices before introducing acquisition uncertainty. |
| `P-PROP-SHORT-03` | “We treat the raw response…” | Put the unknown provider response in auxiliary law R and distinguish arm laws. | The exact-parent case completes the setup needed for cancellation. |
| `P-PROP-SHORT-04` | “The same factor R appears…” | Cancel acquisition law R and bind four-slot and terminal-V2 proposal laws. | Accounts for the implemented transition without pretending to recover an LLM marginal. |
| `P-PROP-SHORT-05` | “An evaluable sampling law…” | Warn that correct q can still have poor target overlap. | Separates discovery from target approximation and hands the denominator to SMC. |
| `P-SMC-01` | “The stage targets combine…” | Define stage energies, scorer scales, and the four finite targets. | Denies a real-world likelihood interpretation before extended-state correction. |
| `P-SMC-02` | “We can now write…” | Derive the extended target/proposal and incremental potential. | Shows why the free-form LLM marginal is unnecessary, leading to particle weights. |
| `P-SMC-03` | “The particle implementation…” | Define child allocation, resampling, and terminal/path-normalizer distinction. | Fixes exact-mass as the calibration endpoint before work accounting. |
| `P-SMC-04` | “The studies also separate…” | Separate online executions from exhaustive references across studies. | Forbids crediting reference discoveries as search, clearing the way for study designs. |

### Study methods

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-EXP-01` | “The evaluation begins…” | State the final study order and endpoint separation. | Sends the reader first to bounded r5 discovery. |
| `P-EXP-R5-01` | “Before task generation…” | Establish r5 freeze, custodial generation, task count, and singleton completeness. | Declares the public finite domain before provider custody details. |
| `P-EXP-R5-02` | “The provider was GPT-OSS-120B…” | Bind provider revision/settings and private-material custody. | Keeps private targets outside the provider environment through public sealing. |
| `P-EXP-R5-03` | “For each task, grammar-random…” | Define matched mechanics and paired seeds. | Identifies shortlist acquisition as the only arm difference. |
| `P-EXP-R5-04` | “Each trajectory began…” | Define the 29-slot schedule, endpoint, task-level test, and no early stopping. | Restricts post-unblind equivalence to the declared finite domain. |
| `P-EXP-CAL-01` | “Calibration V1 was frozen…” | Define four developmental supports and the exhaustive public-evidence oracle. | Denies that the oracle is an LLM or search baseline. |
| `P-EXP-CAL-02` | “The study used N…” | Define V1 particle counts, repetitions, schedule, and two-part gate. | Ends on prespecified criteria before reporting the post-failure diagnostic. |
| `P-EXP-CAL-03` | “Because V1 failed…” | Define terminal V2’s frozen six-arm mechanism design. | Conditions the claim on one terminal step and leaves V1 failed. |
| `P-EXP-CAL-04` | “Fresh calibration V2 asked…” | Define the 20-task sampled-mode bridge and multi-component gate. | Confirms all bindings and failure rules preceded the fresh secret. |
| `P-EXP-CAL-05` | “After V2 failed…” | Isolate factorized mode acquisition as the sole diagnostic change. | Reused tasks make a second fresh suite mandatory. |
| `P-EXP-CAL-06` | “Fresh V3 R2 then tested…” | Define the second fresh suite, primary bridge, descriptive arms, and joint gate. | Ends on all gate criteria without leaking results. |
| `P-EXP-EXEDEC-V2-01` | “ExeDec V2 is a descriptive…” | Define four public targets, repeated paired seeds, probes, and no efficacy gate. | Keeps the adapter unofficial and the endpoint finite. |
| `P-EXP-EXEDEC-V2-02` | “The targets and examples are public…” | Disclose contamination, probe, custody, and packaging limitations. | Classifies ExeDec only as adapter debug and points to the integrity appendix. |

### Results

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-RES-01` | “The r5 study supports…” | Reassert endpoint-specific evidential status before numbers. | Sends the reader through r5, calibration, then ExeDec without pooling. |
| `P-RES-R5-01` | “The r5 LLM arm found…” | Report 10/12 versus 2/12 and the exact paired decision. | Bounds the contrast to complete systems, not an isolated model effect. |
| `P-RES-R5-02` | “The 24 trajectories executed…” | Report logical/physical work, cache use, provider calls, and totalized failures. | Preserves no-retry failure accounting before audit evidence. |
| `P-RES-R5-03` | “The post-unblind audit verified…” | Report commitment checks and syntax-versus-semantic alias recovery. | Limits recovery to finite-domain behavior, not unique ASTs or extrapolation. |
| `P-RES-CAL-01` | “V1 failed both…” | State the frozen V1 failure despite universal primary-cell discovery. | Makes discovery/calibration separation explicit before the table. |
| `P-RES-CAL-02` | “For N=32…” | Report V1 RMSE across particle counts. | Concludes evaluable q was insufficient under that proposal and suite. |
| `P-RES-CAL-03` | “Terminal V2 first verified…” | Report identity checks and exhaustive positive-control mechanism evidence. | Calls the control impractical and leaves V1 unchanged. |
| `P-RES-CAL-04` | “Fresh V2 also failed…” | Report V2 point estimates, uncertainty, convergence, and hard tasks. | Preserves the second failure and forbids task replacement. |
| `P-RES-CAL-05` | “The post-failure factorized diagnostic…” | Report successful reused-task mechanism evidence. | Withholds confirmation and makes fresh V3 necessary. |
| `P-RES-CAL-06` | “Fresh V3 passed…” | Report primary bias/RMSE, uncertainty, task maximum, and monotone RMSE. | Hands off to the full gate table rather than selecting favorable checks. |
| `P-RES-CAL-07` | “The descriptive N=256…” | Compare descriptive arms and expose the target-mean-loss miss. | Restricts the pass to provider-free terminal exact mass, not LLM/full-SMC/PBE calibration. |
| `P-RES-EXEDEC-V2-01` | “All 64 runs formed…” | Report paired private-debug-probe counts and the exact descriptive test. | Refuses significance or confirmation because seeds nest within four targets. |
| `P-RES-EXEDEC-V2-02` | “Public-example exactness was…” | Report public exactness and online/provider work. | Excludes reference enumeration from physical calls without making a speed claim. |
| `P-RES-EXEDEC-V2-03` | “Among runs that passed…” | Report success-conditioned slots and terminal ESS. | Says conditioning is not speed and finite probes are not equivalence. |

### Limitations, conclusion, and disclosure

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-LIM-01` | “The r5 result covers…” | Bound task count, domain, model revision, and comparator. | Denies extrapolation and independent model-replicate inference. |
| `P-LIM-02` | “Calibration remains deliberately narrow.” | Consolidate failure, stage, support, provider, and functional limits. | Exact enumeration explains why V3 does not transfer automatically. |
| `P-LIM-03` | “The ExeDec result has…” | Consolidate released-target, adapter, repetition, contamination, and custody limits. | Keeps the exact test descriptive and the derivative non-remedial. |
| `P-LIM-04` | “Across the studies…” | Deny wall-time/compute equivalence and state execution-review needs. | Sends failure and supersession details to the integrity appendix. |
| `P-CON-01` | “DPC turns four typed…” | Restate the accountable proposal and bounded r5 discovery conclusion. | Denies LLM-only, speed, and broad-generalization conclusions. |
| `P-CON-02` | “The calibration sequence qualifies…” | Reconcile two failures with the narrow second-fresh pass. | Separates future discovery, accuracy by functional, and compute endpoints. |
| `P-ACK-01` | “This draft received…” | Record unresolved funding, compute, conflict, and author metadata. | Makes disclosure completion a submission condition. |

### Appendices

| ID | Opening cue | One purpose | Close and connection |
|---|---|---|---|
| `P-APP-NORM-01` | “For the four-slot method arms…” | Prove four-slot and top-64 normalization/cancellation. | States that normalized laws are not finite-particle calibration proofs. |
| `P-APP-NORM-02` | “The fresh proposal--bridge studies…” | Account for grammar, mode, center, and alias-slice masses. | Separates evaluability from the empirical fresh gate. |
| `P-APP-R5-TASK-01` | “Table 4 reports…” | Define every r5 task--arm row and totalized-slot label. | Ensures the table reproduces provider-failure accounting. |
| `P-APP-CAL-TASK-01` | “Table 5 exposes…” | Define V1 task-level RMSE/bias summaries. | Keeps discovery descriptive while exposing heterogeneity. |
| `P-APP-INTEGRITY-01` | “This appendix records…” | Preserve blind r1--r5 supersession, abort, invalid sealing, and sensitivity history. | Explains why r5 required a new secret without rewriting r4. |
| `P-APP-INTEGRITY-02` | “The provider-free calibration sequence…” | Preserve V1/fresh-V2 failures, both diagnostics, and pre-secret V3 R1. | States that R2 supplements rather than overwrites failures. |
| `P-APP-INTEGRITY-03` | “Fresh V3 R2 acquired…” | Record acquisition-before-reference, reveal-free replay, and unblind bindings. | Authentication returns to the narrow V3 boundary. |
| `P-APP-INTEGRITY-04` | “The adapter studies have…” | Preserve ExeDec V1/V2 import history and original packaging failure. | Keeps content validation separate from the failed packaging contract. |
| `P-APP-INTEGRITY-05` | “A separately audited local derivative…” | Record the mode-only derivative and its noncanonical status. | Preserves all records separately rather than repairing history. |
| `P-APP-ORACLE-01` | “Exact reference enumeration serves…” | Separate exact-reference functionals from online synthesis work. | Denies practical-search or speed credit to exhaustive controls. |
| `P-APP-ORACLE-02` | “Fresh V2 and V3 also use…” | State V3 acquisition-before-reference and sticky-baseline exceptions. | Makes the endpoint observable without turning the primary bank into an oracle. |
| `P-APP-AVAIL-01` | “The data-and-paper companion…” | Bind public repositories, exact source revision, canonical artifacts, failures, and hashes. | Defines all locators relative to the source checkout and companion artifact tree. |

## Section-level handoff

- **Abstract:** proposal -> r5 discovery -> retained V1/fresh-V2 failures ->
  second-fresh V3 pass -> claim boundary.
- **Introduction:** PBE ambiguity -> attribution problem -> proposal-density gap
  -> totalized shortlist solution -> evidence sequence.
- **Setting:** typed skeleton -> sound evidence -> authoritative application.
- **Proposal:** totalize slots -> cancel unknown acquisition law -> warn about
  overlap.
- **SMC:** stage targets -> extended correction -> child/resampling geometry ->
  online/reference boundary.
- **Studies and Results:** r5 -> V1 failure -> terminal V2 diagnosis -> fresh V2
  failure -> reused-task factorized diagnosis -> fresh V3 pass -> ExeDec debug.
- **Limitations:** narrow r5 scope -> functional/stage-specific calibration ->
  ExeDec and compute boundaries.
- **Conclusion:** bounded positive discovery -> retained failures plus narrow
  V3 confirmation -> full-pipeline and external confirmation.
- **Appendices:** normalization -> task records -> integrity history -> oracle
  boundary -> artifact inventory.

## ExeDec claim review

The integrated ExeDec result must satisfy all of the following checks:

1. derive numerical outcomes only from the canonical imported study analysis;
2. report 32 paired seed blocks nested within four public targets, never 64
   independent observations or 32 independent target tasks;
3. state the exact two-sided value and absence of an efficacy gate while
   refusing a statistical-significance or confirmatory interpretation;
4. describe success as exactness on finite private debug probes, not semantic
   equivalence or fresh hidden generalization;
5. report logical slots, physical scoring, provider calls, conditional first
   success, and ESS without converting them into speed claims;
6. repeat the unofficial-adapter, public-contamination, debug-only, and
   nonexclusive-custody boundaries; and
7. keep the verified study import separate from the original failed
   operations-evidence transfer and the noncanonical, unimported derivative.

## Metadata and submission review

Before submission, verify the title, short title, author order, affiliations,
contact addresses, funding, compute support, author contributions, competing
interests, repository revision, archive DOI, licenses, and acknowledgments.
Tri Nguyen's affiliation and contact address and the final disclosure package
remain unresolved in the checked-in draft.

The paper has no exploratory figure attachment plan. Any future figure must be
generated from a canonical verified artifact, add information not already clear
in the tables, and carry the same statistical and integrity boundaries as its
host paragraph.
