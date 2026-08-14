# Paper Paragraph Plan

This is the editorial skeleton for `main.tex`. It records what each rendered
paragraph is meant to do, how it should open, how it should close, and how it
hands the argument to the next paragraph. It is an internal planning document,
not part of the submitted manuscript.

Aligned source:

- `main.tex` SHA-256: `238841f2f43627894b26a03878c57f04d63a497464492a127a568b93d0c2c40a`
- Reviewed manuscript commit: `68c4a7601e720b56ec8888e7938308248af0cf03`

## Paper-level argument

1. Define the attribution problem created by learned search guidance.
2. Show why importance-corrected SMC needs an evaluable proposal.
3. Define the application-computed shortlist proposal and its correction.
4. Test whether LLM shortlist acquisition improves bounded discovery.
5. Test calibration separately and retain the negative result.
6. Use a terminal-only diagnostic to study the failure mechanism.
7. Report ExeDec only as a released-data adapter stress test.
8. End with the narrow discovery claim, the calibration qualification, and the
   evidence still needed.

## Paragraph construction rule

Every paragraph should have one main job.

- **Opening:** name the paragraph's object, question, or claim directly.
- **Middle:** define, justify, or quantify only that opening claim.
- **Closing:** interpret the paragraph or expose the next unresolved question.
- **Handoff:** repeat one key concept from the close in the next opening, without
  repeating the same sentence structure.
- **Claim discipline:** keep a caveat beside the result it qualifies. Do not
  defer all qualifications to a generic limitations paragraph.

## Abstract

### `P-ABS-01` — Complete evidentiary arc

- **Purpose:** Compress the problem, method, primary discovery result,
  calibration failure, terminal diagnosis, and claim boundary into one arc.
- **Opening move:** State the probability-accounting requirement created by
  importance-corrected SMC.
- **Closing move:** State positively what the evidence supports, then distinguish
  it from calibration, general synthesis, and speed.
- **Handoff:** The abstract's attribution problem becomes the first problem
  developed in the Introduction.
- **Guardrail:** Bind 128/128 discovery to the `N=256` V1 cell and bind the
  overlap diagnosis to the conditioned terminal step.

## 1. Introduction

### `P-INT-01` — PBE motivation

- **Purpose:** Introduce PBE, underdetermination, and combinatorial typed search.
- **Opening move:** Define the input-output task in plain terms.
- **Closing move:** Note that learned guidance can reorder search but makes the
  source of improvement harder to identify.
- **Handoff:** Search guidance creates the attribution problem in `P-INT-02`.

### `P-INT-02` — Unit of evaluation

- **Purpose:** Explain why the LLM cannot be evaluated apart from the surrounding
  interpreter, selector, and search procedure.
- **Opening move:** Name attribution as the problem created by learned guidance.
- **Closing move:** Define the complete model-interpreter-SMC system as the unit
  of evaluation, with proposal, target, budget, and comparator separated.
- **Handoff:** SMC makes this accounting requirement mathematical in `P-INT-03`.

### `P-INT-03` — Proposal-probability gap

- **Purpose:** Introduce the particle view and identify the missing evaluable
  canonical-program proposal probability.
- **Opening move:** Say that SMC makes the accounting requirement precise.
- **Closing move:** Explain why a free-form response alone does not supply the
  probability required by the interpreter's canonical AST.
- **Handoff:** `P-INT-04` introduces the mechanism that closes this gap.

### `P-INT-04` — Core DPC idea

- **Purpose:** Explain the application-authoritative transition and the
  auxiliary-variable cancellation at a conceptual level.
- **Opening move:** State that DPC addresses the probability gap without calling
  model output a program distribution.
- **Closing move:** Emphasize that the implemented transition is evaluable even
  though the marginal LLM probability is not recovered.
- **Handoff:** The evaluable transition becomes the replayable methodological
  contribution in `P-INT-05`.

### `P-INT-05` — Method contribution

- **Purpose:** Name the components that make the recorded shortlist replayable.
- **Opening move:** State the contribution as a bounded mechanism, not a broad
  synthesis claim.
- **Closing move:** Describe the recorded shortlist as a replayable proposal.
- **Handoff:** `P-INT-06` turns that proposal into three empirical questions.

### `P-INT-06` — Evidence plan

- **Purpose:** Separate discovery, calibration, diagnosis, and adapter stress
  testing before any results appear.
- **Opening move:** Say explicitly that the proposal is evaluated through
  separate questions.
- **Closing move:** State the paper's central distinction: better discovery can
  coexist with poor target approximation.
- **Handoff:** Related Work locates this contribution before the setting and
  method are formalized.

## 2. Related Work

### `P-REL-01` — Symbolic and learned PBE

- **Purpose:** Place DPC between symbolic constraints and learned search
  priorities.
- **Opening move:** Name the three intersecting traditions.
- **Closing move:** Narrow the gap to the sampling law induced by a finite
  learned shortlist.
- **Handoff:** The sampling-law problem motivates the particle-method context.

### `P-REL-02` — Particle inference

- **Purpose:** Distinguish the contribution from prior uses of SMC.
- **Opening move:** Acknowledge standard SMC tools and prior synthesis work.
- **Closing move:** Identify the four-slot proposal and exact-reference
  calibration test as the new combination.
- **Handoff:** `P-REL-03` addresses the final neighboring line of work, ExeDec.

### `P-REL-03` — ExeDec boundary

- **Purpose:** Explain why the local adapter is not a reproduction of full
  ExeDec or the complete DeepCoder language.
- **Opening move:** State what ExeDec studies.
- **Closing move:** Label the adapter as a released-data stress test and point to
  the narrower filter-map setting that follows.
- **Handoff:** `P-BG-01` defines that setting. If sections move, revise the word
  “below.”

## 3. Typed Filter-Map Synthesis

### `P-BG-01` — Task and semantic scope

- **Purpose:** Define exactness, the bounded typed DSL, structural cost, and the
  declared-domain claim boundary.
- **Opening move:** Introduce the example set and exactness criterion.
- **Closing move:** Require each study to state its evaluation domain.
- **Handoff:** `P-SYM-01` supplies the finite confirmatory domain and skeleton.

### `P-SYM-01` — Confirmatory program family

- **Purpose:** Restrict the primary study to an auditable filter-map skeleton and
  normalized grammar.
- **Opening move:** State the restriction directly, then display the skeleton.
- **Closing move:** Clarify that the provider need not enumerate the support.
- **Handoff:** Execution evidence can now be defined hole by hole.

### `P-SYM-02` — Public execution evidence

- **Purpose:** Define when examples imply predicate or mapper facts and what the
  harness records.
- **Opening move:** State that a hole is constrained only when its effect can be
  isolated.
- **Closing move:** Limit the provider to public execution evidence.
- **Handoff:** `P-SYM-03` assigns validity and execution authority to the
  application.

### `P-SYM-03` — Application-provider boundary

- **Purpose:** Make the application authoritative for parsing, types, execution,
  caching, validity, and success.
- **Opening move:** State that all validity decisions remain with the
  application.
- **Closing move:** Identify the remaining technical problem: the law from which
  repairs are sampled.
- **Handoff:** `P-PROP-SHORT-01` defines that law.

## 4. Probability-Accountable Repair Shortlists

### `P-PROP-SHORT-01` — Totalized proposal definition

- **Purpose:** Convert four response slots into a fixed empirical law and mix it
  with the grammar restart.
- **Opening move:** Name the current program, selected hole, evidence, and four
  requested slots.
- **Closing move:** Let the displayed equations carry the definition of `H_S`
  and `q`.
- **Handoff:** `P-PROP-SHORT-02` proves the operational properties of that law.

### `P-PROP-SHORT-02` — Normalization and failures

- **Purpose:** Establish full support over the declared bounded grammar and
  explain duplicate, retry, and totalization behavior.
- **Opening move:** Derive normalization and positivity from the grammar law.
- **Closing move:** State the epsilon regimes used by the completed studies.
- **Handoff:** The next paragraph explains why the provider-response law need not
  be evaluated.

### `P-PROP-SHORT-03` — Acquisition law as auxiliary variable

- **Purpose:** Define `R_t^a` for LLM, grammar-random, and deterministic oracle
  acquisition without calling it a marginal program probability.
- **Opening move:** Treat the raw response as an auxiliary variable.
- **Closing move:** Explain exact-parent absorption while retaining restart
  support.
- **Handoff:** `P-PROP-SHORT-04` shows how the shared acquisition law cancels.

### `P-PROP-SHORT-04` — Cancellation and proposal variants

- **Purpose:** State the cancellation and distinguish four-slot laws from the
  terminal V2 top-64 law.
- **Opening move:** Put the same `R_t^a` in target and proposal.
- **Closing move:** Reaffirm that the method accounts for the sampled transition,
  not the LLM's marginal program probability.
- **Handoff:** Correct accounting leaves efficiency unresolved.

### `P-PROP-SHORT-05` — Accountability versus efficiency

- **Purpose:** Explain why evaluable probabilities do not ensure low-variance
  importance weights.
- **Opening move:** State that an evaluable sampling law can still be inefficient.
- **Closing move:** Point first to `q` as the SMC denominator, then to separate
  discovery and calibration endpoints.
- **Handoff:** `P-SMC-01` begins the particle correction; the Studies section
  later realizes the second forward link.

## 5. Importance-Corrected SMC

### `P-SMC-01` — Stage targets

- **Purpose:** Define violation counts, capped loss, program cost, study-specific
  scales, and the four stage densities.
- **Opening move:** Name the three ingredients of the target.
- **Closing move:** Bound the densities as finite computational targets rather
  than real-world likelihoods or posteriors.
- **Handoff:** `P-SMC-02` places these densities and the shortlist law on an
  extended state space.

### `P-SMC-02` — Extended-space importance potential

- **Purpose:** Write the target and proposal, cancel `R`, and expose the
  incremental potential.
- **Opening move:** Announce the extended state space containing realized
  shortlists.
- **Closing move:** State that only `q` must be evaluated for the realized
  shortlist.
- **Handoff:** `P-SMC-03` applies the potential to scheduled children.

### `P-SMC-03` — Particle update and terminal marginal

- **Purpose:** Define child weights, offspring-mass allocation, resampling, and
  the terminal target.
- **Opening move:** Move from the potential to each scheduled child.
- **Closing move:** Separate the terminal target from the product-path
  normalizer.
- **Handoff:** `P-SMC-04` turns that conceptual separation into work accounting.

### `P-SMC-04` — Online versus reference work

- **Purpose:** Prevent exhaustive reference computation from being credited as
  online search or discovery.
- **Opening move:** State that online search and exact reference work are counted
  separately.
- **Closing move:** Exclude reference evaluations and their discoveries from the
  online account.
- **Handoff:** `P-EXP-01` assigns discovery and calibration to separate studies.

## 6. Studies

### `P-EXP-01` — Study roadmap

- **Purpose:** Order the four evidential components without pooling them.
- **Opening move:** Begin with the r5 bounded-discovery comparison.
- **Closing move:** Place ExeDec outside the confirmatory/calibration sequence as
  a limited adapter stress test.
- **Handoff:** The next subsection starts with r5 task generation.

### `P-EXP-R5-01` — Fresh task generation

- **Purpose:** Establish that method and analysis were fixed before a fresh
  secret seeded 12 independent target draws.
- **Opening move:** State the freeze-before-generation order.
- **Closing move:** Define public coverage: all domain singletons plus list
  examples.
- **Handoff:** `P-EXP-R5-02` records provider identity, settings, and custody.

### `P-EXP-R5-02` — Provider and custody

- **Purpose:** Bind the model revision, decoding limits, timeout, concurrency,
  no-retry rule, and private-material boundary.
- **Opening move:** Identify GPT-OSS-120B at the sealed revision.
- **Closing move:** End at the public-seal-before-private-material boundary.
- **Handoff:** With provenance fixed, `P-EXP-R5-03` defines the matched arms.

### `P-EXP-R5-03` — Matched acquisition comparison

- **Purpose:** Show that the two arms share every mechanism except shortlist
  acquisition.
- **Opening move:** State execution order and shared components.
- **Closing move:** Interpret the design as a comparison between acquisition
  mechanisms inside the complete shared system.
- **Handoff:** `P-EXP-R5-04` specifies budget, endpoint, and analysis unit.

### `P-EXP-R5-04` — Budget and primary analysis

- **Purpose:** Derive 29 logical slots, define the public endpoint, relate it to
  finite-domain behavior, and specify the paired sign test.
- **Opening move:** Build the slot schedule from the initial execution and four
  stages.
- **Closing move:** State full-budget execution with no outcome-dependent stop.
- **Handoff:** The next subsection asks the separate calibration question.

### `P-EXP-CAL-01` — V1 tasks and deterministic oracle

- **Purpose:** Define the provider-free 36,000-program supports and local
  public-evidence ranking oracle.
- **Opening move:** State that V1 was frozen separately and had no provider.
- **Closing move:** Deny LLM or search-baseline status to the oracle.
- **Handoff:** `P-EXP-CAL-02` defines particle counts, repetitions, and gate.

### `P-EXP-CAL-02` — V1 grid and gate

- **Purpose:** Specify `N`, repetition count, `1+4N` budget, no early stop, and
  both `N=256` thresholds.
- **Opening move:** Define the particle grid and repetitions.
- **Closing move:** End with the exact RMSE and bias criteria, not outcomes.
- **Handoff:** `P-EXP-CAL-03` exists because V1 later fails those criteria.

### `P-EXP-CAL-03` — Terminal V2 diagnostic design

- **Purpose:** Define the post-failure, pre-draw terminal diagnostic and its six
  proposal arms.
- **Opening move:** State the order: V1 failure, then V2 freeze, then new draws.
- **Closing move:** Limit V2 to the conditioned terminal calculation and leave
  V1 and full-SMC validity unchanged.
- **Handoff:** The final design subsection is a separate adapter stress test.

### `P-EXP-EXEDEC-V2-01` — Adapter design

- **Purpose:** Define four released targets, strict adapter subset, nested seeds,
  finite private-debug endpoint, and absence of an efficacy gate.
- **Opening move:** Label the exercise descriptive from the first sentence.
- **Closing move:** State the exact two-sided test and no-gate status.
- **Handoff:** `P-EXP-EXEDEC-V2-02` explains why the result cannot be elevated.

### `P-EXP-EXEDEC-V2-02` — Adapter evidence boundary

- **Purpose:** Attach public-data contamination, finite-probe, nonexclusive
  custody, and packaging-verifier limits to the design.
- **Opening move:** State that targets and examples are released and potentially
  present in training data.
- **Closing move:** Keep the study at adapter-debug status and point to the
  integrity appendix.
- **Handoff:** `P-RES-01` preserves this separate status when results begin.

## 7. Results

### `P-RES-01` — Results boundary

- **Purpose:** State the headline split before presenting any numbers.
- **Opening move:** Attribute bounded discovery only to r5 and calibration failure
  only to V1.
- **Closing move:** Keep ExeDec separate because of released data and repeated
  seeds.
- **Handoff:** The first result subsection reports r5's primary comparison.

### `P-RES-R5-01` — Primary r5 result

- **Purpose:** Report the paired discovery result, frozen decision, descriptive
  checkpoint, and system-level interpretation.
- **Opening move:** Lead with 10/12 versus 2/12 by slot 29.
- **Closing move:** Attribute the contrast to complete systems differing only in
  acquisition, not to the LLM alone.
- **Handoff:** `P-RES-R5-02` reports the work needed to obtain that result.
- **Guardrail:** Preserve eight LLM-favoring discordances, one-sided
  `p=1/256=0.00390625`, both frozen thresholds, and descriptive slot-21 8 versus
  2.

### `P-RES-R5-02` — r5 resource account

- **Purpose:** Report slots, scorer calls, cache hits, provider calls, tokens,
  and response totalization.
- **Opening move:** Start from all 24 trajectories and 696 logical slots.
- **Closing move:** End with the two output-cap responses becoming four sentinels
  each, with no retry or backfill.
- **Handoff:** `P-RES-R5-03` reports the post-unblind semantic audit.
- **Guardrail:** Never turn 367 physical calls, 329 cache hits, 58 provider
  calls, or 137,759 tokens into a speed claim.

### `P-RES-R5-03` — Post-unblind semantic audit

- **Purpose:** Establish the reveal order and distinguish syntax identity from
  finite-domain semantic aliases.
- **Opening move:** State what the audit verified after public sealing.
- **Closing move:** Bound the claim to finite-domain behavior, not unique AST or
  out-of-domain semantics.
- **Handoff:** The calibration subsection asks a different question from semantic
  recovery.
- **Guardrail:** Preserve the 415-file inventory and the 4 syntax-identical / 8
  alias split across 12 successful task-arm trajectories.

### `P-RES-CAL-01` — Failed V1 gate

- **Purpose:** Report the negative calibration decision while retaining discovery
  as descriptive.
- **Opening move:** State that both `N=256` gate components failed despite
  128/128 discovery in that cell.
- **Closing move:** Say that discovery did not answer the calibration question.
- **Handoff:** The table supplies exact criteria and outcomes; `P-RES-CAL-02`
  examines the particle grid.
- **Guardrail:** RMSE `0.2395265` and bias `+0.1130184` both fail their frozen
  criteria.

### `P-RES-CAL-02` — Behavior across particle counts

- **Purpose:** Report the finite RMSE grid and reject an unsupported reference
  rate claim.
- **Opening move:** List RMSE in increasing `N` order.
- **Closing move:** Limit insufficiency to this proposal, schedule, and four-task
  suite.
- **Handoff:** `P-RES-CAL-03` investigates a mechanism rather than revising the
  decision.
- **Guardrail:** Do not infer that larger `N` generally worsens SMC or claim a
  universal convergence rate.

### `P-RES-CAL-03` — Terminal mechanism evidence

- **Purpose:** Report exact identity validation and the top-64 positive control.
- **Opening move:** Lead with maximum identity error
  `2.22 x 10^-16`.
- **Closing move:** Label the control diagnostic rather than practical search and
  leave V1 failed.
- **Handoff:** The next subsection resets to ExeDec; do not force a causal bridge.
- **Guardrail:** Preserve `epsilon=0.50`, pooled `N=512` RMSE `0.2792` to
  `0.0626`, and the conditioned-terminal-step scope.

### `P-RES-EXEDEC-V2-01` — Descriptive private-probe result

- **Purpose:** Report paired-block outcomes and attach the inference boundary to
  the exact test value.
- **Opening move:** Establish 64 runs and 32 complete paired blocks.
- **Closing move:** Explicitly reject statistical-significance and confirmation
  interpretations.
- **Handoff:** `P-RES-EXEDEC-V2-02` reports public success and resource counts.
- **Guardrail:** Preserve 17/32 versus 3/32, difference 0.4375, cells 16/2/1/13,
  18 discordances, and two-sided `p=0.001312255859375`.

### `P-RES-EXEDEC-V2-02` — ExeDec online work

- **Purpose:** Report public exactness, logical slots, online physical scorer
  calls, and provider calls.
- **Opening move:** Lead with public-example exactness 21/32 versus 5/32.
- **Closing move:** Keep reference enumeration outside the online count and state
  zero grammar provider calls.
- **Handoff:** `P-RES-EXEDEC-V2-03` reports conditional summaries.
- **Guardrail:** Preserve 1,856 logical slots, 1,040 online physical calls before
  references, and 188 provider calls under cap 224.

### `P-RES-EXEDEC-V2-03` — Conditional ExeDec summaries

- **Purpose:** Report first-success slots and terminal ESS without implying speed
  or semantic equivalence.
- **Opening move:** State explicitly that the slot means condition on probe
  success.
- **Closing move:** Deny semantic equivalence from finite probes.
- **Handoff:** The target-level table completes the descriptive account before
  Limitations.
- **Guardrail:** Preserve conditional means 12 versus 9 and terminal ESS 4.7512
  versus 3.0567 out of eight.

## 8. Limitations and Integrity

### `P-LIM-01` — r5 scope

- **Purpose:** Bound task family, domain, model revision, comparator, and model
  replication.
- **Opening move:** Call the setting narrow and enumerate it.
- **Closing move:** Reject arbitrary-PBE, out-of-domain, and independent-model
  replicate interpretations.
- **Handoff:** `P-LIM-02` adds the distinct calibration limit.

### `P-LIM-02` — Calibration scope

- **Purpose:** Keep V1 failed and V2 terminal-only while naming what exact
  proposal accounting does not establish.
- **Opening move:** State calibration as an additional limit.
- **Closing move:** Explain that exact references depend on small enumerable
  supports.
- **Handoff:** `P-LIM-03` moves from calibration to the ExeDec evidence boundary.

### `P-LIM-03` — ExeDec scope and custody

- **Purpose:** Collect the adapter's public-data, nested-seed, finite-probe,
  scale, custody, and packaging limits in one place.
- **Opening move:** State that its evidential value is narrower than r5.
- **Closing move:** Make clear that the noncanonical derivative removes none of
  the integrity limits.
- **Handoff:** `P-LIM-04` closes with cross-study compute and safety limits.

### `P-LIM-04` — Compute, safety, and history

- **Purpose:** Deny equal-compute and speed interpretations and require sandboxed
  review of generated code.
- **Opening move:** Separate logical-slot equality from wall time, provider cost,
  and total compute.
- **Closing move:** Point to the failed and superseded protocol record.
- **Handoff:** The Conclusion can now state the positive result without hiding
  its limits.

## 9. Conclusion

### `P-CON-01` — Positive bounded contribution

- **Purpose:** Restate the mechanism and the narrow r5 discovery advantage.
- **Opening move:** Describe the application-computed, declared-grammar proposal
  and explicit SMC correction.
- **Closing move:** Deny LLM-only, speed, and broad-generalization claims.
- **Handoff:** `P-CON-02` qualifies discovery with calibration.

### `P-CON-02` — Calibration qualification and next evidence

- **Purpose:** Make the negative calibration result coequal with the positive
  discovery result and identify the next study.
- **Opening move:** State that calibration qualifies discovery.
- **Closing move:** Require proposal redesign, recalibration, and independently
  selected external tasks while keeping discovery, accuracy, and cost separate.
- **Guardrail:** V2 supports poor overlap only in the conditioned terminal step,
  leaves V1 failed, and does not validate full SMC.

## Acknowledgments

### `P-ACK-01` — Disclosure placeholder

- **Purpose:** Prevent an incomplete draft disclosure from appearing final.
- **Opening move:** State only currently known funding status.
- **Closing move:** List the author-supplied disclosures still required.
- **Handoff:** None; this is a submission-completeness block, not part of the
  scientific argument.
- **Guardrail:** Do not invent affiliation, funding, hardware, compute, conflict,
  or contribution information.

## Appendix A. Proposal Normalization

### `P-APP-NORM-01` — Normalization proof boundary

- **Purpose:** Prove normalization and declared-grammar positivity for four-slot
  and top-64 laws, then restate the cancellation.
- **Opening move:** Begin from the fixed cardinality of the totalized shortlist.
- **Closing move:** Distinguish proposal-law validity from finite-particle
  calibration.
- **Handoff:** Appendix B moves from proof detail to task-level evidence.

## Appendix B. Task-Level Results

### `P-APP-R5-TASK-01` — r5 table guide

- **Purpose:** Define every r5 table column and the provider-totalized-slot
  terminology.
- **Opening move:** State that every sealed task-arm trajectory is reported.
- **Closing move:** Note that grammar-random made no provider calls.
- **Handoff:** The table supplies row-level evidence; the next paragraph explains
  calibration heterogeneity.

### `P-APP-CAL-TASK-01` — Calibration table guide

- **Purpose:** Explain why per-task values are needed alongside the pooled V1
  failure.
- **Opening move:** Name task heterogeneity as the reason for the table.
- **Closing move:** Define RMSE and bias as 32 self-normalized estimates per task
  at `N=256`.
- **Handoff:** Appendix C moves from outcomes to protocol history.

## Appendix C. Preserved Integrity and Failure Timeline

### `P-APP-INTEGRITY-01` — Blind-confirmation chronology

- **Purpose:** Record r1-r5 status without turning invalid runs into evidence.
- **Opening move:** State that revisions are included because they affect
  evidential status.
- **Closing move:** End with the r5 sealer-only change, retained scientific method,
  and fresh secret/suite.
- **Handoff:** `P-APP-INTEGRITY-02` records calibration and ExeDec revision paths.
- **Guardrail:** Keep r4 blinded and excluded; its 11/12 versus 2/12 and 10/12
  versus 0/12 diagnostics are history only. Preserve post hoc Bonferroni
  0.0078125 as sensitivity, not the frozen r5 analysis.

### `P-APP-INTEGRITY-02` — Calibration and adapter revisions

- **Purpose:** Record V1/V2 calibration status and ExeDec V1/V2 source-closure
  history.
- **Opening move:** Separate these revision histories from blind confirmation.
- **Closing move:** End with canonical ExeDec V2 import and run-integrity checks.
- **Handoff:** `P-APP-INTEGRITY-03` documents the remaining operations archive
  failure.

### `P-APP-INTEGRITY-03` — Operations-evidence failure

- **Purpose:** Preserve the original safe-mode failure, content validation, and
  noncanonical derivative as distinct records.
- **Opening move:** Name the exact verifier failure and path.
- **Closing move:** State that no original, derivative, or protocol record is
  rewritten to fit the narrative.
- **Handoff:** Appendix D separates reference computation from online evidence.

## Appendix D. Exact-Reference Claim Boundary

### `P-APP-ORACLE-01` — Reference-work interpretation

- **Purpose:** Explain what enumeration supplies and why it is not synthesis
  speed or online discovery.
- **Opening move:** Name existence checking and target functionals as its two
  purposes.
- **Closing move:** Deny practical-search status to the positive control.
- **Handoff:** Appendix E gives readers the exact public locations of evidence
  and source.

## Appendix E. Artifact Availability

### `P-APP-AVAIL-01` — Reproducibility locator

- **Purpose:** Bind the manuscript to public data, checksums, failure records,
  implementation source, and exact revisions.
- **Opening move:** Separate the Hugging Face data-and-paper companion from the
  GitHub source-and-tests branch.
- **Closing move:** Explain how repository-relative paths connect the source
  checkout and companion artifact tree.
- **Handoff:** The forced page break leads to References on page 13.
- **Guardrail:** Preserve canonical and noncanonical artifact status and every
  published hash exactly; do not imply that the failed operations-evidence
  archive passed or was imported.

## Maintenance checklist

When `main.tex` changes:

1. Update the matching paragraph entry here if its purpose, opening, close, or
   handoff changes.
2. Keep paragraph IDs unique and adjacent to their rendered paragraph.
3. Recheck all Results guardrails against the sealed analyses.
4. Rebuild the PDF and visually inspect every affected page.
5. Update the aligned `main.tex` SHA-256 at the top of this file.
6. Do not use this planning document as a source for numerical claims; the
   sealed analyses and protocols remain authoritative.
