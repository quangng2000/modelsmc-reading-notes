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
fresh-blind r5 discovery, negative calibration V1, terminal diagnostic V2, and a
separately marked ExeDec V2 debug study. Do not restore the staged family/hole
Qwen proposal, joint-semantic slate, Gate-2 grid, blind evidence-frontier V2,
developmental matrix, or unexecuted full-slate study to the rendered paper.

## Front matter and motivation

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-ABS-01` | State the proposal, r5 result, failed V1 gate, terminal V2 diagnosis, and limits in under 200 words. | ExeDec is not an abstract-level result; do not imply calibrated SMC or speed. |
| `P-INT-01` | Locate PBE among symbolic and learned search methods. | Gulwani, Feser, Kalyan, HYSYNTH, and LLM-guided enumeration. |
| `P-INT-02` | Define the system-attribution problem. | Treat model, interpreter, selector, target, and comparator as separate components. |
| `P-INT-03` | Explain why SMC needs an evaluable proposal law. | Free-form text probability is not canonical-program probability. |
| `P-INT-04` | Introduce application totalization and auxiliary-law cancellation. | Claim an accountable implemented transition, not an inferred LLM AST probability. |
| `P-INT-05` | State the completed evidence sequence and bounded ExeDec debug result. | Discovery and calibration remain distinct; ExeDec stays descriptive and integrity-limited. |

## Setting, proposal, and SMC

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-BG-01` | Define PBE exactness, bounded DSL, structural cost, and semantic scope. | Example exactness is not out-of-domain correctness. |
| `P-SYM-01` | Define the strict Filter-then-Map skeleton and normalized recursive grammar. | Do not imply support enumeration is required online. |
| `P-SYM-02` | Explain sound singleton evidence and provider-visible information. | Hidden targets, secrets, catalogs, and comparator outcomes remain unavailable. |
| `P-SYM-03` | Make the application authoritative for parsing, typing, execution, and caching. | The provider neither validates programs nor supplies confidence. |
| `P-PROP-SHORT-01` | Define four-slot totalization and `q=(1-epsilon)H_S+epsilon g`. | `epsilon=0.05` for r5, V1, ExeDec V2, and sticky V2; V2 also has prespecified alternatives. Preserve failed slots. |
| `P-PROP-SHORT-02` | Define acquisition law `R` and its exact cancellation. | Four-slot arms use Equation 2; V2 alternatives have declared evaluable laws, including the top-64 control's 64-slot empirical law. |
| `P-PROP-SHORT-03` | Separate proposal correctness from overlap and finite-particle accuracy. | This paragraph must set up the negative calibration result. |
| `P-SMC-01` | Define the four declared stage targets and frozen scales. | `L_D` is capped soft execution loss; r5/V1/V2 use `(0.75,0.02,2)`, ExeDec uses scorer defaults `(2.0,0.15,2)`; targets are not scale-matched. |
| `P-SMC-02` | Define the initial-program-conditioned extended target, proposal, and incremental potential. | The same acquisition factor must occur in numerator and denominator. |
| `P-SMC-03` | Define child weights, resampling, and terminal/path distinction. | Do not conflate the product-path normalizer with the terminal target. |
| `P-SMC-04` | Separate r5/ExeDec online execution from V1/V2 and ExeDec reference work. | ExeDec's online physical-call count excludes public-reference enumeration; oracle discoveries are never online synthesis. |

## Study methods

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-EXP-01` | Introduce the evidence layers in their final order. | R5, V1, V2, then ExeDec debug; no parallel paper fork. |
| `P-EXP-R5-01` | State fresh task generation, model binding, provider settings, and blind custody. | R5 and ExeDec calls share frozen settings and no retries; long hashes belong in the appendix. |
| `P-EXP-R5-02` | Define paired seeds/arms, the `1+4+3x8=29` slot schedule, endpoint, and analysis unit. | Acquisition seeds differ; shared sampling and resampling seeds make the pairing explicit. |
| `P-EXP-CAL-01` | Define V1's provider-free design, `N`, `1+4N` schedule, and frozen gate. | The ranking oracle is exhaustive and not an LLM or search baseline. |
| `P-EXP-CAL-02` | Define terminal V2's six-arm, two-`N`, 128-repetition post-failure design. | It cannot rescue V1 or validate the four-stage recurrence. |
| `P-EXP-EXEDEC-V2-01` | Define the four-target, eight-seed paired debug design. | Local strict Filter-then-Map adapter; not an official ExeDec split; no efficacy gate. |
| `P-EXP-EXEDEC-V2-02` | Disclose public-data, finite-probe, contamination, custody, and transfer limits. | The study archive is canonical and imported; the original operations-evidence export failed, and its derivative is noncanonical and unimported. Never say that two imports passed. |

## Results

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-RES-01` | Preserve the methods/results ordering and endpoint separation. | Never pool r5 discovery, calibration, or ExeDec debug outcomes. |
| `P-RES-R5-01` | Report 10/12 versus 2/12, the exact paired test, and post hoc two-look sensitivity. | Primary `p=1/256`; `0.0078125` is not the frozen analysis and treats invalid r4 as another look; r4 remains excluded. |
| `P-RES-R5-02` | Report slots, physical scoring, cache reuse, provider work, and totalized failures. | Logical and physical work are different quantities; no speed claim. |
| `P-RES-R5-03` | Report sealing and semantic-alias audit. | Finite-domain recovery is not unique-AST or out-of-domain recovery. |
| `P-RES-CAL-01` | State that V1 failed despite exact-program discovery in all 128 `N=256` runs. | Both gate components remain visible; do not generalize discovery to the full 640-run grid. |
| `P-RES-CAL-02` | Report the particle-count RMSE sequence. | Do not claim an `N^{-1/2}` rate or that more particles intrinsically worsen SMC. |
| `P-RES-CAL-03` | Report V2 identities and global top-64, `epsilon=0.50` mechanism evidence. | The exhaustive positive control is not search and does not revise V1. |
| `P-RES-EXEDEC-V2-01` | Report 17/32 versus 3/32, paired cells, difference, and exact two-sided value. | Eight seeds are nested within only four public targets; the exact value is descriptive, not a significance or confirmation claim. |
| `P-RES-EXEDEC-V2-02` | Report public exactness, online work before reference enumeration, provider calls, conditional first success, and ESS. | The 1,040 count excludes public-reference evaluations; finite probes and slot summaries imply neither equivalence nor speed. |
| `P-RES-EXEDEC-V2-03` | State canonical study integrity and the operations-evidence qualification. | Valid content audit and the noncanonical derivative do not erase the original verifier failure or create exclusive custody. |

## Interpretation and close

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-REL-01` | Position against symbolic, neural-guided, and LLM-guided PBE. | The narrow contribution is application-computed shortlist probability. |
| `P-REL-02` | Position against SMC synthesis and model discovery. | Do not claim first use of SMC; emphasize exact accounting plus a tested calibration failure. |
| `P-REL-03` | Position the ExeDec adapter against the actual ExeDec contribution. | No complete DSL, official split, or execution-decomposition claim. |
| `P-LIM-01` | Bound r5 by task count, generator, integer domain, model, and comparator. | No task-population or independent model-replicate inference. |
| `P-LIM-02` | Preserve the negative calibration interpretation. | V2 is conditioned terminal evidence only. |
| `P-LIM-03` | State ExeDec, compute, safety, and integrity limits. | Point once to the appendix timeline; do not duplicate failed-run numbers. |
| `P-CON-01` | Conclude the positive bounded r5 result. | No LLM-only, speed, or broad-generalization conclusion. |
| `P-CON-02` | Conclude with the negative calibration result and next evidentiary need. | Keep discovery, distributional accuracy, and compute separate. |
| `P-ACK-01` | Record the draft disclosure status. | Replace before submission with complete funding, compute, author, and conflict disclosures. |

## Appendices

| ID | Primary job | Required evidence or boundary |
|---|---|---|
| `P-APP-NORM-01` | Prove four-slot normalization and note V2's analogous 64-slot control law. | Acquisition cancellation is shared; none of this is a finite-particle calibration proof. |
| `P-APP-R5-TASK-01` | Introduce all r5 task--arm records. | Totals must reproduce 10/12 versus 2/12 and the failure-accounting text. |
| `P-APP-CAL-TASK-01` | Expose V1 task heterogeneity. | Values must reproduce the pooled gate and retain discovery as descriptive. |
| `P-APP-INTEGRITY-01` | Preserve r1/r2, r3, r4, and r5 chronology. | R4 values are invalidated history, not evidence. |
| `P-APP-INTEGRITY-02` | Preserve V1/V2, ExeDec V1, and the verified V2 study import chronology. | The 1,143-file study archive and canonical analysis are distinct from operations evidence. |
| `P-APP-INTEGRITY-03` | Preserve the original operations-evidence failure and local derivative status. | All content/cross-bindings validated, but the original failed; the derivative is noncanonical and unimported. |
| `P-APP-ORACLE-01` | State the exact-reference claim boundary. | Exhaustive controls do not establish online discovery or speed. |
| `P-APP-AVAIL-01` | List canonical study, analysis, inventory, failed operations transfer, and derivative locators. | Hashes must match the local artifacts; never label the derivative canonical or imported. Replace local paths before publication. |

## Section-level handoff

- **Abstract:** proposal -> r5 discovery -> failed V1 gate -> narrow terminal V2
  diagnosis -> claim boundary.
- **Introduction:** PBE ambiguity -> attribution problem -> proposal-density gap
  -> totalized shortlist solution -> evidence sequence.
- **Setting:** typed skeleton -> sound evidence -> authoritative application.
- **Proposal:** totalize slots -> cancel unknown acquisition law -> warn about
  overlap.
- **SMC:** stage targets -> extended correction -> child/resampling geometry ->
  online/reference boundary.
- **Studies and Results:** r5 -> V1 -> V2 -> ExeDec debug, in the same order.
- **Limitations:** narrow r5 scope -> negative calibration -> ExeDec and compute
  boundaries.
- **Conclusion:** bounded positive discovery -> central negative calibration ->
  external confirmation.
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
