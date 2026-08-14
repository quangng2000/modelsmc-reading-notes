# Publication experiment plan

Status: completed evidence integrated into the publisher-facing draft. The
fresh-blind r5 study, provider-free calibration V1, terminal diagnostic V2, and
ExeDec V2 released-data debug study are complete. The canonical 1,143-file
ExeDec study archive passed its frozen verifier and was atomically imported.
The separate original operations-evidence archive failed its frozen safe-mode
verifier and remains preserved as a failed transfer; a mode-header-normalized
local derivative is validated but explicitly noncanonical and unimported.

This plan governs the final r5-centered paper. Superseded protocols, failed
runs, developmental studies, and frozen artifact trees remain preserved but do
not define parallel manuscript narratives.

## Evidence register

### Fresh-blind r5: primary bounded-discovery result

- Design: 12 freshly generated finite-domain Filter-then-Map tasks; paired
  LLM-SMC and grammar-random acquisition arms; 29 logical complete-program
  slots per arm.
- Shared mechanics: initial law, execution evidence, hole policy, four-slot
  totalization, 0.05 recursive-grammar restart, stage targets, importance
  weights, resampling, and stopping policy.
- Result: 10/12 LLM successes versus 2/12 grammar-random successes; eight
  LLM-only discordances and none in the reverse direction; one-sided exact sign
  test `p = 1/256 = 0.00390625`.
- Post hoc multiple-attempt sensitivity: treating the observed-but-invalid r4
  diagnostic as an additional numerical look gives a two-look Bonferroni value
  of `2/256 = 0.0078125`. This is not the frozen r5 analysis; r3 had no
  numerical outcome, and r4 remains invalid and excluded.
- Interpretation: bounded system-level discovery on the frozen generator and
  domain. This does not isolate an LLM-only effect, show out-of-domain
  generalization, or establish compute or wall-clock superiority.
- Integrity: reveal-free analysis and public inventory sealed before unblinding;
  post-unblind audit verified commitments, task regeneration, and finite-domain
  target behavior.

### Provider-free calibration V1: frozen negative gate

- Design: four developmental 36,000-program supports, five particle counts, and
  32 repetitions per task--particle-count cell.
- Frozen `N=256` gate: exact-program-mass RMSE below 0.10 and signed bias within
  `[-0.03, 0.03]`.
- Result: RMSE 0.2395265 and bias +0.1130184; both gate components failed.
  Every `N=256` run discovered at least one exact program, which was descriptive
  and not a gate component.
- Interpretation: exact proposal accounting and exact-program discovery were
  insufficient for accurate finite-particle target-mass estimation under the
  frozen proposal and schedule. V1 remains failed.

### Terminal diagnostic V2: post-failure mechanism evidence

- Design: provider-free terminal-only identities and proposal-overlap
  comparisons on the same exactly enumerable supports, frozen after V1 failed
  and before new Monte Carlo draws.
- Monte Carlo grid: six arms, four tasks, `N in {256, 512}`, and 128
  repetitions per task--arm--`N` cell (6,144 runs; 2,359,296 terminal draws).
- Result: finite-state importance identities held to maximum absolute error
  `2.22e-16`. An exhaustive global top-64, `epsilon=0.50` positive control reduced pooled
  `N=512` exact-mass RMSE from 0.2792 for the sticky V1 terminal proposal to
  0.0626.
- Interpretation: poor proposal--terminal-target overlap is a material
  mechanism in the tested terminal calculation. The exhaustive control is not
  a search algorithm. V2 does not rescue V1 or validate the four-stage SMC
  recurrence.

### ExeDec V2: released-data adapter debug

- Frozen design: four deduplicated released targets, eight paired seeds per
  task, 32 paired blocks, two arms, and 64 completed runs.
- Endpoint: success on finite private debug probes; exact two-sided
  discordant-pair test; no efficacy gate.
- Hidden-probe result: LLM-SMC 17/32 versus grammar-random 3/32, with 16
  LLM-only blocks, two grammar-random-only blocks, one success in both arms,
  13 in neither, and a success-fraction difference of 0.4375. The exact
  two-sided conditional sign-test value is `0.001312255859375`.
- Other outcomes: public-example exactness 21/32 versus 5/32; 1,856 logical
  slots; 1,040 online physical scorer calls before reference enumeration
  (exhaustive public-reference evaluations excluded); 188 LLM provider calls under the cap of
  224 versus none for grammar-random; conditional mean first hidden success
  slot 12 versus 9; mean terminal ESS (out of eight particles) 4.7512 versus
  3.0567.
- Per-target hidden-probe successes, LLM/grammar: `0131` 2/0, `0245` 5/1,
  `0258` 4/1, and `0608` 6/1, each over eight repeated seed blocks.
- Classification: public and potentially contaminated released targets;
  unofficial strict Filter-then-Map adapter; finite debug probes; descriptive,
  non-confirmatory, and integrity-limited because staging custody was not
  exclusive.
- Scoring boundary: r5 and calibration V1/V2 use
  `(lambda_L, lambda_C, lambda_E) = (0.75, 0.02, 2)`, whereas ExeDec V2 task
  records use the frozen scorer defaults `(2.0, 0.15, 2)`. It is not a
  scale-matched external confirmation.
- Inference boundary: the 32 blocks are repeated seeds nested within only four
  public targets. The exact test value is descriptive, not a statistical
  significance or confirmatory result. Finite probes do not establish semantic
  equivalence.
- Provenance: the canonical study archive and analysis are verified and
  imported. The auxiliary operations-evidence archive failed the frozen
  safe-mode packaging contract on shared-FUSE tar modes even though all 149
  content hashes and cross-bindings validated. Preserve that failure. The
  three-file local derivative changes only tar mode headers and remains
  noncanonical and unimported.

## Evidence retained outside the main narrative

- Blind-confirmation r1/r2: superseded before secret preparation.
- R3: aborted before execution or provider calls because two required record
  fields were absent.
- R4: completed public execution and reveal-free analysis but failed the frozen
  sealing gate before unblinding; remains blinded and excluded. Invalidated
  diagnostic values belong only in the appendix integrity timeline.
- Developmental three-arm SMC matrix: reused tasks, one seed per task--arm, and
  descriptive only.
- Earlier evidence-frontier blind study, Gate-2 grid, staged family/hole Qwen
  proposal, joint-semantic slate, and proposed full-slate study: retained in
  artifacts and history, but removed from the rendered r5-centered paper.
- ExeDec V1: superseded before provider calls because source closure was
  incomplete.
- ExeDec V2 original operations-evidence export: failed the frozen safe-mode
  verifier and remains the authoritative failed transfer. The independently
  audited local mode-header derivative, remediation report, and remediation
  seal are preserved under a distinct noncanonical directory and were not
  imported.

No frozen protocol or artifact directory may be moved, renamed, rewritten, or
deleted as part of manuscript cleanup. Add indexes or publisher-facing
summaries instead.

## Final manuscript structure

1. Introduction and bounded claim.
2. Typed Filter-then-Map setting and execution evidence.
3. Probability-accountable four-slot repair proposal.
4. SMC targets, auxiliary-law cancellation, weighting, and resampling.
5. Studies, in order: r5; calibration V1; terminal V2; ExeDec V2 debug.
6. Results in the same order, with ExeDec explicitly classified as descriptive
   debug evidence.
7. Related work.
8. Limitations and integrity boundaries.
9. Conclusion.
10. Appendices: normalization, task-level tables, integrity timeline,
    exact-reference boundary, and artifact inventory.

The abstract must remain under 200 words and lead with r5, the failed V1 gate,
and the narrow terminal V2 diagnosis. ExeDec is not an abstract-level efficacy
result.

## Claim policy

The paper may claim:

- the four-slot totalized proposal law is explicit, normalized, full-support on
  the declared grammar, and exactly evaluable by the application;
- r5 supplies bounded paired discovery evidence for the complete LLM-shortlist
  SMC system relative to its matched grammar-acquisition arm;
- V1 failed finite-particle exact-mass calibration under its frozen gate; and
- V2 supports poor terminal proposal--target overlap as a material mechanism in
  its conditioned provider-free calculation; and
- ExeDec V2 descriptively observed 17/32 versus 3/32 hidden-probe exact blocks
  on the unofficial adapter, while treating repeated seeds, the exact test,
  finite probes, public contamination, and custody as explicit limitations.

The paper must not claim:

- a calibrated LLM posterior or calibrated finite-particle SMC procedure;
- an effect attributable to the LLM independently of interpreter, weighting,
  resampling, or other shared mechanics;
- unique-AST recovery, arbitrary-task or out-of-domain generalization;
- wall-clock, provider-compute, or exhaustive-search superiority; or
- a confirmatory, contamination-free, official ExeDec benchmark result.

## Future external-confirmation roadmap

Future work is not part of the completed paper. Any external confirmation
requires a new preregistration after the current manuscript and artifacts are
frozen. The preferred sequence is:

1. implement the full DeepCoder SSA language rather than relabel the narrow
   adapter;
2. generate fresh custodial tasks after method freeze and collision-check them
   against public syntax and behavior;
3. choose the task and paired-seed sample size from an outcome-blind latency and
   power exercise;
4. compare LLM-SMC with matched evidence-only, beam, and grammar-search arms at
   predeclared logical and provider budgets;
5. run a fresh multi-task exact-reference calibration study with task-cluster
   uncertainty and a frozen particle schedule; and
6. replicate in a second domain only after its interpreter and normalized
   grammar proposal are complete.

Detailed sample sizes, gates, and hypotheses belong in a future frozen
preregistration, not in this completed-study publication plan.

## Freeze and leakage controls for future studies

Before any confirmatory call, externally bind:

- task generator, public and hidden manifests, collision audit, and custodial
  secret handling;
- source and dependency closure, prompt templates, analysis code, model and
  tokenizer revisions;
- task, provider, proposal, resampling, and baseline seeds;
- proposal parameters, particle schedule, budgets, and stopping policy;
- invalid, duplicate, no-op, retry, and infrastructure-abort semantics; and
- hypotheses, estimands, exclusions, multiplicity correction, uncertainty
  construction, and pass gates.

The provider may receive public examples, current canonical state, typed local
grammar, and mechanically sound feedback. It must not receive hidden tests,
target syntax, generator secrets, support cardinality, exhaustive catalogs, or
comparator outcomes. Raw requests and responses must be sealed before hidden
evaluation.

## Submission blockers

- Complete Tri Nguyen's affiliation and contact metadata.
- Complete funding, compute, author-contribution, and competing-interest
  disclosures.
- Replace local artifact paths with an immutable public archive and exact source
  revision.
- Preserve the original failed ExeDec operations-evidence transfer and the
  noncanonical, unimported local derivative as distinct provenance records.
- Rebuild the PDF and visually inspect every page after the final evidence and
  metadata update.
