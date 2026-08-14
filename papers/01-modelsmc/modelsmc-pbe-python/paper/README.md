# Paper source

This directory contains the JMLR-style, arXiv-compatible manuscript for the
ModelSMC-PBE study.

- `main.tex` is the clean manuscript. Paragraph identifiers appear only as
  LaTeX comments and are not rendered.
- `AUTHOR_REVIEW_GUIDE.md` is a separate paragraph-level argument map for
  author review. It records each paragraph's purpose, opening-sentence job,
  closing-sentence handoff, evidence, dependency, and likely reviewer
  challenge.
- `references.bib` contains the primary literature cited by the manuscript.
- `generated/` contains tables and provider-free figure bundles generated from
  archived aggregate artifacts. Figure bundles include vector PDF/SVG,
  high-DPI PNG, normalized data, and a checksum manifest.
- `jmlr2e.sty` is the unmodified official JMLR style file at upstream commit
  `f413f638b407af76074813f8f88a82a7a5a81e9d`.

The checked-in manuscript identifies Tri Nguyen and Thanh-Dat Nguyen as
co-authors. Thanh-Dat Nguyen's verified author block lists Harvard University,
Basis Research Institute, and `datnguyen@seas.harvard.edu`. Before an arXiv or
journal submission, add Tri Nguyen's affiliation and contact address and the
authors' final funding, compute, contribution, and competing-interest
disclosures. The independently verified ExeDec V2 study import is complete, and
the manuscript reports the released-data result as debug-only and
integrity-limited. The separate original operations-evidence export failed its
frozen safe-mode verifier; its mode-header-normalized derivative is validated
but noncanonical and unimported. Do not describe this as two successful imports
or as confirmatory ExeDec evidence.

The calibration narrative preserves failed V1 and fresh V2, the conditioned
terminal and reused-task diagnostics, and the authenticated second-fresh V3 R2
pass as one chronology. V3 confirms only provider-free terminal exact-program
mass on the declared singleton-complete finite-support law. It does not
calibrate the LLM, four-stage SMC, full PBE, a large DSL, or target mean loss.
The exact protocol, method-seal, custody, analysis, replay, and unblind hashes
are listed in the manuscript's Artifact Availability appendix.

The intended commands are:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The same source uses JMLR's `preprint` mode, which is suitable for arXiv. For a
journal submission, follow the current JMLR author instructions and editor
metadata requirements without modifying `jmlr2e.sty`.
