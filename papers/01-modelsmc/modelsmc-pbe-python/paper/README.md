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
- `generated/` contains tables generated from archived experiment artifacts.
- `jmlr2e.sty` is the unmodified official JMLR style file at upstream commit
  `f413f638b407af76074813f8f88a82a7a5a81e9d`.

The checked-in manuscript identifies Tri Nguyen and Thanh-Dat Nguyen as
co-authors. Thanh-Dat Nguyen's verified author block lists Harvard University,
Basis Research Institute, and `datnguyen@seas.harvard.edu`. Before an arXiv or
journal submission, add Tri Nguyen's affiliation and contact address and the
authors' final funding/compute disclosure. Do not submit the exploratory pilot
table as confirmatory evidence.

The intended commands are:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The same source uses JMLR's `preprint` mode, which is suitable for arXiv. For a
journal submission, follow the current JMLR author instructions and editor
metadata requirements without modifying `jmlr2e.sty`.
