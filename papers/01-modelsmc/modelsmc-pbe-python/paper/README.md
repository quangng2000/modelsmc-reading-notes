# Paper source

This directory contains the JMLR-style, arXiv-compatible manuscript for the
ModelSMC-PBE study.

- `main.tex` is the clean manuscript. Paragraph identifiers appear only as
  LaTeX comments and are not rendered.
- `AUTHOR_REVIEW_GUIDE.md` is a separate paragraph-level argument map for
  author review. It records each paragraph's purpose, evidence, dependency,
  and likely reviewer challenge.
- `references.bib` contains the primary literature cited by the manuscript.
- `generated/` contains tables generated from archived experiment artifacts.
- `jmlr2e.sty` is the unmodified official JMLR style file at upstream commit
  `f413f638b407af76074813f8f88a82a7a5a81e9d`.

The checked-in manuscript is an anonymous research draft. Before an arXiv or
journal submission, replace the anonymous author block with author names,
affiliations, a corresponding address, and an appropriate funding/compute
disclosure. Do not submit the exploratory pilot table as confirmatory evidence.

The intended commands are:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The same source uses JMLR's `preprint` mode, which is suitable for arXiv. For a
journal submission, follow the current JMLR author instructions and editor
metadata requirements without modifying `jmlr2e.sty`.
