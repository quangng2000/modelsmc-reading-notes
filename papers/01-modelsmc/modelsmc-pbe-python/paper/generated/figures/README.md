# Generated publication figures

Create each figure bundle in its own new subdirectory with
`python -m research.figures`. The bundle contains PDF, SVG, high-DPI PNG,
normalized figure data, and a checksum manifest. PDF paths are already relative
to `paper/main.tex` when the CLI receives a matching `--latex-prefix`.

Do not copy partial Gate-2 figures into the manuscript. A bundle is eligible for
an explicitly exploratory paper figure only when its manifest reports both
`complete_four_checkpoint_gate2: true` and
`eligible_for_exploratory_paper_insertion: true`. This does not make the pilot
confirmatory; `confirmatory_claim_ready` remains false.
