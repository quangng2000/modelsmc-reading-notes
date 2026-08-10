from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).parents[2]
PARAGRAPH_MARKER = re.compile(r"^% \[(P-[A-Z]+-[0-9]+)\]$", re.MULTILINE)
GUIDE_ROW = re.compile(r"^\| (P-[A-Z]+-[0-9]+) \|", re.MULTILINE)


def test_every_manuscript_paragraph_has_both_review_plan_rows() -> None:
    manuscript = (PROJECT / "paper" / "main.tex").read_text(encoding="utf-8")
    guide = (PROJECT / "paper" / "AUTHOR_REVIEW_GUIDE.md").read_text(
        encoding="utf-8"
    )

    manuscript_ids = PARAGRAPH_MARKER.findall(manuscript)
    guide_counts = Counter(GUIDE_ROW.findall(guide))

    assert len(manuscript_ids) == len(set(manuscript_ids)) == 33
    assert set(guide_counts) == set(manuscript_ids)
    assert set(guide_counts.values()) == {2}
