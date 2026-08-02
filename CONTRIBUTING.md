# Contributing

This repository is a collaborative reading notebook. Contributions should make the notes easier to verify against the source papers and easier for another reader to understand.

## Choose the right place

- Open an **Issue** for a question, disputed interpretation, missing definition, possible factual error, or idea that still needs discussion.
- Open a **Pull Request** for a concrete change to the notes that is ready for line-by-line review.

Discussion should normally begin in the issue for the relevant paper. A pull request can then link that issue when the proposed wording is ready.

## Reviewing a complete paper draft

The initial notes for Paper 1 and Paper 2 are submitted as two separate pull requests. Because each README is added as a new file, every line is available for direct review:

1. Open the pull request for the paper.
2. Select **Files changed** and open the paper's `README.md`.
3. Hover beside a line and select the comment button to ask a question or explain a concern.
4. Use a suggestion block when you have exact replacement wording.
5. Select **Review changes** to submit all comments together.

Keep broad interpretation questions in the paper's standing issue, and use inline pull-request comments for feedback tied to exact wording.

## Contribution workflow

1. Choose Paper 1 or Paper 2 and identify the relevant section or page.
2. Create a focused branch, such as `notes/paper-1-potential-functions`.
3. Edit only the relevant paper notes and any index text that must change with them.
4. Check Markdown links and GitHub math rendering.
5. Open a pull request and link the relevant review issue.
6. Request review from a collaborator before merging.

Keep each pull request focused on one topic. Small pull requests make factual and mathematical review much easier.

## Evidence standards

- Prefer the paper itself or another primary source.
- Identify the paper section, figure, table, equation, appendix, or PDF page supporting a claim.
- Clearly distinguish what the authors show from our interpretation or inference.
- Preserve important assumptions, sample sizes, uncertainty, and limitations.
- Do not turn a competitive result into a state-of-the-art claim unless the reported comparison supports it.
- Define notation before using it, and keep notation consistent within each paper page.

## Pull-request checklist

- [ ] The change is filed under the correct paper.
- [ ] New factual claims have a primary-source reference or precise paper location.
- [ ] Mathematical notation is defined before use.
- [ ] Markdown links resolve from the file containing them.
- [ ] Display and inline math render correctly on GitHub.
- [ ] The change does not mix general discussion with an unrelated notes revision.

## Repository structure

```text
README.md                                  # Two-paper index
papers/01-modelsmc/README.md               # Paper 1 notes, introduced by its review PR
papers/02-data-structure-synthesis/README.md # Paper 2 notes, introduced by its review PR
```
