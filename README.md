# Program Synthesis and Model Discovery

Reading notes for two papers on synthesizing executable programs from evidence. The papers use very different search and inference strategies, but both ask how a system can turn an incomplete specification into a useful program.

## Paper index

| # | Paper | Main idea | Review status | Source |
| --- | --- | --- | --- | --- |
| 1 | **A Probabilistic Framework for LLM-Based Model Discovery** | Use an LLM to propose executable scientific models and Sequential Monte Carlo to weight and resample them using observed data. | Full draft proposed in its own pull request | [arXiv:2602.18266](https://arxiv.org/abs/2602.18266) |
| 2 | **Synthesizing Data Structure Transformations from Input-Output Examples** | Use typed program skeletons, deduction, and best-first enumeration to synthesize the simplest functional program consistent with examples. | Full draft proposed in its own pull request | [PLDI 2015](https://doi.org/10.1145/2737924.2737977) |

## How the papers connect

| Question | Paper 1: ModelSMC | Paper 2: Lambda Learner |
| --- | --- | --- |
| What is synthesized? | Scientific simulator programs | Functional list and tree transformations |
| What specifies the goal? | Observed scientific data, context, priors, and prompts | Input-output examples, types, primitives, and a cost model |
| How are candidates proposed? | LLM-based program revision | Type-aware hypotheses and enumerative search |
| How are candidates evaluated? | Likelihood-based potential functions | Deductive consistency checks against examples |
| How does search focus? | Particle weighting and resampling | Best-first expansion by program cost |
| What is returned? | A weighted population of plausible models | A minimum-cost program satisfying the examples |

The shared pattern is:

```text
specification or evidence
        -> propose candidate programs
        -> evaluate candidates
        -> focus search on promising candidates
        -> return an executable program or distribution over programs
```

Paper 2 provides a classical program-synthesis reference point: the language and search space are explicit, and correctness is checked against examples. Paper 1 replaces the fixed symbolic proposal mechanism with open-ended LLM revisions and replaces exact example consistency with probabilistic evidence from simulation.

## Repository layout

```text
.
├── .github
│   ├── ISSUE_TEMPLATE
│   └── pull_request_template.md
├── CONTRIBUTING.md
└── README.md
```

The source PDFs are linked above rather than committed to this repository. Each paper's complete notes are intentionally introduced through a separate pull request. This makes the whole draft visible in GitHub's **Files changed** view so reviewers can comment on individual lines before the notes are merged.

## Suggested reading path

1. Start with Paper 2 for the classical synthesis vocabulary: hypotheses, holes, deduction, types, cost, and enumerative search.
2. Continue with Paper 1 for probabilistic model discovery: kernels, potential functions, particles, likelihood weighting, and resampling.
3. Compare what guarantees are gained or lost when moving from an explicit typed search space to open-ended LLM proposals.

## Collaboration

- Discuss [Paper 1 in issue #2](https://github.com/quangng2000/modelsmc-reading-notes/issues/2) and [Paper 2 in issue #1](https://github.com/quangng2000/modelsmc-reading-notes/issues/1).
- Review each initial paper draft in its own pull request. Open **Files changed** to comment on a specific README line or suggest replacement wording.
- Open a [new paper-review issue](https://github.com/quangng2000/modelsmc-reading-notes/issues/new?template=paper-review.md) for a separate question, disputed interpretation, possible correction, or missing definition.
- Use a focused branch and pull request for a concrete notes change that is ready for line-by-line review.
- Follow the evidence, notation, and review guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

General rule: discuss the claim in an issue, propose exact wording in a pull request, and ask a collaborator to review before merging.
