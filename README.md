# Program Synthesis and Model Discovery

Reading notes for two papers on synthesizing executable programs from evidence. The papers use very different search and inference strategies, but both ask how a system can turn an incomplete specification into a useful program.

## Paper index

| # | Paper | Main idea | Notes | Source |
| --- | --- | --- | --- | --- |
| 1 | **A Probabilistic Framework for LLM-Based Model Discovery** | Use an LLM to propose executable scientific models and Sequential Monte Carlo to weight and resample them using observed data. | [Paper 1 notes](papers/01-modelsmc/README.md) | [arXiv:2602.18266](https://arxiv.org/abs/2602.18266) |
| 2 | **Synthesizing Data Structure Transformations from Input-Output Examples** | Use typed program skeletons, deduction, and best-first enumeration to synthesize the simplest functional program consistent with examples. | [Paper 2 notes](papers/02-data-structure-synthesis/README.md) | [PLDI 2015](https://doi.org/10.1145/2737924.2737977) |

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
├── README.md
└── papers
    ├── 01-modelsmc
    │   └── README.md
    └── 02-data-structure-synthesis
        └── README.md
```

The source PDFs are linked from their paper pages rather than committed to this repository. The LLaMPPL discussion remains inside Paper 1 as related background; it is not counted as a third primary paper.

## Suggested reading path

1. Read the [Paper 2 notes](papers/02-data-structure-synthesis/README.md) for the classical synthesis vocabulary: hypotheses, holes, deduction, types, cost, and enumerative search.
2. Read the [Paper 1 notes](papers/01-modelsmc/README.md) for probabilistic model discovery: kernels, potential functions, particles, likelihood weighting, and resampling.
3. Compare what guarantees are gained or lost when moving from an explicit typed search space to open-ended LLM proposals.
