<div align="center">
  <h1>A Probabilistic Framework for<br/>LLM-Based Model Discovery</h1>
  <p><strong>ModelSMC reading notes and intuition</strong></p>

  <p>
    <a href="https://arxiv.org/abs/2602.18266"><img alt="arXiv paper" src="https://img.shields.io/badge/arXiv-2602.18266-b31b1b?style=flat-square"></a>
    <img alt="Topic: ModelSMC" src="https://img.shields.io/badge/topic-ModelSMC-6f42c1?style=flat-square">
    <img alt="Format: reading notes" src="https://img.shields.io/badge/format-reading_notes-0969da?style=flat-square">
  </p>

  <p><em>The LLM provides informed creativity; real data controls which ideas survive.</em></p>
</div>

<p align="center">
  <a href="#core-idea">Core idea</a> ·
  <a href="#system-roles">Roles</a> ·
  <a href="#methodology">Method</a> ·
  <a href="#glossary">Glossary</a> ·
  <a href="#results">Results</a> ·
  <a href="#notes--questions">Questions</a>
</p>

---

## Core idea

ModelSMC treats scientific model discovery as Bayesian inference over executable programs:

$$
p(m \mid x_o) \propto p(x_o \mid m)p(m)
$$

| Symbol | Meaning |
| --- | --- |
| $m$ | An executable scientific model |
| $x_o$ | Observed real-world data |
| $p(x_o \mid m)$ | How well simulations from $m$ explain the data |
| $p(m \mid x_o)$ | Posterior support for model $m$ |

> [!TIP]
> The LLM generates ideas. The simulator tests them. SMC decides where to spend the next round of computation.

## System roles

| Component | What it does | Mental model |
| --- | --- | --- |
| **Transformer / LLM** | Proposes revisions to scientific code | Creative scientist |
| **Simulator + real data** | Executes and evaluates each proposed mechanism | Experiment |
| **Sequential Monte Carlo** | Maintains, weights, and resamples a population of programs | Research manager |

The Transformer is usually frozen; it is **not trained during the discovery loop**.

<details>
<summary><strong>What the LLM reads before proposing a revision</strong></summary>

- The scientific problem
- The current simulator code
- Previously discovered programs
- Numerical results and textual feedback

</details>

## Methodology

```mermaid
flowchart LR
    A["Scientific problem"] --> B["Create program particles"]
    B --> C["LLM revises code"]
    C --> D["Run every simulator"]
    D --> E["Compare with real data"]
    E --> F["Weight and resample"]
    F --> G["Give feedback"]
    G --> C
```

1. Start with observed data, domain knowledge, and a simulator.
2. Create a population of candidate programs called **particles**.
3. Ask the LLM to revise selected candidates in different ways.
4. Run every candidate simulator at different parameter values.
5. Compare its predictions with the observed data.
6. Give each candidate a likelihood-based weight.
7. Resample: copy promising candidates and remove many weak ones.
8. Return the results to the LLM and repeat.

The output is a weighted collection of plausible scientific models, not only one winner.

## Glossary

### ModelSMC

| Term | Intuition |
| --- | --- |
| **Sequential Monte Carlo (SMC)** | Approximates a target probability distribution with weighted particles, propagation, and resampling. It does not approximate the potential function itself. |
| **Particle** | In ModelSMC, one complete executable scientific program. |
| **Weight** | Relative evidence that a candidate model explains the observations. |
| **Resampling** | Rebuild the population from normalized weights so promising programs receive more descendants. |
| **Simulator** | Executable code that converts model structure, parameters, and conditions into synthetic data. |
| **Hodgkin-Huxley code** | A neuron simulator based on sodium, potassium, and leak currents; the LLM proposes additional ion-channel mechanisms. |

### Related LLaMPPL vocabulary

> [!IMPORTANT]
> Feynman-Kac Transformers, infilling, and prompt intersection come from the related **LLaMPPL** work. They use the same SMC pattern, but their particles are partial strings rather than scientific programs.

| Term | Intuition |
| --- | --- |
| **Feynman-Kac Transformer model** | Not a Transformer architecture, but a stopped and weighted Markov process built around a causal Transformer: $\left(s_0,\{M_t\},\{G_t\}\right)$. |
| **Markov kernel $M_t$** | A normalized probability distribution for the next state given the current state. |
| **Potential function $G_t$** | A nonnegative incremental path weight, such as an observation likelihood, hard condition, or importance correction. |
| **Infilling** | Generate missing text inside a template while using the surrounding known fragments as evidence. |
| **Prompt intersection** | Sample text that is simultaneously probable under multiple prompts. |
| **Resampling** | Copy promising partial strings and remove low-weight partial strings. |

### Same algorithmic pattern, different state

| Framework | One particle represents | Transformer role | Evidence |
| --- | --- | --- | --- |
| **LLaMPPL** | A partial text-generation path | Proposes or scores tokens | Text constraints and observations |
| **ModelSMC** | An entire scientific program | Proposes code revisions | Simulator fit to scientific data |

$$
\text{propose} \rightarrow \text{weight} \rightarrow \text{resample} \rightarrow \text{repeat}
$$

## Results

> [!NOTE]
> **Main result:** LLM-based scientific model discovery can be formulated as Bayesian inference over executable programs.

- ModelSMC found competitive mechanistic models across synthetic and real-world tasks.
- In the neuron experiment, high-weight programs repeatedly added an M-type slow potassium current.
- Its main value is a posterior over plausible mechanisms and their uncertainty, rather than a universal accuracy win over every baseline.

## Opportunities

- Reduce compute with adaptive particle budgets, early rejection, and cheaper simulations.
- Improve proposals with scientific retrieval and domain-specific context.
- Build more reliable, uncertainty-aware likelihood estimators.
- Test when a full SMC population materially outperforms a single LLM revision chain.

## Notes & questions

<details open>
<summary><strong>Is this inference-time probabilistic program synthesis?</strong></summary>

Approximately, yes. A frozen LLM synthesizes executable program revisions, while an outer Bayesian/SMC loop evaluates and selects them.

> **probabilistic program synthesis + simulation-based Bayesian model selection**

</details>

<details>
<summary><strong>What if the LLM never proposes the real mechanism?</strong></summary>

SMC can only select among reachable proposals. If the correct mechanism is outside the LLM's proposal support, ModelSMC cannot recover it.

</details>

<details>
<summary><strong>Where is the prior over programs?</strong></summary>

The clean Bayesian target contains $p(m)$, but the practical LLM proposal is implicit and difficult to evaluate. Prompts, proposal behavior, and likelihood-based weights therefore shape the effective prior.

</details>

---

## Sources

- [A Probabilistic Framework for LLM-Based Model Discovery](https://arxiv.org/abs/2602.18266)
- [Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs](https://arxiv.org/abs/2306.03081)
