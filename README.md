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

#### What exactly is a Feynman-Kac Transformer model?

A Feynman-Kac Transformer is **not a new neural-network architecture**. It is a stopped, weighted Markov process built around a generative Transformer $f_\theta$:

$$
\left(s_0, \{M_t\}_{t \ge 1}, \{G_t\}_{t \ge 1}\right).
$$

For a realized stopped path, the product below is its **unnormalized path weight**:

$$
\widetilde W(s_{0:T})=
\prod_{t=1}^{T}
M_t(s_t \mid s_{t-1}, f_\theta)
G_t(s_{t-1}, s_t, f_\theta).
$$

The paper defines a normalized filtering distribution at step $t$,

$$
P_t(s)=
\frac{
\mathbb E_M\!\left[
\left(\prod_{i=1}^{t\wedge T}G_i(S_{i-1},S_i,f_\theta)\right)
\mathbf 1[S_t=s]
\right]
}{
\mathbb E_M\!\left[
\prod_{i=1}^{t\wedge T}G_i(S_{i-1},S_i,f_\theta)
\right]
},
\qquad
P(s)=\lim_{t\to\infty}P_t(s).
$$

- $f_\theta$ maps every unfinished string to vocabulary logits.
- $M_t$ is a normalized transition distribution over string states. It need not append exactly one token: it can append several tokens or insert known fragments, provided the state contains enough information for the process to remain Markov.
- EOS-terminated strings are absorbing, and the process must reach EOS with probability $1$.
- $G_t$ is a nonnegative, computable incremental weight, and the resulting weighted distribution must have a finite, nonzero normalizer.

##### Markov kernel: notation and properties

In this paper the state space is discrete, so a Markov kernel is simply a conditional probability mass function over the next string state.

| Notation | Meaning |
| --- | --- |
| $V$ | The model's token vocabulary, including EOS. |
| $S=V^*$ | All finite token strings that can be states of the process. |
| $F \subseteq S$ | Finished strings: states whose final token is EOS. |
| $F^{\mathrm c}=S\setminus F$ | Unfinished strings from which generation may continue. |
| $S_t$ / $s_t$ | The random state at step $t$ / one realized value of that state. |
| $x$ | A fixed prompt, when it is kept separate from the generated string state. |
| $f_\theta:F^{\mathrm c}\to\mathbb R^{\lvert V\rvert}$ | The generative Transformer, mapping an unfinished string to next-token logits. |
| $M_t(s'\mid s,f_\theta)$ | The probability of moving from current state $s$ to next state $s'$ at step $t$. |
| $\delta_s(s')$ | A point mass: $1$ when $s'=s$ and $0$ otherwise. |
| $T=\inf\{t\ge 0:S_t\in F\}$ | The first step at which the process reaches an EOS-terminated state. |

For every unfinished state $s$, a valid kernel is nonnegative and normalized:

$$
M_t(s'\mid s,f_\theta)\ge 0,
\qquad
\sum_{s'\in S}M_t(s'\mid s,f_\theta)=1.
$$

The **Markov property** says that, after the current state and step are known, earlier states provide no additional information about the next transition:

$$
\Pr(S_t=s_t\mid S_{0:t-1}=s_{0:t-1},f_\theta)
=M_t(s_t\mid s_{t-1},f_\theta).
$$

Using the entire current string as the state does not mean the model forgets its prefix; the prefix is already contained in $S_{t-1}$. If a generation procedure uses additional history that cannot be recovered from the string and $t$, that information must also be included in the state for the process to remain Markov.

Other useful properties are:

- **Time-inhomogeneous transitions are allowed.** The subscript $t$ means that different steps may use different kernels, as in infilling.
- **A step need not equal one token.** A normalized kernel may append one token, append multiple tokens, or add deterministic fragments around sampled tokens.
- **Finished states are absorbing.** $M_t$ is defined for $s\in F^{\mathrm c}$; the stopped chain extends the transition rule on $s\in F$ with $\delta_s(s')$, so the string cannot change after EOS.
- **Termination is an assumption.** The construction requires $\Pr(T<\infty)=1$; a kernel that can generate forever with positive probability does not satisfy the paper's setup.
- **The kernel is a proposal, not the final target.** $M_t$ determines how particles move. $G_t$ reweights those moves, and SMC uses the resulting weights to decide which partial paths receive descendants.
- **It is not an MCMC transition kernel.** No stationary distribution or invariance property is required here.
- **The factorization is not unique.** Different pairs $(M_t,G_t)$ can define the same posterior but yield very different weight variance and finite-particle efficiency.

For ordinary next-token generation, define

$$
\pi_\theta(w\mid xs)=\mathrm{softmax}(f_\theta(xs))_w.
$$

The corresponding append-one-token kernel is

$$
M_t(s'\mid s,f_\theta)
=\sum_{w\in V}\pi_\theta(w\mid xs)\,\mathbf 1[s'=sw].
$$

As a contrasting example, let $C$ be the set of prefixes that can still lead to a valid complete string. The probability that an ordinary LM transition remains inside $C$ is

$$
Z_C(s)=\sum_{u\in V}\pi_\theta(u\mid xs)\,\mathbf 1[su\in C].
$$

When $Z_C(s)>0$, a token-masked proposal is

$$
M'_t(sw\mid s,f_\theta)
=\frac{\pi_\theta(w\mid xs)\,\mathbf 1[sw\in C]}{Z_C(s)}.
$$

This is a valid locally normalized kernel, but it changes the path distribution. To preserve the original language-model probabilities on allowed paths, the potential must restore the removed normalizer:

$$
G'_t(s,sw,f_\theta)=Z_C(s).
$$

On the proposal's support, $M'_tG'_t=\pi_\theta(w\mid xs)$. If $Z_C(s)=0$, the masked proposal is undefined unless the program handles that dead state explicitly. This is why choosing a legal next token and conditioning the complete generated string are not generally the same operation.

In a LLaMPPL program, one step's potential can combine several factors:

$$
G_t =
\prod_{\text{sample sites}}
\frac{p(v)}{q(v)}
\times
\prod_{\text{observations}} p(y)
\times
\prod_{\text{conditions}} \mathbf{1}[c].
$$

These are, respectively, importance corrections, observation likelihoods, and hard constraints. Therefore, $G_t$ is more than a local preference score.

If $M_t$ is ordinary temperature-1 next-token sampling from the Transformer and reaches EOS almost surely, setting every $G_t=1$ recovers ordinary generation. If $M_t$ is modified, for example by masking illegal tokens, $G_t=1$ instead targets the modified locally normalized proposal. An importance correction is generally needed to preserve the original language model conditioned on the complete constraint.

This definition does not directly cover an encoder-only Transformer, which has no next-token transition kernel. It also assumes access to next-token logits. A black-box generator may still serve as a proposal in a more general SMC construction, but it cannot directly provide corrections or likelihood factors that require token probabilities.

> **Compressed:** the Transformer supplies probabilities, $M_t$ proposes state transitions, $G_t$ corrects and scores them, and SMC approximates the resulting global posterior.

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
