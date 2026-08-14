# Proposal: Execution-Guided LLM Repair for Model-Based Program Synthesis

## Summary

This project investigates how a large language model (LLM) can improve program synthesis when the candidate space is too large for exhaustive evaluation. Our initial approach used LLM likelihoods to rank candidate programs. Preliminary experiments with GPT-OSS-120B show that this ranking signal is not reliably correlated with execution correctness. We therefore propose a different role for the LLM: the interpreter evaluates programs exactly, while the LLM proposes local repairs in response to structured execution feedback.

The proposed loop is:

$$
\text{propose} \rightarrow \text{execute} \rightarrow
\text{diagnose} \rightarrow \text{repair} \rightarrow \text{execute again}.
$$

This preserves the program-synthesis target distribution while using the LLM only to improve the search proposal. A controlled pilot repaired an initially incorrect predicate and mapper in two iterations and reached a zero-loss program. The next step is to replace manually aligned feedback with a sound automatic feedback generator and evaluate the method across tasks and search budgets.

## Motivation

Let a candidate program be

$$
z=(h,p,m),
$$

where $h$ is a skeleton family, $p$ is a predicate, and $m$ is a mapping expression. Given input-output examples $D$, the synthesis target is

$$
\pi(z\mid D) \propto
\exp\left[-\alpha L(z,D)-\beta C(z)\right],
$$

where $L(z,D)$ is execution loss and $C(z)$ is program complexity.

For small finite grammars, every candidate can be executed. For large grammars, exhaustive evaluation becomes impractical. An LLM could help by defining a proposal distribution

$$
q_{\text{LLM}}(z\mid D)
$$

that concentrates computation on plausible programs. The key question is how to obtain a useful proposal without allowing model confidence to replace semantic correctness.

## Previous Method: Global LLM Ranking

The previous method presented candidates to the LLM and used token likelihoods or compatibility labels as program scores:

$$
D \rightarrow \text{LLM candidate scores}
\rightarrow \text{rank or sample}
\rightarrow \text{execute selected programs}.
$$

This method had three practical problems:

1. Large candidate catalogs exceeded the model's 4,096-token serving context.
2. Candidate likelihood was not well aligned with execution loss.
3. The model could solve a direct comparison through reasoning, but that ability did not transfer to global likelihood ranking.

On the sparse bounded-square task, 600 predicate ASTs collapse to 36 distinct predicate behaviors over the relevant integer domain. Nevertheless, the correct behavior ranked 18th of 36 under the pointwise likelihood scorer. In a guided five-candidate test, it ranked fifth. By contrast, GPT-OSS-120B selected the correct predicate in a direct two-candidate reasoning test.

These results suggest

$$
s_{\text{LLM}}(z) \not\approx -L(z,D),
$$

even when the model can articulate the correct solution in generated reasoning.

## Proposed Method: Execution-Guided Local Repair

The proposed method assigns distinct responsibilities to the interpreter and the LLM:

- The interpreter determines behavior and correctness.
- The LLM proposes a program or a local repair.
- The search algorithm maintains exploration and accounts for proposal probabilities.

At iteration $t$, the procedure is:

1. Sample or construct a candidate $z_t$ from proposal $q_t$.
2. Execute $z_t$ on the training examples.
3. Convert the mismatch into structured feedback $F_t$.
4. Freeze components that are provisionally supported by execution evidence.
5. Ask the LLM to revise one failing component.
6. Parse the revision into a grammar-valid AST and execute it.
7. Stop at zero loss or continue until the search budget is exhausted.

Formally,

$$
q_{t+1}(z)=q_{\text{LLM}}(z\mid D,F_{1:t}).
$$

The feedback changes where the algorithm searches, but not the target:

$$
\pi(z\mid D) \propto
\exp\left[-\alpha L(z,D)-\beta C(z)\right].
$$

To preserve support, the learned proposal can be mixed with a grammar-based exploratory proposal:

$$
q_t(z)=(1-\varepsilon)q_{\text{LLM},t}(z)
+\varepsilon q_{\text{grammar}}(z),
\qquad \varepsilon>0.
$$

If importance sampling or SMC is used, sampled programs retain the usual correction:

$$
\widetilde w_t(z)=\frac{\pi(z\mid D)}{q_t(z)}.
$$

Thus the LLM guides search without changing what counts as a good program.

## Structured Execution Feedback

Raw scalar loss is often too weak to guide repair. The feedback generator should instead report observable discrepancies such as:

- expected and actual output lengths;
- the first mismatching output position;
- missing, extra, or incorrect output values;
- input elements whose current mapped values can be aligned soundly;
- constraints supported by multiple examples;
- which program component may be revised.

For example, suppose the current program keeps $\{0,1,2\}$ and maps $x\mapsto x^2$. On

$$
[-3,-1,0,2,4]\rightarrow[1,0,4],
$$

it produces $[0,4]$. A feedback record may state that the result is one element short and that retaining $-1$ would add $(-1)^2=1$ at the missing position. The model can then revise the predicate while the mapper remains frozen.

Feedback must be derived only from the provided training examples and interpreter traces. It must not expose a hidden reference program, held-out behavior, correct catalog index, or oracle-only annotation.

## Preliminary Results

The bounded-square grammar contains:

$$
600\text{ predicates}\times60\text{ mappers}=36{,}000
$$

complete filter-map programs. Exhaustive execution required approximately 0.90 seconds and found exactly two zero-loss ASTs:

$$
(-2<x)\land(x<3),\qquad (x<3)\land(-2<x),
$$

both with

$$
m(x)=x^2.
$$

This exhaustive run validates the grammar, interpreter, and target solution, but does not demonstrate an LLM advantage because this particular space is small.

We then conducted a controlled execution-guided repair test starting from a program in which both components were wrong:

$$
p_0=\{0,1,2\},\qquad m_0(x)=x.
$$

The first feedback step froze the predicate and reported mapped-value evidence:

$$
0\mapsto0,\qquad1\mapsto1,\qquad2\mapsto4.
$$

GPT-OSS-120B revised the mapper to

$$
m_1(x)=x\times x.
$$

Execution then passed three of five examples but remained one output short on two examples. The second feedback step froze the mapper and identified the missing contribution from $-1$. The model revised the predicate mask from

$$
00011100\rightarrow00111100,
$$

which represents retaining $-1,0,1,2$. The final program passed all five examples:

$$
L(z_2,D)=0.
$$

This pilot demonstrates that an LLM can use structured interpreter feedback to repair multiple program components sequentially.

## Current Limitation

The pilot used manually prepared alignment evidence, including facts such as $2\mapsto4$ and the diagnosis that retaining $-1$ would repair a missing output. The LLM did not discover all of these constraints from an unstructured interpreter trace.

Consequently, the current result is a proof of concept for the repair interface, not yet evidence that the full method improves synthesis autonomously. The main technical requirement is a sound feedback generator that extracts only logically justified constraints and represents ambiguity when multiple alignments are possible.

## Proposed Experimental Plan

### Phase 1: Automatic feedback generation

Implement a sequence-alignment procedure between candidate traces and expected outputs. It should return a set of possible explanations rather than committing to an unsupported alignment. Each explanation will record:

- retained input positions;
- candidate mapped values;
- insertions, deletions, and substitutions;
- constraints implied for predicate or mapper holes;
- an ambiguity score or set of alternative alignments.

All suggested repairs will be re-executed, so unsound LLM conclusions cannot be accepted as correct programs.

### Phase 2: Alternating repair policy

At each iteration, select one hole to revise using interpreter evidence. Compare:

1. predicate-first repair;
2. mapper-first repair;
3. LLM-selected repair order;
4. multiple-particle repair in which different hypotheses revise different holes.

### Phase 3: Integration with SMC

Represent each particle as a program plus its repair history. Use the LLM to propose local mutations, execute each mutated particle, compute the unchanged target weight, and resample. Retain a nonzero grammar proposal mixture to preserve exploration.

### Phase 4: Evaluation

Evaluate on grammars large enough that exhaustive search is expensive. Primary metrics should include:

- success probability within a fixed execution budget;
- number of executed programs before the first zero-loss solution;
- wall-clock time and LLM token cost;
- best execution loss versus iteration;
- fraction of LLM repairs that are grammar-valid;
- probability assigned to exact programs when this is measurable;
- robustness across seeds and example permutations.

Required baselines are:

- uniform grammar sampling;
- standard grammar-based SMC without an LLM;
- one-shot LLM program generation;
- global LLM candidate ranking;
- exhaustive search where feasible;
- execution-guided repair without LLM reasoning, using random local mutations.

An ablation should compare scalar-loss feedback, raw expected/actual outputs, automatically aligned feedback, and oracle-aligned feedback. The oracle-aligned condition is an interface ceiling and must not be reported as an autonomous method.

## Research Questions

1. Does structured execution feedback increase the probability that an LLM proposes a lower-loss program?
2. Does alternating hole repair reduce the number of executions required to reach a solution?
3. How much feedback can be derived soundly when output alignment is ambiguous?
4. Does maintaining multiple repair hypotheses outperform committing to one diagnosis?
5. At what grammar size does execution-guided LLM repair become more efficient than exhaustive enumeration?
6. Can the proposal be normalized or logged sufficiently to support valid importance weighting?

## Expected Contribution

The intended contribution is not another LLM-based correctness score. It is an execution-grounded proposal mechanism for probabilistic program synthesis:

$$
\boxed{\text{The interpreter evaluates; the LLM repairs; SMC manages uncertainty.}}
$$

If successful, the method would connect semantic program execution with the LLM's ability to interpret localized failure evidence, while retaining a well-defined synthesis target and explicit exploration guarantees.

