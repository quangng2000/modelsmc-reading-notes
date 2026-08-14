"""Prompt and feedback construction for black-box program proposers."""

from __future__ import annotations

import json

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.search.models import ParticleRecord

_GRAMMAR_GUIDE = """
Complete program wrappers:
- {"kind":"ExpressionProgram","body": EXPR}
- {"kind":"MapProgram","mapper": EXPR}; mapper scope provides Item
- {"kind":"FoldRightProgram","initial": EXPR,"reducer": EXPR}; reducer scope
  provides Item and Accumulator

Expression nodes:
- leaves: Input, Item, Accumulator, EmptyIntList, EmptyBoolList
- literals: IntLiteral with decimal-string intValue; BoolLiteral with boolValue
- arithmetic/comparison: Add, Subtract, Multiply, LessThan, EqualInt, And
- unary boolean: Not
- lists: PrependInt or PrependBool with head and tail
- conditional: IfThenElse with condition, thenExpr, and elseExpr

Every node is a JSON object with exactly the fields described by its kind.
Only use integer literals from integerConstants. Return a complete program,
not a hole, lambda-calculus string, source-code expression, or Markdown fence.
""".strip()


def feedback_for(score: ScoredProgram, *, limit: int = 8) -> str:
    """Produce compact, deterministic counterexample feedback."""

    failures = [evaluation for evaluation in score.evaluations if not evaluation.exact]
    if not failures:
        return (
            f"Exact on all {len(score.evaluations)} examples; inferred type "
            f"{score.inferred_type}; structural cost {score.cost}."
        )
    lines = [
        f"Validated type {score.inferred_type}; structural cost {score.cost}; "
        f"exact examples {score.exact_matches}/{len(score.evaluations)}; "
        f"total loss {score.total_loss:g}."
    ]
    for index, evaluation in enumerate(failures[:limit], start=1):
        lines.append(
            f"Counterexample {index}: input {evaluation.input}; predicted "
            f"{evaluation.predicted}; expected {evaluation.expected}; "
            f"loss {evaluation.loss:g}."
        )
    if len(failures) > limit:
        lines.append(f"{len(failures) - limit} additional failures omitted.")
    return "\n".join(lines)


def proposal_prompt(
    config: ExperimentConfig,
    ancestor: ParticleRecord,
    *,
    iteration: int,
    slot: int,
) -> str:
    """Describe the PBE task and exact revision context for one particle."""

    spec = config.spec.model_dump(mode="json", by_alias=True)
    ancestor_json = json.dumps(ancestor.program, sort_keys=True, separators=(",", ":"))
    ancestry_json = json.dumps(
        [step.as_dict() for step in ancestor.ancestry],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"""
Synthesize a unary program for this programming-by-example task.

Task specification:
{json.dumps(spec, ensure_ascii=False, indent=2)}

This is ModelSMC refinement iteration {iteration}, population slot {slot}.
Revise the ancestor rather than merely describing a possible family.

Ancestor AST:
{ancestor_json}

Scorer feedback for the ancestor:
{ancestor.feedback}

Bounded ancestor trajectory, oldest to newest:
{ancestry_json}

Transport grammar:
{_GRAMMAR_GUIDE}

Goal: propose a well-typed, low-cost AST that exactly matches every example.
The semantic scorer—not your rationale—will determine acceptance.
""".strip()
