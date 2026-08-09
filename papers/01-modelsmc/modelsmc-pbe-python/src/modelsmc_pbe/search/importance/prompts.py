"""Deterministic hole-only prompts for finite Qwen candidate scoring."""

from __future__ import annotations

import json
from collections.abc import Sequence

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.deduction import DeductionReport, render_deduction_guidance
from modelsmc_pbe.domain import ValueType, canonical_key
from modelsmc_pbe.induction import HoleSpec, render_hole_type
from modelsmc_pbe.proposals import ExpressionScope, HoleSpecification

from .records import HoleFilling, ImportanceState


def proposal_hole(hole: HoleSpec, *, input_type: ValueType) -> HoleSpecification:
    """Translate symbolic hole variables into the transport-scoring scope."""

    parameters = {parameter.name: parameter.value_type for parameter in hole.parameters}
    return HoleSpecification(
        expected_type=hole.output_type,
        scope=ExpressionScope(
            input_type=input_type,
            input_available="x" in parameters or "xs" in parameters,
            item_type=parameters.get("item"),
            accumulator_type=parameters.get("acc"),
        ),
    )


def _feedback(ancestor: ImportanceState, *, limit: int = 6) -> str:
    failures = [item for item in ancestor.score.evaluations if not item.exact]
    if not failures:
        return (
            f"exact on all examples; structural cost={ancestor.score.cost}; "
            f"family={ancestor.family}"
        )
    lines = [
        f"loss={ancestor.score.total_loss:g}; cost={ancestor.score.cost}; "
        f"exact={ancestor.score.exact_matches}/{len(ancestor.score.evaluations)}"
    ]
    lines.extend(
        f"input {item.input}: predicted {item.predicted}, expected {item.expected}"
        for item in failures[:limit]
    )
    return "\n".join(lines)


def hole_prompt_prefix(
    *,
    config: ExperimentConfig,
    report: DeductionReport,
    hole: HoleSpec,
    ancestor: ImportanceState,
    previous_fillings: Sequence[HoleFilling],
    stage: int,
    beta: float,
    candidate_count: int,
) -> str:
    """Build the common prefix whose continuation is exactly one candidate AST."""

    signature = config.spec.signature
    if signature is None:  # pragma: no cover - specification validation fills it
        raise ValueError("PBE specification has no signature")
    task = config.spec.model_dump(mode="json", by_alias=True)
    partial = {
        filling.hole_name: json.loads(canonical_key(filling.expression))
        for filling in previous_fillings
    }
    variables = proposal_hole(hole, input_type=signature.input_type).scope.as_prompt_record()
    return (
        "Complete one typed expression hole for a programming-by-example task.\n"
        "The continuation after CANDIDATE_AST_JSON= must be exactly one canonical "
        "JSON expression AST: no Markdown, prose, rationale, or wrapper.\n"
        f"Task={json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        f"Deduction=\n{render_deduction_guidance(report)}\n"
        f"Hole=?{hole.name}\n"
        f"HoleType={render_hole_type(hole)}\n"
        f"Variables={json.dumps(variables, sort_keys=True, separators=(',', ':'))}\n"
        f"AllowedIntegerConstants={json.dumps(config.spec.integer_constants)}\n"
        f"AncestorAST={canonical_key(ancestor.program)}\n"
        f"AncestorFeedback={_feedback(ancestor)}\n"
        f"PreviousFillings={json.dumps(partial, sort_keys=True, separators=(',', ':'))}\n"
        f"SMCStage={stage}; beta={beta:.12g}; finiteCandidates={candidate_count}\n"
        "CANDIDATE_AST_JSON="
    )
