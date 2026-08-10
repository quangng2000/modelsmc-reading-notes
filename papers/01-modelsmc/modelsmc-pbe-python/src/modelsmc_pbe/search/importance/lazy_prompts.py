"""Finite-candidate prompts for factorized lazy construction."""

from __future__ import annotations

import json
from collections.abc import Sequence

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.deduction import render_deduction_guidance
from modelsmc_pbe.domain import canonical_key
from modelsmc_pbe.induction import render_hole_type, render_hypothesis

from .lazy_records import FactorizedFamily, LazyImportanceState
from .prompts import proposal_hole
from .records import HoleFilling


def _family_candidate(family: FactorizedFamily) -> str:
    return json.dumps(family.hypothesis.kind.value, ensure_ascii=False)


def _ancestor_feedback(ancestor: LazyImportanceState, *, limit: int = 6) -> str:
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


def lazy_family_prompt_prefix(
    *,
    config: ExperimentConfig,
    families: Sequence[FactorizedFamily],
    ancestor: LazyImportanceState,
    stage: int,
    beta: float,
) -> str:
    """Build a structural prompt without accessing complete support states."""

    task = config.spec.model_dump(mode="json", by_alias=True)
    choices = [
        {
            "candidate": _family_candidate(family),
            "hypothesis": render_hypothesis(family.hypothesis),
            "deduction": render_deduction_guidance(family.deduction),
            "completeConstructionTraces": family.support_count,
        }
        for family in families
    ]
    rendered_choices = json.dumps(
        choices,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Choose one viable typed structural hypothesis for a "
        "programming-by-example task.\n"
        "The continuation after SKELETON_JSON= must be exactly one candidate JSON "
        "string from ViableSkeletons: no Markdown, prose, or rationale.\n"
        f"Task={json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        f"ViableSkeletons={rendered_choices}\n"
        f"AncestorAST={canonical_key(ancestor.program)}\n"
        f"AncestorFeedback={_ancestor_feedback(ancestor)}\n"
        f"SMCStage={stage}; beta={beta:.12g}; finiteCandidates={len(families)}\n"
        "SKELETON_JSON="
    )


def lazy_hole_prompt_prefix(
    *,
    config: ExperimentConfig,
    family: FactorizedFamily,
    hole_index: int,
    ancestor: LazyImportanceState,
    previous_fillings: Sequence[HoleFilling],
    stage: int,
    beta: float,
    candidate_count: int,
) -> str:
    """Build one hole prompt from catalogs and a sampled ancestor only."""

    signature = config.spec.signature
    if signature is None:  # pragma: no cover
        raise ValueError("PBE specification has no signature")
    hole = family.hypothesis.holes[hole_index]
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
        f"Deduction=\n{render_deduction_guidance(family.deduction)}\n"
        f"Hole=?{hole.name}\n"
        f"HoleType={render_hole_type(hole)}\n"
        f"Variables={json.dumps(variables, sort_keys=True, separators=(',', ':'))}\n"
        f"AllowedIntegerConstants={json.dumps(config.spec.integer_constants)}\n"
        f"AncestorAST={canonical_key(ancestor.program)}\n"
        f"AncestorFeedback={_ancestor_feedback(ancestor)}\n"
        f"PreviousFillings={json.dumps(partial, sort_keys=True, separators=(',', ':'))}\n"
        f"SMCStage={stage}; beta={beta:.12g}; finiteCandidates={candidate_count}\n"
        "CANDIDATE_AST_JSON="
    )
