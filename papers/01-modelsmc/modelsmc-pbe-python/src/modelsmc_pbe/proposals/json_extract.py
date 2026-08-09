"""Strict extraction of a model proposal envelope from message content."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from modelsmc_pbe.core.typecheck import Scope, infer_expression, uses_input
from modelsmc_pbe.core.types import Node, child, static_type
from modelsmc_pbe.domain.ast import (
    AstNode,
    AstTransportError,
    ProgramAst,
    canonical_key,
    normalize_program,
)
from modelsmc_pbe.proposals.base import ProgramProposal, ProposalError, ProposalRequest
from modelsmc_pbe.proposals.candidate_scoring import CandidateScoreRequest


def strip_json_fence(content: str) -> str:
    """Accept one optional JSON Markdown fence, but no surrounding prose."""

    trimmed = content.strip()
    if not trimmed.startswith("```"):
        return trimmed
    lines = trimmed.splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
        raise ProposalError("proposal has an unsupported Markdown fence")
    if lines[-1].strip() != "```":
        raise ProposalError("proposal JSON fence is not closed")
    return "\n".join(lines[1:-1]).strip()


def parse_proposal_content(
    content: str,
    request: ProposalRequest,
    *,
    source: str,
    model: str | None = None,
) -> ProgramProposal:
    """Parse exactly ``{\"program\": AST, \"rationale\": string}``."""

    if not content.strip():
        raise ProposalError("provider returned empty message content")
    try:
        decoded: Any = json.loads(strip_json_fence(content))
    except json.JSONDecodeError as error:
        raise ProposalError(f"provider content is not one valid JSON value: {error.msg}") from error
    if not isinstance(decoded, Mapping):
        raise ProposalError("proposal must be a JSON object")
    envelope = cast(Mapping[str, object], decoded)
    if set(envelope) != {"program", "rationale"}:
        raise ProposalError(
            f"proposal must contain exactly 'program' and 'rationale'; received {sorted(envelope)}"
        )
    rationale = envelope["rationale"]
    if not isinstance(rationale, str):
        raise ProposalError("proposal rationale must be a string")
    try:
        program = normalize_program(
            envelope["program"],
            allowed_integer_constants=request.integer_constants,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
    except AstTransportError as error:
        raise ProposalError(f"invalid proposal AST: {error}") from error
    return ProgramProposal(
        program=program,
        rationale=rationale.strip()[:2_000],
        source=source,
        model=model,
    )


def _canonical_object(content: str, label: str) -> Mapping[str, object]:
    if not content:
        raise ProposalError(f"{label} is empty")
    try:
        decoded: Any = json.loads(content)
    except json.JSONDecodeError as error:
        raise ProposalError(f"{label} is not one valid JSON value: {error.msg}") from error
    if not isinstance(decoded, Mapping):
        raise ProposalError(f"{label} must be a JSON object")
    record = cast(Mapping[str, object], decoded)
    if content != canonical_key(record):
        raise ProposalError(
            f"{label} must use canonical JSON: sorted keys, no insignificant whitespace"
        )
    return record


def parse_canonical_program_content(
    content: str,
    request: ProposalRequest,
) -> ProgramAst:
    """Parse one direct complete-program AST in the canonical JSON encoding."""

    record = _canonical_object(content, "canonical program proposal")
    try:
        return normalize_program(
            record,
            allowed_integer_constants=request.integer_constants,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
    except AstTransportError as error:
        raise ProposalError(f"invalid canonical program AST: {error}") from error


def parse_canonical_expression_content(
    content: str,
    request: CandidateScoreRequest,
) -> AstNode:
    """Parse and type-check one direct expression AST for a declared hole."""

    record = _canonical_object(content, "canonical expression candidate")
    wrapper = {"kind": "ExpressionProgram", "body": record}
    try:
        normalized = normalize_program(
            wrapper,
            allowed_integer_constants=request.integer_constants,
            max_depth=request.max_depth + 1,
            max_nodes=request.max_nodes + 1,
        )
    except AstTransportError as error:
        raise ProposalError(f"invalid canonical expression AST: {error}") from error
    expression = cast(AstNode, child(cast(Node, normalized), "body"))
    scope_metadata = request.hole.scope
    if not scope_metadata.input_available and uses_input(cast(Node, expression)):
        raise ProposalError("canonical expression references Input outside its declared scope")
    inferred = infer_expression(
        cast(Node, expression),
        Scope(
            input_type=static_type(scope_metadata.input_type),
            item_type=(
                None if scope_metadata.item_type is None else static_type(scope_metadata.item_type)
            ),
            accumulator_type=(
                None
                if scope_metadata.accumulator_type is None
                else static_type(scope_metadata.accumulator_type)
            ),
        ),
    )
    expected = static_type(request.hole.expected_type)
    if inferred is None:
        raise ProposalError("canonical expression is ill-typed in the declared scope")
    if inferred is not expected:
        raise ProposalError(
            f"canonical expression has type {inferred.value}; expected {expected.value}"
        )
    return expression
