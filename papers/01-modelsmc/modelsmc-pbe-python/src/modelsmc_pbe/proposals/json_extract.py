"""Strict extraction of a model proposal envelope from message content."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from modelsmc_pbe.domain.ast import AstTransportError, normalize_program
from modelsmc_pbe.proposals.base import ProgramProposal, ProposalError, ProposalRequest


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
