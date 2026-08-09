"""Strict decoding of vLLM teacher-forced prompt-logprob responses."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import httpx

from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.candidate_scoring import CandidateSequenceScore


@dataclass(frozen=True, slots=True)
class _PromptPath:
    leading_unscored: int
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]


def decode_candidate_scores(
    response: httpx.Response,
    prompts: Sequence[str],
    candidates: Sequence[str],
    expressions: Sequence[AstNode],
) -> tuple[CandidateSequenceScore, ...]:
    """Decode one complete teacher-forced prompt per canonical candidate."""

    choices = _decode_choices(response, prompts)
    return tuple(
        _candidate_score(
            candidate,
            expression,
            _parse_prompt_path(choices[index].get("prompt_logprobs"), f"choices[{index}]"),
        )
        for index, (candidate, expression) in enumerate(zip(candidates, expressions, strict=True))
    )


def _decode_choices(
    response: httpx.Response, prompts: Sequence[str]
) -> dict[int, Mapping[str, object]]:
    if response.is_error:
        digest = hashlib.sha256(response.content).hexdigest()
        raise ProposalError(
            f"vLLM returned HTTP {response.status_code} "
            f"(body bytes={len(response.content)}, sha256={digest})"
        )
    try:
        body = response.json()
    except (ValueError, TypeError) as error:
        raise ProposalError("vLLM candidate-score response is not valid JSON") from error
    raw_choices = body.get("choices") if isinstance(body, Mapping) else None
    if not isinstance(raw_choices, list) or len(raw_choices) != len(prompts):
        raise ProposalError("vLLM response does not contain one choice per scored prompt")
    choices: dict[int, Mapping[str, object]] = {}
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, Mapping):
            raise ProposalError("vLLM response choice must be an object")
        choice = cast(Mapping[str, object], raw_choice)
        index = choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in choices:
            raise ProposalError("vLLM response choice indices must be unique integers")
        if index < 0 or index >= len(prompts) or choice.get("text") != prompts[index]:
            raise ProposalError("vLLM response choice does not match its echoed prompt")
        choices[index] = choice
    if len(choices) != len(prompts):
        raise ProposalError("vLLM response choice indices are incomplete")
    return choices


def _parse_prompt_path(value: object, path: str) -> _PromptPath:
    if not isinstance(value, list) or not value:
        raise ProposalError(f"{path}.prompt_logprobs must be a nonempty array")
    leading_unscored = 0
    token_ids: list[int] = []
    logprobs: list[float] = []
    for index, raw_entry in enumerate(value):
        if raw_entry is None:
            if index != 0:
                raise ProposalError(f"{path}.prompt_logprobs may omit only the first token score")
            leading_unscored = 1
            continue
        if not isinstance(raw_entry, Mapping) or len(raw_entry) != 1:
            raise ProposalError(
                f"{path}.prompt_logprobs[{index}] must contain exactly the observed token"
            )
        raw_token_id, raw_detail = next(iter(raw_entry.items()))
        token_id = _parse_token_id(raw_token_id, f"{path}.prompt_logprobs[{index}]")
        if not isinstance(raw_detail, Mapping):
            raise ProposalError(f"{path}.prompt_logprobs[{index}] detail must be an object")
        raw_logprob = raw_detail.get("logprob")
        if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
            raise ProposalError(f"{path}.prompt_logprobs[{index}].logprob must be a number")
        logprob = float(raw_logprob)
        if not math.isfinite(logprob) or logprob > 0:
            raise ProposalError(
                f"{path}.prompt_logprobs[{index}].logprob must be finite and nonpositive"
            )
        token_ids.append(token_id)
        logprobs.append(logprob)
    return _PromptPath(leading_unscored, tuple(token_ids), tuple(logprobs))


def _parse_token_id(value: object, path: str) -> int:
    if not isinstance(value, str):
        raise ProposalError(f"{path} token ID must be a JSON object key")
    digits = value.removeprefix("token_id:")
    if not digits.isdigit():
        raise ProposalError(f"{path} has invalid token ID {value!r}")
    return int(digits)


def _candidate_score(
    candidate: str,
    expression: AstNode,
    full: _PromptPath,
) -> CandidateSequenceScore:
    if not full.token_ids:
        raise ProposalError("candidate prompt must contain at least one scored token")
    return CandidateSequenceScore(
        candidate=candidate,
        expression=expression,
        token_ids=full.token_ids,
        token_logprobs=full.logprobs,
        sequence_logprob=math.fsum(full.logprobs),
    )
