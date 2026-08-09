from __future__ import annotations

import asyncio
import json
from typing import cast

import httpx
import pytest

from modelsmc_pbe.domain import ValueType, canonical_key
from modelsmc_pbe.proposals import (
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
    ExpressionScope,
    HoleSpecification,
    ProposalError,
    ProposalRequest,
    VLLMPromptLogprobConfig,
    VLLMPromptLogprobScorer,
)
from modelsmc_pbe.proposals.json_extract import (
    parse_canonical_expression_content,
    parse_canonical_program_content,
)

ITEM = {"kind": "Item"}
ADD_ONE = {
    "kind": "Add",
    "left": {"kind": "Item"},
    "right": {"kind": "IntLiteral", "intValue": "1"},
}
SUB_ONE = {
    "kind": "Subtract",
    "left": {"kind": "Item"},
    "right": {"kind": "IntLiteral", "intValue": "1"},
}


def _request(*candidates: object) -> CandidateScoreRequest:
    return CandidateScoreRequest(
        prompt_prefix="Task and grammar\nCanonical expression: ",
        candidates=tuple(canonical_key(cast(dict[str, object], value)) for value in candidates),
        hole=HoleSpecification(
            expected_type=ValueType.INT,
            scope=ExpressionScope(
                input_type=ValueType.INT_LIST,
                input_available=False,
                item_type=ValueType.INT,
            ),
        ),
        integer_constants=(0, 1),
    )


def _entry(token_id: int, logprob: float) -> dict[str, object]:
    return {
        str(token_id): {
            "logprob": logprob,
            "rank": 1,
            "decoded_token": f"token-{token_id}",
        }
    }


def test_canonical_expression_parser_enforces_encoding_type_and_scope() -> None:
    request = _request(ADD_ONE)
    canonical = request.candidates[0]

    assert parse_canonical_expression_content(canonical, request) == ADD_ONE

    with pytest.raises(ProposalError, match="canonical JSON"):
        parse_canonical_expression_content(json.dumps(ADD_ONE), request)
    with pytest.raises(ProposalError, match="outside its declared scope"):
        parse_canonical_expression_content(canonical_key({"kind": "Input"}), request)
    with pytest.raises(ProposalError, match="expected IntType"):
        parse_canonical_expression_content(
            canonical_key({"kind": "BoolLiteral", "boolValue": True}), request
        )
    with pytest.raises(ProposalError, match="allowed constant"):
        parse_canonical_expression_content(
            canonical_key({"kind": "IntLiteral", "intValue": "9"}), request
        )


def test_direct_canonical_program_parser_has_no_envelope_or_prose() -> None:
    program = {"kind": "ExpressionProgram", "body": {"kind": "Input"}}
    request = ProposalRequest(prompt="return identity", integer_constants=(0,))

    assert parse_canonical_program_content(canonical_key(program), request) == program
    with pytest.raises(ProposalError, match="canonical JSON"):
        parse_canonical_program_content(f" {canonical_key(program)}", request)


def test_vllm_scores_finite_candidates_in_batches_and_preserves_order() -> None:
    request = _request(ITEM, ADD_ONE, SUB_ONE)
    paths = {
        request.candidates[0]: [(20, -0.25)],
        request.candidates[1]: [(21, -0.5), (22, -0.75)],
        request.candidates[2]: [(23, -1.25)],
    }
    seen_bodies: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(http_request.content))
        seen_bodies.append(body)
        prompts = cast(list[str], body["prompt"])
        choices: list[dict[str, object]] = []
        for index, prompt in enumerate(prompts):
            suffix = prompt.removeprefix(request.prompt_prefix)
            path = [
                None,
                _entry(10, -0.1),
                _entry(11, -0.2),
                *(_entry(token_id, score) for token_id, score in paths[suffix]),
            ]
            choices.append({"index": index, "text": prompt, "prompt_logprobs": path})
        return httpx.Response(200, json={"choices": list(reversed(choices))})

    async def run() -> CandidateScoreBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            scorer = VLLMPromptLogprobScorer(
                VLLMPromptLogprobConfig(
                    model="Qwen/Qwen3-Coder",
                    base_url="http://gpu.test/v1/",
                    max_batch_size=2,
                ),
                client=client,
            )
            assert isinstance(scorer, CandidateScorer)
            return await scorer.score_candidates(request)

    result = asyncio.run(run())

    assert [score.candidate for score in result.scores] == list(request.candidates)
    assert [score.sequence_logprob for score in result.scores] == [-0.55, -1.55, -1.55]
    assert result.scores[1].token_ids == (10, 11, 21, 22)
    assert result.semantics is CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT
    assert len(seen_bodies) == 2
    assert all(len(cast(list[str], body["prompt"])) <= 2 for body in seen_bodies)
    assert all(body["max_tokens"] == 0 for body in seen_bodies)
    assert all(body["echo"] is True for body in seen_bodies)
    assert all(body["prompt_logprobs"] == 0 for body in seen_bodies)
    assert all(body["stream"] is False for body in seen_bodies)


def test_vllm_scores_the_full_prompt_without_a_token_boundary_assumption() -> None:
    request = _request(ITEM)

    def handler(http_request: httpx.Request) -> httpx.Response:
        prompts = cast(list[str], json.loads(http_request.content)["prompt"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "text": prompts[0],
                        "prompt_logprobs": [None, _entry(99, -0.2), _entry(20, -0.3)],
                    },
                ]
            },
        )

    async def run() -> CandidateScoreBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            scorer = VLLMPromptLogprobScorer(
                VLLMPromptLogprobConfig(model="test-model"), client=client
            )
            return await scorer.score_candidates(request)

    result = asyncio.run(run())

    assert result.scores[0].token_ids == (99, 20)
    assert result.scores[0].sequence_logprob == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "override",
    [
        {"max_batch_size": 0},
        {"max_batch_size": True},
        {"max_concurrency": 0},
        {"timeout_seconds": float("nan")},
        {"semantics": "raw-model"},
        {"extra_body": {"prompt_logprobs": 3}},
    ],
)
def test_vllm_candidate_config_rejects_invalid_semantics_and_limits(
    override: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        VLLMPromptLogprobConfig(model="test", **override)  # type: ignore[arg-type]


def test_candidate_request_rejects_duplicate_serializations() -> None:
    with pytest.raises(ValueError, match="unique"):
        _request(ITEM, ITEM)
