from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from modelsmc_pbe.grammar import enumerate_skeleton
from modelsmc_pbe.proposals import (
    CatalogProposer,
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
    ProgramProposal,
    ProposalError,
    ProposalRequest,
    ScriptedProposer,
)
from modelsmc_pbe.proposals.json_extract import parse_proposal_content


def _request(index: int = 0) -> ProposalRequest:
    return ProposalRequest(
        prompt="Synthesize x + 1 as a complete AST.",
        integer_constants=(0, 1),
        request_index=index,
    )


def _program() -> dict[str, object]:
    return {
        "kind": "ExpressionProgram",
        "body": {
            "kind": "Add",
            "left": {"kind": "Input"},
            "right": {"kind": "IntLiteral", "intValue": "1"},
        },
    }


def _content() -> str:
    return json.dumps({"program": _program(), "rationale": "add one"})


def test_strict_json_extraction_accepts_one_optional_fence() -> None:
    proposal = parse_proposal_content(
        f"```json\n{_content()}\n```",
        _request(),
        source="test",
        model="unit-model",
    )

    assert proposal.program == _program()
    assert proposal.rationale == "add one"
    assert proposal.model == "unit-model"


@pytest.mark.parametrize(
    "content",
    [
        f"Here is the answer: {_content()}",
        json.dumps({"program": _program(), "rationale": "ok", "extra": None}),
        json.dumps({"program": _program()}),
    ],
)
def test_strict_json_extraction_rejects_prose_and_envelope_variants(content: str) -> None:
    with pytest.raises(ProposalError):
        parse_proposal_content(content, _request(), source="test")


def test_catalog_and_scripted_proposers_are_deterministic() -> None:
    catalog = enumerate_skeleton("expression-arithmetic", [0, 1], 100)

    async def run() -> None:
        proposer = CatalogProposer(catalog[:2])
        selected = await proposer.propose_many([_request(0), _request(1), _request(2)])
        assert selected[0].program == selected[2].program
        assert selected[0].program != selected[1].program

        scripted = ScriptedProposer([_program(), ProposalError("scripted failure")])
        assert (await scripted.propose(_request())).program == _program()
        with pytest.raises(ProposalError, match="scripted failure"):
            await scripted.propose(_request(1))
        with pytest.raises(ProposalError, match="exhausted"):
            await scripted.propose(_request(2))

    asyncio.run(run())


def test_openai_compatible_proposer_decodes_mock_response_and_sets_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": _content()},
                    }
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            proposer = OpenAICompatibleProposer(
                OpenAICompatibleConfig(
                    model="Qwen/Qwen3-Coder",
                    base_url="http://gpu.test/v1/",
                    api_key="secret",
                    max_concurrency=4,
                ),
                client=client,
            )
            proposals = await proposer.propose_many([_request(0), _request(1)])
        assert len(proposals) == 2
        assert all(proposal.program == _program() for proposal in proposals)

    asyncio.run(run())

    assert len(seen) == 2
    assert all(str(request.url) == "http://gpu.test/v1/chat/completions" for request in seen)
    bodies = [json.loads(request.content) for request in seen]
    assert all(body["model"] == "Qwen/Qwen3-Coder" for body in bodies)
    assert all(body["response_format"] == {"type": "json_object"} for body in bodies)
    assert all(request.headers["authorization"] == "Bearer secret" for request in seen)


def test_batch_outcomes_preserve_one_provider_failure() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": _content()},
                    }
                ]
            },
        )

    async def run() -> list[ProgramProposal | BaseException]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            proposer = OpenAICompatibleProposer(
                OpenAICompatibleConfig(model="test-model"),
                client=client,
            )
            return await proposer.propose_many_outcomes([_request(0), _request(1)])

    outcomes = asyncio.run(run())

    assert isinstance(outcomes[0], ProposalError)
    assert isinstance(outcomes[1], ProgramProposal)
    assert "temporarily unavailable" not in str(outcomes[0])
    assert "sha256=" in str(outcomes[0])


@pytest.mark.parametrize(
    "override",
    [
        {"base_url": ""},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"temperature": float("nan")},
        {"max_tokens": True},
        {"max_concurrency": False},
    ],
)
def test_openai_compatible_configuration_rejects_unsafe_values(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleConfig(model="test-model", **override)  # type: ignore[arg-type]
