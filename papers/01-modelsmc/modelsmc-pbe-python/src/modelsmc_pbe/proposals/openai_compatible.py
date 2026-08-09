"""Async OpenAI-compatible transport for Ollama and GPU-backed vLLM.

Submitting particle requests concurrently lets vLLM continuously batch them on
the accelerator.  This module treats returned ASTs only as untrusted JSON;
semantic scoring happens later through the local program scorer.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from modelsmc_pbe.proposals.base import (
    ProgramProposal,
    ProposalError,
    ProposalOutcome,
    ProposalRequest,
)
from modelsmc_pbe.proposals.json_extract import parse_proposal_content


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    """Connection and generation settings for Ollama or vLLM."""

    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    timeout_seconds: float = 180.0
    temperature: float = 0.7
    max_tokens: int = 2_048
    max_concurrency: int = 8
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and in [0, 2]")
        if (
            isinstance(self.max_tokens, bool)
            or isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_tokens, int)
            or not isinstance(self.max_concurrency, int)
            or self.max_tokens < 1
            or self.max_concurrency < 1
        ):
            raise ValueError("max_tokens and max_concurrency must be positive")


class OpenAICompatibleProposer:
    """Request strict proposal envelopes from an OpenAI-compatible server."""

    name = "openai-compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _payload(self, request: ProposalRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{request.prompt}\n\n"
                        "Return exactly one JSON object with keys 'program' and 'rationale'. "
                        "The program value must be a complete JSON AST. Integer literal "
                        "values must be decimal strings."
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        protected = set(payload)
        collisions = protected.intersection(self.config.extra_body)
        if collisions:
            raise ValueError(
                f"extra_body cannot override managed request fields: {sorted(collisions)}"
            )
        payload.update(self.config.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _decode_response(
        self,
        response: httpx.Response,
        request: ProposalRequest,
    ) -> ProgramProposal:
        if response.is_error:
            body_digest = hashlib.sha256(response.content).hexdigest()
            raise ProposalError(
                f"OpenAI-compatible server returned HTTP {response.status_code} "
                f"(body bytes={len(response.content)}, sha256={body_digest})"
            )
        try:
            body = cast(dict[str, Any], response.json())
        except (ValueError, TypeError) as error:
            raise ProposalError("OpenAI-compatible response is not valid JSON") from error
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProposalError("OpenAI-compatible response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ProposalError("OpenAI-compatible response did not finish with 'stop'")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProposalError("OpenAI-compatible choice has no string message content")
        return parse_proposal_content(
            message["content"],
            request,
            source=self.name,
            model=self.config.model,
        )

    async def _propose_with_client(
        self,
        request: ProposalRequest,
        client: httpx.AsyncClient,
    ) -> ProgramProposal:
        async with self._semaphore:
            try:
                response = await client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=self._payload(request),
                    timeout=self.config.timeout_seconds,
                )
            except httpx.TimeoutException as error:
                raise ProposalError(
                    f"OpenAI-compatible request timed out after "
                    f"{self.config.timeout_seconds:g} seconds"
                ) from error
            except httpx.HTTPError as error:
                raise ProposalError(f"OpenAI-compatible request failed: {error}") from error
        return await self._decode_response(response, request)

    async def propose(self, request: ProposalRequest) -> ProgramProposal:
        if self._client is not None:
            return await self._propose_with_client(request, self._client)
        async with httpx.AsyncClient() as client:
            return await self._propose_with_client(request, client)

    async def propose_many(self, requests: list[ProposalRequest]) -> list[ProgramProposal]:
        """Issue concurrent requests so a vLLM server can GPU-batch particles."""

        outcomes = await self.propose_many_outcomes(requests)
        proposals: list[ProgramProposal] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            proposals.append(outcome)
        return proposals

    async def propose_many_outcomes(
        self, requests: list[ProposalRequest]
    ) -> list[ProposalOutcome]:
        """Batch with connection pooling while preserving per-request failures."""

        if not requests:
            return []
        if self._client is not None:
            outcomes = await asyncio.gather(
                *(self._propose_with_client(request, self._client) for request in requests),
                return_exceptions=True,
            )
            return list(outcomes)
        async with httpx.AsyncClient() as shared_client:
            outcomes = await asyncio.gather(
                *(
                    self._propose_with_client(request, shared_client)
                    for request in requests
                ),
                return_exceptions=True,
            )
            return list(outcomes)
