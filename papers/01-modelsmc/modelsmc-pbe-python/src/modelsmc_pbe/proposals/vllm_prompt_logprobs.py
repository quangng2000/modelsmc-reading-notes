"""Finite-candidate teacher-forced scoring through vLLM completions."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScoreRequest,
    CandidateSequenceScore,
    ProviderScoreMetrics,
)
from modelsmc_pbe.proposals.json_extract import parse_canonical_expression_content
from modelsmc_pbe.proposals.vllm_response import decode_candidate_scores

_MANAGED_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "max_tokens",
        "echo",
        "prompt_logprobs",
        "stream",
        "add_special_tokens",
    }
)


@dataclass(frozen=True, slots=True)
class VLLMPromptLogprobConfig:
    """Connection and batching policy for teacher-forced candidate scoring."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    timeout_seconds: float = 180.0
    max_concurrency: int = 8
    max_batch_size: int = 128
    add_special_tokens: bool = True
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    semantics: CandidateLogprobSemantics = CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("model and base_url must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (
            ("max_concurrency", self.max_concurrency),
            ("max_batch_size", self.max_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.add_special_tokens, bool):
            raise TypeError("add_special_tokens must be a boolean")
        for name, revision_value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if revision_value is not None and (
                not isinstance(revision_value, str) or not revision_value.strip()
            ):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if self.semantics is not CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT:
            raise ValueError("vLLM candidate scoring requires full-prompt semantics")
        if not isinstance(self.extra_body, dict):
            raise TypeError("extra_body must be a dictionary")
        collisions = _MANAGED_FIELDS.intersection(self.extra_body)
        if collisions:
            raise ValueError(
                f"extra_body cannot override managed request fields: {sorted(collisions)}"
            )


class VLLMPromptLogprobScorer:
    """Score canonical candidates without sampling a free-form continuation."""

    name = "vllm-prompt-logprobs"

    def __init__(
        self,
        config: VLLMPromptLogprobConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._http_requests = 0
        self._http_failures = 0
        self._scored_token_positions = 0
        self._http_request_seconds_sum = 0.0

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/completions"

    def provider_metrics(self) -> ProviderScoreMetrics:
        """Return actual HTTP request and token-position telemetry."""

        return ProviderScoreMetrics(
            http_requests=self._http_requests,
            http_failures=self._http_failures,
            scored_token_positions=self._scored_token_positions,
            http_request_seconds_sum=self._http_request_seconds_sum,
        )

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        if self._client is not None:
            return await self._score_request_with_client(request, self._client)
        async with httpx.AsyncClient() as client:
            return await self._score_request_with_client(request, client)

    async def _score_request_with_client(
        self,
        request: CandidateScoreRequest,
        client: httpx.AsyncClient,
    ) -> CandidateScoreBatch:
        expressions: tuple[AstNode | None, ...] = (
            tuple(
                parse_canonical_expression_content(candidate, request)
                for candidate in request.candidates
            )
            if request.candidate_kind is CandidateKind.EXPRESSION
            else tuple(None for _ in request.candidates)
        )
        return await self._score_with_client(request, expressions, client)

    async def score_many(self, requests: list[CandidateScoreRequest]) -> list[CandidateScoreBatch]:
        if self._client is not None:
            return list(
                await asyncio.gather(*(self.score_candidates(request) for request in requests))
            )
        async with httpx.AsyncClient() as client:
            return list(
                await asyncio.gather(
                    *(self._score_request_with_client(request, client) for request in requests)
                )
            )

    async def _score_with_client(
        self,
        request: CandidateScoreRequest,
        expressions: tuple[AstNode | None, ...],
        client: httpx.AsyncClient,
    ) -> CandidateScoreBatch:
        chunks = [
            (
                request.candidates[start : start + self.config.max_batch_size],
                expressions[start : start + self.config.max_batch_size],
            )
            for start in range(0, len(request.candidates), self.config.max_batch_size)
        ]
        scored_chunks = await asyncio.gather(
            *(self._score_chunk(request.prompt_prefix, *chunk, client) for chunk in chunks)
        )
        scores = tuple(score for chunk in scored_chunks for score in chunk)
        return CandidateScoreBatch(
            scores=scores,
            source=self.name,
            model=self.config.model,
            semantics=self.config.semantics,
            model_revision=self.config.model_revision,
            tokenizer_revision=self.config.tokenizer_revision,
            provenance=CandidateScoreProvenance(CandidateScoreOrigin.PROVIDER),
        )

    async def _score_chunk(
        self,
        prefix: str,
        candidates: Sequence[str],
        expressions: Sequence[AstNode | None],
        client: httpx.AsyncClient,
    ) -> tuple[CandidateSequenceScore, ...]:
        prompts = [prefix + candidate for candidate in candidates]
        async with self._semaphore:
            self._http_requests += 1
            started = time.perf_counter()
            try:
                response = await client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=self._payload(prompts),
                    timeout=self.config.timeout_seconds,
                )
            except httpx.TimeoutException as error:
                self._http_failures += 1
                raise ProposalError(
                    f"vLLM candidate scoring timed out after "
                    f"{self.config.timeout_seconds:g} seconds"
                ) from error
            except httpx.HTTPError as error:
                self._http_failures += 1
                raise ProposalError(f"vLLM candidate scoring request failed: {error}") from error
            finally:
                self._http_request_seconds_sum += time.perf_counter() - started
        try:
            scores = decode_candidate_scores(response, prompts, candidates, expressions)
        except ProposalError:
            self._http_failures += 1
            raise
        self._scored_token_positions += sum(len(score.token_logprobs) for score in scores)
        return scores

    def _payload(self, prompts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompts,
            "max_tokens": 0,
            "echo": True,
            "prompt_logprobs": 0,
            "stream": False,
            "add_special_tokens": self.config.add_special_tokens,
        }
        payload.update(self.config.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        return headers
