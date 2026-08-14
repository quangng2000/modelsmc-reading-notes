from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from modelsmc_pbe.domain import ValueType, canonical_key
from modelsmc_pbe.proposals import (
    CachedCandidateScorer,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScoreRequest,
    CandidateSequenceScore,
    ExpressionScope,
    HoleSpecification,
    LLMEnergyNormalization,
    ScoreCacheCorruptionError,
    ScoreCacheIdentity,
    ScoreCacheMissError,
    ScoreCacheMode,
)
from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.json_extract import parse_canonical_expression_content
from modelsmc_pbe.search.importance.proposal_distribution import candidate_distribution

ITEM = canonical_key({"kind": "Item"})
ADD_ONE = canonical_key(
    {
        "kind": "Add",
        "left": {"kind": "Item"},
        "right": {"kind": "IntLiteral", "intValue": "1"},
    }
)


def _request(*, prefix: str = "Task\nCanonical expression: ") -> CandidateScoreRequest:
    return CandidateScoreRequest(
        prompt_prefix=prefix,
        candidates=(ITEM, ADD_ONE),
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


def _identity(**changes: object) -> ScoreCacheIdentity:
    base = ScoreCacheIdentity(
        scorer_name="vllm-prompt-logprobs",
        model_alias="qwen-coder",
        model_repository="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_revision="488639f1",
        tokenizer_revision="488639f1",
        server_config="vllm=0.11.0;logprobs=processed_logprobs",
        semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
        energy_normalization=LLMEnergyNormalization.TOTAL_FULL_PROMPT_LOGPROB,
    )
    return replace(base, **changes)


class _Provider:
    name = "vllm-prompt-logprobs"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        return (await self.score_many([request]))[0]

    async def score_many(self, requests: list[CandidateScoreRequest]) -> list[CandidateScoreBatch]:
        self.calls += 1
        if self.fail:
            raise AssertionError("provider must not be called")
        return [self._batch(request) for request in requests]

    @staticmethod
    def _batch(request: CandidateScoreRequest) -> CandidateScoreBatch:
        paths = (((10, 11), (-0.25, -0.5)), ((10, 12, 13), (-0.25, -0.75, -1.0)))
        scores = tuple(
            CandidateSequenceScore(
                candidate=candidate,
                expression=parse_canonical_expression_content(candidate, request),
                token_ids=token_ids,
                token_logprobs=token_logprobs,
                sequence_logprob=sum(token_logprobs),
            )
            for candidate, (token_ids, token_logprobs) in zip(
                request.candidates, paths, strict=True
            )
        )
        return CandidateScoreBatch(
            scores=scores,
            source="vllm-prompt-logprobs",
            model="qwen-coder",
            semantics=CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT,
            model_revision="488639f1",
            tokenizer_revision="488639f1",
            provenance=CandidateScoreProvenance(CandidateScoreOrigin.PROVIDER),
        )


def test_cold_write_then_replay_is_provider_free_and_categorical_exact(
    tmp_path: Path,
) -> None:
    request = _request()
    cold_provider = _Provider()
    cold_cache = CachedCandidateScorer(
        cold_provider,
        cache_dir=tmp_path,
        mode=ScoreCacheMode.READ_WRITE,
        identity=_identity(),
    )
    cold = asyncio.run(cold_cache.score_candidates(request))

    replay_provider = _Provider(fail=True)
    replay_cache = CachedCandidateScorer(
        replay_provider,
        cache_dir=tmp_path,
        mode=ScoreCacheMode.REPLAY_ONLY,
        identity=_identity(),
    )
    warm = asyncio.run(replay_cache.score_candidates(request))

    assert cold_provider.calls == 1
    assert replay_provider.calls == 0
    assert warm.scores == cold.scores
    assert warm.source == cold.source
    assert warm.model == cold.model
    assert warm.semantics == cold.semantics
    assert warm.model_revision == cold.model_revision
    assert warm.tokenizer_revision == cold.tokenizer_revision
    assert cold.provenance is not None and cold.provenance.cache_hit is False
    assert warm.provenance is not None and warm.provenance.cache_hit is True
    assert warm.provenance.cache_key_sha256 == cold.provenance.cache_key_sha256

    kwargs = {
        "q_deduction": torch.tensor([0.5, 0.5], dtype=torch.float64),
        "temperature": 0.7,
        "epsilon": 0.05,
        "deduction_mix": 0.5,
    }
    cold_q = candidate_distribution(
        sequence_logprobs=tuple(score.sequence_logprob for score in cold.scores),
        **kwargs,
    )
    warm_q = candidate_distribution(
        sequence_logprobs=tuple(score.sequence_logprob for score in warm.scores),
        **kwargs,
    )
    assert torch.equal(cold_q.q_llm, warm_q.q_llm)
    assert torch.equal(cold_q.probabilities, warm_q.probabilities)

    metrics = replay_cache.metrics()
    assert metrics.hit_requests == 1
    assert metrics.hit_candidates == 2
    assert metrics.miss_requests == 0
    assert metrics.provider_invocations == 0
    assert metrics.cache_served_scored_tokens == 5


def test_corruption_fails_closed_without_provider_fallback(tmp_path: Path) -> None:
    request = _request()
    writer = CachedCandidateScorer(
        _Provider(),
        cache_dir=tmp_path,
        mode=ScoreCacheMode.READ_WRITE,
        identity=_identity(),
    )
    asyncio.run(writer.score_candidates(request))
    entry = writer.entry_path(request)
    record = json.loads(entry.read_text(encoding="utf-8"))
    record["batch"]["scores"][0]["token_ids"][0] = 999
    entry.write_text(json.dumps(record), encoding="utf-8")

    provider = _Provider(fail=True)
    reader = CachedCandidateScorer(
        provider,
        cache_dir=tmp_path,
        mode=ScoreCacheMode.READ_WRITE,
        identity=_identity(),
    )
    with pytest.raises(ScoreCacheCorruptionError, match="checksum mismatch"):
        asyncio.run(reader.score_candidates(request))
    assert provider.calls == 0


def test_replay_only_miss_never_calls_provider(tmp_path: Path) -> None:
    provider = _Provider(fail=True)
    cache = CachedCandidateScorer(
        provider,
        cache_dir=tmp_path,
        mode=ScoreCacheMode.REPLAY_ONLY,
        identity=_identity(),
    )

    with pytest.raises(ScoreCacheMissError, match="replay-only"):
        asyncio.run(cache.score_candidates(_request()))
    assert provider.calls == 0
    assert cache.metrics().miss_candidates == 2


def test_cache_key_binds_provider_and_validation_contract(tmp_path: Path) -> None:
    request = _request()
    base = CachedCandidateScorer(
        _Provider(),
        cache_dir=tmp_path,
        mode=ScoreCacheMode.READ_WRITE,
        identity=_identity(),
    )
    digests = {
        base.key_sha256(request),
        CachedCandidateScorer(
            _Provider(),
            cache_dir=tmp_path,
            mode=ScoreCacheMode.READ_WRITE,
            identity=_identity(model_repository="Qwen/Qwen2.5-Coder-7B-Instruct"),
        ).key_sha256(request),
        CachedCandidateScorer(
            _Provider(),
            cache_dir=tmp_path,
            mode=ScoreCacheMode.READ_WRITE,
            identity=_identity(model_revision="different"),
        ).key_sha256(request),
        CachedCandidateScorer(
            _Provider(),
            cache_dir=tmp_path,
            mode=ScoreCacheMode.READ_WRITE,
            identity=_identity(tokenizer_revision="different"),
        ).key_sha256(request),
        CachedCandidateScorer(
            _Provider(),
            cache_dir=tmp_path,
            mode=ScoreCacheMode.READ_WRITE,
            identity=_identity(server_config="vllm=0.11.1"),
        ).key_sha256(request),
        CachedCandidateScorer(
            _Provider(),
            cache_dir=tmp_path,
            mode=ScoreCacheMode.READ_WRITE,
            identity=_identity(
                energy_normalization=(LLMEnergyNormalization.MEAN_FULL_PROMPT_CONDITIONAL_LOGPROB)
            ),
        ).key_sha256(request),
        base.key_sha256(_request(prefix="Changed prefix: ")),
        base.key_sha256(replace(request, integer_constants=(-1, 0, 1))),
        base.key_sha256(replace(request, max_depth=64)),
    }
    assert len(digests) == 9


def test_failed_provider_call_creates_no_entry_and_records_failure(tmp_path: Path) -> None:
    request = _request()

    class _BrokenProvider(_Provider):
        async def score_many(
            self, requests: list[CandidateScoreRequest]
        ) -> list[CandidateScoreBatch]:
            self.calls += 1
            raise ProposalError("provider unavailable")

    cache = CachedCandidateScorer(
        _BrokenProvider(),
        cache_dir=tmp_path,
        mode=ScoreCacheMode.READ_WRITE,
        identity=_identity(),
    )
    with pytest.raises(ProposalError, match="unavailable"):
        asyncio.run(cache.score_candidates(request))
    assert not cache.entry_path(request).exists()
    metrics = cache.metrics()
    assert metrics.provider_failures == 1
    assert metrics.provider_invocations == 1
    assert metrics.provider_scored_tokens == 0
