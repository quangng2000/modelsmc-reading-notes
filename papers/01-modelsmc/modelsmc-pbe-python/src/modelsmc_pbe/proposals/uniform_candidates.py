"""Provider-free equal scoring for the finite importance-proposal control."""

from __future__ import annotations

from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreRequest,
    CandidateSequenceScore,
)
from modelsmc_pbe.proposals.json_extract import parse_canonical_expression_content


class UniformCandidateScorer:
    """Assign equal energy to every already canonical finite candidate."""

    name = "uniform-finite-candidates"

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        expressions = (
            tuple(
                parse_canonical_expression_content(candidate, request)
                for candidate in request.candidates
            )
            if request.candidate_kind is CandidateKind.EXPRESSION
            else tuple(None for _ in request.candidates)
        )
        return CandidateScoreBatch(
            scores=tuple(
                CandidateSequenceScore(
                    candidate=candidate,
                    expression=expression,
                    token_ids=(),
                    token_logprobs=(),
                    sequence_logprob=0.0,
                )
                for candidate, expression in zip(
                    request.candidates,
                    expressions,
                    strict=True,
                )
            ),
            source=self.name,
            model="none",
            semantics=CandidateLogprobSemantics.EXPLICIT_UNIFORM,
        )

    async def score_many(self, requests: list[CandidateScoreRequest]) -> list[CandidateScoreBatch]:
        return [await self.score_candidates(request) for request in requests]
