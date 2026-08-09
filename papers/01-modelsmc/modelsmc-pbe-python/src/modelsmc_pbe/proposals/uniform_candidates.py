"""Provider-free equal scoring for the finite importance-proposal control."""

from __future__ import annotations

from modelsmc_pbe.proposals.candidate_scoring import (
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
        return CandidateScoreBatch(
            scores=tuple(
                CandidateSequenceScore(
                    candidate=candidate,
                    expression=parse_canonical_expression_content(candidate, request),
                    token_ids=(),
                    token_logprobs=(),
                    sequence_logprob=0.0,
                )
                for candidate in request.candidates
            ),
            source=self.name,
            model="none",
            semantics=CandidateLogprobSemantics.EXPLICIT_UNIFORM,
        )

    async def score_many(
        self, requests: list[CandidateScoreRequest]
    ) -> list[CandidateScoreBatch]:
        return [await self.score_candidates(request) for request in requests]
