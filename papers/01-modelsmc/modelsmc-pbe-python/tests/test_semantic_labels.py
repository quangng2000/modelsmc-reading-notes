from __future__ import annotations

import asyncio
import math

import pytest

from modelsmc_pbe.proposals import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScoreRequest,
    CandidateSequenceScore,
)
from modelsmc_pbe.proposals.labels import (
    COMPATIBILITY_TEMPLATE_VERSION,
    HARMONY_GPT_OSS_TEMPLATE_VERSION,
    CompatibilityMapping,
    CompatibilityProgram,
    CompatibilityScoringRequest,
    LabelBoundaryError,
    SemanticPromptProtocol,
    SymmetrizedLabelCompatibilityScorer,
    build_compatibility_prompt,
    build_program_paths,
    compatibility_add_special_tokens,
    extract_label_pair,
)


def _raw_score(
    candidate: str,
    token_ids: tuple[int, ...],
    token_logprobs: tuple[float, ...],
) -> CandidateSequenceScore:
    return CandidateSequenceScore(
        candidate=candidate,
        expression=None,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        sequence_logprob=math.fsum(token_logprobs),
    )


class _ScriptedRawScorer:
    name = "scripted-label-paths"

    def __init__(
        self,
        *,
        semantics: CandidateLogprobSemantics = (
            CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT
        ),
    ) -> None:
        self.semantics = semantics
        self.requests: list[CandidateScoreRequest] = []
        self.score_many_calls = 0

    async def score_candidates(self, request: CandidateScoreRequest) -> CandidateScoreBatch:
        self.requests.append(request)
        final_logprobs = (-0.1, -2.1, -1.0, -0.4)
        scores = tuple(
            _raw_score(
                candidate,
                (100 + index // 2, 900 + index % 2),
                (-0.2 - 0.05 * (index % 2), final_logprobs[index % 4]),
            )
            for index, candidate in enumerate(request.candidates)
        )
        return CandidateScoreBatch(
            scores=scores,
            source=self.name,
            model="semantic-test-model",
            semantics=self.semantics,
            model_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
            provenance=CandidateScoreProvenance(CandidateScoreOrigin.SYNTHETIC),
        )

    async def score_many(self, requests: list[CandidateScoreRequest]) -> list[CandidateScoreBatch]:
        self.score_many_calls += 1
        return [await self.score_candidates(request) for request in requests]


def test_symmetrized_semantic_score_uses_one_global_request_and_four_paths() -> None:
    raw_scorer = _ScriptedRawScorer()
    scorer = SymmetrizedLabelCompatibilityScorer(raw_scorer)
    programs = (
        CompatibilityProgram("program-1", '{"kind":"Input"}'),
        CompatibilityProgram("program-2", '{"body":{"kind":"Item"}}'),
    )

    result = asyncio.run(
        scorer.score(
            CompatibilityScoringRequest(
                dataset_context="[1,2] -> [1,2]\n[3] -> [3]",
                programs=programs,
                integer_constants=(0, 1),
            )
        )
    )

    assert len(raw_scorer.requests) == 1
    request = raw_scorer.requests[0]
    assert request.candidate_kind is CandidateKind.LABEL
    assert request.hole is None
    assert len(request.candidates) == 4 * len(programs)
    assert result.raw_candidate_count == 8
    assert result.template_version == COMPATIBILITY_TEMPLATE_VERSION
    assert [score.program_key for score in result.scores] == ["program-1", "program-2"]
    assert [score.compatibility_log_score for score in result.scores] == pytest.approx([1.3, 1.3])
    assert all(len(score.paths) == 4 for score in result.scores)
    assert all(
        proof.max_abs_prefix_logprob_delta == pytest.approx(0.05)
        for score in result.scores
        for proof in score.boundary_proofs
    )
    assert result.scores[0].raw_identity.source == raw_scorer.name
    assert result.scores[0].raw_identity.model_revision == "model-revision"
    assert request.candidates[0][:-1] == request.candidates[1][:-1]
    assert request.candidates[2][:-1] == request.candidates[3][:-1]


def test_swapped_mapping_cancels_a_fixed_label_prior() -> None:
    """0.5[(delta+t) + (t-delta)] recovers semantic evidence t."""

    result = asyncio.run(
        SymmetrizedLabelCompatibilityScorer(_ScriptedRawScorer()).score(
            CompatibilityScoringRequest(
                dataset_context="x -> x",
                programs=(CompatibilityProgram("identity", "identity(input)"),),
            )
        )
    )

    score = result.scores[0]
    plus, minus = score.boundary_proofs
    assert plus.label_a_logprob - plus.label_b_logprob == pytest.approx(2.0)
    assert minus.label_b_logprob - minus.label_a_logprob == pytest.approx(0.6)
    assert score.compatibility_log_score == pytest.approx(1.3)
    assert score.compatibility_log_score > 0


def test_harmony_gpt_oss_paths_teacher_force_terminal_assistant_labels() -> None:
    protocol = SemanticPromptProtocol.HARMONY_GPT_OSS_V1
    prompt = build_compatibility_prompt("x -> x", protocol)
    paths = build_program_paths(CompatibilityProgram("p", "identity(input)"), protocol)

    assert prompt.startswith("<|start|>system<|message|>")
    assert prompt.endswith("x -> x")
    assert all(
        "<|end|><|start|>assistant<|channel|>final<|message|>" in path.candidate for path in paths
    )
    assert [path.candidate[-1] for path in paths] == ["A", "B", "A", "B"]
    assert all(not path.candidate.endswith("<|return|>") for path in paths)
    assert compatibility_add_special_tokens(protocol) is False
    assert compatibility_add_special_tokens(SemanticPromptProtocol.RAW_V2) is True


def test_harmony_semantic_adapter_archives_its_distinct_template_version() -> None:
    result = asyncio.run(
        SymmetrizedLabelCompatibilityScorer(
            _ScriptedRawScorer(),
            prompt_protocol=SemanticPromptProtocol.HARMONY_GPT_OSS_V1,
        ).score(
            CompatibilityScoringRequest(
                dataset_context="x -> x",
                programs=(CompatibilityProgram("p", "identity(input)"),),
            )
        )
    )

    assert result.template_version == HARMONY_GPT_OSS_TEMPLATE_VERSION
    assert result.scores[0].template_version == HARMONY_GPT_OSS_TEMPLATE_VERSION


def test_semantic_adapter_chunks_only_at_whole_program_boundaries() -> None:
    raw_scorer = _ScriptedRawScorer()
    programs = tuple(
        CompatibilityProgram(f"program-{index}", f"program({index})") for index in range(3)
    )

    result = asyncio.run(
        SymmetrizedLabelCompatibilityScorer(raw_scorer).score(
            CompatibilityScoringRequest(
                dataset_context="x -> x",
                programs=programs,
                request_index=7,
                raw_candidate_batch_size=5,
                raw_request_batch_size=1,
            )
        )
    )

    assert [request.request_index for request in raw_scorer.requests] == [7, 8, 9]
    assert raw_scorer.score_many_calls == 3
    assert [len(request.candidates) for request in raw_scorer.requests] == [4, 4, 4]
    assert len({request.prompt_prefix for request in raw_scorer.requests}) == 1
    assert [score.program_key for score in result.scores] == [
        "program-0",
        "program-1",
        "program-2",
    ]
    assert result.raw_candidate_count == 12


@pytest.mark.parametrize(
    ("a_ids", "a_logs", "b_ids", "b_logs", "message"),
    [
        ((10, 20), (-0.2, -0.3), (11, 21), (-0.2, -0.4), "only at the final"),
        ((10, 20), (-0.2, -0.3), (10, 20), (-0.2, -0.4), "distinct final"),
        ((10, 20), (-0.2, -0.3), (10, 11, 21), (-0.2, -0.1, -0.4), "equal"),
    ],
)
def test_label_boundary_rejects_nonisolated_final_tokens(
    a_ids: tuple[int, ...],
    a_logs: tuple[float, ...],
    b_ids: tuple[int, ...],
    b_logs: tuple[float, ...],
    message: str,
) -> None:
    specs = build_program_paths(CompatibilityProgram("p", "program"))

    with pytest.raises(LabelBoundaryError, match=message):
        extract_label_pair(
            CompatibilityMapping.A_IS_COMPATIBLE,
            specs[0],
            specs[1],
            _raw_score(specs[0].candidate, a_ids, a_logs),
            _raw_score(specs[1].candidate, b_ids, b_logs),
        )


def test_label_boundary_retains_distinct_prefix_logprob_commitments() -> None:
    specs = build_program_paths(CompatibilityProgram("p", "program"))

    evidence_a, evidence_b, proof = extract_label_pair(
        CompatibilityMapping.A_IS_COMPATIBLE,
        specs[0],
        specs[1],
        _raw_score(specs[0].candidate, (10, 20), (-0.2, -0.3)),
        _raw_score(specs[1].candidate, (10, 21), (-0.25, -0.4)),
    )

    assert evidence_a.prefix_token_ids_sha256 == evidence_b.prefix_token_ids_sha256
    assert evidence_a.prefix_token_logprobs_sha256 != evidence_b.prefix_token_logprobs_sha256
    assert proof.label_a_prefix_token_logprobs_sha256 == (evidence_a.prefix_token_logprobs_sha256)
    assert proof.label_b_prefix_token_logprobs_sha256 == (evidence_b.prefix_token_logprobs_sha256)
    assert proof.max_abs_prefix_logprob_delta == pytest.approx(0.05)


def test_semantic_adapter_rejects_non_teacher_forced_scores() -> None:
    scorer = SymmetrizedLabelCompatibilityScorer(
        _ScriptedRawScorer(semantics=CandidateLogprobSemantics.EXPLICIT_UNIFORM)
    )

    with pytest.raises(Exception, match="teacher-forced full-prompt"):
        asyncio.run(
            scorer.score(
                CompatibilityScoringRequest(
                    dataset_context="x -> x",
                    programs=(CompatibilityProgram("p", "program"),),
                )
            )
        )
