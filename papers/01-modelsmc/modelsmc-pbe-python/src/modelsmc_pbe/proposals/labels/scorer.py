"""Adapter from raw teacher-forced paths to semantic compatibility log odds."""

from __future__ import annotations

import hashlib

from modelsmc_pbe.proposals.base import ProposalError
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
    CandidateSequenceScore,
)
from modelsmc_pbe.proposals.labels.boundary import extract_label_pair
from modelsmc_pbe.proposals.labels.contracts import (
    CompatibilityMapping,
    CompatibilityPathSpec,
    CompatibilityProgram,
    CompatibilityScore,
    CompatibilityScoreBatch,
    CompatibilityScoringRequest,
    LabelPathEvidence,
    RawCompatibilityScoreIdentity,
)
from modelsmc_pbe.proposals.labels.prompts import (
    COMPATIBILITY_TEMPLATE_VERSION,
    build_compatibility_prompt,
    build_program_paths,
)


class SymmetrizedLabelCompatibilityScorer:
    """Score each program through two swapped A/B teacher-forced comparisons."""

    name = "symmetrized-binary-compatibility"
    raw_candidates_per_program = 4

    def __init__(self, scorer: CandidateScorer) -> None:
        self._scorer = scorer

    async def score(self, request: CompatibilityScoringRequest) -> CompatibilityScoreBatch:
        """Score one slate in bounded requests that all use one shared prefix."""

        prompt = build_compatibility_prompt(request.dataset_context)
        programs_per_request = request.raw_candidate_batch_size // self.raw_candidates_per_program
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        derived_scores: list[CompatibilityScore] = []
        programs_per_wave = programs_per_request * request.raw_request_batch_size
        request_offset = 0
        for wave_start in range(0, len(request.programs), programs_per_wave):
            wave = request.programs[wave_start : wave_start + programs_per_wave]
            program_chunks = tuple(
                wave[start : start + programs_per_request]
                for start in range(0, len(wave), programs_per_request)
            )
            spec_chunks = tuple(
                tuple(path for program in programs for path in build_program_paths(program))
                for programs in program_chunks
            )
            raw_requests = [
                CandidateScoreRequest(
                    prompt_prefix=prompt,
                    candidates=tuple(spec.candidate for spec in specs),
                    hole=None,
                    integer_constants=request.integer_constants,
                    request_index=request.request_index + request_offset + index,
                    max_depth=request.max_depth,
                    max_nodes=request.max_nodes,
                    candidate_kind=CandidateKind.LABEL,
                )
                for index, specs in enumerate(spec_chunks)
            ]
            raw_batches = await self._scorer.score_many(raw_requests)
            if len(raw_batches) != len(raw_requests):
                raise ProposalError(
                    "semantic scorer returned the wrong number of request batches"
                )
            self._derive_wave(
                program_chunks,
                spec_chunks,
                raw_requests,
                raw_batches,
                prompt_sha256,
                derived_scores,
            )
            request_offset += len(raw_requests)
        return CompatibilityScoreBatch(
            scores=tuple(derived_scores),
            template_version=COMPATIBILITY_TEMPLATE_VERSION,
            prompt_sha256=prompt_sha256,
        )

    def _derive_wave(
        self,
        program_chunks: tuple[tuple[CompatibilityProgram, ...], ...],
        spec_chunks: tuple[tuple[CompatibilityPathSpec, ...], ...],
        raw_requests: list[CandidateScoreRequest],
        raw_batches: list[CandidateScoreBatch],
        prompt_sha256: str,
        derived_scores: list[CompatibilityScore],
    ) -> None:
        for programs, specs, raw_request, raw_batch in zip(
            program_chunks,
            spec_chunks,
            raw_requests,
            raw_batches,
            strict=True,
        ):
            self._validate_raw_batch(raw_request, raw_batch)
            identity = RawCompatibilityScoreIdentity(
                source=raw_batch.source,
                model=raw_batch.model,
                semantics=raw_batch.semantics,
                model_revision=raw_batch.model_revision,
                tokenizer_revision=raw_batch.tokenizer_revision,
                provenance=raw_batch.provenance,
            )
            derived_scores.extend(
                self._derive_score(
                    program,
                    specs[4 * index : 4 * index + 4],
                    raw_batch.scores[4 * index : 4 * index + 4],
                    prompt_sha256,
                    identity,
                )
                for index, program in enumerate(programs)
            )

    @staticmethod
    def _validate_raw_batch(
        request: CandidateScoreRequest,
        raw_batch: CandidateScoreBatch,
    ) -> None:
        if raw_batch.semantics is not CandidateLogprobSemantics.TEACHER_FORCED_FULL_PROMPT:
            raise ProposalError("semantic compatibility requires teacher-forced full-prompt paths")
        if len(raw_batch.scores) != len(request.candidates):
            raise ProposalError("semantic scorer returned the wrong number of raw label paths")
        if tuple(score.candidate for score in raw_batch.scores) != request.candidates:
            raise ProposalError("semantic scorer did not preserve raw label-path order")

    @staticmethod
    def _derive_score(
        program: CompatibilityProgram,
        specs: tuple[CompatibilityPathSpec, ...],
        raw_scores: tuple[CandidateSequenceScore, ...],
        prompt_sha256: str,
        identity: RawCompatibilityScoreIdentity,
    ) -> CompatibilityScore:
        plus_a, plus_b, plus_proof = extract_label_pair(
            CompatibilityMapping.A_IS_COMPATIBLE,
            specs[0],
            specs[1],
            raw_scores[0],
            raw_scores[1],
        )
        minus_a, minus_b, minus_proof = extract_label_pair(
            CompatibilityMapping.B_IS_COMPATIBLE,
            specs[2],
            specs[3],
            raw_scores[2],
            raw_scores[3],
        )
        paths: tuple[LabelPathEvidence, ...] = (plus_a, plus_b, minus_a, minus_b)
        compatibility_log_odds = 0.5 * (
            (plus_proof.label_a_logprob - plus_proof.label_b_logprob)
            + (minus_proof.label_b_logprob - minus_proof.label_a_logprob)
        )
        return CompatibilityScore(
            program_key=program.program_key,
            program_sha256=program.sha256,
            compatibility_log_odds=compatibility_log_odds,
            paths=paths,
            boundary_proofs=(plus_proof, minus_proof),
            template_version=COMPATIBILITY_TEMPLATE_VERSION,
            prompt_sha256=prompt_sha256,
            raw_identity=identity,
        )
