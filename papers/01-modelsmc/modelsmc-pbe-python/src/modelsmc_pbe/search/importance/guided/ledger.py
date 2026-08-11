"""Persistent score evidence for the finite guided proposal."""

from __future__ import annotations

from dataclasses import replace

from modelsmc_pbe.proposals import CandidateScoreBatch, CandidateScoreRequest, llm_energy

from ..proposal_distribution import CandidateDistribution
from ..records import ImportanceSMCOptions
from ..score_ledger import (
    LLMScoreCandidateLedger,
    LLMScoreSelectionLedger,
    LLMScoreWaveLedger,
    prompt_prefix_sha256,
)


class GuidedScoreLedger:
    """De-duplicate score evidence while retaining each categorical use."""

    def __init__(self, options: ImportanceSMCOptions) -> None:
        self._options = options
        self._entries: dict[
            tuple[object, ...],
            tuple[LLMScoreWaveLedger, list[LLMScoreSelectionLedger]],
        ] = {}

    def freeze(self) -> tuple[LLMScoreWaveLedger, ...]:
        return tuple(
            replace(record, selections=tuple(selections))
            for record, selections in self._entries.values()
        )

    def record(
        self,
        *,
        stage: int,
        beta: float,
        wave: str,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        distribution: CandidateDistribution,
        selected: int,
        slot: int,
        ancestor_state_index: int,
        forced: bool,
        cloned: bool,
        deduction_mix: float,
    ) -> None:
        qwen = tuple(float(value) for value in distribution.q_llm.tolist())
        deduction = tuple(float(value) for value in distribution.q_deduction.tolist())
        proposal = tuple(float(value) for value in distribution.probabilities.tolist())
        key = (
            stage,
            wave,
            request.prompt_prefix,
            request.candidates,
            batch.source,
            batch.model,
            batch.model_revision,
            batch.tokenizer_revision,
            batch.semantics,
            self._options.llm_energy_normalization,
            self._options.proposal_temperature,
            self._options.proposal_epsilon,
            deduction_mix,
            deduction,
            proposal,
        )
        entry = self._entries.get(key)
        if entry is None:
            candidates = tuple(
                LLMScoreCandidateLedger(
                    canonical_candidate=score.candidate,
                    token_ids=score.token_ids,
                    token_logprobs=score.token_logprobs,
                    scored_token_count=len(score.token_logprobs),
                    total_sequence_logprob=score.sequence_logprob,
                    normalized_energy=llm_energy(score, self._options.llm_energy_normalization),
                    qwen_probability=qwen[index],
                    deduction_probability=deduction[index],
                    proposal_probability=proposal[index],
                )
                for index, score in enumerate(batch.scores)
            )
            record = LLMScoreWaveLedger(
                stage=stage,
                beta=beta,
                wave=wave,
                request_index=request.request_index,
                prompt_prefix=request.prompt_prefix,
                prompt_prefix_sha256=prompt_prefix_sha256(request.prompt_prefix),
                candidate_kind=request.candidate_kind.value,
                source=batch.source,
                model=batch.model,
                model_revision=batch.model_revision,
                tokenizer_revision=batch.tokenizer_revision,
                score_semantics=batch.semantics.value,
                score_origin=(
                    "unspecified" if batch.provenance is None else batch.provenance.origin.value
                ),
                cache_key_sha256=(
                    None if batch.provenance is None else batch.provenance.cache_key_sha256
                ),
                cache_hit=(None if batch.provenance is None else batch.provenance.cache_hit),
                energy_normalization=self._options.llm_energy_normalization,
                temperature=self._options.proposal_temperature,
                proposal_epsilon=self._options.proposal_epsilon,
                deduction_mix=deduction_mix,
                candidates=candidates,
                selections=(),
            )
            entry = (record, [])
            self._entries[key] = entry
        entry[1].append(
            LLMScoreSelectionLedger(
                slot=slot,
                ancestor_state_index=ancestor_state_index,
                selected_index=selected,
                selected_probability=proposal[selected],
                selected_qwen_probability=qwen[selected],
                selected_deduction_probability=deduction[selected],
                forced=forced,
                cloned=cloned,
            )
        )
