"""Shared scoring runtime for family and hole proposal waves."""

from __future__ import annotations

import time
from typing import Any

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.proposals import (
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
    llm_energy,
)

from ..deduction_guide import FiniteDeductionGuide
from ..proposal_distribution import CandidateDistribution, candidate_distribution
from ..records import (
    ImportanceProposalBudgetExceeded,
    ImportanceSMCOptions,
    ImportanceSupport,
)
from ..score_ledger import LLMScoreWaveLedger
from .ledger import GuidedScoreLedger
from .models import EventEmitter, emit
from .requests import deduplicate_requests, request_batch_sha256


class GuidedProposalRuntime:
    """Own shared proposal dependencies, score budget, and evidence ledger."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: ImportanceSupport,
        scorer: CandidateScorer,
        generator: torch.Generator,
        emitter: EventEmitter | None,
    ) -> None:
        self.config = config
        self.options = options
        self.support = support
        self.scorer = scorer
        self.generator = generator
        self.emitter = emitter
        self.family_by_hypothesis = {family.hypothesis_index: family for family in support.families}
        self._request_index = 0
        self._scored_candidates = 0
        self._ledger = GuidedScoreLedger(options)
        self._deduction_guide = FiniteDeductionGuide.build(
            support,
            smc=config.smc,
            strength=options.deduction_strength,
            beta_max=options.beta_max,
        )
        self._deduction_cache: dict[tuple[float, tuple[tuple[int, ...], ...]], torch.Tensor] = {}
        guide = self._deduction_guide.distribution(options.beta_max)
        self.final_deduction_guide = guide
        exact_indices = tuple(
            state.state_index for state in support.states if state.score.exact_program
        )
        self.event(
            "importance.proposal.deduction_guide.ready",
            message="exact Occam/deduction subtree guide constructed",
            level="debug",
            deduction_mix=options.deduction_mix,
            family_deduction_mix=options.resolved_family_deduction_mix,
            hole_deduction_mix=options.resolved_hole_deduction_mix,
            deduction_strength=options.deduction_strength,
            minimum_violations=float(self._deduction_guide.violations.min().item()),
            maximum_violations=float(self._deduction_guide.violations.max().item()),
            exact_program_mass=(
                0.0 if not exact_indices else float(guide[list(exact_indices)].sum().item())
            ),
            family_masses={
                family.hypothesis.kind.value: float(guide[list(family.state_indices)].sum().item())
                for family in support.families
            },
        )

    @property
    def source(self) -> str:
        return self.scorer.name

    @property
    def scored_candidates(self) -> int:
        return self._scored_candidates

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        return self._ledger.freeze()

    def next_request_index(self) -> int:
        request_index = self._request_index
        self._request_index += 1
        return request_index

    async def score_requests(
        self,
        requests: list[CandidateScoreRequest],
        *,
        stage: int,
        wave: str,
        event_scope: str,
        event_message: str,
        **event_data: Any,
    ) -> tuple[CandidateScoreBatch, ...]:
        unique, inverse = deduplicate_requests(requests)
        request_hash = request_batch_sha256(unique)
        candidate_total = sum(len(request.candidates) for request in unique)
        self._reserve_candidate_budget(candidate_total, stage=stage, wave=wave)
        common = {
            "stage": stage,
            **event_data,
            "requests": len(requests),
            "unique_requests": len(unique),
            "candidates": candidate_total,
            "request_batch_sha256": request_hash,
            "source": self.source,
        }
        self.event(
            f"{event_scope}.started",
            message=f"{event_message} started",
            level="debug",
            **common,
        )
        started = time.perf_counter()
        try:
            unique_batches = await self.scorer.score_many(unique)
        except Exception:
            self.event(
                f"{event_scope}.failed",
                message=f"{event_message} failed",
                level="error",
                elapsed_seconds=time.perf_counter() - started,
                **common,
            )
            raise
        self.event(
            f"{event_scope}.completed",
            message=f"{event_message} completed",
            level="debug",
            elapsed_seconds=time.perf_counter() - started,
            **common,
        )
        if len(unique_batches) != len(unique):
            raise RuntimeError("candidate scorer did not return one batch per unique request")
        return tuple(unique_batches[position] for position in inverse)

    def candidate_probabilities(
        self,
        request: CandidateScoreRequest,
        batch: CandidateScoreBatch,
        *,
        groups: tuple[tuple[int, ...], ...],
        beta: float,
        deduction_mix: float,
    ) -> CandidateDistribution:
        if tuple(score.candidate for score in batch.scores) != request.candidates:
            raise RuntimeError("candidate scorer reordered or changed the finite catalog")
        q_deduction = self._deduction_probabilities(groups, beta=beta)
        return candidate_distribution(
            sequence_logprobs=tuple(
                llm_energy(score, self.options.llm_energy_normalization) for score in batch.scores
            ),
            q_deduction=q_deduction,
            temperature=self.options.proposal_temperature,
            epsilon=self.options.proposal_epsilon,
            deduction_mix=deduction_mix,
        )

    def record_score_wave(
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
        self._ledger.record(
            stage=stage,
            beta=beta,
            wave=wave,
            request=request,
            batch=batch,
            distribution=distribution,
            selected=selected,
            slot=slot,
            ancestor_state_index=ancestor_state_index,
            forced=forced,
            cloned=cloned,
            deduction_mix=deduction_mix,
        )

    def event(self, name: str, *, message: str, level: str, **data: Any) -> None:
        emit(self.emitter, name, message=message, level=level, **data)

    def _deduction_probabilities(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        beta: float,
    ) -> torch.Tensor:
        cache_key = (beta, groups)
        cached = self._deduction_cache.get(cache_key)
        if cached is not None:
            return cached
        probabilities = self._deduction_guide.subtree_probabilities(groups, beta=beta)
        self._deduction_cache[cache_key] = probabilities
        return probabilities

    def _reserve_candidate_budget(self, candidates: int, *, stage: int, wave: str) -> None:
        attempted = self._scored_candidates + candidates
        if attempted > self.options.max_scored_candidates:
            self.event(
                "importance.proposal.budget_exceeded",
                message="candidate-scoring wave rejected before scorer I/O",
                level="error",
                stage=stage,
                wave=wave,
                requested_candidates=candidates,
                scored_candidates=self._scored_candidates,
                max_scored_candidates=self.options.max_scored_candidates,
            )
            raise ImportanceProposalBudgetExceeded(
                "candidate scoring would exceed --max-scored-candidates="
                f"{self.options.max_scored_candidates} "
                f"(used {self._scored_candidates}, next wave {candidates})"
            )
        self._scored_candidates = attempted
        self.event(
            "importance.proposal.budget_reserved",
            message="candidate-scoring work reserved within the run budget",
            level="debug",
            stage=stage,
            wave=wave,
            requested_candidates=candidates,
            scored_candidates=self._scored_candidates,
            max_scored_candidates=self.options.max_scored_candidates,
        )
