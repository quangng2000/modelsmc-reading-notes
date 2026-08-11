"""Public finite guided proposal kernel orchestration."""

from __future__ import annotations

import math

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.proposals import CandidateScorer

from ..records import ImportanceSMCOptions, ImportanceSupport, ProposedState
from ..score_ledger import LLMScoreWaveLedger
from .families import select_families
from .holes import complete_paths
from .models import EventEmitter, PathSeed, ProposalPath, logaddexp
from .runtime import GuidedProposalRuntime


class FiniteGuidedProposalKernel:
    """Sample finite canonical traces and evaluate their complete guided ``q``."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        options: ImportanceSMCOptions,
        support: ImportanceSupport,
        scorer: CandidateScorer,
        generator: torch.Generator,
        emit: EventEmitter | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("importance proposal sampling requires a CPU generator")
        if config.smc.alpha >= 1.0:
            raise ValueError(
                "importance-smc requires alpha < 1 so every target state remains reachable"
            )
        if options.proposal_strategy != "guided":
            raise ValueError("finite guided kernel requires proposal_strategy='guided'")
        self._config = config
        self._generator = generator
        self._runtime = GuidedProposalRuntime(
            config=config,
            options=options,
            support=support,
            scorer=scorer,
            generator=generator,
            emitter=emit,
        )

    @property
    def source(self) -> str:
        return self._runtime.source

    @property
    def scored_candidates(self) -> int:
        """Number of unique teacher-forced candidate prompts reserved so far."""

        return self._runtime.scored_candidates

    @property
    def final_deduction_guide(self) -> torch.Tensor:
        """Return the normalized final-stage guide in fixed support order."""

        return self._runtime.final_deduction_guide.clone()

    @property
    def score_ledger(self) -> tuple[LLMScoreWaveLedger, ...]:
        """Freeze unique scored requests and their sampled uses in call order."""

        return self._runtime.score_ledger

    async def sample_many(
        self,
        ancestor_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Draw one child per ancestor, including exact clone-mixture mass."""

        seeds = tuple(
            self._sampling_seed(slot, ancestor_state_index)
            for slot, ancestor_state_index in enumerate(ancestor_state_indices)
        )
        return await self._propose(seeds, stage=stage, beta=beta)

    async def evaluate_many(
        self,
        ancestor_state_index: int,
        target_state_indices: tuple[int, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        """Evaluate the exact mixture mass of specified target programs."""

        seeds = tuple(
            PathSeed(
                slot=slot,
                ancestor_state_index=ancestor_state_index,
                cloned=False,
                target_state_index=target_state_index,
            )
            for slot, target_state_index in enumerate(target_state_indices)
        )
        return await self._propose(seeds, stage=stage, beta=beta)

    async def _propose(
        self,
        seeds: tuple[PathSeed, ...],
        *,
        stage: int,
        beta: float,
    ) -> tuple[ProposedState, ...]:
        paths = await select_families(self._runtime, seeds, stage=stage, beta=beta)
        paths = await complete_paths(self._runtime, paths, stage=stage, beta=beta)
        return tuple(self._finish(path) for path in paths)

    def _sampling_seed(self, slot: int, ancestor_state_index: int) -> PathSeed:
        alpha = self._config.smc.alpha
        cloned = (
            alpha > 0.0
            and float(torch.rand((), dtype=torch.float64, generator=self._generator).item()) < alpha
        )
        return PathSeed(
            slot=slot,
            ancestor_state_index=ancestor_state_index,
            cloned=cloned,
            target_state_index=ancestor_state_index if cloned else None,
        )

    def _finish(self, path: ProposalPath) -> ProposedState:
        if len(path.compatible_state_indices) != 1:
            raise RuntimeError(
                "construction trace is not one-to-one with a canonical complete program"
            )
        state_index = path.compatible_state_indices[0]
        if path.target_state_index is not None and state_index != path.target_state_index:
            raise RuntimeError("probability evaluation did not recover its target state")
        alpha = self._config.smc.alpha
        if state_index == path.ancestor_state_index and alpha > 0.0:
            log_q_mixture = logaddexp(
                math.log(alpha),
                math.log1p(-alpha) + path.log_q_construct,
            )
        elif alpha > 0.0:
            log_q_mixture = math.log1p(-alpha) + path.log_q_construct
        else:
            log_q_mixture = path.log_q_construct
        return ProposedState(
            state_index=state_index,
            ancestor_state_index=path.ancestor_state_index,
            log_q_construct=path.log_q_construct,
            log_q_mixture=log_q_mixture,
            cloned=path.cloned,
            family=path.family.hypothesis.kind.value,
        )
