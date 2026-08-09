"""Top-level orchestration for practical ModelSMC-style PBE search."""

from __future__ import annotations

import math

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.core import ProgramScorer
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import Proposer
from modelsmc_pbe.runtime import DeviceInfo
from modelsmc_pbe.search.models import PaperSearchResult, ParticleRecord
from modelsmc_pbe.search.paper.errors import PaperSearchError
from modelsmc_pbe.search.paper.events import emit
from modelsmc_pbe.search.paper.initial import initialize_population
from modelsmc_pbe.search.paper.objective import is_better, score_weights, select_ancestors
from modelsmc_pbe.search.paper.particles import ParticleFactory
from modelsmc_pbe.search.paper.propagation import PropagationPolicy
from modelsmc_pbe.search.paper.proposal_batch import ProposalBatchRunner
from modelsmc_pbe.search.paper.state import SearchState
from modelsmc_pbe.smc import effective_sample_size


class PaperSearchEngine:
    """Run the paper's practical resample, clone/revise, and reweight loop.

    The LLM proposal probability is unknown, so normalized scores are
    population-allocation weights rather than a calibrated posterior.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        scorer: ProgramScorer,
        proposer: Proposer,
        device: DeviceInfo,
        generator: torch.Generator,
        logger: RunLogger | None = None,
    ) -> None:
        if generator.device.type != "cpu":
            raise ValueError("paper search requires a CPU generator for deterministic decisions")
        self.config = config
        self.scorer = scorer
        self.proposer = proposer
        self.device = device
        self.generator = generator
        self.logger = logger
        self.state = SearchState()
        self.factory = ParticleFactory(self.state)
        batch_runner = ProposalBatchRunner(
            scorer=scorer,
            proposer=proposer,
            factory=self.factory,
            state=self.state,
            logger=logger,
        )
        self.propagation = PropagationPolicy(
            config=config,
            proposer=proposer,
            generator=generator,
            factory=self.factory,
            state=self.state,
            batch_runner=batch_runner,
            logger=logger,
        )

    async def run(self) -> PaperSearchResult:
        particles = await initialize_population(
            config=self.config,
            scorer=self.scorer,
            factory=self.factory,
            logger=self.logger,
        )
        champion = particles[0]
        first_exact = 0 if champion.score.exact_program else None
        resampling_steps = 0

        for iteration in range(1, self.config.smc.iterations + 1):
            weights_before = self._weights(particles)
            ess_before = effective_sample_size(weights_before)
            ancestors, relative_ess, resampled = select_ancestors(
                particles,
                threshold=self.config.smc.ess_threshold,
                generator=self.generator,
            )
            resampling_steps += int(resampled)
            emit(
                self.logger,
                "iteration.started",
                message="paper-search iteration started",
                iteration=iteration,
                ess=ess_before,
                relative_ess=relative_ess,
                ess_threshold=self.config.smc.ess_threshold,
                resampled=resampled,
                unique_ancestors=len({ancestor.particle_id for ancestor in ancestors}),
            )

            propagated = await self.propagation.propagate(
                ancestors,
                iteration=iteration,
            )
            particles = score_weights(self.config, propagated)
            champion, first_exact = self._update_champion(
                particles,
                champion=champion,
                first_exact=first_exact,
                iteration=iteration,
            )
            emit(
                self.logger,
                "iteration.completed",
                message="paper-search iteration completed",
                iteration=iteration,
                ess=effective_sample_size(self._weights(particles)),
                unique_programs=len({particle.program_key for particle in particles}),
                exact_programs=sum(particle.score.exact_program for particle in particles),
                best_loss=champion.score.total_loss,
                best_cost=champion.score.cost,
                best_exact=champion.score.exact_program,
                proposal_calls=self.state.proposal_calls,
            )

        return self._finish(
            particles,
            champion=champion,
            first_exact=first_exact,
            resampling_steps=resampling_steps,
        )

    def _update_champion(
        self,
        particles: list[ParticleRecord],
        *,
        champion: ParticleRecord,
        first_exact: int | None,
        iteration: int,
    ) -> tuple[ParticleRecord, int | None]:
        for particle in particles:
            if is_better(particle, champion):
                champion = particle
                emit(
                    self.logger,
                    "champion.improved",
                    message="best-so-far program improved",
                    iteration=iteration,
                    particle_id=particle.particle_id,
                    total_loss=particle.score.total_loss,
                    cost=particle.score.cost,
                    exact=particle.score.exact_program,
                    program=particle.program,
                )
            if particle.score.exact_program and first_exact is None:
                first_exact = iteration
                emit(
                    self.logger,
                    "solution.first_exact",
                    message="first exact program found",
                    iteration=iteration,
                    particle_id=particle.particle_id,
                    proposal_calls=self.state.proposal_calls,
                    program=particle.program,
                )
        return champion, first_exact

    def _finish(
        self,
        particles: list[ParticleRecord],
        *,
        champion: ParticleRecord,
        first_exact: int | None,
        resampling_steps: int,
    ) -> PaperSearchResult:
        final_ess = effective_sample_size(self._weights(particles))
        if not math.isfinite(final_ess):
            raise PaperSearchError("final ESS is not finite")
        result = PaperSearchResult(
            champion=champion,
            particles=tuple(particles),
            proposal_calls=self.state.proposal_calls,
            iterations_completed=self.config.smc.iterations,
            resampling_steps=resampling_steps,
            first_exact_iteration=first_exact,
            final_ess=final_ess,
            unique_programs=len({particle.program_key for particle in particles}),
            proposal_responses=self.state.proposal_responses,
            proposal_errors=self.state.proposal_errors,
            scorer_rejections=self.state.scorer_rejections,
            accepted_proposals=self.state.accepted_proposals,
        )
        if result.degraded:
            emit(
                self.logger,
                "search.degraded",
                message="every proposal call failed before returning an AST",
                level="warning",
                proposal_calls=result.proposal_calls,
                proposal_errors=result.proposal_errors,
            )
        return result

    @staticmethod
    def _weights(particles: list[ParticleRecord]) -> torch.Tensor:
        return torch.tensor(
            [particle.weight for particle in particles], dtype=torch.float64
        )
