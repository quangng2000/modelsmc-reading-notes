"""Clone-or-revise propagation policy for one paper-search iteration."""

from __future__ import annotations

import torch

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import ProposalRequest, Proposer
from modelsmc_pbe.search.models import ParticleRecord
from modelsmc_pbe.search.paper.errors import PaperSearchError
from modelsmc_pbe.search.paper.events import emit
from modelsmc_pbe.search.paper.particles import ParticleFactory
from modelsmc_pbe.search.paper.prompts import proposal_prompt
from modelsmc_pbe.search.paper.proposal_batch import PendingProposal, ProposalBatchRunner
from modelsmc_pbe.search.paper.state import SearchState


class PropagationPolicy:
    """Draw clone/revise decisions and preserve slot order across async work."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        proposer: Proposer,
        generator: torch.Generator,
        factory: ParticleFactory,
        state: SearchState,
        batch_runner: ProposalBatchRunner,
        logger: RunLogger | None,
    ) -> None:
        self.config = config
        self.proposer = proposer
        self.generator = generator
        self.factory = factory
        self.state = state
        self.batch_runner = batch_runner
        self.logger = logger

    async def propagate(
        self,
        ancestors: list[ParticleRecord],
        *,
        iteration: int,
    ) -> list[ParticleRecord]:
        children: list[ParticleRecord | None] = [None] * len(ancestors)
        pending: list[PendingProposal] = []
        for slot, ancestor in enumerate(ancestors):
            draw = float(torch.rand((), generator=self.generator).item())
            if draw < self.config.smc.alpha:
                children[slot] = self.factory.clone(
                    ancestor,
                    iteration=iteration,
                    slot=slot,
                )
                emit(
                    self.logger,
                    "proposal.cloned",
                    message="cloned selected ancestor",
                    level="trace",
                    iteration=iteration,
                    slot=slot,
                    ancestor_id=ancestor.particle_id,
                    draw=draw,
                    alpha=self.config.smc.alpha,
                )
                continue
            pending.append(self._pending(ancestor, iteration=iteration, slot=slot, draw=draw))

        resolved = await self.batch_runner.run(pending, iteration=iteration)
        for slot, child in resolved.items():
            children[slot] = child
        if any(child is None for child in children):
            raise PaperSearchError("internal propagation error left a particle slot empty")
        return [child for child in children if child is not None]

    def _pending(
        self,
        ancestor: ParticleRecord,
        *,
        iteration: int,
        slot: int,
        draw: float,
    ) -> PendingProposal:
        proposal_call = self.state.allocate_proposal_call()
        request = ProposalRequest(
            prompt=proposal_prompt(
                self.config,
                ancestor,
                iteration=iteration,
                slot=slot,
            ),
            integer_constants=tuple(self.config.spec.integer_constants),
            request_index=proposal_call,
            max_depth=self.config.smc.max_depth,
            max_nodes=self.config.smc.max_nodes,
        )
        emit(
            self.logger,
            "proposal.requested",
            message="requesting a revised AST",
            level="trace",
            iteration=iteration,
            slot=slot,
            ancestor_id=ancestor.particle_id,
            proposal_call=proposal_call,
            proposer=self.proposer.name,
            draw=draw,
            prompt=request.prompt,
        )
        return PendingProposal(slot, ancestor, request, proposal_call, draw)
