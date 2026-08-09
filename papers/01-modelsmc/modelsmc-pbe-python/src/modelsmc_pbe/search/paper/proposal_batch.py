"""Provider execution, semantic scoring, and fallback policy for one batch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter

from modelsmc_pbe.core import ProgramScorer, RejectedProgram
from modelsmc_pbe.observability import RunLogger
from modelsmc_pbe.proposals import ProgramProposal, ProposalRequest, Proposer
from modelsmc_pbe.search.models import ParticleRecord
from modelsmc_pbe.search.paper.events import emit
from modelsmc_pbe.search.paper.particles import ParticleFactory
from modelsmc_pbe.search.paper.state import SearchState


@dataclass(frozen=True, slots=True)
class PendingProposal:
    """One slot whose clone-or-revise draw selected revision."""

    slot: int
    ancestor: ParticleRecord
    request: ProposalRequest
    proposal_call: int
    draw: float


class ProposalBatchRunner:
    """Resolve provider outcomes without letting one failure abort a batch."""

    def __init__(
        self,
        *,
        scorer: ProgramScorer,
        proposer: Proposer,
        factory: ParticleFactory,
        state: SearchState,
        logger: RunLogger | None,
    ) -> None:
        self.scorer = scorer
        self.proposer = proposer
        self.factory = factory
        self.state = state
        self.logger = logger

    async def run(
        self,
        pending: list[PendingProposal],
        *,
        iteration: int,
    ) -> dict[int, ParticleRecord]:
        if not pending:
            return {}
        started = perf_counter()
        outcomes = await self.proposer.propose_many_outcomes(
            [item.request for item in pending]
        )
        emit(
            self.logger,
            "proposal.batch_completed",
            message="proposal batch completed",
            level="debug",
            iteration=iteration,
            requests=len(pending),
            latency_seconds=perf_counter() - started,
            proposer=self.proposer.name,
        )

        children: dict[int, ParticleRecord] = {}
        successful: list[tuple[PendingProposal, ProgramProposal]] = []
        for item, outcome in zip(pending, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                self.state.proposal_errors += 1
                children[item.slot] = self._provider_fallback(item, outcome, iteration)
            else:
                self.state.proposal_responses += 1
                successful.append((item, outcome))

        if not successful:
            return children
        scores = await asyncio.to_thread(
            self.scorer.score_batch,
            [proposal.program for _, proposal in successful],
        )
        for (item, proposal), score in zip(successful, scores, strict=True):
            if isinstance(score, RejectedProgram):
                self.state.scorer_rejections += 1
                children[item.slot] = self.factory.clone(
                    item.ancestor,
                    iteration=iteration,
                    slot=item.slot,
                    source="scorer-rejection-fallback",
                    rationale=f"semantic core rejected proposal: {score.reason}",
                    proposal_call=item.proposal_call,
                )
                emit(
                    self.logger,
                    "proposal.rejected",
                    message="semantic core rejected proposal; retained ancestor",
                    level="warning",
                    iteration=iteration,
                    slot=item.slot,
                    ancestor_id=item.ancestor.particle_id,
                    proposal_call=item.proposal_call,
                    reason=score.reason,
                    inferred_type=score.inferred_type,
                    cost=score.cost,
                )
                continue
            self.state.accepted_proposals += 1
            child = self.factory.proposed(
                item.ancestor,
                proposal,
                score,
                iteration=iteration,
                slot=item.slot,
                proposal_call=item.proposal_call,
            )
            children[item.slot] = child
            emit(
                self.logger,
                "proposal.scored",
                message="accepted and scored proposed AST",
                level="trace",
                iteration=iteration,
                slot=item.slot,
                particle_id=child.particle_id,
                ancestor_id=item.ancestor.particle_id,
                proposal_call=item.proposal_call,
                source=proposal.source,
                total_loss=score.total_loss,
                cost=score.cost,
                exact=score.exact_program,
                program=proposal.program,
                rationale=proposal.rationale,
            )
        return children

    def _provider_fallback(
        self,
        item: PendingProposal,
        outcome: BaseException,
        iteration: int,
    ) -> ParticleRecord:
        child = self.factory.clone(
            item.ancestor,
            iteration=iteration,
            slot=item.slot,
            source="proposal-error-fallback",
            rationale=f"proposal failed: {type(outcome).__name__}: {outcome}",
            proposal_call=item.proposal_call,
        )
        emit(
            self.logger,
            "proposal.failed",
            message="proposal failed; retained ancestor",
            level="warning",
            iteration=iteration,
            slot=item.slot,
            ancestor_id=item.ancestor.particle_id,
            proposal_call=item.proposal_call,
            draw=item.draw,
            error_type=type(outcome).__name__,
            error_message=str(outcome),
        )
        return child
