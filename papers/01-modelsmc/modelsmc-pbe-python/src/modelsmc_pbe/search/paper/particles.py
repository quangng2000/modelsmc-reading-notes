"""Immutable particle construction and bounded ancestry maintenance."""

from __future__ import annotations

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.domain import ProgramAst, clone_program
from modelsmc_pbe.proposals import ProgramProposal
from modelsmc_pbe.search.models import ParticleRecord, lineage_step
from modelsmc_pbe.search.paper.prompts import feedback_for
from modelsmc_pbe.search.paper.state import SearchState


class ParticleFactory:
    """Allocate deterministic identities and construct immutable records."""

    def __init__(self, state: SearchState) -> None:
        self.state = state

    def initial(
        self,
        *,
        program: ProgramAst,
        score: ScoredProgram,
        slot: int,
        population_size: int,
    ) -> ParticleRecord:
        particle_id = self.state.allocate_particle_id()
        feedback = feedback_for(score)
        return ParticleRecord(
            particle_id=particle_id,
            parent_id=None,
            iteration=0,
            slot=slot,
            program=program,
            score=score,
            weight=1.0 / population_size,
            source="initial",
            rationale="fixed base program derived from the declared signature",
            feedback=feedback,
            lineage=(particle_id,),
            ancestry=(
                lineage_step(
                    particle_id=particle_id,
                    program=clone_program(program),
                    score=score,
                    feedback=feedback,
                ),
            ),
        )

    def clone(
        self,
        ancestor: ParticleRecord,
        *,
        iteration: int,
        slot: int,
        source: str = "clone",
        rationale: str = "cloned selected ancestor",
        proposal_call: int | None = None,
    ) -> ParticleRecord:
        particle_id = self.state.allocate_particle_id()
        program = clone_program(ancestor.program)
        return ParticleRecord(
            particle_id=particle_id,
            parent_id=ancestor.particle_id,
            iteration=iteration,
            slot=slot,
            program=program,
            score=ancestor.score,
            weight=ancestor.weight,
            source=source,
            rationale=rationale,
            feedback=ancestor.feedback,
            lineage=(*ancestor.lineage, particle_id),
            ancestry=(
                *ancestor.ancestry[-3:],
                lineage_step(
                    particle_id=particle_id,
                    program=clone_program(program),
                    score=ancestor.score,
                    feedback=ancestor.feedback,
                ),
            ),
            proposal_call=proposal_call,
        )

    def proposed(
        self,
        ancestor: ParticleRecord,
        proposal: ProgramProposal,
        score: ScoredProgram,
        *,
        iteration: int,
        slot: int,
        proposal_call: int,
    ) -> ParticleRecord:
        particle_id = self.state.allocate_particle_id()
        program = clone_program(proposal.program)
        feedback = feedback_for(score)
        return ParticleRecord(
            particle_id=particle_id,
            parent_id=ancestor.particle_id,
            iteration=iteration,
            slot=slot,
            program=program,
            score=score,
            weight=ancestor.weight,
            source=proposal.source,
            rationale=proposal.rationale,
            feedback=feedback,
            lineage=(*ancestor.lineage, particle_id),
            ancestry=(
                *ancestor.ancestry[-3:],
                lineage_step(
                    particle_id=particle_id,
                    program=clone_program(program),
                    score=score,
                    feedback=feedback,
                ),
            ),
            proposal_call=proposal_call,
        )
