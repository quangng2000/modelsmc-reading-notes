"""Serializable records shared by the paper-style PBE search shell."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.core import ScoredProgram
from modelsmc_pbe.domain import ProgramAst, canonical_key


def score_record(score: ScoredProgram) -> dict[str, object]:
    """Convert a semantic score into a stable artifact record."""

    return {
        "inferred_type": score.inferred_type,
        "total_loss": score.total_loss,
        "exact_matches": score.exact_matches,
        "cost": score.cost,
        "scorer_log_target": score.log_target,
        "exact_program": score.exact_program,
        "evaluations": [
            {
                "input": evaluation.input,
                "expected": evaluation.expected,
                "predicted": evaluation.predicted,
                "exact": evaluation.exact,
                "loss": evaluation.loss,
            }
            for evaluation in score.evaluations
        ],
    }


@dataclass(frozen=True, slots=True)
class LineageStep:
    """Bounded proposal context retained from one ancestor."""

    particle_id: int
    program: ProgramAst
    feedback: str
    total_loss: float
    cost: int
    exact_program: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "program": self.program,
            "feedback": self.feedback,
            "total_loss": self.total_loss,
            "cost": self.cost,
            "exact_program": self.exact_program,
        }


def lineage_step(
    *,
    particle_id: int,
    program: ProgramAst,
    score: ScoredProgram,
    feedback: str,
) -> LineageStep:
    return LineageStep(
        particle_id=particle_id,
        program=program,
        feedback=feedback,
        total_loss=score.total_loss,
        cost=score.cost,
        exact_program=score.exact_program,
    )


@dataclass(frozen=True, slots=True)
class ParticleRecord:
    """One complete program particle plus its ancestry and semantic score."""

    particle_id: int
    parent_id: int | None
    iteration: int
    slot: int
    program: ProgramAst
    score: ScoredProgram
    weight: float
    source: str
    rationale: str
    feedback: str
    lineage: tuple[int, ...]
    ancestry: tuple[LineageStep, ...]
    proposal_call: int | None = None

    @property
    def program_key(self) -> str:
        return canonical_key(self.program)

    def as_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "parent_id": self.parent_id,
            "iteration": self.iteration,
            "slot": self.slot,
            "program": self.program,
            "program_key": self.program_key,
            "score": score_record(self.score),
            "weight": self.weight,
            "source": self.source,
            "rationale": self.rationale,
            "feedback": self.feedback,
            "lineage": list(self.lineage),
            "ancestry": [step.as_dict() for step in self.ancestry],
            "proposal_call": self.proposal_call,
        }


@dataclass(frozen=True, slots=True)
class PaperSearchResult:
    """Outcome of the practical ModelSMC-inspired resample/revise search."""

    champion: ParticleRecord
    particles: tuple[ParticleRecord, ...]
    proposal_calls: int
    iterations_completed: int
    resampling_steps: int
    first_exact_iteration: int | None
    final_ess: float
    unique_programs: int
    proposal_responses: int
    proposal_errors: int
    scorer_rejections: int
    accepted_proposals: int

    @property
    def exact(self) -> bool:
        return self.champion.score.exact_program

    @property
    def degraded(self) -> bool:
        """True when every provider call failed before returning an AST."""

        return self.proposal_calls > 0 and self.proposal_errors == self.proposal_calls

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "paper-search",
            "probabilistic_claim": "heuristic_search_uncorrected_proposal_kernel",
            "exact": self.exact,
            "degraded": self.degraded,
            "champion": self.champion.as_dict(),
            "proposal_calls": self.proposal_calls,
            "iterations_completed": self.iterations_completed,
            "resampling_steps": self.resampling_steps,
            "first_exact_iteration": self.first_exact_iteration,
            "final_ess": self.final_ess,
            "unique_programs": self.unique_programs,
            "proposal_responses": self.proposal_responses,
            "proposal_errors": self.proposal_errors,
            "scorer_rejections": self.scorer_rejections,
            "accepted_proposals": self.accepted_proposals,
        }

    def final_particle_records(self) -> list[dict[str, object]]:
        ordered = sorted(
            self.particles,
            key=lambda particle: (
                particle.score.exact_program,
                -particle.score.total_loss,
                -particle.score.cost,
                particle.weight,
            ),
            reverse=True,
        )
        return [particle.as_dict() for particle in ordered]
