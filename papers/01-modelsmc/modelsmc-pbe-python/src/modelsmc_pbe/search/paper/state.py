"""Mutable counters isolated from immutable particle records."""

from dataclasses import dataclass


@dataclass(slots=True)
class SearchState:
    """Run-local identity allocation and proposal accounting."""

    next_particle_id: int = 0
    proposal_calls: int = 0
    proposal_responses: int = 0
    proposal_errors: int = 0
    scorer_rejections: int = 0
    accepted_proposals: int = 0

    def allocate_particle_id(self) -> int:
        identifier = self.next_particle_id
        self.next_particle_id += 1
        return identifier

    def allocate_proposal_call(self) -> int:
        identifier = self.proposal_calls
        self.proposal_calls += 1
        return identifier
