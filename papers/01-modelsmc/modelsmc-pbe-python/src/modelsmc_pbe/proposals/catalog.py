"""Deterministic proposers for controls, replay, and unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from modelsmc_pbe.domain.ast import ProgramAst, clone_program, normalize_program
from modelsmc_pbe.proposals.base import (
    ProgramProposal,
    ProposalError,
    ProposalOutcome,
    ProposalRequest,
)


class CatalogProposer:
    """Select from a fixed catalog by request index, without mutable state."""

    name = "catalog"

    def __init__(self, programs: Sequence[ProgramAst]) -> None:
        if not programs:
            raise ValueError("catalog must contain at least one program")
        self._programs = tuple(clone_program(program) for program in programs)

    async def propose(self, request: ProposalRequest) -> ProgramProposal:
        selected = self._programs[request.request_index % len(self._programs)]
        program = normalize_program(
            selected,
            allowed_integer_constants=request.integer_constants,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
        return ProgramProposal(
            program=program,
            rationale=f"deterministic catalog entry {request.request_index % len(self._programs)}",
            source=self.name,
        )

    async def propose_many(self, requests: list[ProposalRequest]) -> list[ProgramProposal]:
        return [await self.propose(request) for request in requests]

    async def propose_many_outcomes(
        self, requests: list[ProposalRequest]
    ) -> list[ProposalOutcome]:
        return list(await self.propose_many(requests))


ScriptStep = ProgramAst | ProgramProposal | Exception


class ScriptedProposer:
    """Consume a scripted sequence exactly once for deterministic failure tests."""

    name = "scripted"

    def __init__(self, steps: Sequence[ScriptStep]) -> None:
        if not steps:
            raise ValueError("script must contain at least one step")
        self._steps = list(steps)
        self._next = 0
        self._lock = asyncio.Lock()

    async def propose(self, request: ProposalRequest) -> ProgramProposal:
        async with self._lock:
            if self._next >= len(self._steps):
                raise ProposalError("scripted proposer is exhausted")
            step = self._steps[self._next]
            self._next += 1
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ProgramProposal):
            raw_program = step.program
            rationale = step.rationale
        else:
            raw_program = step
            rationale = f"scripted proposal {self._next - 1}"
        program = normalize_program(
            raw_program,
            allowed_integer_constants=request.integer_constants,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
        return ProgramProposal(program=program, rationale=rationale, source=self.name)

    async def propose_many(self, requests: list[ProposalRequest]) -> list[ProgramProposal]:
        return await asyncio.gather(*(self.propose(request) for request in requests))

    async def propose_many_outcomes(
        self, requests: list[ProposalRequest]
    ) -> list[ProposalOutcome]:
        outcomes = await asyncio.gather(
            *(self.propose(request) for request in requests),
            return_exceptions=True,
        )
        return list(outcomes)
