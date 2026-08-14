"""Small async interface shared by catalog and model-backed proposers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from modelsmc_pbe.domain.ast import ProgramAst


class ProposalError(RuntimeError):
    """A provider failed or returned a malformed proposal."""


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """Provider-independent information for one proposal call."""

    prompt: str
    integer_constants: tuple[int, ...]
    request_index: int = 0
    system_prompt: str = (
        "Propose a JSON AST for the bounded program-synthesis grammar. "
        "Return only the requested JSON object, never source code."
    )
    max_depth: int = 32
    max_nodes: int = 256

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("proposal prompt must not be empty")
        if self.request_index < 0:
            raise ValueError("request_index must be nonnegative")
        if self.max_depth < 1 or self.max_nodes < 1:
            raise ValueError("AST depth and node limits must be positive")


@dataclass(frozen=True, slots=True)
class ProgramProposal:
    """A transport AST plus non-semantic provider diagnostics."""

    program: ProgramAst
    rationale: str
    source: str
    model: str | None = None


type ProposalOutcome = ProgramProposal | BaseException


@runtime_checkable
class Proposer(Protocol):
    """Structural protocol consumed by the ModelSMC propagation shell."""

    name: str

    async def propose(self, request: ProposalRequest) -> ProgramProposal: ...

    async def propose_many(self, requests: list[ProposalRequest]) -> list[ProgramProposal]: ...

    async def propose_many_outcomes(
        self, requests: list[ProposalRequest]
    ) -> list[ProposalOutcome]: ...
