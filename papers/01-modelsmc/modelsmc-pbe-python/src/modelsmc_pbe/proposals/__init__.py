"""Program proposal interfaces and provider implementations."""

from modelsmc_pbe.proposals.base import (
    ProgramProposal,
    ProposalError,
    ProposalOutcome,
    ProposalRequest,
    Proposer,
)
from modelsmc_pbe.proposals.catalog import CatalogProposer, ScriptedProposer
from modelsmc_pbe.proposals.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
)

__all__ = [
    "CatalogProposer",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProposer",
    "ProgramProposal",
    "ProposalError",
    "ProposalOutcome",
    "ProposalRequest",
    "Proposer",
    "ScriptedProposer",
]
