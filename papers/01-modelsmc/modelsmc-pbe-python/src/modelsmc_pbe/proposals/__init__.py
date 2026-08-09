"""Program proposal interfaces and provider implementations."""

from modelsmc_pbe.proposals.base import (
    ProgramProposal,
    ProposalError,
    ProposalOutcome,
    ProposalRequest,
    Proposer,
)
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScorer,
    CandidateScoreRequest,
    CandidateSequenceScore,
)
from modelsmc_pbe.proposals.catalog import CatalogProposer, ScriptedProposer
from modelsmc_pbe.proposals.hole import ExpressionScope, HoleSpecification
from modelsmc_pbe.proposals.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
)
from modelsmc_pbe.proposals.uniform_candidates import UniformCandidateScorer
from modelsmc_pbe.proposals.vllm_prompt_logprobs import (
    VLLMPromptLogprobConfig,
    VLLMPromptLogprobScorer,
)

__all__ = [
    "CandidateLogprobSemantics",
    "CandidateScoreBatch",
    "CandidateScoreRequest",
    "CandidateScorer",
    "CandidateSequenceScore",
    "CatalogProposer",
    "ExpressionScope",
    "HoleSpecification",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProposer",
    "ProgramProposal",
    "ProposalError",
    "ProposalOutcome",
    "ProposalRequest",
    "Proposer",
    "ScriptedProposer",
    "UniformCandidateScorer",
    "VLLMPromptLogprobConfig",
    "VLLMPromptLogprobScorer",
]
