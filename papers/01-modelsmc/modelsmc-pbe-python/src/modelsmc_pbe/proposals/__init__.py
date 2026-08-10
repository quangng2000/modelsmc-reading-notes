"""Program proposal interfaces and provider implementations."""

from modelsmc_pbe.proposals.base import (
    ProgramProposal,
    ProposalError,
    ProposalOutcome,
    ProposalRequest,
    Proposer,
)
from modelsmc_pbe.proposals.candidate_scoring import (
    CandidateKind,
    CandidateLogprobSemantics,
    CandidateScoreBatch,
    CandidateScoreOrigin,
    CandidateScoreProvenance,
    CandidateScorer,
    CandidateScoreRequest,
    CandidateSequenceScore,
    LLMEnergyNormalization,
    ProviderMetricSource,
    ProviderScoreMetrics,
    llm_energy,
)
from modelsmc_pbe.proposals.catalog import CatalogProposer, ScriptedProposer
from modelsmc_pbe.proposals.hole import ExpressionScope, HoleSpecification
from modelsmc_pbe.proposals.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProposer,
)
from modelsmc_pbe.proposals.score_cache import (
    CachedCandidateScorer,
    ScoreCacheCorruptionError,
    ScoreCacheError,
    ScoreCacheIdentity,
    ScoreCacheMetrics,
    ScoreCacheMissError,
    ScoreCacheMode,
    ScoreCacheProvenance,
)
from modelsmc_pbe.proposals.uniform_candidates import UniformCandidateScorer
from modelsmc_pbe.proposals.vllm_prompt_logprobs import (
    VLLMPromptLogprobConfig,
    VLLMPromptLogprobScorer,
)

__all__ = [
    "CachedCandidateScorer",
    "CandidateKind",
    "CandidateLogprobSemantics",
    "CandidateScoreBatch",
    "CandidateScoreOrigin",
    "CandidateScoreProvenance",
    "CandidateScoreRequest",
    "CandidateScorer",
    "CandidateSequenceScore",
    "CatalogProposer",
    "ExpressionScope",
    "HoleSpecification",
    "LLMEnergyNormalization",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProposer",
    "ProgramProposal",
    "ProposalError",
    "ProposalOutcome",
    "ProposalRequest",
    "Proposer",
    "ProviderMetricSource",
    "ProviderScoreMetrics",
    "ScoreCacheCorruptionError",
    "ScoreCacheError",
    "ScoreCacheIdentity",
    "ScoreCacheMetrics",
    "ScoreCacheMissError",
    "ScoreCacheMode",
    "ScoreCacheProvenance",
    "ScriptedProposer",
    "UniformCandidateScorer",
    "VLLMPromptLogprobConfig",
    "VLLMPromptLogprobScorer",
    "llm_energy",
]
