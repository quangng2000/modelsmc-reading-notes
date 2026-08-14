"""Pure mathematics for defensive, slate-based joint semantic proposals."""

from .kernel import JOINT_SEMANTIC_PROPOSAL_SOURCE, LazyJointSemanticProposalKernel
from .law import SemanticStageLaw
from .ledger import SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION, SemanticProposalLedger
from .records import (
    SemanticBranchDistribution,
    SemanticSlateProbability,
    SemanticTracePath,
)
from .slate import (
    SemanticSlateEntry,
    attach_semantic_scores,
    enumerate_construction_traces,
    iter_construction_traces,
    select_deterministic_slate,
    trace_sort_key,
)

__all__ = [
    "JOINT_SEMANTIC_PROPOSAL_SOURCE",
    "SEMANTIC_PROPOSAL_LEDGER_SCHEMA_VERSION",
    "LazyJointSemanticProposalKernel",
    "SemanticBranchDistribution",
    "SemanticProposalLedger",
    "SemanticSlateEntry",
    "SemanticSlateProbability",
    "SemanticStageLaw",
    "SemanticTracePath",
    "attach_semantic_scores",
    "enumerate_construction_traces",
    "iter_construction_traces",
    "select_deterministic_slate",
    "trace_sort_key",
]
