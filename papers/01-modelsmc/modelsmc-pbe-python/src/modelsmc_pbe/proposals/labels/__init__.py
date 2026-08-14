"""Contrastive binary-label semantic compatibility scoring."""

from modelsmc_pbe.proposals.labels.boundary import LabelBoundaryError, extract_label_pair
from modelsmc_pbe.proposals.labels.contracts import (
    CompatibilityLabel,
    CompatibilityMapping,
    CompatibilityPathSpec,
    CompatibilityProgram,
    CompatibilityScore,
    CompatibilityScoreBatch,
    CompatibilityScoringRequest,
    LabelBoundaryProof,
    LabelPathEvidence,
    RawCompatibilityScoreIdentity,
)
from modelsmc_pbe.proposals.labels.prompts import (
    COMPATIBILITY_TEMPLATE_VERSION,
    HARMONY_GPT_OSS_TEMPLATE_VERSION,
    SemanticPromptProtocol,
    build_compatibility_prompt,
    build_program_paths,
    compatibility_add_special_tokens,
    compatibility_template_version,
)
from modelsmc_pbe.proposals.labels.scorer import SymmetrizedLabelCompatibilityScorer

__all__ = [
    "COMPATIBILITY_TEMPLATE_VERSION",
    "HARMONY_GPT_OSS_TEMPLATE_VERSION",
    "CompatibilityLabel",
    "CompatibilityMapping",
    "CompatibilityPathSpec",
    "CompatibilityProgram",
    "CompatibilityScore",
    "CompatibilityScoreBatch",
    "CompatibilityScoringRequest",
    "LabelBoundaryError",
    "LabelBoundaryProof",
    "LabelPathEvidence",
    "RawCompatibilityScoreIdentity",
    "SemanticPromptProtocol",
    "SymmetrizedLabelCompatibilityScorer",
    "build_compatibility_prompt",
    "build_program_paths",
    "compatibility_add_special_tokens",
    "compatibility_template_version",
    "extract_label_pair",
]
