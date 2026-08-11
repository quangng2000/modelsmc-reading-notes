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
    build_compatibility_prompt,
    build_program_paths,
)
from modelsmc_pbe.proposals.labels.scorer import SymmetrizedLabelCompatibilityScorer

__all__ = [
    "COMPATIBILITY_TEMPLATE_VERSION",
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
    "SymmetrizedLabelCompatibilityScorer",
    "build_compatibility_prompt",
    "build_program_paths",
    "extract_label_pair",
]
