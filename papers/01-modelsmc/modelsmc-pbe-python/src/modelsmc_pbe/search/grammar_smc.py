"""Compatibility imports for the modular finite-grammar SMC control.

New code may import from :mod:`modelsmc_pbe.search.grammar_control`; this
module keeps the original public path stable for callers and experiments.
"""

from modelsmc_pbe.search.grammar_control.engine import GrammarSMCEngine
from modelsmc_pbe.search.grammar_control.records import (
    CALIBRATED_CLAIM,
    EmptyGrammarSupportError,
    GrammarParticle,
    GrammarReferenceMetrics,
    GrammarSMCOptions,
    GrammarSMCResult,
    GrammarStageDiagnostic,
    GrammarStateSummary,
)
from modelsmc_pbe.search.grammar_control.transition import (
    prior_independent_log_acceptance,
)

__all__ = [
    "CALIBRATED_CLAIM",
    "EmptyGrammarSupportError",
    "GrammarParticle",
    "GrammarReferenceMetrics",
    "GrammarSMCEngine",
    "GrammarSMCOptions",
    "GrammarSMCResult",
    "GrammarStageDiagnostic",
    "GrammarStateSummary",
    "prior_independent_log_acceptance",
]
