"""Paper-2-style refutation and hole-example inference."""

from modelsmc_pbe.deduction.engine import deduce_hypotheses, deduce_hypothesis
from modelsmc_pbe.deduction.records import (
    DeductionFact,
    DeductionFactKind,
    DeductionReport,
    FrozenValue,
    HoleExample,
    Refutation,
    RefutationKind,
    TypedValue,
)
from modelsmc_pbe.deduction.render import (
    render_deduction_fact,
    render_deduction_guidance,
    render_deduction_trace,
    render_hole_example,
    render_refutation,
    render_typed_value,
)

__all__ = [
    "DeductionFact",
    "DeductionFactKind",
    "DeductionReport",
    "FrozenValue",
    "HoleExample",
    "Refutation",
    "RefutationKind",
    "TypedValue",
    "deduce_hypotheses",
    "deduce_hypothesis",
    "render_deduction_fact",
    "render_deduction_guidance",
    "render_deduction_trace",
    "render_hole_example",
    "render_refutation",
    "render_typed_value",
]
