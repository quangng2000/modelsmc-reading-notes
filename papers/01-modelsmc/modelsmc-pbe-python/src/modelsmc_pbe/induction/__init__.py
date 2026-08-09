"""Paper-2-inspired type-directed inductive generalization."""

from modelsmc_pbe.induction.assembly import assemble_program
from modelsmc_pbe.induction.engine import induce_hypotheses
from modelsmc_pbe.induction.records import (
    HoleSpec,
    InductionReport,
    SkeletonKind,
    StructuralFact,
    StructuralRelation,
    TypedSkeleton,
    TypedVariable,
)
from modelsmc_pbe.induction.render import (
    render_hole_type,
    render_hypothesis,
    render_induction_trace,
    render_structural_fact,
)

__all__ = [
    "HoleSpec",
    "InductionReport",
    "SkeletonKind",
    "StructuralFact",
    "StructuralRelation",
    "TypedSkeleton",
    "TypedVariable",
    "assemble_program",
    "induce_hypotheses",
    "render_hole_type",
    "render_hypothesis",
    "render_induction_trace",
    "render_structural_fact",
]
