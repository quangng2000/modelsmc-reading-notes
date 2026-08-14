"""Condition symbolic support on one explicit finite program family."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.core.cost import expression_cost
from modelsmc_pbe.domain import ValueType
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.enumeration import HoleEnumerationLimitExceeded
from modelsmc_pbe.grammar import (
    SkeletonName,
    arithmetic_expression_count,
    arithmetic_expressions,
    filter_predicate_count,
    filter_predicates,
    signed_piecewise_expression_count,
    signed_piecewise_expressions,
)
from modelsmc_pbe.induction import (
    HoleSpec,
    InductionReport,
    SkeletonKind,
    TypedSkeleton,
)


@dataclass(frozen=True, slots=True)
class ConditionedHoleCatalog:
    """One exact compact catalog supplied by a declared skeleton family."""

    expressions: tuple[AstNode, ...]
    generated_states: int
    max_cost: int


_SKELETON_KINDS: dict[SkeletonName, SkeletonKind] = {
    "expression-arithmetic": SkeletonKind.EXPRESSION,
    "map-arithmetic": SkeletonKind.MAP,
    "foldr-filter-map": SkeletonKind.FOLD_RIGHT_FILTER_MAP,
    "foldr-filter-piecewise-map": SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
}


def condition_induction(
    report: InductionReport,
    skeleton: SkeletonName | None,
    *,
    multi_family: bool = False,
) -> InductionReport:
    """Select one declared family, or retain the original generic families."""

    if multi_family and skeleton is not None:
        raise ValueError("multi-family induction cannot also select one skeleton")
    if multi_family:
        order = (
            SkeletonKind.EXPRESSION,
            SkeletonKind.MAP,
            SkeletonKind.FOLD_RIGHT_FILTER_MAP,
            SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
            SkeletonKind.FOLD_RIGHT,
        )
        available = {hypothesis.kind: hypothesis for hypothesis in report.hypotheses}
        return InductionReport(
            signature=report.signature,
            facts=report.facts,
            hypotheses=tuple(available[kind] for kind in order if kind in available),
        )
    if skeleton is None:
        hypotheses = tuple(
            hypothesis
            for hypothesis in report.hypotheses
            if hypothesis.kind
            not in {
                SkeletonKind.FOLD_RIGHT_FILTER_MAP,
                SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP,
            }
        )
        return InductionReport(
            signature=report.signature,
            facts=report.facts,
            hypotheses=hypotheses,
        )

    expected_signatures = {
        "expression-arithmetic": (ValueType.INT, ValueType.INT),
        "map-arithmetic": (ValueType.INT_LIST, ValueType.INT_LIST),
        "foldr-filter-map": (ValueType.INT_LIST, ValueType.INT_LIST),
        "foldr-filter-piecewise-map": (ValueType.INT_LIST, ValueType.INT_LIST),
    }
    expected_input, expected_output = expected_signatures[skeleton]
    if (
        report.signature.input_type is not expected_input
        or report.signature.output_type is not expected_output
    ):
        raise ValueError(
            f"conditioned skeleton {skeleton!r} requires "
            f"{expected_input.value} -> {expected_output.value}; received "
            f"{report.signature.input_type.value} -> {report.signature.output_type.value}"
        )
    kind = _SKELETON_KINDS[skeleton]
    hypothesis = report.hypothesis(kind)
    return InductionReport(
        signature=report.signature,
        facts=report.facts,
        hypotheses=(hypothesis,),
    )


def conditioned_hole_catalog(
    skeleton: SkeletonName | None,
    hypothesis: TypedSkeleton,
    hole: HoleSpec,
    integer_constants: tuple[int, ...],
    *,
    multi_family: bool = False,
    state_limit: int,
) -> ConditionedHoleCatalog | None:
    """Return a skeleton-specific catalog, leaving generic support unchanged."""

    expressions: tuple[AstNode, ...]
    expected_states: int
    maximum_cost: int
    catalog_kind: str
    if skeleton == "expression-arithmetic" and hole.name == "body":
        expected_states = arithmetic_expression_count(integer_constants)
        maximum_cost = 3
        catalog_kind = "input-arithmetic"
    elif skeleton == "map-arithmetic" and hole.name == "mapper":
        expected_states = arithmetic_expression_count(integer_constants)
        maximum_cost = 3
        catalog_kind = "item-arithmetic"
    elif (
        skeleton == "foldr-filter-map"
        or (multi_family and hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_MAP)
    ) and hole.name == "predicate":
        expected_states = filter_predicate_count(integer_constants)
        maximum_cost = 7
        catalog_kind = "filter-predicate"
    elif (
        skeleton == "foldr-filter-map"
        or (multi_family and hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_MAP)
    ) and hole.name == "mapped_value":
        expected_states = arithmetic_expression_count(integer_constants)
        maximum_cost = 3
        catalog_kind = "item-arithmetic"
    elif (
        skeleton == "foldr-filter-piecewise-map"
        or (
            multi_family
            and hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP
        )
    ) and hole.name == "predicate":
        expected_states = filter_predicate_count(integer_constants)
        maximum_cost = 7
        catalog_kind = "filter-predicate"
    elif (
        skeleton == "foldr-filter-piecewise-map"
        or (
            multi_family
            and hypothesis.kind is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP
        )
    ) and hole.name == "piecewise_mapped_value":
        expected_states = signed_piecewise_expression_count(integer_constants)
        maximum_cost = 10
        catalog_kind = "signed-piecewise"
    elif skeleton is None:
        return None
    else:  # pragma: no cover - condition_induction and hole-order invariant
        raise ValueError(
            f"conditioned skeleton {skeleton!r} has no catalog for "
            f"{hypothesis.kind.value}.?{hole.name}"
        )
    if expected_states > state_limit:
        raise HoleEnumerationLimitExceeded(
            state_limit=state_limit,
            attempted_states=expected_states,
            cost=maximum_cost,
        )
    if catalog_kind == "input-arithmetic":
        expressions = arithmetic_expressions("Input", integer_constants)
    elif catalog_kind == "item-arithmetic":
        expressions = arithmetic_expressions("Item", integer_constants)
    elif catalog_kind == "signed-piecewise":
        expressions = signed_piecewise_expressions(integer_constants)
    else:
        expressions = filter_predicates(integer_constants)
    if len(expressions) != expected_states:  # pragma: no cover - catalog invariant
        raise RuntimeError("conditioned catalog size preflight does not match construction")
    if not expressions:
        raise ValueError(
            f"conditioned skeleton {skeleton!r} produced an empty ?{hole.name} catalog"
        )
    return ConditionedHoleCatalog(
        expressions=expressions,
        generated_states=expected_states,
        max_cost=max(expression_cost(expression) for expression in expressions),
    )
