"""Build typed factorized support without constructing complete programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from modelsmc_pbe.config import SMCConfig
from modelsmc_pbe.core.cost import expression_cost
from modelsmc_pbe.core.types import Node, child, kind
from modelsmc_pbe.deduction import DeductionReport, deduce_hypotheses
from modelsmc_pbe.domain import PBESpec, canonical_key, clone_program
from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.enumeration import HoleEnumerationOptions, enumerate_hole_expressions
from modelsmc_pbe.grammar import arithmetic_expressions
from modelsmc_pbe.induction import SkeletonKind, induce_hypotheses

from .conditioning import condition_induction, conditioned_hole_catalog
from .factorized import count_cost_constrained_products
from .lazy_records import (
    FactorizedFamily,
    FactorizedHoleCatalog,
    FactorizedImportanceSupport,
)
from .proposal_distribution import deduction_mismatch_counts
from .records import (
    EmptyImportanceSupportError,
    HoleCatalogSummary,
    HoleFilling,
    ImportanceSMCOptions,
    ImportanceSupportLimitExceeded,
)

type EventEmitter = Callable[..., None]


def _emit(emit: EventEmitter | None, name: str, **data: Any) -> None:
    if emit is not None:
        emit(name, **data)


def _expression_depth(expression: AstNode) -> int:
    node = cast(Node, expression)
    node_kind = kind(node)
    if node_kind in {
        "Input",
        "Item",
        "Accumulator",
        "IntLiteral",
        "BoolLiteral",
        "EmptyIntList",
        "EmptyBoolList",
    }:
        return 1
    if node_kind == "Not":
        return 1 + _expression_depth(cast(AstNode, child(node, "operand")))
    if node_kind == "IfThenElse":
        return 1 + max(
            _expression_depth(cast(AstNode, child(node, field)))
            for field in ("condition", "thenExpr", "elseExpr")
        )
    left_field = "head" if node_kind in {"PrependInt", "PrependBool"} else "left"
    right_field = "tail" if node_kind in {"PrependInt", "PrependBool"} else "right"
    return 1 + max(
        _expression_depth(cast(AstNode, child(node, left_field))),
        _expression_depth(cast(AstNode, child(node, right_field))),
    )


def _family_shape(kind_value: SkeletonKind) -> tuple[int, int, Mapping[str, int]]:
    """Return base program cost, fixed transport nodes, and hole depth offsets."""

    if kind_value is SkeletonKind.EXPRESSION:
        return 0, 1, {"body": 1}
    if kind_value is SkeletonKind.MAP:
        return 2, 1, {"mapper": 1}
    if kind_value is SkeletonKind.FOLD_RIGHT:
        return 3, 1, {"initial": 1, "reducer": 1}
    if kind_value is SkeletonKind.FOLD_RIGHT_FILTER_MAP:
        return 8, 6, {"predicate": 2, "mapped_value": 3}
    if kind_value is SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP:
        return 8, 6, {"predicate": 2, "piecewise_mapped_value": 3}
    raise ValueError(f"unsupported factorized skeleton {kind_value.value}")


class FactorizedSupportBuilder:
    """Enumerate typed holes and count, but never materialize, their product."""

    def __init__(
        self,
        *,
        spec: PBESpec,
        smc: SMCConfig,
        options: ImportanceSMCOptions,
        emit: EventEmitter | None = None,
    ) -> None:
        self._spec = spec
        self._smc = smc
        self._options = options
        self._emit = emit

    def build(self) -> FactorizedImportanceSupport:
        induction = condition_induction(
            induce_hypotheses(self._spec),
            self._options.conditioned_skeleton,
            multi_family=self._options.multi_family,
        )
        deductions = deduce_hypotheses(self._spec, induction.hypotheses)
        excluded = self._auto_specialized_exclusions(deductions)
        families: list[FactorizedFamily] = []
        summaries: list[HoleCatalogSummary] = []
        support_states = 0

        for hypothesis_index, report in enumerate(deductions):
            if not report.viable or hypothesis_index in excluded:
                continue
            base_cost, base_nodes, depth_offsets = _family_shape(report.hypothesis.kind)
            combination_budget = min(
                self._smc.max_cost - base_cost,
                self._smc.max_nodes - base_nodes,
            )
            if combination_budget < 0:
                continue
            catalogs: list[FactorizedHoleCatalog] = []
            for hole in report.hypothesis.holes:
                expressions, generated_states, maximum_cost, source = self._hole_expressions(
                    report,
                    hole.name,
                )
                depth_offset = depth_offsets[hole.name]
                allowed = tuple(
                    expression
                    for expression in expressions
                    if depth_offset + _expression_depth(expression) <= self._smc.max_depth
                )
                if not allowed:
                    catalogs = []
                    break
                fillings = tuple(
                    HoleFilling(
                        hole_name=hole.name,
                        expression=clone_program(expression),
                        key=canonical_key(expression),
                    )
                    for expression in sorted(allowed, key=canonical_key)
                )
                costs = tuple(
                    expression_cost(cast(Node, filling.expression)) for filling in fillings
                )
                mismatches = deduction_mismatch_counts(
                    tuple(filling.expression for filling in fillings),
                    report.examples_for(hole.name),
                )
                catalogs.append(
                    FactorizedHoleCatalog(
                        hole=hole,
                        fillings=fillings,
                        costs=costs,
                        deduction_mismatches=mismatches,
                    )
                )
                summaries.append(
                    HoleCatalogSummary(
                        family=report.hypothesis.kind.value,
                        hole_name=hole.name,
                        returned_expressions=len(fillings),
                        generated_states=generated_states,
                        max_cost=maximum_cost,
                    )
                )
                _emit(
                    self._emit,
                    "importance.lazy.hole.enumerated",
                    message="typed hole catalog enumerated without complete-program assembly",
                    level="debug",
                    family=report.hypothesis.kind.value,
                    hole=hole.name,
                    expressions=len(fillings),
                    generated_states=generated_states,
                    catalog_source=source,
                )
            if len(catalogs) != len(report.hypothesis.holes):
                continue
            count = count_cost_constrained_products(
                tuple(catalog.costs for catalog in catalogs),
                combination_budget,
            )
            if count < 1:
                continue
            if support_states + count > self._options.support_limit:
                raise ImportanceSupportLimitExceeded(
                    "factorized complete construction support exceeds "
                    f"{self._options.support_limit} traces; no complete programs were "
                    "materialized"
                )
            families.append(
                FactorizedFamily(
                    hypothesis_index=hypothesis_index,
                    hypothesis=report.hypothesis,
                    deduction=report,
                    catalogs=tuple(catalogs),
                    base_cost=base_cost,
                    combination_budget=combination_budget,
                    support_count=count,
                )
            )
            support_states += count

        if not families:
            raise EmptyImportanceSupportError(
                "no factorized family has a complete program within the declared bounds"
            )
        support = FactorizedImportanceSupport(
            induction=induction,
            deductions=deductions,
            families=tuple(families),
            hole_catalogs=tuple(summaries),
            support_states=support_states,
            conditioned_skeleton=self._options.conditioned_skeleton,
            multi_family=self._options.multi_family,
        )
        _emit(
            self._emit,
            "importance.lazy.support.ready",
            message="factorized construction support counted without materializing states",
            support_states=support_states,
            viable_hypotheses=len(families),
            refuted_hypotheses=sum(not report.viable for report in deductions),
            materialized_complete_programs=0,
        )
        return support

    def _hole_expressions(
        self,
        report: DeductionReport,
        hole_name: str,
    ) -> tuple[tuple[AstNode, ...], int, int, str]:
        hole = report.hypothesis.hole(hole_name)
        conditioned = conditioned_hole_catalog(
            self._options.conditioned_skeleton,
            report.hypothesis,
            hole,
            tuple(self._spec.integer_constants),
            multi_family=self._options.multi_family,
            state_limit=self._options.hole_state_limit,
        )
        if conditioned is not None:
            return (
                conditioned.expressions,
                conditioned.generated_states,
                conditioned.max_cost,
                "conditioned-skeleton",
            )
        catalog = enumerate_hole_expressions(
            hole,
            self._spec.integer_constants,
            HoleEnumerationOptions(
                max_cost=self._options.hole_max_cost,
                state_limit=self._options.hole_state_limit,
            ),
        )
        return (
            tuple(entry.expression for entry in catalog.entries),
            catalog.generated_states,
            self._options.hole_max_cost,
            "typed-cost-enumerator",
        )

    def _auto_specialized_exclusions(
        self,
        deductions: tuple[DeductionReport, ...],
    ) -> frozenset[int]:
        if not self._options.multi_family:
            return frozenset()
        by_kind = {
            report.hypothesis.kind: (index, report)
            for index, report in enumerate(deductions)
        }
        simple = by_kind.get(SkeletonKind.FOLD_RIGHT_FILTER_MAP)
        piecewise = by_kind.get(SkeletonKind.FOLD_RIGHT_FILTER_PIECEWISE_MAP)
        if simple is None or piecewise is None:
            return frozenset()
        simple_index, simple_report = simple
        piecewise_index, piecewise_report = piecewise
        if not simple_report.viable or not piecewise_report.viable:
            return frozenset()

        constants = tuple(self._spec.integer_constants)
        mapped_examples = simple_report.examples_for("mapped_value")
        if not mapped_examples:
            arithmetic_is_adequate = True
            reason = "mapped-value evidence is underconstrained"
        else:
            arithmetic = arithmetic_expressions("Item", constants)
            mismatches = deduction_mismatch_counts(arithmetic, mapped_examples)
            arithmetic_is_adequate = any(mismatch == 0 for mismatch in mismatches)
            reason = (
                "compact arithmetic mapping fits derived evidence"
                if arithmetic_is_adequate
                else "piecewise mapping is required by derived evidence"
            )
        if 0 not in constants:
            arithmetic_is_adequate = True
            reason = "piecewise catalog requires declared constant 0"
        excluded_index = piecewise_index if arithmetic_is_adequate else simple_index
        retained_index = simple_index if arithmetic_is_adequate else piecewise_index
        _emit(
            self._emit,
            "importance.lazy.family.catalog_selected",
            message="auto mode selected one bounded specialized mapping catalog",
            level="info",
            retained_family=deductions[retained_index].hypothesis.kind.value,
            excluded_family=deductions[excluded_index].hypothesis.kind.value,
            reason=reason,
        )
        return frozenset((excluded_index,))
