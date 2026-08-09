"""Type-directed dynamic programming over exact structural expression cost."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import product

from modelsmc_pbe.core.cost import expression_cost
from modelsmc_pbe.domain import ValueType, canonical_key
from modelsmc_pbe.domain.ast import AstNode, clone_program
from modelsmc_pbe.induction import HoleSpec, TypedVariable

from .errors import HoleEnumerationLimitExceeded
from .operators import ExpressionOperator, expression_operators, types_required_for
from .records import EnumeratedExpression, HoleEnumerationOptions, HoleExpressionCatalog

type _Bucket = dict[str, AstNode]
type _CostTable = dict[int, dict[ValueType, _Bucket]]

_VARIABLE_KINDS = {
    "x": "Input",
    "xs": "Input",
    "item": "Item",
    "acc": "Accumulator",
}


def _stable_constants(constants: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    values = constants if isinstance(constants, Sequence) else tuple(constants)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("integer constants must contain integers, not booleans or other values")
    return tuple(sorted(set(values)))


def _scope_leaves(parameters: tuple[TypedVariable, ...]) -> tuple[tuple[ValueType, AstNode], ...]:
    leaves: list[tuple[ValueType, AstNode]] = []
    used_node_kinds: set[str] = set()
    for parameter in parameters:
        try:
            node_kind = _VARIABLE_KINDS[parameter.name]
        except KeyError as error:
            choices = ", ".join(sorted(_VARIABLE_KINDS))
            raise ValueError(
                f"hole parameter {parameter.name!r} has no DSL variable node; "
                f"expected one of: {choices}"
            ) from error
        if node_kind in used_node_kinds:
            raise ValueError(
                f"hole scope maps more than one parameter to AST variable {node_kind}"
            )
        used_node_kinds.add(node_kind)
        leaves.append((parameter.value_type, {"kind": node_kind}))
    return tuple(leaves)


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield ordered positive integer tuples summing to ``total``."""

    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for remainder in _positive_compositions(total - first, parts - 1):
            yield (first, *remainder)


def _operator_is_relevant(
    operator: ExpressionOperator,
    required_types: frozenset[ValueType],
) -> bool:
    return operator.result_type in required_types and all(
        operand_type in required_types for operand_type in operator.operand_types
    )


class _CatalogBuilder:
    """Mutable, run-local DP state hidden behind the pure public function."""

    def __init__(
        self,
        hole: HoleSpec,
        constants: tuple[int, ...],
        options: HoleEnumerationOptions,
    ) -> None:
        self.hole = hole
        self.constants = constants
        self.options = options
        self.required_types = types_required_for(hole.output_type)
        self.table: _CostTable = {}
        self.seen: set[str] = set()
        self.generated_states = 0

    def build(self) -> HoleExpressionCatalog:
        for cost in range(1, self.options.max_cost + 1):
            self.table[cost] = {
                value_type: {} for value_type in self.required_types
            }
            if cost == 1:
                self._add_leaves()
            self._add_composites(cost)

        entries: list[EnumeratedExpression] = []
        for cost in range(1, self.options.max_cost + 1):
            bucket = self.table[cost][self.hole.output_type]
            for key in sorted(bucket):
                expression = clone_program(bucket[key])
                actual_cost = expression_cost(expression)
                if actual_cost != cost:  # pragma: no cover - constructor invariant
                    raise RuntimeError(
                        f"enumerator built cost-{actual_cost} expression in cost-{cost} bucket"
                    )
                entries.append(
                    EnumeratedExpression(
                        expression=expression,
                        cost=cost,
                        canonical_key=key,
                    )
                )
        return HoleExpressionCatalog(
            hole=self.hole,
            options=self.options,
            entries=tuple(entries),
            generated_states=self.generated_states,
        )

    def _add_leaves(self) -> None:
        for value_type, expression in _scope_leaves(self.hole.parameters):
            if value_type in self.required_types:
                self._add(1, value_type, expression)
        for constant in self.constants:
            self._add(
                1,
                ValueType.INT,
                {"kind": "IntLiteral", "intValue": str(constant)},
            )
        self._add(1, ValueType.BOOL, {"kind": "BoolLiteral", "boolValue": False})
        self._add(1, ValueType.BOOL, {"kind": "BoolLiteral", "boolValue": True})
        if ValueType.INT_LIST in self.required_types:
            self._add(1, ValueType.INT_LIST, {"kind": "EmptyIntList"})
        if ValueType.BOOL_LIST in self.required_types:
            self._add(1, ValueType.BOOL_LIST, {"kind": "EmptyBoolList"})

    def _add_composites(self, cost: int) -> None:
        child_cost = cost - 1
        if child_cost < 1:
            return
        for operator in expression_operators():
            if not _operator_is_relevant(operator, self.required_types):
                continue
            for costs in _positive_compositions(child_cost, len(operator.fields)):
                pools = self._operand_pools(operator, costs)
                if pools is None:
                    continue
                for operands in product(*pools):
                    expression: AstNode = {"kind": operator.kind}
                    for field, operand in zip(operator.fields, operands, strict=True):
                        expression[field] = clone_program(operand)
                    self._add(cost, operator.result_type, expression)

    def _operand_pools(
        self,
        operator: ExpressionOperator,
        costs: tuple[int, ...],
    ) -> tuple[tuple[AstNode, ...], ...] | None:
        pools: list[tuple[AstNode, ...]] = []
        for operand_type, cost in zip(operator.operand_types, costs, strict=True):
            bucket = self.table[cost][operand_type]
            if not bucket:
                return None
            pools.append(tuple(bucket[key] for key in sorted(bucket)))
        return tuple(pools)

    def _add(self, cost: int, value_type: ValueType, expression: AstNode) -> None:
        key = canonical_key(expression)
        if key in self.seen:
            return
        attempted = self.generated_states + 1
        if attempted > self.options.state_limit:
            raise HoleEnumerationLimitExceeded(
                state_limit=self.options.state_limit,
                attempted_states=attempted,
                cost=cost,
            )
        self.seen.add(key)
        self.table[cost][value_type][key] = expression
        self.generated_states = attempted


def enumerate_hole_expressions(
    hole: HoleSpec,
    integer_constants: Sequence[int] | Iterable[int],
    options: HoleEnumerationOptions,
) -> HoleExpressionCatalog:
    """Enumerate the complete bounded expression catalog for one typed hole.

    Expressions are constructed by exact unit-node cost and returned in
    increasing cost, with canonical-key order inside each cost bucket.  The
    ``state_limit`` applies to every distinct typed DP state needed to build the
    target catalog, not merely the returned output-type expressions.
    """

    if not isinstance(hole, HoleSpec):
        raise TypeError("hole must be a HoleSpec")
    if not isinstance(options, HoleEnumerationOptions):
        raise TypeError("options must be HoleEnumerationOptions")
    return _CatalogBuilder(hole, _stable_constants(integer_constants), options).build()
