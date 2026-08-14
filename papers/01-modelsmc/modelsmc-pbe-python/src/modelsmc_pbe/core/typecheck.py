"""Static type and variable-scope checking for the bounded PBE DSL."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Node, StaticType, child, kind, list_item_type, list_type

_INT_BINARY = frozenset({"Add", "Subtract", "Multiply", "LessThan", "EqualInt"})
_ARITHMETIC = frozenset({"Add", "Subtract", "Multiply"})
_LEAVES_WITHOUT_INPUT = frozenset(
    {"Item", "Accumulator", "IntLiteral", "BoolLiteral", "EmptyIntList", "EmptyBoolList"}
)


@dataclass(frozen=True, slots=True)
class Scope:
    """Types available to scoped expression variables."""

    input_type: StaticType
    item_type: StaticType | None = None
    accumulator_type: StaticType | None = None


def infer_expression(expression: Node, scope: Scope) -> StaticType | None:
    """Infer one expression type, returning ``None`` on any type/scope error."""

    node_kind = kind(expression)
    if node_kind == "Input":
        return scope.input_type
    if node_kind == "Item":
        return scope.item_type
    if node_kind == "Accumulator":
        return scope.accumulator_type
    if node_kind == "IntLiteral":
        return StaticType.INT
    if node_kind == "BoolLiteral":
        return StaticType.BOOL
    if node_kind == "EmptyIntList":
        return StaticType.INT_LIST
    if node_kind == "EmptyBoolList":
        return StaticType.BOOL_LIST

    if node_kind in _INT_BINARY:
        operands_are_int = (
            infer_expression(child(expression, "left"), scope) is StaticType.INT
            and infer_expression(child(expression, "right"), scope) is StaticType.INT
        )
        if not operands_are_int:
            return None
        return StaticType.INT if node_kind in _ARITHMETIC else StaticType.BOOL

    if node_kind == "Not":
        operand = infer_expression(child(expression, "operand"), scope)
        return StaticType.BOOL if operand is StaticType.BOOL else None
    if node_kind == "And":
        left = infer_expression(child(expression, "left"), scope)
        right = infer_expression(child(expression, "right"), scope)
        return StaticType.BOOL if left is right is StaticType.BOOL else None
    if node_kind in {"PrependInt", "PrependBool"}:
        scalar = StaticType.INT if node_kind == "PrependInt" else StaticType.BOOL
        sequence = StaticType.INT_LIST if node_kind == "PrependInt" else StaticType.BOOL_LIST
        head = infer_expression(child(expression, "head"), scope)
        tail = infer_expression(child(expression, "tail"), scope)
        return sequence if head is scalar and tail is sequence else None

    condition = infer_expression(child(expression, "condition"), scope)
    then_type = infer_expression(child(expression, "thenExpr"), scope)
    else_type = infer_expression(child(expression, "elseExpr"), scope)
    if condition is StaticType.BOOL and then_type is not None and then_type is else_type:
        return then_type
    return None


def uses_input(expression: Node) -> bool:
    """Detect forbidden capture of the outer list by map and fold bodies."""

    node_kind = kind(expression)
    if node_kind == "Input":
        return True
    if node_kind in _LEAVES_WITHOUT_INPUT:
        return False
    if node_kind == "Not":
        return uses_input(child(expression, "operand"))
    if node_kind == "IfThenElse":
        return any(
            uses_input(child(expression, field))
            for field in ("condition", "thenExpr", "elseExpr")
        )
    is_prepend = node_kind in {"PrependInt", "PrependBool"}
    left_field = "head" if is_prepend else "left"
    right_field = "tail" if is_prepend else "right"
    return uses_input(child(expression, left_field)) or uses_input(
        child(expression, right_field)
    )


def infer_program(program: Node, input_type: StaticType) -> StaticType | None:
    """Infer a complete program's result type under its wrapper-defined scope."""

    node_kind = kind(program)
    if node_kind == "ExpressionProgram":
        return infer_expression(child(program, "body"), Scope(input_type))

    item_type = list_item_type(input_type)
    if item_type is None:
        return None
    if node_kind == "MapProgram":
        mapper = child(program, "mapper")
        if uses_input(mapper):
            return None
        mapper_type = infer_expression(mapper, Scope(input_type, item_type=item_type))
        return None if mapper_type is None else list_type(mapper_type)

    initial = child(program, "initial")
    reducer = child(program, "reducer")
    if uses_input(initial) or uses_input(reducer):
        return None
    initial_type = infer_expression(initial, Scope(input_type))
    if initial_type is None:
        return None
    reducer_type = infer_expression(
        reducer,
        Scope(input_type, item_type=item_type, accumulator_type=initial_type),
    )
    return initial_type if reducer_type is initial_type else None
