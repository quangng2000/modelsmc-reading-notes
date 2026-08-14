"""Structural Occam cost for the bounded PBE language."""

from __future__ import annotations

from .types import Node, child, kind

_LEAF_KINDS = frozenset(
    {
        "Input",
        "Item",
        "Accumulator",
        "IntLiteral",
        "BoolLiteral",
        "EmptyIntList",
        "EmptyBoolList",
    }
)


def expression_cost(expression: Node) -> int:
    """Count expression AST nodes, assigning every node unit cost."""

    node_kind = kind(expression)
    if node_kind in _LEAF_KINDS:
        return 1
    if node_kind == "Not":
        return 1 + expression_cost(child(expression, "operand"))
    if node_kind == "IfThenElse":
        return 1 + sum(
            expression_cost(child(expression, field))
            for field in ("condition", "thenExpr", "elseExpr")
        )
    if node_kind in {"PrependInt", "PrependBool"}:
        return (
            1
            + expression_cost(child(expression, "head"))
            + expression_cost(child(expression, "tail"))
        )
    return (
        1
        + expression_cost(child(expression, "left"))
        + expression_cost(child(expression, "right"))
    )


def program_cost(program: Node) -> int:
    """Return the wrapper-aware structural cost of one complete program."""

    node_kind = kind(program)
    if node_kind == "ExpressionProgram":
        return expression_cost(child(program, "body"))
    if node_kind == "MapProgram":
        return 2 + expression_cost(child(program, "mapper"))
    return (
        3
        + expression_cost(child(program, "initial"))
        + expression_cost(child(program, "reducer"))
    )
