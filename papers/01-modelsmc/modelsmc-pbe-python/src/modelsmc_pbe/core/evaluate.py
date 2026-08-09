"""Deterministic interpreter for well-typed PBE programs."""

from __future__ import annotations

from typing import Final, TypeGuard, cast

from .types import Node, RuntimeValue, child, kind


class _EvalError:
    pass


_ERROR: Final = _EvalError()
type Evaluation = RuntimeValue | _EvalError


def evaluate_expression(
    expression: Node,
    input_value: RuntimeValue,
    *,
    item: RuntimeValue | _EvalError = _ERROR,
    accumulator: RuntimeValue | _EvalError = _ERROR,
) -> Evaluation:
    """Evaluate an expression, returning an internal sentinel on failure."""

    node_kind = kind(expression)
    if node_kind == "Input":
        return input_value
    if node_kind == "Item":
        return item
    if node_kind == "Accumulator":
        return accumulator
    if node_kind == "IntLiteral":
        return int(cast(str, expression["intValue"]))
    if node_kind == "BoolLiteral":
        return cast(bool, expression["boolValue"])
    if node_kind in {"EmptyIntList", "EmptyBoolList"}:
        return []

    if node_kind in {"Add", "Subtract", "Multiply", "LessThan", "EqualInt"}:
        left = evaluate_expression(
            child(expression, "left"), input_value, item=item, accumulator=accumulator
        )
        right = evaluate_expression(
            child(expression, "right"), input_value, item=item, accumulator=accumulator
        )
        if not _is_int(left) or not _is_int(right):
            return _ERROR
        if node_kind == "Add":
            return left + right
        if node_kind == "Subtract":
            return left - right
        if node_kind == "Multiply":
            return left * right
        if node_kind == "LessThan":
            return left < right
        return left == right

    if node_kind == "Not":
        operand = evaluate_expression(
            child(expression, "operand"), input_value, item=item, accumulator=accumulator
        )
        return not operand if isinstance(operand, bool) else _ERROR
    if node_kind == "And":
        left = evaluate_expression(
            child(expression, "left"), input_value, item=item, accumulator=accumulator
        )
        right = evaluate_expression(
            child(expression, "right"), input_value, item=item, accumulator=accumulator
        )
        if isinstance(left, bool) and isinstance(right, bool):
            return left and right
        return _ERROR
    if node_kind in {"PrependInt", "PrependBool"}:
        head = evaluate_expression(
            child(expression, "head"), input_value, item=item, accumulator=accumulator
        )
        tail = evaluate_expression(
            child(expression, "tail"), input_value, item=item, accumulator=accumulator
        )
        if not isinstance(tail, list):
            return _ERROR
        if node_kind == "PrependInt" and _is_int(head):
            return [head, *tail]
        if node_kind == "PrependBool" and isinstance(head, bool):
            return [head, *tail]
        return _ERROR

    condition = evaluate_expression(
        child(expression, "condition"), input_value, item=item, accumulator=accumulator
    )
    if not isinstance(condition, bool):
        return _ERROR
    branch = "thenExpr" if condition else "elseExpr"
    return evaluate_expression(
        child(expression, branch), input_value, item=item, accumulator=accumulator
    )


def evaluate_program(program: Node, input_value: RuntimeValue) -> RuntimeValue | None:
    """Evaluate a complete, already type-checked program."""

    node_kind = kind(program)
    if node_kind == "ExpressionProgram":
        result = evaluate_expression(child(program, "body"), input_value)
        return _runtime_or_none(result)
    if not isinstance(input_value, list):
        return None
    if node_kind == "MapProgram":
        mapper = child(program, "mapper")
        mapped: list[int | bool] = []
        for value in input_value:
            result = evaluate_expression(mapper, input_value, item=value)
            if isinstance(result, (_EvalError, list)):
                return None
            mapped.append(result)
        return cast(RuntimeValue, mapped)

    initial = evaluate_expression(child(program, "initial"), input_value)
    if initial is _ERROR:
        return None
    result = initial
    reducer = child(program, "reducer")
    for value in reversed(input_value):
        result = evaluate_expression(reducer, input_value, item=value, accumulator=result)
        if result is _ERROR:
            return None
    return _runtime_or_none(result)


def _runtime_or_none(value: Evaluation) -> RuntimeValue | None:
    if value is _ERROR:
        return None
    return cast(RuntimeValue, value)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
