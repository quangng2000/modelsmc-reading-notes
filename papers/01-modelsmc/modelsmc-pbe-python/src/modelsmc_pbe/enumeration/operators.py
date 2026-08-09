"""Typed expression-form declarations for the bounded PBE DSL."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.domain import ValueType


@dataclass(frozen=True, slots=True)
class ExpressionOperator:
    """One AST constructor with ordered child fields and their types."""

    kind: str
    result_type: ValueType
    fields: tuple[str, ...]
    operand_types: tuple[ValueType, ...]

    def __post_init__(self) -> None:
        if len(self.fields) != len(self.operand_types):
            raise ValueError("operator fields and operand types must be aligned")


_FIXED_OPERATORS = (
    ExpressionOperator("Not", ValueType.BOOL, ("operand",), (ValueType.BOOL,)),
    ExpressionOperator(
        "Add",
        ValueType.INT,
        ("left", "right"),
        (ValueType.INT, ValueType.INT),
    ),
    ExpressionOperator(
        "Subtract",
        ValueType.INT,
        ("left", "right"),
        (ValueType.INT, ValueType.INT),
    ),
    ExpressionOperator(
        "Multiply",
        ValueType.INT,
        ("left", "right"),
        (ValueType.INT, ValueType.INT),
    ),
    ExpressionOperator(
        "LessThan",
        ValueType.BOOL,
        ("left", "right"),
        (ValueType.INT, ValueType.INT),
    ),
    ExpressionOperator(
        "EqualInt",
        ValueType.BOOL,
        ("left", "right"),
        (ValueType.INT, ValueType.INT),
    ),
    ExpressionOperator(
        "And",
        ValueType.BOOL,
        ("left", "right"),
        (ValueType.BOOL, ValueType.BOOL),
    ),
    ExpressionOperator(
        "PrependInt",
        ValueType.INT_LIST,
        ("head", "tail"),
        (ValueType.INT, ValueType.INT_LIST),
    ),
    ExpressionOperator(
        "PrependBool",
        ValueType.BOOL_LIST,
        ("head", "tail"),
        (ValueType.BOOL, ValueType.BOOL_LIST),
    ),
)


def expression_operators() -> tuple[ExpressionOperator, ...]:
    """Return every typed constructor, including polymorphic conditionals."""

    conditionals = tuple(
        ExpressionOperator(
            "IfThenElse",
            result_type,
            ("condition", "thenExpr", "elseExpr"),
            (ValueType.BOOL, result_type, result_type),
        )
        for result_type in ValueType
    )
    return (*_FIXED_OPERATORS, *conditionals)


def types_required_for(output_type: ValueType) -> frozenset[ValueType]:
    """Return types that can occur in a well-typed expression of ``output_type``."""

    if output_type in {ValueType.INT, ValueType.BOOL}:
        return frozenset((ValueType.INT, ValueType.BOOL))
    if output_type is ValueType.INT_LIST:
        return frozenset((ValueType.INT, ValueType.BOOL, ValueType.INT_LIST))
    return frozenset((ValueType.INT, ValueType.BOOL, ValueType.BOOL_LIST))
