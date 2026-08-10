"""Canonical finite expression catalogs shared by conditioned grammars."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from modelsmc_pbe.domain.ast import AstNode, canonical_key, clone_program

ArithmeticVariable = Literal["Input", "Item"]

_ARITHMETIC_OPERATORS = ("Add", "Subtract", "Multiply")


def stable_integer_constants(
    constants: Sequence[int] | Iterable[int],
) -> tuple[int, ...]:
    """Validate, deduplicate, and sort an integer constant catalog."""

    values = constants if isinstance(constants, Sequence) else tuple(constants)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("integer constants must contain integers, not booleans or other values")
    return tuple(sorted(set(values)))


def arithmetic_expression_count(
    constants: Sequence[int] | Iterable[int],
) -> int:
    """Return the compact scalar catalog size without constructing ASTs."""

    constant_count = len(stable_integer_constants(constants))
    return 4 + 7 * constant_count


def filter_predicate_count(
    constants: Sequence[int] | Iterable[int],
) -> int:
    """Return the atom-plus-ordered-conjunction size without constructing ASTs."""

    atom_count = 3 * len(stable_integer_constants(constants))
    return atom_count + atom_count * atom_count


def signed_transform_count(
    constants: Sequence[int] | Iterable[int],
) -> int:
    """Return the compact scalar branch catalog size.

    The catalog contains every declared constant plus identity, negation, and
    squaring.  These three reusable transforms cover the signed-window study
    case without admitting the much larger unrestricted arithmetic grammar in
    both branches of a conditional.
    """

    return len(stable_integer_constants(constants)) + 3


def signed_piecewise_expression_count(
    constants: Sequence[int] | Iterable[int],
) -> int:
    """Return the exact number of canonical signed piecewise expressions.

    Two sign orientations are available (negative/nonnegative and
    positive/nonpositive).  Equal branches are excluded because they are
    ordinary arithmetic expressions rather than genuinely piecewise ones.
    A zero literal must be declared by the specification so every constructed
    AST remains inside the declared grammar.
    """

    stable_constants = stable_integer_constants(constants)
    if 0 not in stable_constants:
        return 0
    transforms = signed_transform_count(stable_constants)
    return 2 * transforms * (transforms - 1)


def _constant(value: int) -> AstNode:
    return {"kind": "IntLiteral", "intValue": str(value)}


def _binary(kind: str, left: AstNode, right: AstNode) -> AstNode:
    return {
        "kind": kind,
        "left": clone_program(left),
        "right": clone_program(right),
    }


def _canonical(expressions: Iterable[AstNode]) -> tuple[AstNode, ...]:
    unique = {canonical_key(expression): expression for expression in expressions}
    return tuple(clone_program(unique[key]) for key in sorted(unique))


def arithmetic_expressions(
    variable: ArithmeticVariable,
    constants: Sequence[int] | Iterable[int],
) -> tuple[AstNode, ...]:
    """Return the complete compact arithmetic catalog for one scalar variable."""

    stable_constants = stable_integer_constants(constants)
    variable_node: AstNode = {"kind": variable}
    constant_nodes = [_constant(value) for value in stable_constants]
    expressions: list[AstNode] = [variable_node, *constant_nodes]
    operand_pairs = [
        (variable_node, variable_node),
        *((variable_node, constant) for constant in constant_nodes),
        *((constant, variable_node) for constant in constant_nodes),
    ]
    for operator in _ARITHMETIC_OPERATORS:
        expressions.extend(_binary(operator, left, right) for left, right in operand_pairs)
    return _canonical(expressions)


def filter_predicates(
    constants: Sequence[int] | Iterable[int],
) -> tuple[AstNode, ...]:
    """Return comparison atoms and every ordered conjunction of two atoms."""

    stable_constants = stable_integer_constants(constants)
    item: AstNode = {"kind": "Item"}
    atoms: list[AstNode] = []
    for value in stable_constants:
        constant = _constant(value)
        atoms.extend(
            (
                _binary("LessThan", item, constant),
                _binary("LessThan", constant, item),
                _binary("EqualInt", item, constant),
            )
        )
    return _canonical(
        (
            *atoms,
            *(_binary("And", left, right) for left in atoms for right in atoms),
        )
    )


def signed_transforms(
    constants: Sequence[int] | Iterable[int],
) -> tuple[AstNode, ...]:
    """Return identity, negation, squaring, and declared constants."""

    stable_constants = stable_integer_constants(constants)
    item: AstNode = {"kind": "Item"}
    zero = _constant(0)
    return _canonical(
        (
            item,
            _binary("Subtract", zero, item),
            _binary("Multiply", item, item),
            *(_constant(value) for value in stable_constants),
        )
    )


def signed_piecewise_expressions(
    constants: Sequence[int] | Iterable[int],
) -> tuple[AstNode, ...]:
    """Return the complete compact conditional-mapper catalog.

    Every expression splits at zero and chooses two distinct branch transforms.
    Keeping the conditional as one catalogued expression lets deduction score it
    against direct ``item -> output`` examples without inventing unsound branch
    labels for the unknown condition.
    """

    stable_constants = stable_integer_constants(constants)
    if 0 not in stable_constants:
        return ()
    item: AstNode = {"kind": "Item"}
    zero = _constant(0)
    conditions = (
        _binary("LessThan", item, zero),
        _binary("LessThan", zero, item),
    )
    transforms = signed_transforms(stable_constants)
    expressions: list[AstNode] = []
    for condition in conditions:
        for then_expression in transforms:
            for else_expression in transforms:
                if canonical_key(then_expression) == canonical_key(else_expression):
                    continue
                expressions.append(
                    {
                        "kind": "IfThenElse",
                        "condition": clone_program(condition),
                        "thenExpr": clone_program(then_expression),
                        "elseExpr": clone_program(else_expression),
                    }
                )
    return _canonical(expressions)
