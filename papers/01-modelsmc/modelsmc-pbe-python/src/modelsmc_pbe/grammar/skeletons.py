"""Finite syntactic catalogs for four PBE program families.

The catalogs construct JSON ASTs only.  They do not decide whether an AST is
well typed, executable, within the configured cost bound, or correct on examples;
the semantic scorer remains authoritative for all of those questions.

Every catalog is finite by construction and enumeration fails loudly if the
caller-provided state limit is too small.  This is preferable to silently
dropping support from a grammar used as a probabilistic proposal.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from modelsmc_pbe.domain.ast import AstNode, ProgramAst, canonical_key, clone_program
from modelsmc_pbe.grammar.fragments import (
    arithmetic_expressions,
    filter_predicates,
    signed_piecewise_expressions,
    stable_integer_constants,
)

SkeletonName = Literal[
    "expression-arithmetic",
    "map-arithmetic",
    "foldr-filter-map",
    "foldr-filter-piecewise-map",
]

_SKELETONS: tuple[SkeletonName, ...] = (
    "expression-arithmetic",
    "map-arithmetic",
    "foldr-filter-map",
    "foldr-filter-piecewise-map",
)


class EnumerationLimitExceeded(RuntimeError):
    """Raised rather than returning a silently truncated proposal space."""


def available_skeletons() -> tuple[SkeletonName, ...]:
    """Return names in stable CLI/display order."""

    return _SKELETONS


def _constant(value: int) -> AstNode:
    return {"kind": "IntLiteral", "intValue": str(value)}


def _leaf(kind: str) -> AstNode:
    return {"kind": kind}


def _binary(kind: str, left: AstNode, right: AstNode) -> AstNode:
    return {
        "kind": kind,
        "left": clone_program(left),
        "right": clone_program(right),
    }


def _deduplicate(programs: Iterable[ProgramAst], limit: int) -> list[ProgramAst]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("enumeration limit must be a positive integer")
    unique: dict[str, ProgramAst] = {}
    for program in programs:
        key = canonical_key(program)
        unique.setdefault(key, program)
        if len(unique) > limit:
            raise EnumerationLimitExceeded(
                f"skeleton contains more than {limit} states; raise the enumeration limit "
                "or choose a narrower skeleton"
            )
    return [unique[key] for key in sorted(unique)]


def _expression_arithmetic(constants: tuple[int, ...]) -> Iterable[ProgramAst]:
    for body in arithmetic_expressions("Input", constants):
        yield {"kind": "ExpressionProgram", "body": body}


def _map_arithmetic(constants: tuple[int, ...]) -> Iterable[ProgramAst]:
    for mapper in arithmetic_expressions("Item", constants):
        yield {"kind": "MapProgram", "mapper": mapper}


def _foldr_filter_map(constants: tuple[int, ...]) -> Iterable[ProgramAst]:
    accumulator = _leaf("Accumulator")
    for condition in filter_predicates(constants):
        for mapped_value in arithmetic_expressions("Item", constants):
            yield {
                "kind": "FoldRightProgram",
                "initial": _leaf("EmptyIntList"),
                "reducer": {
                    "kind": "IfThenElse",
                    "condition": clone_program(condition),
                    "thenExpr": {
                        "kind": "PrependInt",
                        "head": clone_program(mapped_value),
                        "tail": clone_program(accumulator),
                    },
                    "elseExpr": clone_program(accumulator),
                },
            }


def _foldr_filter_piecewise_map(constants: tuple[int, ...]) -> Iterable[ProgramAst]:
    accumulator = _leaf("Accumulator")
    for condition in filter_predicates(constants):
        for mapped_value in signed_piecewise_expressions(constants):
            yield {
                "kind": "FoldRightProgram",
                "initial": _leaf("EmptyIntList"),
                "reducer": {
                    "kind": "IfThenElse",
                    "condition": clone_program(condition),
                    "thenExpr": {
                        "kind": "PrependInt",
                        "head": clone_program(mapped_value),
                        "tail": clone_program(accumulator),
                    },
                    "elseExpr": clone_program(accumulator),
                },
            }


def bounded_square_target() -> ProgramAst:
    """Return the intended target of ``foldr-bounded-square.json``.

    The predicate is ``-2 < item && item < 3`` and the retained value is
    ``item * item``.  Both holes are members of ``foldr-filter-map`` whenever
    ``-2`` and ``3`` are in the supplied integer-constant catalog.
    """

    item = _leaf("Item")
    accumulator = _leaf("Accumulator")
    return {
        "kind": "FoldRightProgram",
        "initial": _leaf("EmptyIntList"),
        "reducer": {
            "kind": "IfThenElse",
            "condition": _binary(
                "And",
                _binary("LessThan", _constant(-2), item),
                _binary("LessThan", item, _constant(3)),
            ),
            "thenExpr": {
                "kind": "PrependInt",
                "head": _binary("Multiply", item, item),
                "tail": accumulator,
            },
            "elseExpr": _leaf("Accumulator"),
        },
    }


def signed_window_target() -> ProgramAst:
    """Return the intended target of ``foldr-signed-window.json``."""

    item = _leaf("Item")
    accumulator = _leaf("Accumulator")
    zero = _constant(0)
    mapped_value: AstNode = {
        "kind": "IfThenElse",
        "condition": _binary("LessThan", item, zero),
        "thenExpr": _binary("Subtract", zero, item),
        "elseExpr": _binary("Multiply", item, item),
    }
    return {
        "kind": "FoldRightProgram",
        "initial": _leaf("EmptyIntList"),
        "reducer": {
            "kind": "IfThenElse",
            "condition": _binary(
                "And",
                _binary("LessThan", _constant(-3), item),
                _binary("LessThan", item, _constant(3)),
            ),
            "thenExpr": {
                "kind": "PrependInt",
                "head": mapped_value,
                "tail": accumulator,
            },
            "elseExpr": _leaf("Accumulator"),
        },
    }


def enumerate_skeleton(
    name: SkeletonName | str,
    constants: Sequence[int] | Iterable[int],
    limit: int,
) -> list[ProgramAst]:
    """Enumerate one complete deterministic AST catalog.

    ``limit`` is a safety ceiling, not a request to take a prefix.  Returning a
    prefix would silently change proposal support and make probability
    accounting dependent on incidental traversal order.
    """

    stable_constants = stable_integer_constants(constants)
    if name == "expression-arithmetic":
        generated = _expression_arithmetic(stable_constants)
    elif name == "map-arithmetic":
        generated = _map_arithmetic(stable_constants)
    elif name == "foldr-filter-map":
        generated = _foldr_filter_map(stable_constants)
    elif name == "foldr-filter-piecewise-map":
        generated = _foldr_filter_piecewise_map(stable_constants)
    else:
        choices = ", ".join(_SKELETONS)
        raise ValueError(f"unknown skeleton {name!r}; expected one of: {choices}")
    return _deduplicate(generated, limit)
