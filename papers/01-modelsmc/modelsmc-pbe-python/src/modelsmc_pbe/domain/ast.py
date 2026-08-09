"""Strict JSON transport for the bounded program AST.

This module deliberately performs only transport validation: node tags, exact
field names, JSON scalar shapes, depth, and node-count limits.  It does *not*
infer types, evaluate programs, calculate program cost, or enforce variable
scope. Those semantic decisions remain in the pure-Python semantic core.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from typing import Any, cast

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type AstNode = dict[str, JsonValue]
type ProgramAst = dict[str, JsonValue]

_DECIMAL_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_LEAF_KINDS = frozenset(
    {
        "Input",
        "Item",
        "Accumulator",
        "EmptyIntList",
        "EmptyBoolList",
    }
)
_BINARY_KINDS = frozenset({"Add", "Subtract", "Multiply", "LessThan", "EqualInt", "And"})
_PROGRAM_FIELDS: dict[str, tuple[str, ...]] = {
    "ExpressionProgram": ("kind", "body"),
    "MapProgram": ("kind", "mapper"),
    "FoldRightProgram": ("kind", "initial", "reducer"),
}


class AstTransportError(ValueError):
    """Raised when an object is not a bounded, canonical transport AST."""


def canonical_key(value: Mapping[str, object]) -> str:
    """Return a stable JSON key suitable for deduplication and replay logs."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise AstTransportError(f"AST is not JSON serializable: {error}") from error


def clone_program(program: ProgramAst) -> ProgramAst:
    """Clone a transport AST so particle mutations cannot alias catalog data."""

    return deepcopy(program)


def _record(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AstTransportError(f"{path} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def _exact_keys(record: Mapping[str, object], expected: Collection[str], path: str) -> None:
    actual = set(record)
    wanted = set(expected)
    if actual != wanted:
        raise AstTransportError(
            f"{path} must contain exactly {sorted(wanted)}; received {sorted(actual)}"
        )


def _literal(
    value: object,
    allowed_integer_constants: frozenset[str] | None,
    path: str,
) -> str:
    if not isinstance(value, str) or _DECIMAL_INTEGER.fullmatch(value) is None:
        raise AstTransportError(f"{path} must be a canonical decimal integer string")
    if allowed_integer_constants is not None and value not in allowed_integer_constants:
        raise AstTransportError(f"{path}={value} is not in the allowed constant catalog")
    return value


def normalize_program(
    value: object,
    *,
    allowed_integer_constants: Collection[int | str] | None = None,
    max_depth: int = 32,
    max_nodes: int = 256,
) -> ProgramAst:
    """Validate and copy a program AST without making semantic judgments.

    Integer literals use decimal strings to preserve lossless transport. The
    returned dictionaries and lists contain only fresh JSON values.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    constants = (
        None
        if allowed_integer_constants is None
        else frozenset(str(constant) for constant in allowed_integer_constants)
    )
    nodes_seen = 0

    def expression(candidate: object, depth: int, path: str) -> AstNode:
        nonlocal nodes_seen
        if depth > max_depth:
            raise AstTransportError(f"{path} exceeds maximum AST depth {max_depth}")
        record = _record(candidate, path)
        nodes_seen += 1
        if nodes_seen > max_nodes:
            raise AstTransportError(f"AST exceeds maximum node count {max_nodes}")
        kind = record.get("kind")
        if not isinstance(kind, str):
            raise AstTransportError(f"{path}.kind must be a string")

        if kind in _LEAF_KINDS:
            _exact_keys(record, ("kind",), path)
            return {"kind": kind}
        if kind == "IntLiteral":
            _exact_keys(record, ("kind", "intValue"), path)
            return {
                "kind": kind,
                "intValue": _literal(record["intValue"], constants, f"{path}.intValue"),
            }
        if kind == "BoolLiteral":
            _exact_keys(record, ("kind", "boolValue"), path)
            bool_value = record["boolValue"]
            if not isinstance(bool_value, bool):
                raise AstTransportError(f"{path}.boolValue must be a boolean")
            return {"kind": kind, "boolValue": bool_value}
        if kind in {"PrependInt", "PrependBool"}:
            _exact_keys(record, ("kind", "head", "tail"), path)
            return {
                "kind": kind,
                "head": expression(record["head"], depth + 1, f"{path}.head"),
                "tail": expression(record["tail"], depth + 1, f"{path}.tail"),
            }
        if kind == "Not":
            _exact_keys(record, ("kind", "operand"), path)
            return {
                "kind": kind,
                "operand": expression(record["operand"], depth + 1, f"{path}.operand"),
            }
        if kind in _BINARY_KINDS:
            _exact_keys(record, ("kind", "left", "right"), path)
            return {
                "kind": kind,
                "left": expression(record["left"], depth + 1, f"{path}.left"),
                "right": expression(record["right"], depth + 1, f"{path}.right"),
            }
        if kind == "IfThenElse":
            _exact_keys(record, ("kind", "condition", "thenExpr", "elseExpr"), path)
            return {
                "kind": kind,
                "condition": expression(record["condition"], depth + 1, f"{path}.condition"),
                "thenExpr": expression(record["thenExpr"], depth + 1, f"{path}.thenExpr"),
                "elseExpr": expression(record["elseExpr"], depth + 1, f"{path}.elseExpr"),
            }
        raise AstTransportError(f"{path}.kind={kind!r} is not in the transport grammar")

    root = _record(value, "program")
    nodes_seen += 1
    if nodes_seen > max_nodes:
        raise AstTransportError(f"AST exceeds maximum node count {max_nodes}")
    root_kind = root.get("kind")
    if not isinstance(root_kind, str) or root_kind not in _PROGRAM_FIELDS:
        raise AstTransportError(f"program.kind={root_kind!r} is not a complete program wrapper")
    _exact_keys(root, _PROGRAM_FIELDS[root_kind], "program")
    if root_kind == "ExpressionProgram":
        return {"kind": root_kind, "body": expression(root["body"], 2, "program.body")}
    if root_kind == "MapProgram":
        return {"kind": root_kind, "mapper": expression(root["mapper"], 2, "program.mapper")}
    return {
        "kind": root_kind,
        "initial": expression(root["initial"], 2, "program.initial"),
        "reducer": expression(root["reducer"], 2, "program.reducer"),
    }


def parse_program_json(
    content: str,
    *,
    allowed_integer_constants: Collection[int | str] | None = None,
    max_depth: int = 32,
    max_nodes: int = 256,
) -> ProgramAst:
    """Parse exactly one JSON value and normalize it as a transport program."""

    try:
        decoded: Any = json.loads(content)
    except json.JSONDecodeError as error:
        raise AstTransportError(f"program is not valid JSON: {error.msg}") from error
    return normalize_program(
        decoded,
        allowed_integer_constants=allowed_integer_constants,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
