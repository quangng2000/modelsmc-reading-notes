"""Small semantic types shared by the pure-Python PBE core."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import cast

from modelsmc_pbe.domain.models import ValueType

type RuntimeValue = int | bool | list[int] | list[bool]
type Node = Mapping[str, object]


class StaticType(StrEnum):
    """Static type names stored in scorer records and run artifacts."""

    INT = "IntType"
    BOOL = "BoolType"
    INT_LIST = "IntListType"
    BOOL_LIST = "BoolListType"


def static_type(value_type: ValueType) -> StaticType:
    """Translate the public specification type into an internal static type."""

    return {
        ValueType.INT: StaticType.INT,
        ValueType.BOOL: StaticType.BOOL,
        ValueType.INT_LIST: StaticType.INT_LIST,
        ValueType.BOOL_LIST: StaticType.BOOL_LIST,
    }[value_type]


def list_item_type(value_type: StaticType) -> StaticType | None:
    """Return a list's scalar element type, or ``None`` for scalar inputs."""

    if value_type is StaticType.INT_LIST:
        return StaticType.INT
    if value_type is StaticType.BOOL_LIST:
        return StaticType.BOOL
    return None


def list_type(value_type: StaticType) -> StaticType | None:
    """Lift a scalar type to its list type."""

    if value_type is StaticType.INT:
        return StaticType.INT_LIST
    if value_type is StaticType.BOOL:
        return StaticType.BOOL_LIST
    return None


def kind(node: Node) -> str:
    """Read a normalized node tag."""

    return cast(str, node["kind"])


def child(node: Node, field: str) -> Node:
    """Read a normalized child node."""

    return cast(Node, node[field])
