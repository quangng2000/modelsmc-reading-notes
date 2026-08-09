"""Stable human-readable rendering for score artifacts and feedback."""

from __future__ import annotations

from typing import cast

from .types import RuntimeValue, StaticType


def render_value(value: RuntimeValue, value_type: StaticType) -> str:
    """Render a runtime value in the stable artifact format."""

    if value_type is StaticType.BOOL:
        return "true" if value else "false"
    if value_type is StaticType.INT:
        return str(value)
    if value_type is StaticType.BOOL_LIST:
        items = cast(list[bool], value)
        return "[" + ", ".join("true" if item else "false" for item in items) + "]"
    items_int = cast(list[int], value)
    return "[" + ", ".join(str(item) for item in items_int) + "]"
