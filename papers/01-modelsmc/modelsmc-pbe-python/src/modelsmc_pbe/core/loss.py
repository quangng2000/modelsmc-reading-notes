"""Bounded scalar and sequence losses used by ModelSMC potentials."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeGuard, cast

from .errors import CoreInvariantError
from .types import RuntimeValue, StaticType


def _edit_distance[Item](
    predicted: Sequence[Item],
    expected: Sequence[Item],
    substitution_cost: Callable[[Item, Item], int],
    cap: int,
) -> int:
    previous = [min(index, cap) for index in range(len(expected) + 1)]
    for left_index, left in enumerate(predicted, start=1):
        current = [min(left_index, cap)]
        for right_index, right in enumerate(expected, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost(left, right),
                    cap,
                )
            )
        previous = current
    return previous[-1]


def soft_loss(
    predicted: RuntimeValue,
    expected: RuntimeValue,
    output_type: StaticType,
    cap: int,
) -> int:
    """Compute the language's capped loss for one example."""

    if output_type is StaticType.INT:
        if _is_int(predicted) and _is_int(expected):
            return min(abs(predicted - expected), cap)
    elif output_type is StaticType.BOOL:
        if isinstance(predicted, bool) and isinstance(expected, bool):
            return int(predicted != expected)
    elif output_type is StaticType.INT_LIST:
        if isinstance(predicted, list) and isinstance(expected, list):
            left = cast(list[int], predicted)
            right = cast(list[int], expected)
            return _edit_distance(left, right, lambda a, b: min(abs(a - b), cap, 2), cap)
    elif isinstance(predicted, list) and isinstance(expected, list):
        left_bool = cast(list[bool], predicted)
        right_bool = cast(list[bool], expected)
        return _edit_distance(left_bool, right_bool, lambda a, b: int(a != b), cap)
    raise CoreInvariantError("evaluation returned a value with the wrong output type")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
