"""Small state records shared by the finite guided proposal modules."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..records import FamilySupport, HoleFilling

type EventEmitter = Callable[..., None]


@dataclass(slots=True)
class ProposalPath:
    slot: int
    ancestor_state_index: int
    family: FamilySupport
    compatible_state_indices: tuple[int, ...]
    cloned: bool
    target_state_index: int | None
    previous_fillings: list[HoleFilling] = field(default_factory=list)
    log_q_construct: float = 0.0


@dataclass(frozen=True, slots=True)
class PathSeed:
    slot: int
    ancestor_state_index: int
    cloned: bool
    target_state_index: int | None


def logaddexp(left: float, right: float) -> float:
    """Stable scalar log-add-exp without introducing a tensor dependency."""

    maximum = max(left, right)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def emit(
    emitter: EventEmitter | None,
    name: str,
    *,
    message: str,
    level: str,
    **data: Any,
) -> None:
    if emitter is not None:
        emitter(name, message=message, level=level, **data)
