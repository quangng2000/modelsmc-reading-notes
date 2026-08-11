"""Shared finite construction-trie grouping."""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import family_candidate
from .records import FamilySupport, HoleFilling, ImportanceSupport


@dataclass(frozen=True, slots=True)
class GroupedHoleChoice:
    """One canonical filling and all compatible terminal support states."""

    key: str
    filling: HoleFilling
    state_indices: tuple[int, ...]


def ordered_families(support: ImportanceSupport) -> tuple[FamilySupport, ...]:
    """Return the canonical family order used by every materialized proposal."""

    return tuple(sorted(support.families, key=family_candidate))


def grouped_hole_choices(
    support: ImportanceSupport,
    family: FamilySupport,
    compatible_state_indices: tuple[int, ...],
    *,
    hole_index: int,
) -> tuple[GroupedHoleChoice, ...]:
    """Partition the current construction subtree by one canonical filling."""

    hole_name = family.hypothesis.holes[hole_index].name
    fillings: dict[str, HoleFilling] = {}
    indices: dict[str, list[int]] = {}
    for state_index in compatible_state_indices:
        filling = support.states[state_index].filling(hole_name)
        fillings.setdefault(filling.key, filling)
        indices.setdefault(filling.key, []).append(state_index)
    return tuple(
        GroupedHoleChoice(
            key=key,
            filling=fillings[key],
            state_indices=tuple(indices[key]),
        )
        for key in sorted(fillings)
    )
