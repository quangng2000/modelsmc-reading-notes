"""Construction-path records and finite-trie grouping for the joint oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..records import FamilySupport


@dataclass(frozen=True, slots=True)
class JointPathSeed:
    slot: int
    ancestor_state_index: int
    target_state_index: int | None


@dataclass(slots=True)
class JointPath:
    slot: int
    ancestor_state_index: int
    family: FamilySupport
    compatible_state_indices: tuple[int, ...]
    target_state_index: int | None
    log_q_construct: float = 0.0


def validate_seeds(seeds: tuple[JointPathSeed, ...], *, state_count: int) -> None:
    """Reject ancestor or forced-target references outside materialized support."""

    for seed in seeds:
        if not 0 <= seed.ancestor_state_index < state_count:
            raise IndexError("ancestor state index is outside the finite support")
        if seed.target_state_index is not None and not 0 <= seed.target_state_index < state_count:
            raise IndexError("target state index is outside the finite support")


def positive_probability(
    probabilities: torch.Tensor,
    selected: int,
    label: str,
) -> float:
    """Read one finite categorical mass and enforce strict positivity."""

    probability = float(probabilities[selected].item())
    if not math.isfinite(probability) or probability <= 0.0:
        raise RuntimeError(f"joint target assigned nonpositive {label} mass")
    return probability
