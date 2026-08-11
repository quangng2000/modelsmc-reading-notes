"""Immutable records for the joint semantic construction law."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from modelsmc_pbe.smc import categorical_sample

from ..lazy_records import ConstructionTrace
from .slate import SemanticSlateEntry


@dataclass(frozen=True, slots=True)
class SemanticBranchDistribution:
    """One exact family or hole conditional in canonical branch order."""

    values: tuple[int, ...]
    log_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.log_probabilities):
            raise ValueError("semantic branch values and log probabilities must align")
        if len(set(self.values)) != len(self.values):
            raise ValueError("semantic branch values must be unique")
        if any(not math.isfinite(value) for value in self.log_probabilities):
            raise ValueError("defensive semantic conditionals must be finite")

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Return ordinary probabilities for inspection or categorical sampling."""

        return tuple(math.exp(value) for value in self.log_probabilities)

    def log_probability(self, value: int) -> float:
        """Return the conditional log mass of one branch value."""

        try:
            position = self.values.index(value)
        except ValueError as error:
            raise ValueError(f"{value} is not a valid semantic proposal branch") from error
        return self.log_probabilities[position]

    def sample(self, *, generator: torch.Generator) -> int:
        """Draw one branch using deterministic CPU-float64 categorical sampling."""

        position = int(
            categorical_sample(
                torch.tensor(self.probabilities, dtype=torch.float64),
                1,
                generator=generator,
            )[0].item()
        )
        return self.values[position]


@dataclass(frozen=True, slots=True)
class SemanticTracePath:
    """A complete trace and the conditionals whose product produced its mass."""

    trace: ConstructionTrace
    family_log_probability: float
    hole_log_probabilities: tuple[float, ...]
    log_probability: float


@dataclass(frozen=True, slots=True)
class SemanticSlateProbability:
    """Normalized semantic and defensive masses for one slate trace."""

    entry: SemanticSlateEntry
    log_q_a: float
    log_q: float
