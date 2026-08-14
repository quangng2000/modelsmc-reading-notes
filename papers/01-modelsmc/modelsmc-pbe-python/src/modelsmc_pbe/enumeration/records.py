"""Immutable public records for a bounded hole-expression catalog."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.domain.ast import AstNode
from modelsmc_pbe.induction import HoleSpec


@dataclass(frozen=True, slots=True)
class HoleEnumerationOptions:
    """Safety bounds for one complete increasing-cost enumeration."""

    max_cost: int
    state_limit: int

    def __post_init__(self) -> None:
        if isinstance(self.max_cost, bool) or not isinstance(self.max_cost, int):
            raise TypeError("max_cost must be an integer")
        if isinstance(self.state_limit, bool) or not isinstance(self.state_limit, int):
            raise TypeError("state_limit must be an integer")
        if self.max_cost < 1:
            raise ValueError("max_cost must be positive")
        if self.state_limit < 1:
            raise ValueError("state_limit must be positive")


@dataclass(frozen=True, slots=True)
class EnumeratedExpression:
    """One canonical expression and its exact structural cost."""

    expression: AstNode
    cost: int
    canonical_key: str


@dataclass(frozen=True, slots=True)
class HoleExpressionCatalog:
    """All requested-output expressions plus internal DP state accounting."""

    hole: HoleSpec
    options: HoleEnumerationOptions
    entries: tuple[EnumeratedExpression, ...]
    generated_states: int

    def entries_at_cost(self, cost: int) -> tuple[EnumeratedExpression, ...]:
        """Return the exact-cost bucket without constructing a prefix."""

        return tuple(entry for entry in self.entries if entry.cost == cost)

    def expressions_at_cost(self, cost: int) -> tuple[AstNode, ...]:
        """Return only AST nodes from one exact-cost bucket."""

        return tuple(entry.expression for entry in self.entries_at_cost(cost))
