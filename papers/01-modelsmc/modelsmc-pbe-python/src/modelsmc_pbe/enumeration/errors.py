"""Explicit failures for bounded expression enumeration."""

from __future__ import annotations


class HoleEnumerationLimitExceeded(RuntimeError):
    """The complete requested catalog is larger than its safety ceiling."""

    def __init__(self, *, state_limit: int, attempted_states: int, cost: int) -> None:
        self.state_limit = state_limit
        self.attempted_states = attempted_states
        self.cost = cost
        super().__init__(
            f"hole enumeration exceeded state_limit={state_limit} while building "
            f"exact cost {cost} (attempted state {attempted_states}); raise the "
            "limit or lower max_cost"
        )
