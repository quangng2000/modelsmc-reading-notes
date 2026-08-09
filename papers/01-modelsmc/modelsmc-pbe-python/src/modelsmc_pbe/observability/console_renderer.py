"""Human-readable projection of structured experiment events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Console

EVENT_LEVELS = {
    "trace": 5,
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "quiet": 100,
}


@dataclass(frozen=True, slots=True)
class ConsoleEventRenderer:
    """Render event summaries while the JSONL stream remains authoritative."""

    minimum_level: str
    console: Console

    def __post_init__(self) -> None:
        if self.minimum_level not in EVENT_LEVELS:
            raise ValueError(f"unknown console level {self.minimum_level!r}")

    def render(
        self,
        *,
        level: str,
        event: str,
        message: str | None,
        data: Mapping[str, Any],
    ) -> None:
        if EVENT_LEVELS[level] < EVENT_LEVELS[self.minimum_level]:
            return
        label = "trace" if level == "trace" else level
        text = message or event
        scalar_data = [
            f"{key}={value}"
            for key, value in data.items()
            if value is None or isinstance(value, (str, int, float, bool))
        ]
        suffix = f" {' '.join(scalar_data[:8])}" if scalar_data else ""
        style = {
            "error": "bold red",
            "warning": "yellow",
            "debug": "dim cyan",
            "trace": "dim",
        }.get(level)
        self.console.print(
            f"[{label}] {text}{suffix}",
            style=style,
            highlight=False,
            markup=False,
        )
