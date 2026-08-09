"""Typed metadata for one expression hole in a program skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from modelsmc_pbe.domain.models import ValueType


@dataclass(frozen=True, slots=True)
class ExpressionScope:
    """Variables and their types at one expression-hole location."""

    input_type: ValueType
    input_available: bool = True
    item_type: ValueType | None = None
    accumulator_type: ValueType | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_type, ValueType):
            raise TypeError("input_type must be a ValueType")
        if not isinstance(self.input_available, bool):
            raise TypeError("input_available must be a boolean")
        if self.item_type is not None and not isinstance(self.item_type, ValueType):
            raise TypeError("item_type must be a ValueType or None")
        if self.accumulator_type is not None and not isinstance(self.accumulator_type, ValueType):
            raise TypeError("accumulator_type must be a ValueType or None")

    def as_prompt_record(self) -> dict[str, str]:
        """Return only variables that the hole is permitted to reference."""

        variables: dict[str, str] = {}
        if self.input_available:
            variables["Input"] = self.input_type.value
        if self.item_type is not None:
            variables["Item"] = self.item_type.value
        if self.accumulator_type is not None:
            variables["Accumulator"] = self.accumulator_type.value
        return variables


@dataclass(frozen=True, slots=True)
class HoleSpecification:
    """Expected result type and lexical scope of one expression hole."""

    expected_type: ValueType
    scope: ExpressionScope

    def __post_init__(self) -> None:
        if not isinstance(self.expected_type, ValueType):
            raise TypeError("expected_type must be a ValueType")
        if not isinstance(self.scope, ExpressionScope):
            raise TypeError("scope must be an ExpressionScope")
