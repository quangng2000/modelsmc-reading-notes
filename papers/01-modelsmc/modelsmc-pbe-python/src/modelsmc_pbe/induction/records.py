"""Immutable records for type-directed inductive generalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsmc_pbe.domain.models import TypeSignature, ValueType


class SkeletonKind(StrEnum):
    """Top-level program forms considered by the symbolic front end."""

    EXPRESSION = "expression"
    MAP = "map"
    FOLD_RIGHT = "foldr"
    FOLD_RIGHT_FILTER_MAP = "foldr-filter-map"
    FOLD_RIGHT_FILTER_PIECEWISE_MAP = "foldr-filter-piecewise-map"


class StructuralRelation(StrEnum):
    """Observable list relationships recorded before deduction."""

    LENGTH_PRESERVED = "length-preserved"
    EMPTY_INPUT_PRODUCES_EMPTY_OUTPUT = "empty-input-produces-empty-output"


@dataclass(frozen=True, slots=True)
class TypedVariable:
    """One variable visible while filling a hole."""

    name: str
    value_type: ValueType

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name must not be empty")


@dataclass(frozen=True, slots=True)
class HoleSpec:
    """The exact first-order type and scope of an unknown expression."""

    name: str
    parameters: tuple[TypedVariable, ...]
    output_type: ValueType

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hole name must not be empty")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError(f"hole {self.name!r} has duplicate parameter names")


@dataclass(frozen=True, slots=True)
class TypedSkeleton:
    """A top-level program hypothesis with typed, scoped holes."""

    kind: SkeletonKind
    input_variable: TypedVariable
    output_type: ValueType
    holes: tuple[HoleSpec, ...]

    def __post_init__(self) -> None:
        names = tuple(hole.name for hole in self.holes)
        if len(names) != len(set(names)):
            raise ValueError(f"{self.kind.value} skeleton has duplicate hole names")

    def hole(self, name: str) -> HoleSpec:
        """Look up one hole without exposing mutable indexing state."""

        for hole in self.holes:
            if hole.name == name:
                return hole
        raise KeyError(f"{self.kind.value} skeleton has no hole {name!r}")


@dataclass(frozen=True, slots=True)
class StructuralFact:
    """One aggregate relationship observed across numbered examples."""

    relation: StructuralRelation
    holds: bool
    supporting_examples: tuple[int, ...]
    contradicting_examples: tuple[int, ...]

    def __post_init__(self) -> None:
        all_indices = (*self.supporting_examples, *self.contradicting_examples)
        if any(index < 1 for index in all_indices):
            raise ValueError("example indices are one-based positive integers")
        if set(self.supporting_examples) & set(self.contradicting_examples):
            raise ValueError("an example cannot both support and contradict one fact")


@dataclass(frozen=True, slots=True)
class InductionReport:
    """All typed hypotheses and structural observations for one specification."""

    signature: TypeSignature
    facts: tuple[StructuralFact, ...]
    hypotheses: tuple[TypedSkeleton, ...]

    def hypothesis(self, kind: SkeletonKind) -> TypedSkeleton:
        """Return a generated hypothesis by kind."""

        for hypothesis in self.hypotheses:
            if hypothesis.kind is kind:
                return hypothesis
        raise KeyError(f"no {kind.value} hypothesis is viable for this signature")
