"""Immutable evidence and refutation records produced by deduction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsmc_pbe.domain.models import RuntimeValue, ValueType, value_has_type
from modelsmc_pbe.induction import HoleSpec, TypedSkeleton

type FrozenValue = int | bool | tuple[int, ...] | tuple[bool, ...]


def _value_matches(value: FrozenValue, value_type: ValueType) -> bool:
    if value_type is ValueType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type is ValueType.BOOL:
        return isinstance(value, bool)
    if not isinstance(value, tuple):
        return False
    if value_type is ValueType.INT_LIST:
        return all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    return all(isinstance(item, bool) for item in value)


@dataclass(frozen=True, slots=True)
class TypedValue:
    """A hashable, lossless runtime value paired with its non-ambiguous type."""

    value_type: ValueType
    value: FrozenValue

    def __post_init__(self) -> None:
        if not _value_matches(self.value, self.value_type):
            raise TypeError(f"value {self.value!r} does not have type {self.value_type.value}")

    @classmethod
    def freeze(cls, value: RuntimeValue, value_type: ValueType) -> TypedValue:
        """Copy a possibly mutable PBE value into an immutable typed value."""

        if not value_has_type(value, value_type):
            raise TypeError(f"value {value!r} does not have type {value_type.value}")
        frozen: FrozenValue
        if isinstance(value, list):
            frozen = tuple(value)
        else:
            frozen = value
        return cls(value_type=value_type, value=frozen)


class DeductionFactKind(StrEnum):
    """Kinds of sound observations derived for a skeleton or hole."""

    BODY_EXAMPLES = "body-examples"
    MAP_LENGTH_PRESERVED = "map-length-preserved"
    MAPPER_EXAMPLES = "mapper-examples"
    FOLDR_INITIAL_EXAMPLES = "foldr-initial-examples"
    FOLDR_SUFFIX_EXAMPLES = "foldr-suffix-examples"
    FOLDR_FILTER_PREDICATE_EXAMPLES = "foldr-filter-predicate-examples"
    FOLDR_FILTER_MAPPED_VALUE_EXAMPLES = "foldr-filter-mapped-value-examples"
    FOLDR_FILTER_PIECEWISE_MAPPED_VALUE_EXAMPLES = (
        "foldr-filter-piecewise-mapped-value-examples"
    )
    HOLE_UNDERCONSTRAINED = "hole-underconstrained"


@dataclass(frozen=True, slots=True)
class DeductionFact:
    """A compact typed summary of one deduction step."""

    kind: DeductionFactKind
    hole: HoleSpec | None
    source_examples: tuple[int, ...]
    derived_examples: int

    def __post_init__(self) -> None:
        if any(index < 1 for index in self.source_examples):
            raise ValueError("example indices are one-based positive integers")
        if self.derived_examples < 0:
            raise ValueError("derived example count must be nonnegative")


@dataclass(frozen=True, slots=True)
class HoleExample:
    """One typed input/output requirement for a skeleton hole."""

    hole: HoleSpec
    inputs: tuple[TypedValue, ...]
    output: TypedValue
    source_examples: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.inputs) != len(self.hole.parameters):
            raise ValueError(
                f"hole {self.hole.name!r} expects {len(self.hole.parameters)} inputs; "
                f"received {len(self.inputs)}"
            )
        for parameter, value in zip(self.hole.parameters, self.inputs, strict=True):
            if parameter.value_type is not value.value_type:
                raise TypeError(
                    f"hole {self.hole.name!r} parameter {parameter.name!r} expects "
                    f"{parameter.value_type.value}; received {value.value_type.value}"
                )
        if self.output.value_type is not self.hole.output_type:
            raise TypeError(
                f"hole {self.hole.name!r} output expects {self.hole.output_type.value}; "
                f"received {self.output.value_type.value}"
            )
        if not self.source_examples or any(index < 1 for index in self.source_examples):
            raise ValueError("hole evidence requires one-based source example indices")


class RefutationKind(StrEnum):
    """Sound reasons that no completion of a hypothesis can fit the examples."""

    INCONSISTENT_SPEC = "inconsistent-spec"
    MAP_LENGTH_MISMATCH = "map-length-mismatch"
    MAP_FUNCTION_CONFLICT = "map-function-conflict"
    FOLDR_INITIAL_CONFLICT = "foldr-initial-conflict"
    FOLDR_REDUCER_CONFLICT = "foldr-reducer-conflict"
    FOLDR_FILTER_MAP_SHAPE_MISMATCH = "foldr-filter-map-shape-mismatch"
    FOLDR_FILTER_PREDICATE_CONFLICT = "foldr-filter-predicate-conflict"
    FOLDR_FILTER_MAPPED_VALUE_CONFLICT = "foldr-filter-mapped-value-conflict"


@dataclass(frozen=True, slots=True)
class Refutation:
    """A machine-readable refutation with stable human-readable detail."""

    kind: RefutationKind
    source_examples: tuple[int, ...]
    detail: str

    def __post_init__(self) -> None:
        if not self.source_examples or any(index < 1 for index in self.source_examples):
            raise ValueError("a refutation requires one-based source example indices")
        if not self.detail:
            raise ValueError("refutation detail must not be empty")


@dataclass(frozen=True, slots=True)
class DeductionReport:
    """Deductive evidence for, or a sound refutation of, one hypothesis."""

    hypothesis: TypedSkeleton
    facts: tuple[DeductionFact, ...]
    hole_examples: tuple[HoleExample, ...]
    refutation: Refutation | None = None

    @property
    def viable(self) -> bool:
        """Whether deduction found no contradiction in the observed examples."""

        return self.refutation is None

    def examples_for(self, hole_name: str) -> tuple[HoleExample, ...]:
        """Return derived examples for one known hole in stable order."""

        self.hypothesis.hole(hole_name)
        return tuple(example for example in self.hole_examples if example.hole.name == hole_name)
