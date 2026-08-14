"""Pydantic models for the PBE task format used by this package.

The paper examples encode integers as decimal strings so that they can be
handled without fixed-width number rounding. Python has arbitrary-precision
integers, so the boundary parser converts those strings into ``int`` values.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


class ValueType(StrEnum):
    """The closed set of first-order types supported by the PBE DSL."""

    INT = "Int"
    BOOL = "Bool"
    INT_LIST = "List<Int>"
    BOOL_LIST = "List<Bool>"


type ScalarValue = StrictInt | StrictBool
type ListValue = list[StrictInt] | list[StrictBool]
type RuntimeValue = ScalarValue | ListValue


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


def _decode_value(value: Any) -> Any:
    """Decode the repository's lossless JSON integer representation."""

    if isinstance(value, str) and _INTEGER.fullmatch(value):
        return int(value)
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def value_has_type(value: RuntimeValue, expected: ValueType) -> bool:
    """Check a runtime value without relying on ``bool`` being an ``int``."""

    if expected is ValueType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is ValueType.BOOL:
        return isinstance(value, bool)
    if not isinstance(value, list):
        return False
    if expected is ValueType.INT_LIST:
        return all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    return all(isinstance(item, bool) for item in value)


def infer_value_type(values: list[RuntimeValue]) -> ValueType:
    """Infer one consistent type from one side of a set of PBE examples.

    An all-empty list collection is intentionally ambiguous.  Such a task must
    carry an explicit signature instead of silently choosing an element type.
    """

    if not values:
        raise ValueError("cannot infer a type from zero values")

    scalar_kinds: set[ValueType] = set()
    list_kinds: set[ValueType] = set()
    saw_empty_list = False
    for value in values:
        if isinstance(value, bool):
            scalar_kinds.add(ValueType.BOOL)
        elif isinstance(value, int):
            scalar_kinds.add(ValueType.INT)
        elif isinstance(value, list):
            if not value:
                saw_empty_list = True
            elif all(isinstance(item, bool) for item in value):
                list_kinds.add(ValueType.BOOL_LIST)
            elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                list_kinds.add(ValueType.INT_LIST)
            else:
                raise ValueError("list values must contain only integers or only booleans")
        else:  # pragma: no cover - the Pydantic union normally rejects this first
            raise ValueError(f"unsupported PBE value: {value!r}")

    if scalar_kinds and (list_kinds or saw_empty_list):
        raise ValueError("examples mix scalar and list values")
    if len(scalar_kinds) > 1 or len(list_kinds) > 1:
        raise ValueError("examples do not have one consistent value type")
    if scalar_kinds:
        return next(iter(scalar_kinds))
    if list_kinds:
        return next(iter(list_kinds))
    raise ValueError("an explicit signature is required when every list is empty")


class TypeSignature(_FrozenModel):
    """Input and output types of the synthesized unary program."""

    input_type: ValueType = Field(alias="input")
    output_type: ValueType = Field(alias="output")


class Example(_FrozenModel):
    """One normalized input/output observation."""

    input_value: RuntimeValue = Field(alias="input")
    output_value: RuntimeValue = Field(alias="output")

    @field_validator("input_value", "output_value", mode="before")
    @classmethod
    def decode_lossless_integers(cls, value: Any) -> Any:
        return _decode_value(value)


class PBESpec(_FrozenModel):
    """A complete programming-by-example synthesis specification."""

    name: Annotated[str, Field(min_length=1)] = "unnamed-synthesis-task"
    signature: TypeSignature | None = None
    examples: Annotated[list[Example], Field(min_length=1)]
    integer_constants: list[StrictInt] = Field(
        default_factory=lambda: [-2, -1, 0, 1, 2], alias="integerConstants"
    )

    @field_validator("integer_constants", mode="before")
    @classmethod
    def decode_constants(cls, value: Any) -> Any:
        return _decode_value(value)

    @model_validator(mode="after")
    def infer_and_validate_signature(self) -> PBESpec:
        if not self.integer_constants:
            raise ValueError("integerConstants must contain at least one value")

        signature = self.signature
        if signature is None:
            signature = TypeSignature(
                input=infer_value_type([example.input_value for example in self.examples]),
                output=infer_value_type([example.output_value for example in self.examples]),
            )
            object.__setattr__(self, "signature", signature)

        for index, example in enumerate(self.examples, start=1):
            if not value_has_type(example.input_value, signature.input_type):
                raise ValueError(
                    f"example {index} input does not match signature type "
                    f"{signature.input_type.value}"
                )
            if not value_has_type(example.output_value, signature.output_type):
                raise ValueError(
                    f"example {index} output does not match signature type "
                    f"{signature.output_type.value}"
                )

        # Preserve user order while removing duplicates.  A stable grammar is
        # important for reproducibility across runs.
        constants = list(dict.fromkeys(self.integer_constants))
        object.__setattr__(self, "integer_constants", constants)
        return self
